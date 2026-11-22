"""Train the experiment's best configuration once, for the 'Screen an upload' page.

Best combination from the report: ResNet18 transfer learning + class-weighted loss + threshold tuned on
validation data, trained at a 10:1 real-to-AI ratio. To run on a laptop CPU in a few minutes, the
ImageNet backbone is kept frozen and only the new output layer is trained (same as the live demo).

Usage (from the live_app folder):
    python train_best_model.py "C:/path/to/ipcv_subset.zip"      # or the unzipped ipcv_subset folder

Output: best_model/head.pt, best_model/meta.json, best_model/test_scores.npz
"""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

import live_core as core

RATIO = 10                 # real : AI in training, as in the report's best result
N_REAL = 5000              # training pool of real images
SEED = 42
OUT = Path(__file__).parent / "best_model"
EXT = {".png", ".jpg", ".jpeg", ".webp"}


class Source:
    """Reads images from the exported zip or from the unzipped folder (train/real, train/fake, test/...)."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.suffix.lower() == ".zip" else None
        names = self.zip.namelist() if self.zip else [str(p.relative_to(self.path)).replace("\\", "/")
                                                      for p in self.path.rglob("*") if p.is_file()]
        self.names = sorted(n for n in names if Path(n).suffix.lower() in EXT)

    def list(self, split: str, cls: str):
        return [n for n in self.names if [p.lower() for p in Path(n).parts[-3:-1]] == [split, cls]]

    def load(self, name: str) -> np.ndarray:
        data = self.zip.read(name) if self.zip else (self.path / name).read_bytes()
        return core.square_resize(Image.open(io.BytesIO(data)))


def embed(bb, src, names, label):
    out = []
    for i in range(0, len(names), 64):
        batch = np.stack([src.load(n) for n in names[i:i + 64]])
        out.append(core.resnet_embed(bb, batch))
        print(f"\r  {label}: {min(i + 64, len(names))}/{len(names)}", end="", flush=True)
    print()
    return np.concatenate(out)


def train_head(E, y, epochs=60):
    torch.manual_seed(SEED)
    head = nn.Linear(E.shape[1], 2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(core.class_weights(y), dtype=torch.float32))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    X, Y = torch.from_numpy(E), torch.from_numpy(y).long()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(head(X[idx]), Y[idx]).backward()
            opt.step()
    return head.eval()


@torch.no_grad()
def scores(head, E):
    return torch.softmax(head(torch.from_numpy(E)), 1)[:, 1].numpy()


def export_feed(src, names, y, s, thr, n_real=36, n_ai=12):
    """Save a random sample of TEST images (never used in training) as the app's simulated feed,
    plus the model's real mistakes for the 'complaint' chapter. No cherry-picking: the feed sample is
    drawn with a fixed seed; mistakes are the most confident errors on the whole test set."""
    feed_dir = OUT / "feed"
    feed_dir.mkdir(parents=True, exist_ok=True)
    for old in feed_dir.glob("*.png"):
        old.unlink()
    rng = np.random.default_rng(SEED)
    real_i, ai_i = np.where(y == 0)[0], np.where(y == 1)[0]
    pick = list(rng.choice(real_i, min(n_real, len(real_i)), replace=False)) + \
        list(rng.choice(ai_i, min(n_ai, len(ai_i)), replace=False))
    false_flags = [i for i in real_i[np.argsort(-s[real_i])] if s[i] >= thr][:4]
    missed = [i for i in ai_i[np.argsort(s[ai_i])] if s[i] < thr][:4]
    items = []
    for role, idx in (("feed", pick), ("false_flag", false_flags), ("missed", missed)):
        for i in idx:
            fname = f"{role}_{len(items):03d}.png"
            Image.fromarray(src.load(names[i])).save(feed_dir / fname)
            items.append({"file": fname, "role": role, "label": int(y[i]), "score": float(s[i])})
    (OUT / "feed.json").write_text(json.dumps(items, indent=1))
    print(f"Feed: {len(pick)} posts, {len(false_flags)} false flags, {len(missed)} missed fakes -> {feed_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="ipcv_subset.zip or the unzipped folder")
    args = ap.parse_args()
    t0 = time.time()
    src = Source(args.source)
    rng = np.random.default_rng(SEED)
    real = list(rng.permutation(src.list("train", "real")))[:N_REAL]
    fake = list(rng.permutation(src.list("train", "fake")))[: N_REAL // RATIO]
    te_real, te_fake = src.list("test", "real"), src.list("test", "fake")
    print(f"Training pool: {len(real)} real + {len(fake)} AI ({RATIO}:1). Test: {len(te_real)} real + {len(te_fake)} AI")
    if not (real and fake and te_real and te_fake):
        raise SystemExit("Could not find train/real, train/fake, test/real and test/fake inside the source.")

    print("Loading ResNet18 ImageNet weights…")
    bb = core.load_resnet18_backbone()
    E_pool = np.concatenate([embed(bb, src, real, "train real"), embed(bb, src, fake, "train AI")])
    y_pool = np.array([0] * len(real) + [1] * len(fake))
    E_te = np.concatenate([embed(bb, src, te_real, "test real"), embed(bb, src, te_fake, "test AI")])
    y_te = np.array([0] * len(te_real) + [1] * len(te_fake))

    tr, va = train_test_split(np.arange(len(y_pool)), test_size=0.15, stratify=y_pool, random_state=SEED)
    print("Training the weighted output layer…")
    head = train_head(E_pool[tr], y_pool[tr])
    s_va = scores(head, E_pool[va])
    p, r, t = precision_recall_curve(y_pool[va], s_va)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    thr = float(t[np.argmax(f1[:-1])])

    s_te = scores(head, E_te)
    m = core.evaluate(y_te, s_te, thr)
    fpr = m["cm"][0, 1] / max(m["cm"][0].sum(), 1)
    meta = {"model": "ResNet18 (frozen ImageNet backbone) + class-weighted output layer",
            "train_ratio": f"{RATIO}:1", "n_train": [int((y_pool[tr] == 0).sum()), int((y_pool[tr] == 1).sum())],
            "threshold": thr, "test_recall": m["ai_recall"], "test_precision": m["ai_precision"],
            "test_fpr": float(fpr), "test_macro_f1": m["macro_f1"], "test_mcc": float(m["mcc"]),
            "test_accuracy": m["accuracy"], "test_pr_auc": float(average_precision_score(y_te, s_te)),
            "confusion_matrix": m["cm"].tolist(), "minutes": round((time.time() - t0) / 60, 1)}
    OUT.mkdir(exist_ok=True)
    torch.save(head.state_dict(), OUT / "head.pt")
    np.savez(OUT / "test_scores.npz", scores=s_te, labels=y_te)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    export_feed(src, te_real + te_fake, y_te, s_te, thr)
    print(json.dumps(meta, indent=2))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
