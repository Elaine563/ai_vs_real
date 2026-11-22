"""Core logic for the live 'teachable machine' version of the AI-vs-Real project.

Same preprocessing, features and model families as the Colab experiment:
  * classical: SVM (RBF, C=10) and Random Forest (300 trees) on 1,846 hand-crafted features
  * neural:    Small CNN trained from scratch, ResNet18 with ImageNet weights
Everything is sized to train on a laptop CPU in well under a minute.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMG = 128          # same size as the report
CNN_IMG = 64       # Small CNN input on CPU (keeps live training fast)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
torch.set_num_threads(max(1, torch.get_num_threads()))

MODEL_INFO = {
    "svm": ("SVM", "classical"),
    "rf": ("Random Forest", "classical"),
    "smallcnn": ("Small CNN", "neural"),
    "resnet18": ("ResNet18", "neural"),
}


# ----------------------------------------------------------------------------- preprocessing
def square_resize(img: Image.Image | np.ndarray, size: int = IMG) -> np.ndarray:
    """RGB, centre-crop to a square on the shorter side, bilinear resize -> uint8 HxWx3."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    w, h = img.size
    m = min(w, h)
    left, top = (w - m) // 2, (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def to_tensor(batch: np.ndarray, size: int | None = None) -> torch.Tensor:
    """uint8 NHWC -> ImageNet-normalised float NCHW (optionally resized)."""
    x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float() / 255.0
    if size and x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


# ----------------------------------------------------------------------------- features
class FeatureExtractor:
    """Colour, LBP, GLCM, HOG and FFT descriptors -> 1,846 numbers at 128 px."""

    def color_hist(self, img):
        hsv = rgb2hsv(img)
        return np.concatenate([np.histogram(hsv[..., c], bins=16, range=(0, 1), density=True)[0]
                               for c in range(3)])

    def lbp(self, g8):
        codes = local_binary_pattern(g8, 8, 1, method="uniform")
        return np.histogram(codes, bins=10, range=(0, 10), density=True)[0]

    def glcm(self, g8):
        q = (g8 // 8).astype(np.uint8)
        m = graycomatrix(q, [1, 2], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32,
                         symmetric=True, normed=True)
        return np.nan_to_num(np.concatenate(
            [graycoprops(m, p).mean(axis=1) for p in ("contrast", "homogeneity", "energy", "correlation")]))

    def hog(self, gf):
        return hog(gf, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")

    def fft(self, gf):
        mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gf))))
        yy, xx = np.indices(mag.shape)
        r = np.hypot(yy - mag.shape[0] / 2, xx - mag.shape[1] / 2)
        bins = np.minimum((r / r.max() * 16).astype(int), 15)
        return np.bincount(bins.ravel(), mag.ravel(), 16) / np.bincount(bins.ravel(), minlength=16)

    def one(self, img):
        gf = rgb2gray(img)
        g8 = (gf * 255).astype(np.uint8)
        return np.concatenate([self.color_hist(img), self.lbp(g8), self.glcm(g8), self.hog(gf), self.fft(gf)])

    def __call__(self, images):
        return np.stack([self.one(im) for im in images]).astype(np.float32)


# ----------------------------------------------------------------------------- networks
class SmallCNN(nn.Module):
    """Same architecture as the report: 3 blocks (32/64/128), GAP, dropout 0.3, 2 outputs."""

    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.features(x))


def load_resnet18_backbone() -> nn.Module:
    """ImageNet ResNet18 with the classifier removed (outputs 512 numbers per image)."""
    import torchvision
    from pathlib import Path
    local = Path(__file__).parent / "resnet18-f37072fd.pth"     # optional manual download
    if local.exists():
        net = torchvision.models.resnet18()
        net.load_state_dict(torch.load(local, map_location="cpu"))
    else:
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Identity()
    return net.eval()


@torch.no_grad()
def resnet_embed(backbone: nn.Module, images: np.ndarray, batch: int = 64) -> np.ndarray:
    out = [backbone(to_tensor(images[i:i + batch])).numpy() for i in range(0, len(images), batch)]
    return np.concatenate(out).astype(np.float32)


def class_weights(y: np.ndarray) -> np.ndarray:
    """n / (2 x count) for classes 0 and 1, the same formula as the report."""
    counts = np.bincount(y, minlength=2).astype(float)
    return len(y) / (2 * np.maximum(counts, 1))


# ----------------------------------------------------------------------------- one trained model
@dataclass
class Trained:
    key: str
    fix: bool                      # True = class weights / weighted loss, False = no handling
    predict: callable = field(repr=False)    # uint8 images (N,128,128,3) -> AI score in [0, 1]
    seconds: float = 0.0
    threshold: float = 0.5


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_classical(key, Xtr, ytr, fix, extractor, seed=42):
    t0 = time.time()
    scaler = StandardScaler().fit(Xtr)
    cw = "balanced" if fix else None
    if key == "svm":
        m = SVC(kernel="rbf", C=10, gamma="scale", class_weight=cw, random_state=seed)
        m.fit(scaler.transform(Xtr), ytr)
        score = lambda imgs: _sigmoid(m.decision_function(scaler.transform(extractor(imgs))))
    else:
        m = RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=seed)
        m.fit(scaler.transform(Xtr), ytr)
        score = lambda imgs: m.predict_proba(scaler.transform(extractor(imgs)))[:, 1]
    return Trained(key, fix, score, time.time() - t0)


def train_resnet_head(Etr, ytr, fix, backbone, epochs=60, seed=42):
    """Transfer learning on CPU: frozen ImageNet ResNet18 + a new 2-output layer."""
    t0 = time.time()
    torch.manual_seed(seed)
    head = nn.Linear(Etr.shape[1], 2)
    w = torch.tensor(class_weights(ytr), dtype=torch.float32) if fix else None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    X, Y = torch.from_numpy(Etr), torch.from_numpy(ytr).long()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss_fn(head(X[idx]), Y[idx]).backward()
            opt.step()
    head.eval()

    @torch.no_grad()
    def score(imgs):
        return torch.softmax(head(torch.from_numpy(resnet_embed(backbone, imgs))), 1)[:, 1].numpy()

    return Trained("resnet18", fix, score, time.time() - t0)


def train_smallcnn(imgs, ytr, fix, epochs=10, seed=42, progress=None):
    """Small CNN from scratch at 64 px with horizontal flips (as in the report, scaled down)."""
    t0 = time.time()
    torch.manual_seed(seed)
    net = SmallCNN()
    w = torch.tensor(class_weights(ytr), dtype=torch.float32) if fix else None
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    X = to_tensor(imgs, CNN_IMG)
    Y = torch.from_numpy(ytr).long()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            xb = X[idx]
            flip = torch.rand(len(xb)) < 0.5
            xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(-1), xb)
            opt.zero_grad()
            loss_fn(net(xb), Y[idx]).backward()
            opt.step()
        if progress:
            progress((ep + 1) / epochs)
    net.eval()

    @torch.no_grad()
    def score(imgs):
        return torch.softmax(net(to_tensor(imgs, CNN_IMG)), 1)[:, 1].numpy()

    return Trained("smallcnn", fix, score, time.time() - t0)


# ----------------------------------------------------------------------------- metrics
def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1) if (tp + fp) else 0.0
    f1_ai = 2 * precision * recall / max(precision + recall, 1e-12)
    rec_real = tn / max(tn + fp, 1)
    prec_real = tn / max(tn + fn, 1) if (tn + fn) else 0.0
    f1_real = 2 * prec_real * rec_real / max(prec_real + rec_real, 1e-12)
    return {
        "accuracy": (tp + tn) / len(y_true),
        "ai_recall": recall,
        "ai_precision": precision,
        "macro_f1": (f1_ai + f1_real) / 2,
        "mcc": matthews_corrcoef(y_true, pred) if len(set(pred)) > 1 else 0.0,
        "cm": np.array([[tn, fp], [fn, tp]]),
    }


# ----------------------------------------------------------------------------- Grad-CAM for the screening page
def gradcam(backbone: nn.Module, head: nn.Module, img: np.ndarray) -> np.ndarray:
    """Heatmap (HxW in [0, 1]) of where ResNet18 looked when pushing the score towards 'AI'."""
    for m in backbone.modules():                      # in-place ReLU breaks gradient capture
        if isinstance(m, nn.ReLU):
            m.inplace = False
    store = {}

    def keep(module, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = backbone.layer4[-1].register_forward_hook(keep)
    x = to_tensor(img[None]).requires_grad_(True)
    head(backbone(x))[:, 1].sum().backward()
    h.remove()
    w = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().numpy()


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap (warm = high attention) over the image."""
    import matplotlib.cm as cm
    heat = (cm.inferno(cam)[..., :3] * 255).astype(np.float32)
    return np.clip((1 - alpha) * img + alpha * heat, 0, 255).astype(np.uint8)
