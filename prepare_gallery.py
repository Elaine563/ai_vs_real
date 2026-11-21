"""Fill ai_gallery/ with AI-generated images from the subset you exported for the assignment.

Usage:
    python prepare_gallery.py path/to/ipcv_subset.zip          # the zip from Step1 (Kaggle)
    python prepare_gallery.py path/to/unzipped_folder --n 200   # or the unzipped folder

It copies N random images (default 150) from the */fake/ folders, already centre-cropped and resized
to 128 x 128, so the app loads quickly. Seed 42, so the same images are picked every time.
"""
import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image

EXT = {".png", ".jpg", ".jpeg", ".webp"}
OUT = Path(__file__).parent / "ai_gallery"


def is_fake(path: str) -> bool:
    parts = [p.lower() for p in Path(path).parts]
    return "fake" in parts and Path(path).suffix.lower() in EXT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="ipcv_subset.zip or the unzipped folder")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()
    src = Path(args.source)
    OUT.mkdir(exist_ok=True)

    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            names = sorted(n for n in z.namelist() if is_fake(n))
            random.Random(42).shuffle(names)
            for i, n in enumerate(names[: args.n]):
                Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((128, 128)).save(OUT / f"ai_{i:03d}.png")
    else:
        names = sorted(str(p) for p in src.rglob("*") if is_fake(str(p.relative_to(src))))
        random.Random(42).shuffle(names)
        for i, n in enumerate(names[: args.n]):
            Image.open(n).convert("RGB").resize((128, 128)).save(OUT / f"ai_{i:03d}.png")
    print(f"Copied {min(len(names), args.n)} of {len(names)} AI images into {OUT}")


if __name__ == "__main__":
    main()
