"""Download the four selected PlantVillage leaf-condition classes."""

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


REPOSITORY = "https://github.com/spMohanty/PlantVillage-Dataset.git"
SOURCE_CLASSES = {
    "Bacterial_spot": "Tomato___Bacterial_spot",
    "Early_blight": "Tomato___Early_blight",
    "Late_blight": "Tomato___Late_blight",
    "healthy": "Tomato___healthy",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"


def image_hash(path):
    digest = hashlib.md5()
    with path.open("rb") as image_file:
        for block in iter(lambda: image_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def main():
    """Fetch source folders sparsely, preserving local images and labels."""
    with tempfile.TemporaryDirectory(prefix="cropai-plantvillage-") as temporary:
        source_root = Path(temporary) / "PlantVillage-Dataset"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             REPOSITORY, str(source_root)],
            check=True,
        )
        sparse_paths = [f"raw/color/{source_name}" for source_name in SOURCE_CLASSES.values()]
        subprocess.run(
            ["git", "-C", str(source_root), "sparse-checkout", "set", *sparse_paths],
            check=True,
        )

        added = 0
        duplicates = 0
        for class_name, source_name in SOURCE_CLASSES.items():
            source_class = source_root / "raw" / "color" / source_name
            target_class = DATASET_ROOT / class_name
            target_class.mkdir(parents=True, exist_ok=True)
            known_hashes = {
                image_hash(path)
                for path in target_class.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }

            for source_image in source_class.iterdir():
                if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                digest = image_hash(source_image)
                if digest in known_hashes:
                    duplicates += 1
                    continue

                target_image = target_class / f"plantvillage_{source_image.name}"
                shutil.copy2(source_image, target_image)
                known_hashes.add(digest)
                added += 1

    print(f"Added {added:,} unique PlantVillage images; skipped {duplicates:,} duplicates.")
    print("Review dataset/README.md for source details, then run: python train.py")


if __name__ == "__main__":
    main()
