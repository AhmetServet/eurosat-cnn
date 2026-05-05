"""Generate stratified train/val/test CSV manifests from the EuroSAT dataset.

Produces three CSV files (splits/train.csv, splits/val.csv, splits/test.csv)
with columns: index, filename, label, className.
"""

import csv
import json
import random
import sys
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_label_map(label_map_path: str) -> dict:
    with open(label_map_path) as f:
        return json.load(f)


def collect_samples(dataset_root: Path, label_map: dict) -> list[dict]:
    samples = []
    seen = set()
    for class_name, label in label_map.items():
        class_dir = dataset_root / class_name
        if not class_dir.is_dir():
            print(f"  [WARN] Directory not found: {class_dir}")
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            rel_path = str(img_path).replace("\\", "/")
            if rel_path in seen:
                continue
            seen.add(rel_path)
            samples.append(
                {
                    "filename": rel_path,
                    "label": label,
                    "className": class_name,
                }
            )
    return samples


def split_and_save(
    samples: list[dict],
    ratios: dict,
    seed: int,
    output_dir: Path,
) -> None:
    random.seed(seed)

    filepaths = [s["filename"] for s in samples]
    labels = [s["label"] for s in samples]
    class_names = [s["className"] for s in samples]

    # First split: train vs temp (val+test)
    train_paths, temp_paths, train_labels, temp_labels, train_classes, temp_classes = (
        train_test_split(
            filepaths,
            labels,
            class_names,
            test_size=ratios["val"] + ratios["test"],
            stratify=labels,
            random_state=seed,
        )
    )

    # Second split: val vs test (from temp)
    val_ratio = ratios["val"] / (ratios["val"] + ratios["test"])
    val_paths, test_paths, val_labels, test_labels, val_classes, test_classes = (
        train_test_split(
            temp_paths,
            temp_labels,
            temp_classes,
            test_size=1.0 - val_ratio,
            stratify=temp_labels,
            random_state=seed,
        )
    )

    splits = {
        "train": (train_paths, train_labels, train_classes),
        "val": (val_paths, val_labels, val_classes),
        "test": (test_paths, test_labels, test_classes),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "filename", "label", "className"]

    for split_name, (paths, lbls, clss) in splits.items():
        out_path = output_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx, (p, l, c) in enumerate(zip(paths, lbls, clss)):
                writer.writerow(
                    {"index": idx, "filename": p, "label": l, "className": c}
                )
        print(f"  ✓ {out_path} — {len(paths)} samples")

    # Summary
    print(f"\n  Split summary: train={len(train_paths)}, val={len(val_paths)}, "
          f"test={len(test_paths)}, total={len(samples)}")

    # Verify stratification
    total = len(train_paths) + len(val_paths) + len(test_paths)
    print(f"  Verification: {total} total (expected {len(samples)})")


def main():
    config = load_config()
    dataset_root = Path(config["dataset"]["root"])
    label_map = load_label_map(config["dataset"]["label_map"])
    ratios = config["splits"]["ratios"]
    seed = config["splits"]["random_seed"]
    output_dir = Path(config["splits"]["dir"])

    print(f"Collecting samples from {dataset_root.resolve()} ...")
    samples = collect_samples(dataset_root, label_map)
    print(f"  Found {len(samples)} images across {len(label_map)} classes\n")

    print("Performing stratified split ...")
    split_and_save(samples, ratios, seed, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
