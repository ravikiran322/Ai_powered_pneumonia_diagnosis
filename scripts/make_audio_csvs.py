#!/usr/bin/env python3
"""
Generate train/val CSVs for AudioCsvDataset.
Scans a directory where each subdirectory is a class label containing audio files.
Outputs CSVs with columns: filepath,label (absolute paths).

Example usage (PowerShell):
    python ./ml/scripts/make_audio_csvs.py --src_dir C:/data/audio_by_class --out_dir ml/data/audio --val_frac 0.2

The generated files will be: {out_dir}/train.csv and {out_dir}/val.csv
"""
import argparse
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def collect_files(src_dir):
    items = []
    for label in sorted(next(os.walk(src_dir))[1]):
        labdir = os.path.join(src_dir, label)
        for ext in AUDIO_EXTS:
            for p in glob.glob(os.path.join(labdir, f"**/*{ext}"), recursive=True):
                items.append((os.path.abspath(p), label))
    return items


def write_csv(items, out_path):
    df = pd.DataFrame(items, columns=["filepath", "label"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Root dir where each subfolder is a class label with audio files")
    parser.add_argument("--out_dir", default="ml/data/audio", help="Directory to write train/val CSVs")
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isdir(args.src_dir):
        raise SystemExit(f"src_dir not found: {args.src_dir}")

    items = collect_files(args.src_dir)
    if not items:
        raise SystemExit(f"No audio files found under {args.src_dir} (expected class subfolders with audio files)")

    filepaths, labels = zip(*items)
    train_fp, val_fp, train_lbl, val_lbl = train_test_split(
        filepaths, labels, test_size=args.val_frac, random_state=args.seed, stratify=labels
    )

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_csv = os.path.join(out_dir, "train.csv")
    val_csv = os.path.join(out_dir, "val.csv")

    write_csv(list(zip(train_fp, train_lbl)), train_csv)
    write_csv(list(zip(val_fp, val_lbl)), val_csv)

    print(f"Wrote {len(train_fp)} train rows -> {train_csv}")
    print(f"Wrote {len(val_fp)} val rows -> {val_csv}")


if __name__ == "__main__":
    main()
