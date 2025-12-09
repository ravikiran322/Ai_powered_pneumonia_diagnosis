import argparse
import csv
from pathlib import Path
from typing import List, Tuple
import random


VALID_AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def gather_files(root: Path) -> List[Tuple[str, str]]:
	rows = []
	# Expect class-named subfolders
	for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
		label = class_dir.name
		for p in class_dir.rglob("*"):
			if p.suffix.lower() in VALID_AUDIO:
				rows.append((str(p.resolve()), label))
	return rows


def write_csv(rows: List[Tuple[str, str]], out_csv: Path) -> None:
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(["filepath", "label"])
		for r in rows:
			w.writerow(r)
	print(f"Wrote {out_csv} with {len(rows)} items")


def main():
	parser = argparse.ArgumentParser(description="Index audio dataset into CSVs."
	)
	parser.add_argument("--source", required=True, help="Path containing class subfolders, or train/val[/test] folders")
	parser.add_argument("--out_dir", default="data/audio", help="Output directory for CSV files")
	parser.add_argument("--val_ratio", type=float, default=0.2, help="Used only if no explicit val folder is provided")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	source = Path(args.source)
	random.seed(args.seed)

	train_dir = source / "train"
	val_dir = source / "val"
	test_dir = source / "test"

	if train_dir.exists() and val_dir.exists():
		train_rows = gather_files(train_dir)
		val_rows = gather_files(val_dir)
		write_csv(train_rows, Path(args.out_dir) / "train.csv")
		write_csv(val_rows, Path(args.out_dir) / "val.csv")
		if test_dir.exists():
			test_rows = gather_files(test_dir)
			write_csv(test_rows, Path(args.out_dir) / "test.csv")
		return

	# Otherwise, split from a single folder with class subfolders
	rows = gather_files(source)
	random.shuffle(rows)
	n = len(rows)
	n_val = int(n * args.val_ratio)
	val_rows = rows[:n_val]
	train_rows = rows[n_val:]
	write_csv(train_rows, Path(args.out_dir) / "train.csv")
	write_csv(val_rows, Path(args.out_dir) / "val.csv")


if __name__ == "__main__":
	main()

