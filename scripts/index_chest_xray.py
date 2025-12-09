import argparse
import os
import csv
from pathlib import Path


def list_images(root: Path):
	valid_ext = {".png", ".jpg", ".jpeg", ".bmp"}
	for p in root.rglob("*"):
		if p.suffix.lower() in valid_ext:
			yield p


def write_split(split_dir: Path, out_csv: Path):
	rows = [("filepath", "label")]
	for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
		label = class_dir.name
		for img_path in list_images(class_dir):
			rows.append((str(img_path.resolve()), label))
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerows(rows)
	print(f"Wrote {out_csv} with {len(rows)-1} items")


def main():
	parser = argparse.ArgumentParser(description="Index chest_xray dataset into CSVs.")
	parser.add_argument("--source", required=True, help="Path to chest_xray folder (contains train/val/test)")
	parser.add_argument("--out_dir", default="data/images", help="Output directory for CSV files")
	args = parser.parse_args()

	source = Path(args.source)
	assert (source / "train").exists(), "train folder not found under source"
	assert (source / "val").exists(), "val folder not found under source"
	train_dir = source / "train"
	val_dir = source / "val"
	test_dir = source / "test" if (source / "test").exists() else None

	out_root = Path(args.out_dir)
	write_split(train_dir, out_root / "train.csv")
	write_split(val_dir, out_root / "val.csv")
	if test_dir is not None:
		write_split(test_dir, out_root / "test.csv")


if __name__ == "__main__":
	main()

