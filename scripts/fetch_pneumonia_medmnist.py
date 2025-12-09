import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def save_split(ds, out_csv: Path, out_img_dir: Path) -> int:
	out_img_dir.mkdir(parents=True, exist_ok=True)
	rows = [("filepath", "label")]
	for i in range(len(ds)):
		img, label = ds[i]
		# img: PIL Image or ndarray; ensure 3-channel PNG
		if isinstance(img, np.ndarray):
			img = Image.fromarray(img)
		if img.mode != "RGB":
			img = img.convert("RGB")
		img_path = out_img_dir / f"{i:06d}.png"
		img.save(img_path)
		# label can be array-like; take int
		if isinstance(label, (list, tuple, np.ndarray)):
			label = int(label[0])
		rows.append((str(img_path.resolve()), str(label)))
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerows(rows)
	return len(rows) - 1


def main():
	parser = argparse.ArgumentParser(description="Fetch PneumoniaMNIST and export to images + CSVs")
	parser.add_argument("--out_dir", default="data/images", help="Output directory")
	args = parser.parse_args()

	from medmnist import PneumoniaMNIST
	from medmnist import INFO

	info = INFO["pneumoniamnist"]
	label_names = info["label"]  # {"0":"normal","1":"pneumonia"}
	print(f"Labels: {label_names}")

	train_ds = PneumoniaMNIST(split="train", download=True, transform=None, as_rgb=True)
	val_ds = PneumoniaMNIST(split="val", download=True, transform=None, as_rgb=True)
	test_ds = PneumoniaMNIST(split="test", download=True, transform=None, as_rgb=True)

	out_root = Path(args.out_dir)
	n_train = save_split(train_ds, out_root / "train.csv", out_root / "files_train")
	n_val = save_split(val_ds, out_root / "val.csv", out_root / "files_val")
	n_test = save_split(test_ds, out_root / "test.csv", out_root / "files_test")

	print(f"Exported: train={n_train}, val={n_val}, test={n_test}")
	print("Note: CSV labels are 0/1. Map: 0=normal, 1=pneumonia")


if __name__ == "__main__":
	main()

