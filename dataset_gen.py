import os
import shutil
import json
import numpy as np
import kagglehub
from tqdm import tqdm

DATA_ROOT = "./data"
RAW_DATA = os.path.join(DATA_ROOT, "raw")
BIN_DATA = os.path.join(DATA_ROOT, "bin")

os.makedirs(DATA_ROOT, exist_ok=True)

download_path = kagglehub.dataset_download("majdouline20/shapenetpart-dataset")

if not os.path.exists(RAW_DATA):
    shutil.copytree(download_path, RAW_DATA)

RAW_DATA = os.path.join(RAW_DATA, "PartAnnotation")

for cat in os.listdir(RAW_DATA):
    cat_path = os.path.join(RAW_DATA, cat)
    if not os.path.isdir(cat_path):
        continue

    pts_dir = os.path.join(cat_path, "points")
    label_base_dir = os.path.join(cat_path, "points_label")
    if not os.path.exists(label_base_dir):
        continue

    os.makedirs(os.path.join(BIN_DATA, cat, "points"), exist_ok=True)
    os.makedirs(os.path.join(BIN_DATA, cat, "points_label"), exist_ok=True)

    part_folders = [
        d for d in os.listdir(label_base_dir)
        if os.path.isdir(os.path.join(label_base_dir, d))
    ]
    part_to_id = {name: i + 1 for i, name in enumerate(part_folders)}

    with open(os.path.join(BIN_DATA, cat, "part_classes.json"), "w") as f:
        json.dump(part_to_id, f, indent=4)

    if not os.path.exists(pts_dir):
        continue

    for filename in tqdm(
        [f for f in os.listdir(pts_dir) if f.endswith(".pts")],
        desc=cat,
    ):
        base_name = filename.replace(".pts", "")
        pts = np.genfromtxt(os.path.join(pts_dir, filename)).astype(np.float32)
        merged_labels = np.zeros(pts.shape[0], dtype=np.int64)
        found_any_label = False

        for part_name, part_id in part_to_id.items():
            seg_path = os.path.join(label_base_dir, part_name, base_name + ".seg")
            if not os.path.exists(seg_path):
                continue
            found_any_label = True
            lbl = np.genfromtxt(seg_path).astype(np.int64)
            merged_labels[lbl == 1] = part_id

        if found_any_label:
            np.save(os.path.join(BIN_DATA, cat, "points", base_name + ".npy"), pts)
            np.save(os.path.join(BIN_DATA, cat, "points_label", base_name + ".npy"), merged_labels)