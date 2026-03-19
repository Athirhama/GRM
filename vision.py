import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from data3 import ShapeNetPart
from model2 import DGCNN_PartSeg


# INPUT VARIABLES

MODEL_PATH   = "checkpoints/best_model.pth"
NUM_POINTS   = 2048
K            = 20
OUT_DIR      = "viz_output"
SAMPLES_PER_CLASS = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ELEV         = 20
AZIMS        = [0, 90, 200]    # rotation angles
CELL_SIZE    = 5


# PREDICTION AND EVALUATION FUNCTIONS

def load_model(path: str, k: int, device: torch.device) -> DGCNN_PartSeg:
    """loads the trained model"""
    model = DGCNN_PartSeg(k=k).to(device)
    state = torch.load(path, map_location=device)
    if all(key.startswith("module.") for key in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def get_n_per_class(dataset: ShapeNetPart, n: int) -> dict:
    """returns n samples per class as {cat_id: [idx0, idx1, ...]}"""
    seen = {i: [] for i in range(len(dataset.categories))}
    for idx in range(len(dataset)):
        _, cat_id, _ = dataset[idx]
        if isinstance(cat_id, (np.ndarray,)):
            cat_id = int(cat_id)
        if len(seen[cat_id]) < n:
            seen[cat_id].append(idx)
        if all(len(v) >= n for v in seen.values()):
            break
    return seen


def predict(model: DGCNN_PartSeg, data_np: np.ndarray, cat_id: int, num_cat: int, device: torch.device) -> np.ndarray:
    """computes predictions for a given point cloud and category"""
    x = torch.tensor(data_np).unsqueeze(0).to(device)
    one_hot = torch.zeros(1, num_cat, device=device)
    one_hot[0, cat_id] = 1.0
    with torch.no_grad():
        logits = model(x, one_hot)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def compute_instance_iou(pred: np.ndarray, gt: np.ndarray, valid_labels: list) -> float:
    """computes the mean IoU across all parts present in either prediction or ground truth"""
    part_ious = []
    for part in valid_labels:
        I = np.sum((pred == part) & (gt == part))
        U = np.sum((pred == part) | (gt == part))
        part_ious.append(1.0 if U == 0 else I / float(U))
    return float(np.mean(part_ious))


# VISUALIZATION FUNCTIONS

def scatter_ax(ax: plt.Axes, pts: np.ndarray, seg: np.ndarray, part_to_color: dict, azim: int) -> None:
    """
    plots a single 3D scatter of the point cloud colored by part labels,
    with a given azimuth angle, making sure there is no distortion
    """
    ax.set_facecolor('white')
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')
    ax.set_axis_off()

    colors = np.array([part_to_color[p] for p in seg])

    # we inverted y and z to get them as we are used to see
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
               c=colors, s=3, alpha=0.95, depthshade=False)

    ranges = pts.max(axis=0) - pts.min(axis=0)
    max_range = ranges.max() / 2
    mid = pts.min(axis=0) + ranges / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.view_init(elev=ELEV, azim=azim)


def plot_and_save(cat_name: str, pts: np.ndarray, gt: np.ndarray, pred: np.ndarray, iou: float, out_path: str) -> None:
    """
    creates a figure with 2 rows: GT and Prediction; and len(AZIMS) columns for each of the different angles
    """
    n_cols = len(AZIMS)
    n_rows = 2

    fig_w = CELL_SIZE * n_cols
    fig_h = CELL_SIZE * n_rows + 0.6

    all_parts = np.union1d(np.unique(gt), np.unique(pred))
    cmap = cm.get_cmap('tab20', max(len(all_parts), 1))
    part_to_color = {p: cmap(i) for i, p in enumerate(sorted(all_parts))}

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
    fig.suptitle(
        f"{cat_name}  -  Instance IoU : {iou:.4f}",
        fontsize=12, fontweight='bold', color='#111111',
        y=1.0 - 0.18 / fig_h
    )

    for row, (seg, row_label) in enumerate([(gt, 'Ground Truth'), (pred, 'Prediction')]):
        for col, azim in enumerate(AZIMS):
            ax = fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1,
                                 projection='3d')
            scatter_ax(ax, pts, seg, part_to_color, azim)

            # column title
            if row == 0:
                ax.set_title(f"{azim}°", fontsize=9, color='#444444', pad=2)

            # row title
            if col == 0:
                ax.text2D(0.1, 1.08, row_label,
                          transform=ax.transAxes,
                          fontsize=10, fontweight='bold',
                          color='#111111', va='bottom', ha='left')

    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.0,
                        wspace=0.0, hspace=0.05)

    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"device: {DEVICE}")

    dataset      = ShapeNetPart(NUM_POINTS, 'test')
    cls_to_label = dataset.get_seg_mapping()
    categories   = dataset.categories
    num_cat      = len(categories)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH, K, DEVICE)

    class_to_indices = get_n_per_class(dataset, SAMPLES_PER_CLASS)

    for cat_id in sorted(class_to_indices.keys()):
        cat_name     = categories[cat_id]
        valid_labels = cls_to_label[cat_id]
        indices      = class_to_indices[cat_id]

        for sample_idx, ds_idx in enumerate(indices):
            data, _, seg = dataset[ds_idx]

            data_np = data if isinstance(data, np.ndarray) else data.numpy()
            seg_np  = seg  if isinstance(seg,  np.ndarray) else seg.numpy()
            pts     = data_np.T

            pred = predict(model, data_np, cat_id, num_cat, DEVICE)
            iou  = compute_instance_iou(pred, seg_np, valid_labels)

            out_path = os.path.join(OUT_DIR, f"{cat_id:02d}_{cat_name}_{sample_idx + 1}.png")
            plot_and_save(cat_name, pts, seg_np, pred, iou, out_path)
            print(f"[{cat_id:02d}] {cat_name:<20} #{sample_idx+1}  IoU={iou:.4f}")


if __name__ == "__main__":
    main()