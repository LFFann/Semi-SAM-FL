import csv
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


TRAIN_PATTERN = re.compile(
    r"iteration (?P<iteration>\d+) :\s+sam_loss : (?P<sam_loss>[-+0-9.eE]+)\s+sam_lr_ :\s+(?P<sam_lr>[-+0-9.eE]+)\s+"
    r"SGDL_loss : (?P<sgdl_loss>[-+0-9.eE]+)\s+UNet_VNet_loss : (?P<unet_vnet_loss>[-+0-9.eE]+)\s+"
    r"fusion_loss : (?P<fusion_loss>[-+0-9.eE]+)\s+UNet_lr_ :\s+(?P<unet_lr>[-+0-9.eE]+)"
)

VAL_PATTERN = re.compile(
    r"iteration (?P<iteration>\d+) :\s+sam_mean_dice : (?P<sam_mean_dice>[-+0-9.eE]+)\s+"
    r"SGDL_mean_dice : (?P<sgdl_mean_dice>[-+0-9.eE]+)\s+unet_mean_dice : (?P<unet_mean_dice>[-+0-9.eE]+)\s+"
    r"vnet_mean_dice : (?P<vnet_mean_dice>[-+0-9.eE]+)"
    r"(?:\s+sam_val_loss : (?P<sam_val_loss>[-+0-9.eE]+)\s+"
    r"sgdl_val_loss : (?P<sgdl_val_loss>[-+0-9.eE]+)\s+"
    r"unet_val_loss : (?P<unet_val_loss>[-+0-9.eE]+)\s+"
    r"vnet_val_loss : (?P<vnet_val_loss>[-+0-9.eE]+)\s+"
    r"fusion_val_loss : (?P<fusion_val_loss>[-+0-9.eE]+))?"
)


def _float_or_none(value):
    if value is None:
        return None
    if isinstance(value, float):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_valid_number(value):
    numeric = _float_or_none(value)
    return numeric is not None and not math.isnan(numeric)


def _sort_value(record, key, default_value):
    numeric = _float_or_none(record.get(key))
    if numeric is None or math.isnan(numeric):
        return default_value
    return numeric


def _sanitize_records(records):
    sanitized = []
    for record in records:
        item = {}
        for key, value in record.items():
            if isinstance(value, (int, float)) or value is None:
                item[key] = value
            else:
                try:
                    item[key] = float(value)
                except (TypeError, ValueError):
                    item[key] = value
        sanitized.append(item)
    return sanitized


def _write_csv(csv_path: Path, records):
    records = _sanitize_records(records)
    if not records:
        return
    fieldnames = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _moving_average(values, window=7):
    averaged = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = [value for value in values[start:index + 1] if value is not None and not math.isnan(value)]
        averaged.append(sum(window_values) / len(window_values) if window_values else None)
    return averaged


def _configure_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#D0D0D0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
    })


def _plot_train_curves(save_dir: Path, train_records):
    if not train_records:
        return
    _configure_plot_style()
    iterations = [record["iteration"] for record in train_records]
    metric_specs = [
        ("sam_loss", "SAM Loss", "#1f77b4"),
        ("sgdl_loss", "SGDL Loss", "#d62728"),
        ("unet_vnet_loss", "UNet/VNet Loss", "#2ca02c"),
        ("fusion_loss", "Fusion Loss", "#9467bd"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)

    for key, label, color in metric_specs:
        values = [_float_or_none(record.get(key)) for record in train_records]
        smooth_values = _moving_average(values, window=7)
        axes[0].plot(iterations, values, color=color, alpha=0.25)
        axes[0].plot(iterations, smooth_values, color=color, label=label)
    axes[0].set_title("Training Loss Curves")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(iterations, [_float_or_none(record.get("sam_lr")) for record in train_records],
                 color="#4c4c4c", label="SAM LR")
    axes[1].plot(iterations, [_float_or_none(record.get("unet_lr")) for record in train_records],
                 color="#bcbd22", label="SGDL LR")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Learning Rate")
    axes[1].legend(frameon=False)

    fig.savefig(save_dir / "train_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_val_curves(save_dir: Path, val_records):
    if not val_records:
        return
    _configure_plot_style()
    iterations = [record["iteration"] for record in val_records]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    metric_specs = [
        ("sam_mean_dice", "SAM Dice", "#1f77b4"),
        ("sgdl_mean_dice", "SGDL Dice", "#d62728"),
        ("unet_mean_dice", "UNet Dice", "#2ca02c"),
        ("vnet_mean_dice", "VNet Dice", "#9467bd"),
    ]
    for key, label, color in metric_specs:
        values = [_float_or_none(record.get(key)) for record in val_records]
        ax.plot(iterations, values, marker="o", markersize=4.5, color=color, label=label)
    ax.set_title("Validation Dice Curves")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Dice")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(save_dir / "val_dice_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_val_loss_curves(save_dir: Path, val_records):
    if not val_records:
        return
    metric_specs = [
        ("sam_val_loss", "SAM Val Loss", "#1f77b4"),
        ("sgdl_val_loss", "SGDL Val Loss", "#d62728"),
        ("unet_val_loss", "UNet Val Loss", "#2ca02c"),
        ("vnet_val_loss", "VNet Val Loss", "#9467bd"),
        ("fusion_val_loss", "Fusion Val Loss", "#8c564b"),
    ]
    has_values = any(
        any(_float_or_none(record.get(key)) is not None for record in val_records)
        for key, _, _ in metric_specs
    )
    if not has_values:
        return
    _configure_plot_style()
    iterations = [record["iteration"] for record in val_records]
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for key, label, color in metric_specs:
        values = [_float_or_none(record.get(key)) for record in val_records]
        if not any(value is not None for value in values):
            continue
        ax.plot(iterations, values, marker="o", markersize=4.5, color=color, label=label)
    ax.set_title("Validation Loss Curves")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(save_dir / "val_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_train_val_loss_curves(save_dir: Path, train_records, val_records):
    if not train_records or not val_records:
        return
    _configure_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    paired_specs = [
        ("sam_loss", "sam_val_loss", "SAM", "#1f77b4"),
        ("sgdl_loss", "sgdl_val_loss", "SGDL", "#d62728"),
    ]
    for axis, (train_key, val_key, title, color) in zip(axes, paired_specs):
        train_iterations = [record["iteration"] for record in train_records]
        train_values = [_float_or_none(record.get(train_key)) for record in train_records]
        val_iterations = [record["iteration"] for record in val_records]
        val_values = [_float_or_none(record.get(val_key)) for record in val_records]
        if not any(value is not None for value in val_values):
            axis.axis("off")
            continue
        axis.plot(train_iterations, _moving_average(train_values, window=7), color=color, label=f"{title} Train")
        axis.plot(val_iterations, val_values, marker="o", markersize=4.5, linestyle="--",
                  color="#2f2f2f", label=f"{title} Val")
        axis.set_title(f"{title} Train-vs-Val Loss")
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Loss")
        axis.legend(frameon=False)
    fig.savefig(save_dir / "train_val_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(save_dir: Path, train_records, val_records):
    if not train_records and not val_records:
        return
    _configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    if train_records:
        iterations = [record["iteration"] for record in train_records]
        sgdl_loss = [_float_or_none(record.get("sgdl_loss")) for record in train_records]
        sam_loss = [_float_or_none(record.get("sam_loss")) for record in train_records]
        axes[0].plot(iterations, _moving_average(sgdl_loss, window=7), color="#d62728", label="SGDL Loss")
        axes[0].plot(iterations, _moving_average(sam_loss, window=7), color="#1f77b4", label="SAM Loss")
        axes[0].set_title("Optimization Overview")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Loss")
        axes[0].legend(frameon=False)
    else:
        axes[0].axis("off")

    if val_records:
        val_iterations = [record["iteration"] for record in val_records]
        axes[1].plot(val_iterations, [_float_or_none(record.get("sgdl_mean_dice")) for record in val_records],
                     marker="o", color="#d62728", label="SGDL Dice")
        axes[1].plot(val_iterations, [_float_or_none(record.get("sam_mean_dice")) for record in val_records],
                     marker="o", color="#1f77b4", label="SAM Dice")
        axes[1].set_title("Validation Performance")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Dice")
        axes[1].set_ylim(bottom=0.0)
        axes[1].legend(frameon=False)
    else:
        axes[1].axis("off")

    val_loss_specs = [
        ("sam_val_loss", "SAM Val Loss", "#1f77b4"),
        ("sgdl_val_loss", "SGDL Val Loss", "#d62728"),
    ]
    if val_records and any(any(_float_or_none(record.get(key)) is not None for record in val_records)
                           for key, _, _ in val_loss_specs):
        val_iterations = [record["iteration"] for record in val_records]
        for key, label, color in val_loss_specs:
            values = [_float_or_none(record.get(key)) for record in val_records]
            if any(value is not None for value in values):
                axes[2].plot(val_iterations, values, marker="o", color=color, label=label)
        axes[2].set_title("Validation Loss")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Loss")
        axes[2].legend(frameon=False)
    else:
        axes[2].axis("off")

    fig.savefig(save_dir / "training_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_training_artifacts(snapshot_path, train_records, val_records, args_dict=None):
    save_dir = Path(snapshot_path) / "monitor"
    save_dir.mkdir(parents=True, exist_ok=True)
    train_records = _sanitize_records(train_records)
    val_records = _sanitize_records(val_records)

    _write_csv(save_dir / "train_metrics.csv", train_records)
    _write_csv(save_dir / "val_metrics.csv", val_records)
    _plot_train_curves(save_dir, train_records)
    _plot_val_curves(save_dir, val_records)
    _plot_val_loss_curves(save_dir, val_records)
    _plot_train_val_loss_curves(save_dir, train_records, val_records)
    _plot_overview(save_dir, train_records, val_records)

    best_sgdl = None
    if val_records:
        best_sgdl = max(val_records, key=lambda item: _float_or_none(item.get("sgdl_mean_dice")) or float("-inf"))

    summary = {
        "args": args_dict or {},
        "num_train_points": len(train_records),
        "num_val_points": len(val_records),
        "best_val_sgdl": best_sgdl,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_dir


def _plot_test_metric_curves(save_dir: Path, case_records):
    if not case_records:
        return
    _configure_plot_style()
    indices = [int(record["index"]) for record in case_records]
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(indices, [_float_or_none(record.get("dice")) for record in case_records],
                 marker="o", color="#1f77b4", label="Dice")
    axes[0].plot(indices, [_float_or_none(record.get("iou")) for record in case_records],
                 marker="s", color="#2ca02c", label="IoU")
    axes[0].set_title("Per-Case Segmentation Accuracy")
    axes[0].set_xlabel("Case Index")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(bottom=0.0)
    axes[0].legend(frameon=False)

    hd95_values = [_float_or_none(record.get("hd95")) for record in case_records]
    axes[1].plot(indices, hd95_values, marker="^", color="#d62728", label="HD95")
    axes[1].set_title("Per-Case Boundary Error")
    axes[1].set_xlabel("Case Index")
    axes[1].set_ylabel("HD95")
    axes[1].legend(frameon=False)
    fig.savefig(save_dir / "test_metric_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_test_metric_distribution(save_dir: Path, case_records):
    if not case_records:
        return
    _configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    specs = [
        ("dice", "Dice Distribution", "#1f77b4"),
        ("iou", "IoU Distribution", "#2ca02c"),
        ("hd95", "HD95 Distribution", "#d62728"),
    ]
    for axis, (key, title, color) in zip(axes, specs):
        values = [_float_or_none(record.get(key)) for record in case_records]
        values = [value for value in values if value is not None and not math.isnan(value)]
        if not values:
            axis.axis("off")
            continue
        axis.hist(values, bins=min(10, max(3, len(values))), color=color, alpha=0.85, edgecolor="white")
        axis.axvline(sum(values) / len(values), color="#2f2f2f", linestyle="--", linewidth=1.2, label="Mean")
        axis.set_title(title)
        axis.set_xlabel(key.upper() if key != "hd95" else "HD95")
        axis.set_ylabel("Count")
        axis.legend(frameon=False)
    fig.savefig(save_dir / "test_metric_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_image(image_path: Path, grayscale=False):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), flag)
    if image is None:
        return None
    if grayscale:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _select_visual_cases(case_records, limit=6):
    if len(case_records) <= limit:
        return case_records
    valid_records = [record for record in case_records if _is_valid_number(record.get("dice"))]
    if len(valid_records) <= limit:
        return valid_records[:limit]
    ordered = sorted(valid_records, key=lambda item: _sort_value(item, "dice", float("-inf")))
    worst = ordered[: limit // 2]
    best = ordered[-(limit - len(worst)):]
    selected = worst + best
    deduped = []
    seen = set()
    for record in selected:
        case_name = record.get("case_name")
        if case_name in seen:
            continue
        seen.add(case_name)
        deduped.append(record)
    return deduped[:limit]


def _plot_case_visual_summary(save_dir: Path, prediction_dir: Path, case_records):
    selected_cases = _select_visual_cases(case_records, limit=6)
    if not selected_cases:
        return
    _configure_plot_style()
    fig, axes = plt.subplots(len(selected_cases), 4, figsize=(14, 2.8 * len(selected_cases)), constrained_layout=True)
    if len(selected_cases) == 1:
        axes = np.array([axes])
    column_titles = ["Original", "Prediction", "Ground Truth", "Overlay"]
    for col_index, title in enumerate(column_titles):
        axes[0, col_index].set_title(title)

    for row_index, record in enumerate(selected_cases):
        case_name = record["case_name"]
        original = _load_image(prediction_dir / "original" / case_name, grayscale=False)
        pred_mask = _load_image(prediction_dir / "pred_mask" / case_name, grayscale=True)
        gt_mask = _load_image(prediction_dir / "gt_mask" / case_name, grayscale=True)
        overlay = _load_image(prediction_dir / "overlay" / case_name, grayscale=False)
        panel_images = [original, pred_mask, gt_mask, overlay]
        for col_index, image in enumerate(panel_images):
            axis = axes[row_index, col_index]
            if image is None:
                axis.text(0.5, 0.5, "Missing", ha="center", va="center")
                axis.set_axis_off()
                continue
            if image.ndim == 2:
                axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            else:
                axis.imshow(image)
            axis.set_axis_off()
        metrics_text = f"{case_name}\nDice={record['dice']:.3f}  IoU={record['iou']:.3f}  HD95={record['hd95']:.3f}"
        axes[row_index, 0].text(0.02, -0.12, metrics_text, transform=axes[row_index, 0].transAxes,
                                ha="left", va="top", fontsize=10)
    fig.savefig(save_dir / "case_visual_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_artifacts(prediction_dir, case_records, summary, metadata=None):
    prediction_dir = Path(prediction_dir)
    save_dir = prediction_dir / "monitor"
    save_dir.mkdir(parents=True, exist_ok=True)
    case_records = _sanitize_records(case_records)

    _write_csv(save_dir / "case_metrics.csv", case_records)
    _plot_test_metric_curves(save_dir, case_records)
    _plot_test_metric_distribution(save_dir, case_records)
    _plot_case_visual_summary(save_dir, prediction_dir, case_records)

    payload = {
        "summary": summary,
        "metadata": metadata or {},
        "num_cases": len(case_records),
        "best_case_by_dice": max(case_records, key=lambda item: _sort_value(item, "dice", float("-inf")))
        if case_records else None,
        "worst_case_by_dice": min(case_records, key=lambda item: _sort_value(item, "dice", float("inf")))
        if case_records else None,
    }
    (save_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return save_dir


def parse_log_to_records(log_path):
    train_records = []
    val_records = []
    log_text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    for line in log_text.splitlines():
        train_match = TRAIN_PATTERN.search(line)
        if train_match:
            record = {"iteration": int(train_match.group("iteration"))}
            for key in ("sam_loss", "sam_lr", "sgdl_loss", "unet_vnet_loss", "fusion_loss", "unet_lr"):
                record[key] = float(train_match.group(key))
            train_records.append(record)
            continue
        val_match = VAL_PATTERN.search(line)
        if val_match:
            record = {"iteration": int(val_match.group("iteration"))}
            for key in (
                "sam_mean_dice", "sgdl_mean_dice", "unet_mean_dice", "vnet_mean_dice",
                "sam_val_loss", "sgdl_val_loss", "unet_val_loss", "vnet_val_loss", "fusion_val_loss"
            ):
                value = val_match.groupdict().get(key)
                record[key] = float(value) if value is not None else None
            val_records.append(record)
    return train_records, val_records
