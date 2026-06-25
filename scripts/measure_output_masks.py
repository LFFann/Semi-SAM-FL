import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.lateral_fissure_measurement import (
    annotate_longitudinal_fissure_measurement,
    annotate_lateral_fissure_measurement,
    longitudinal_measurement_to_row,
    locate_third_ventricle_apex,
    measure_longitudinal_fissure,
    measure_lateral_fissure,
    measurement_to_row,
    parse_pixel_spacing,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MEASUREMENT_OUTPUTS = {
    "lateral_opening_width": {
        "folder": "外侧裂开口宽度",
        "target": "lateral",
        "draw": ("width",),
        "suffix": "_opening_width.png",
    },
    "lateral_max_depth": {
        "folder": "外侧裂最大深度",
        "target": "lateral",
        "draw": ("depth",),
        "suffix": "_max_depth.png",
    },
    "lateral_curvature": {
        "folder": "裂隙弯曲度",
        "target": "lateral",
        "draw": ("curvature",),
        "suffix": "_curvature.png",
    },
    "lateral_angle": {
        "folder": "角度",
        "target": "lateral",
        "draw": ("angle",),
        "suffix": "_angle.png",
    },
    "longitudinal_branch_max_depth": {
        "folder": "纵裂分支最大深度",
        "target": "longitudinal",
        "draw": ("branch_depth",),
        "suffix": "_longitudinal_branch_max_depth.png",
    },
    "longitudinal_full_length": {
        "folder": "纵裂全长",
        "target": "longitudinal",
        "draw": ("full_length",),
        "suffix": "_longitudinal_full_length.png",
    },
    "longitudinal_area": {
        "folder": "纵裂面积",
        "target": "longitudinal",
        "draw": ("area",),
        "suffix": "_longitudinal_area.png",
    },
}
COMMON_TABLE_COLUMNS = [
    "index",
    "mask_path",
    "source_image_path",
    "status",
]
METRIC_TABLE_COLUMNS = {
    "lateral_opening_width": COMMON_TABLE_COLUMNS + [
        "lateral_opening_width_image_overlay_path",
        "lateral_opening_width_mask_overlay_path",
        "fissure_measurement_status",
        "fissure_component_count",
        "fissure_measured_component_count",
        "fissure_width_px",
        "fissure_mean_width_px",
        "fissure_opening_width_measurement_status",
        "fissure_opening_width_mode",
        "fissure_opening_width_yellow_dashed_length_px",
        "fissure_opening_width_segment_a_px",
        "fissure_opening_width_segment_b_px",
        "fissure_left_width_px",
        "fissure_left_opening_width_measurement_status",
        "fissure_left_opening_width_mode",
        "fissure_left_opening_width_yellow_dashed_length_px",
        "fissure_left_opening_width_segment_a_px",
        "fissure_left_opening_width_segment_b_px",
        "fissure_right_width_px",
        "fissure_right_opening_width_measurement_status",
        "fissure_right_opening_width_mode",
        "fissure_right_opening_width_yellow_dashed_length_px",
        "fissure_right_opening_width_segment_a_px",
        "fissure_right_opening_width_segment_b_px",
    ],
    "lateral_max_depth": COMMON_TABLE_COLUMNS + [
        "lateral_max_depth_image_overlay_path",
        "lateral_max_depth_mask_overlay_path",
        "fissure_measurement_status",
        "fissure_component_count",
        "fissure_measured_component_count",
        "fissure_depth_px",
        "fissure_depth_measurement_status",
        "fissure_depth_y",
        "fissure_depth_point_count",
        "fissure_depth_smooth_px",
        "fissure_left_depth_px",
        "fissure_left_depth_measurement_status",
        "fissure_left_depth_y",
        "fissure_left_depth_point_count",
        "fissure_left_depth_smooth_px",
        "fissure_right_depth_px",
        "fissure_right_depth_measurement_status",
        "fissure_right_depth_y",
        "fissure_right_depth_point_count",
        "fissure_right_depth_smooth_px",
    ],
    "lateral_curvature": COMMON_TABLE_COLUMNS + [
        "lateral_curvature_image_overlay_path",
        "lateral_curvature_mask_overlay_path",
        "fissure_measurement_status",
        "fissure_component_count",
        "fissure_measured_component_count",
        "fissure_curvature_ratio",
        "fissure_mean_curvature_ratio",
        "fissure_curvature_measurement_status",
        "fissure_curve_length_px",
        "fissure_curve_chord_px",
        "fissure_curve_bulge_offset_px",
        "fissure_curve_bulge_point_x",
        "fissure_curve_bulge_point_y",
        "fissure_curvature_angle_deg",
        "fissure_left_curvature_ratio",
        "fissure_left_curvature_measurement_status",
        "fissure_left_curve_length_px",
        "fissure_left_curve_chord_px",
        "fissure_left_curve_bulge_offset_px",
        "fissure_left_curve_bulge_point_x",
        "fissure_left_curve_bulge_point_y",
        "fissure_left_curvature_angle_deg",
        "fissure_right_curvature_ratio",
        "fissure_right_curvature_measurement_status",
        "fissure_right_curve_length_px",
        "fissure_right_curve_chord_px",
        "fissure_right_curve_bulge_offset_px",
        "fissure_right_curve_bulge_point_x",
        "fissure_right_curve_bulge_point_y",
        "fissure_right_curvature_angle_deg",
    ],
    "lateral_angle": COMMON_TABLE_COLUMNS + [
        "lateral_angle_image_overlay_path",
        "lateral_angle_mask_overlay_path",
        "fissure_measurement_status",
        "fissure_angle_measurement_status",
        "fissure_angle_deg",
        "fissure_mean_angle_deg",
        "fissure_left_angle_measurement_status",
        "fissure_left_angle_deg",
        "fissure_right_angle_measurement_status",
        "fissure_right_angle_deg",
        "fissure_reference_line_mode",
        "fissure_reference_point_x",
        "fissure_reference_point_y",
        "fissure_reference_direction_x",
        "fissure_reference_direction_y",
        "fissure_left_measurement_anchor_x",
        "fissure_left_measurement_anchor_y",
        "fissure_right_measurement_anchor_x",
        "fissure_right_measurement_anchor_y",
        "fissure_left_local_tangent_status",
        "fissure_right_local_tangent_status",
    ],
    "longitudinal_branch_max_depth": COMMON_TABLE_COLUMNS + [
        "longitudinal_branch_max_depth_image_overlay_path",
        "longitudinal_branch_max_depth_mask_overlay_path",
        "longitudinal_fissure_measurement_status",
        "longitudinal_fissure_component_count",
        "longitudinal_fissure_branch_depth_px",
        "longitudinal_fissure_branch_euclidean_px",
        "longitudinal_fissure_branch_measurement_status",
        "longitudinal_fissure_branch_candidate_count",
        "longitudinal_fissure_branch_start_x",
        "longitudinal_fissure_branch_start_y",
        "longitudinal_fissure_branch_tip_x",
        "longitudinal_fissure_branch_tip_y",
    ],
    "longitudinal_full_length": COMMON_TABLE_COLUMNS + [
        "longitudinal_full_length_image_overlay_path",
        "longitudinal_full_length_mask_overlay_path",
        "longitudinal_fissure_measurement_status",
        "longitudinal_fissure_component_count",
        "longitudinal_fissure_full_length_px",
        "longitudinal_fissure_full_length_measurement_status",
        "longitudinal_fissure_full_length_contour_point_count",
    ],
    "longitudinal_area": COMMON_TABLE_COLUMNS + [
        "longitudinal_area_image_overlay_path",
        "longitudinal_area_mask_overlay_path",
        "longitudinal_fissure_measurement_status",
        "longitudinal_fissure_component_count",
        "longitudinal_fissure_area_px",
    ],
}
METRIC_TABLE_NAMES = {
    "lateral_opening_width": "外侧裂开口宽度",
    "lateral_max_depth": "外侧裂最大深度",
    "lateral_curvature": "裂隙弯曲度",
    "lateral_angle": "角度",
    "longitudinal_branch_max_depth": "纵裂分支最大深度",
    "longitudinal_full_length": "纵裂全长",
    "longitudinal_area": "纵裂面积",
}
CSV_HEADER_ZH = {
    "index": "序号",
    "mask_path": "掩膜路径",
    "source_image_path": "原图路径",
    "status": "处理状态",
    "lateral_opening_width_image_overlay_path": "外侧裂开口宽度原图叠加路径",
    "lateral_opening_width_mask_overlay_path": "外侧裂开口宽度掩膜叠加路径",
    "lateral_max_depth_image_overlay_path": "外侧裂最大深度原图叠加路径",
    "lateral_max_depth_mask_overlay_path": "外侧裂最大深度掩膜叠加路径",
    "lateral_curvature_image_overlay_path": "裂隙弯曲度原图叠加路径",
    "lateral_curvature_mask_overlay_path": "裂隙弯曲度掩膜叠加路径",
    "lateral_angle_image_overlay_path": "角度原图叠加路径",
    "lateral_angle_mask_overlay_path": "角度掩膜叠加路径",
    "longitudinal_branch_max_depth_image_overlay_path": "纵裂分支最大深度原图叠加路径",
    "longitudinal_branch_max_depth_mask_overlay_path": "纵裂分支最大深度掩膜叠加路径",
    "longitudinal_full_length_image_overlay_path": "纵裂全长原图叠加路径",
    "longitudinal_full_length_mask_overlay_path": "纵裂全长掩膜叠加路径",
    "longitudinal_area_image_overlay_path": "纵裂面积原图叠加路径",
    "longitudinal_area_mask_overlay_path": "纵裂面积掩膜叠加路径",
    "fissure_measurement_status": "外侧裂测量状态",
    "fissure_component_count": "外侧裂连通域数量",
    "fissure_measured_component_count": "外侧裂已测连通域数量",
    "fissure_width_px": "外侧裂开口宽度像素",
    "fissure_mean_width_px": "外侧裂平均开口宽度像素",
    "fissure_opening_width_measurement_status": "外侧裂开口宽度测量状态",
    "fissure_opening_width_mode": "外侧裂开口宽度测量模式",
    "fissure_opening_width_yellow_dashed_length_px": "外侧裂黄色虚线长度像素",
    "fissure_opening_width_segment_a_px": "外侧裂黄色虚线第一段像素",
    "fissure_opening_width_segment_b_px": "外侧裂黄色虚线第二段像素",
    "fissure_left_width_px": "左侧外侧裂开口宽度像素",
    "fissure_left_opening_width_measurement_status": "左侧外侧裂开口宽度测量状态",
    "fissure_left_opening_width_mode": "左侧外侧裂开口宽度测量模式",
    "fissure_left_opening_width_yellow_dashed_length_px": "左侧外侧裂黄色虚线长度像素",
    "fissure_left_opening_width_segment_a_px": "左侧外侧裂黄色虚线第一段像素",
    "fissure_left_opening_width_segment_b_px": "左侧外侧裂黄色虚线第二段像素",
    "fissure_right_width_px": "右侧外侧裂开口宽度像素",
    "fissure_right_opening_width_measurement_status": "右侧外侧裂开口宽度测量状态",
    "fissure_right_opening_width_mode": "右侧外侧裂开口宽度测量模式",
    "fissure_right_opening_width_yellow_dashed_length_px": "右侧外侧裂黄色虚线长度像素",
    "fissure_right_opening_width_segment_a_px": "右侧外侧裂黄色虚线第一段像素",
    "fissure_right_opening_width_segment_b_px": "右侧外侧裂黄色虚线第二段像素",
    "fissure_depth_px": "外侧裂最大深度像素",
    "fissure_depth_measurement_status": "外侧裂最大深度测量状态",
    "fissure_depth_y": "外侧裂最大深度所在行",
    "fissure_depth_point_count": "外侧裂最大深度候选行数",
    "fissure_depth_smooth_px": "外侧裂最大深度平滑值像素",
    "fissure_left_depth_px": "左侧外侧裂最大深度像素",
    "fissure_left_depth_measurement_status": "左侧外侧裂最大深度测量状态",
    "fissure_left_depth_y": "左侧外侧裂最大深度所在行",
    "fissure_left_depth_point_count": "左侧外侧裂最大深度候选行数",
    "fissure_left_depth_smooth_px": "左侧外侧裂最大深度平滑值像素",
    "fissure_right_depth_px": "右侧外侧裂最大深度像素",
    "fissure_right_depth_measurement_status": "右侧外侧裂最大深度测量状态",
    "fissure_right_depth_y": "右侧外侧裂最大深度所在行",
    "fissure_right_depth_point_count": "右侧外侧裂最大深度候选行数",
    "fissure_right_depth_smooth_px": "右侧外侧裂最大深度平滑值像素",
    "fissure_curvature_ratio": "裂隙弯曲度比值",
    "fissure_mean_curvature_ratio": "平均裂隙弯曲度比值",
    "fissure_curvature_measurement_status": "裂隙弯曲度测量状态",
    "fissure_curve_length_px": "裂隙曲线长度像素",
    "fissure_curve_chord_px": "裂隙弦长像素",
    "fissure_curve_bulge_offset_px": "裂隙最大凸点偏移像素",
    "fissure_curve_bulge_point_x": "裂隙最大凸点X",
    "fissure_curve_bulge_point_y": "裂隙最大凸点Y",
    "fissure_curvature_angle_deg": "裂隙弯曲角度度",
    "fissure_angle_measurement_status": "外侧裂角度测量状态",
    "fissure_angle_deg": "外侧裂角度度",
    "fissure_mean_angle_deg": "平均外侧裂角度度",
    "fissure_left_angle_measurement_status": "左侧外侧裂角度测量状态",
    "fissure_left_angle_deg": "左侧外侧裂角度度",
    "fissure_right_angle_measurement_status": "右侧外侧裂角度测量状态",
    "fissure_right_angle_deg": "右侧外侧裂角度度",
    "fissure_reference_line_mode": "参考线模式",
    "fissure_reference_point_x": "参考点X",
    "fissure_reference_point_y": "参考点Y",
    "fissure_reference_direction_x": "参考方向X",
    "fissure_reference_direction_y": "参考方向Y",
    "fissure_left_measurement_anchor_x": "左侧角度测量锚点X",
    "fissure_left_measurement_anchor_y": "左侧角度测量锚点Y",
    "fissure_right_measurement_anchor_x": "右侧角度测量锚点X",
    "fissure_right_measurement_anchor_y": "右侧角度测量锚点Y",
    "fissure_left_local_tangent_status": "左侧局部切线状态",
    "fissure_right_local_tangent_status": "右侧局部切线状态",
    "longitudinal_fissure_measurement_status": "纵裂测量状态",
    "longitudinal_fissure_component_count": "纵裂连通域数量",
    "longitudinal_fissure_branch_depth_px": "纵裂分支最大深度像素",
    "longitudinal_fissure_branch_euclidean_px": "纵裂分支直线距离像素",
    "longitudinal_fissure_branch_measurement_status": "纵裂分支测量状态",
    "longitudinal_fissure_branch_candidate_count": "纵裂候选分支数量",
    "longitudinal_fissure_branch_start_x": "纵裂分支起点X",
    "longitudinal_fissure_branch_start_y": "纵裂分支起点Y",
    "longitudinal_fissure_branch_tip_x": "纵裂分支顶点X",
    "longitudinal_fissure_branch_tip_y": "纵裂分支顶点Y",
    "longitudinal_fissure_full_length_px": "纵裂全长像素",
    "longitudinal_fissure_full_length_measurement_status": "纵裂全长测量状态",
    "longitudinal_fissure_full_length_contour_point_count": "纵裂全长轮廓点数",
    "longitudinal_fissure_area_px": "纵裂面积像素",
}
SKIP_DIR_NAMES = {
    ".ipynb_checkpoints",
    "measurement_results",
    "measurement_results_by_type",
    "mask_measurement_results",
    "typed_measurement_results",
    "measure_output_masks",
    "measurement_overlay",
    "image_overlay",
    "mask_overlay",
    "tables",
    "logs",
    "外侧裂开口宽度",
    "外侧裂最大深度",
    "裂隙弯曲度",
    "角度",
    "纵裂分支最大深度",
    "纵裂全长",
    "纵裂面积",
    "测量结果",
    "测量结果_分类",
    "测量结果_按指标分类",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure lateral fissure geometry directly from saved mask images."
    )
    parser.add_argument(
        "--input-root",
        default="./Results/data_260513",
        help="Root folder containing saved mask outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder. Defaults to <input-root>/measurement_results_by_type.",
    )
    parser.add_argument(
        "--mask-pattern",
        default="*_pred_mask.png",
        help="Glob pattern for masks relative to input root.",
    )
    parser.add_argument(
        "--pixel-spacing",
        default="",
        help="Optional pixel spacing in mm, either one value or row,col.",
    )
    parser.add_argument(
        "--foreground-class",
        type=int,
        default=1,
        help="Backward-compatible alias for --lateral-class.",
    )
    parser.add_argument(
        "--lateral-class",
        type=int,
        default=None,
        help="Class label used for lateral fissure metrics. Defaults to --foreground-class.",
    )
    parser.add_argument(
        "--longitudinal-class",
        type=int,
        default=2,
        help="Class label used for longitudinal fissure metrics.",
    )
    parser.add_argument(
        "--third-ventricle-apex",
        choices=("inferior", "superior"),
        default="inferior",
        help="Which endpoint of the third-ventricle/longitudinal mask is used as P0 for angle measurement.",
    )
    parser.add_argument(
        "--reference-line-mode",
        choices=("auto", "upper_arc_prejunction_terminals", "manual", "apex_parallel"),
        default="auto",
        help="Reference line mode for lateral fissure angle measurement.",
    )
    parser.add_argument(
        "--baseline-left-point",
        default="",
        help="Optional baseline point x,y for manual/apex_parallel reference direction.",
    )
    parser.add_argument(
        "--baseline-right-point",
        default="",
        help="Optional baseline point x,y for manual/apex_parallel reference direction.",
    )
    parser.add_argument(
        "--reference-anchor-ratio",
        type=float,
        default=0.60,
        help="Deprecated compatibility option.",
    )
    parser.add_argument(
        "--local-tangent-ratio",
        type=float,
        default=0.60,
        help="Deprecated compatibility option.",
    )
    parser.add_argument(
        "--local-tangent-half-arc",
        type=float,
        default=30.0,
        help="Deprecated compatibility option. Use --local-arc-length instead.",
    )
    parser.add_argument(
        "--upper-arc-end-ratio",
        type=float,
        default=0.55,
        help="For non-branching arc masks, use this main-path arc ratio as the upper-arc pre-junction terminal.",
    )
    parser.add_argument(
        "--branch-cut-radius",
        type=float,
        default=6.0,
        help="Skeleton geodesic radius around branchpoints excluded from upper-arc terminal selection.",
    )
    parser.add_argument(
        "--terminal-max-depth-ratio",
        type=float,
        default=0.62,
        help="Maximum terminal depth as a ratio of each side mask bbox height.",
    )
    parser.add_argument(
        "--terminal-tangent-back-length",
        type=float,
        default=28.0,
        help="Backward arc length in pixels before the pre-junction terminal for local PCA tangent fitting.",
    )
    parser.add_argument(
        "--local-arc-length",
        type=float,
        default=None,
        help="Deprecated alias for --terminal-tangent-back-length.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=8,
        help="Minimum connected component area in pixels.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=2,
        help="Maximum fissure components to measure per mask.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing measurement images.",
    )
    return parser.parse_args()


def setup_logger(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "measure_output_masks.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def imread_unicode(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise OSError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def should_skip(path: Path, input_root: Path):
    return bool(set(path.relative_to(input_root).parts) & SKIP_DIR_NAMES)


def collect_masks(input_root: Path, pattern: str):
    return sorted(
        path for path in input_root.rglob(pattern)
        if path.is_file() and not should_skip(path, input_root)
    )


def mask_to_binary(mask: np.ndarray, foreground_class: int, fallback_binary: bool = False):
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    unique_values = np.unique(mask)
    positive_values = unique_values[unique_values > 0]
    if len(positive_values) == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    if foreground_class in positive_values:
        return (mask == foreground_class).astype(np.uint8)
    is_binary_like = len(positive_values) == 1 and int(positive_values[0]) in (1, 255)
    if fallback_binary and is_binary_like:
        return (mask > 0).astype(np.uint8)
    return np.zeros(mask.shape, dtype=np.uint8)


def parse_point(value: str):
    if value is None or str(value).strip() == "":
        return None
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"point should be x,y, got: {value}")
    return float(parts[0]), float(parts[1])


def original_candidates(mask_path: Path):
    name = mask_path.name
    suffixes = [
        "_pred_mask.png",
        "_mask.png",
        "_pred_mask.tif",
        "_pred_mask.tiff",
    ]
    stems = [mask_path.stem]
    for suffix in suffixes:
        if name.endswith(suffix):
            stems.insert(0, name[:-len(suffix)])
    for stem in dict.fromkeys(stems):
        for ext in IMAGE_EXTENSIONS:
            candidate = mask_path.with_name(f"{stem}{ext}")
            if candidate.exists() and candidate != mask_path:
                yield candidate


def read_display_image(mask_path: Path, binary_mask: np.ndarray):
    for candidate in original_candidates(mask_path):
        image = imread_unicode(candidate, cv2.IMREAD_COLOR)
        if image.shape[:2] == binary_mask.shape:
            return image, candidate
    image = cv2.cvtColor((binary_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return image, None


def relative_output_path(
    output_dir: Path,
    measurement_folder: str,
    image_type: str,
    input_root: Path,
    mask_path: Path,
    suffix: str,
):
    relative = mask_path.relative_to(input_root)
    return output_dir / measurement_folder / image_type / relative.parent / f"{mask_path.stem}{suffix}"


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    ordered = [
        "index",
        "mask_path",
        "source_image_path",
        "status",
    ]
    for output_key in MEASUREMENT_OUTPUTS:
        ordered.append(f"{output_key}_image_overlay_path")
        ordered.append(f"{output_key}_mask_overlay_path")
    fieldnames = ordered + [key for key in fieldnames if key not in ordered]
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chinese_header(column: str) -> str:
    if column in CSV_HEADER_ZH:
        return CSV_HEADER_ZH[column]

    label = column
    replacements = [
        ("fissure_left_", "左侧外侧裂"),
        ("fissure_right_", "右侧外侧裂"),
        ("fissure_", "外侧裂"),
        ("longitudinal_fissure_", "纵裂"),
        ("curvature_ratio", "弯曲度比值"),
        ("curvature_measurement_status", "弯曲度测量状态"),
        ("curve_length_px", "曲线长度像素"),
        ("curve_chord_px", "弦长像素"),
        ("curve_bulge_offset_px", "最大凸点偏移像素"),
        ("curve_bulge_point_x", "最大凸点X"),
        ("curve_bulge_point_y", "最大凸点Y"),
        ("curvature_angle_deg", "弯曲角度度"),
        ("width_px", "宽度像素"),
        ("depth_px", "深度像素"),
        ("measurement_status", "测量状态"),
        ("image_overlay_path", "原图叠加路径"),
        ("mask_overlay_path", "掩膜叠加路径"),
        ("_px", "像素"),
        ("_deg", "度"),
        ("_", ""),
    ]
    for old, new in replacements:
        label = label.replace(old, new)
    return label


def write_table_with_headers(path: Path, rows, columns, header_mode: str = "en"):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = columns if header_mode == "en" else [chinese_header(column) for column in columns]
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(column, "") for column in columns])


def write_metric_csvs(table_dir: Path, rows):
    output_dir = table_dir / "by_metric"
    written = {}
    for metric_key, columns in METRIC_TABLE_COLUMNS.items():
        metric_name = METRIC_TABLE_NAMES.get(metric_key, metric_key)
        en_path = output_dir / f"{metric_key}_en.csv"
        zh_path = output_dir / f"{metric_key}_zh.csv"
        cn_path = output_dir / f"{metric_name}_中文表头.csv"
        write_table_with_headers(en_path, rows, columns, header_mode="en")
        write_table_with_headers(zh_path, rows, columns, header_mode="zh")
        write_table_with_headers(cn_path, rows, columns, header_mode="zh")
        written[metric_key] = {
            "english_header_csv": str(en_path.resolve()),
            "chinese_header_csv": str(zh_path.resolve()),
            "chinese_named_csv": str(cn_path.resolve()),
        }
    return written


def to_jsonable(value):
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize(rows, input_root: Path, output_dir: Path, log_path: Path):
    measurable = [row for row in rows if row.get("fissure_measurement_status") == "ok"]
    angle_measurable = [
        row for row in rows
        if row.get("fissure_angle_measurement_status") == "ok"
    ]
    longitudinal_measurable = [
        row for row in rows
        if row.get("longitudinal_fissure_measurement_status") == "ok"
    ]
    summary = {
        "input_root": str(input_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "measurement_type_dirs": {
            key: str((output_dir / spec["folder"]).resolve())
            for key, spec in MEASUREMENT_OUTPUTS.items()
        },
        "tables_dir": str((output_dir / "tables").resolve()),
        "logs_dir": str((output_dir / "logs").resolve()),
        "log_path": str(log_path.resolve()),
        "num_masks": len(rows),
        "num_lateral_measurable": len(measurable),
        "num_lateral_angle_measurable": len(angle_measurable),
        "num_longitudinal_measurable": len(longitudinal_measurable),
        "avg_fissure_width_px": float(np.mean([row["fissure_width_px"] for row in measurable])) if measurable else float("nan"),
        "avg_fissure_depth_px": float(np.mean([row["fissure_depth_px"] for row in measurable])) if measurable else float("nan"),
        "avg_fissure_mean_width_px": float(np.mean([row["fissure_mean_width_px"] for row in measurable])) if measurable else float("nan"),
        "avg_fissure_curvature_ratio": float(np.mean([row["fissure_curvature_ratio"] for row in measurable])) if measurable else float("nan"),
        "avg_fissure_left_angle_deg": float(np.mean([row["fissure_left_angle_deg"] for row in angle_measurable if row.get("fissure_left_angle_deg") not in ("", None)])) if angle_measurable else float("nan"),
        "avg_fissure_right_angle_deg": float(np.mean([row["fissure_right_angle_deg"] for row in angle_measurable if row.get("fissure_right_angle_deg") not in ("", None)])) if angle_measurable else float("nan"),
        "avg_longitudinal_fissure_branch_depth_px": float(np.mean([row["longitudinal_fissure_branch_depth_px"] for row in longitudinal_measurable])) if longitudinal_measurable else float("nan"),
        "avg_longitudinal_fissure_full_length_px": float(np.mean([row["longitudinal_fissure_full_length_px"] for row in longitudinal_measurable])) if longitudinal_measurable else float("nan"),
        "avg_longitudinal_fissure_area_px": float(np.mean([row["longitudinal_fissure_area_px"] for row in longitudinal_measurable])) if longitudinal_measurable else float("nan"),
    }
    if measurable and "fissure_width_mm" in measurable[0]:
        summary.update({
            "avg_fissure_width_mm": float(np.mean([row["fissure_width_mm"] for row in measurable])),
            "avg_fissure_depth_mm": float(np.mean([row["fissure_depth_mm"] for row in measurable])),
            "avg_fissure_mean_width_mm": float(np.mean([row["fissure_mean_width_mm"] for row in measurable])),
        })
    return summary


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_root / "measurement_results_by_type"
    log_path = setup_logger(output_dir)
    pixel_spacing = parse_pixel_spacing(args.pixel_spacing)
    lateral_class = args.lateral_class if args.lateral_class is not None else args.foreground_class
    baseline_left_point = parse_point(args.baseline_left_point)
    baseline_right_point = parse_point(args.baseline_right_point)
    terminal_tangent_back_length = (
        args.local_arc_length
        if args.local_arc_length is not None
        else args.terminal_tangent_back_length
    )

    mask_paths = collect_masks(input_root, args.mask_pattern)
    logging.info("Input root: %s", input_root)
    logging.info("Output dir: %s", output_dir)
    logging.info("Found %d mask(s) with pattern %s", len(mask_paths), args.mask_pattern)

    rows = []
    for index, mask_path in enumerate(mask_paths, start=1):
        try:
            raw_mask = imread_unicode(mask_path, cv2.IMREAD_UNCHANGED)
            lateral_mask = mask_to_binary(raw_mask, lateral_class, fallback_binary=True)
            longitudinal_mask = mask_to_binary(raw_mask, args.longitudinal_class, fallback_binary=False)
            third_ventricle_apex = locate_third_ventricle_apex(
                longitudinal_mask,
                apex=args.third_ventricle_apex,
            )
            display_mask = ((lateral_mask > 0) | (longitudinal_mask > 0)).astype(np.uint8)
            display_image, source_image_path = read_display_image(mask_path, display_mask)
            lateral_measurement = measure_lateral_fissure(
                lateral_mask,
                pixel_spacing=pixel_spacing,
                min_area=args.min_area,
                max_components=args.max_components,
                angle_reference_point=third_ventricle_apex,
                baseline_left_point=baseline_left_point,
                baseline_right_point=baseline_right_point,
                reference_line_mode=args.reference_line_mode,
                reference_anchor_ratio=args.reference_anchor_ratio,
                local_tangent_ratio=args.local_tangent_ratio,
                local_tangent_half_arc_length=args.local_tangent_half_arc,
                local_arc_length=terminal_tangent_back_length,
                branch_cut_radius=args.branch_cut_radius,
                terminal_max_depth_ratio=args.terminal_max_depth_ratio,
                upper_arc_end_ratio=args.upper_arc_end_ratio,
                reference_pass_through_apex=args.reference_line_mode == "apex_parallel",
            )
            longitudinal_measurement = measure_longitudinal_fissure(
                longitudinal_mask,
                pixel_spacing=pixel_spacing,
                min_area=args.min_area,
            )
            output_paths = {}
            for output_key, output_spec in MEASUREMENT_OUTPUTS.items():
                if output_spec["target"] == "lateral":
                    metric_mask = lateral_mask
                    metric_measurement = lateral_measurement
                    image_overlay = annotate_lateral_fissure_measurement(
                        display_image,
                        metric_mask,
                        measurement=metric_measurement,
                        pixel_spacing=pixel_spacing,
                        draw_measurements=output_spec["draw"],
                    )
                    mask_display = cv2.cvtColor((metric_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    mask_overlay = annotate_lateral_fissure_measurement(
                        mask_display,
                        metric_mask,
                        measurement=metric_measurement,
                        pixel_spacing=pixel_spacing,
                        alpha=0.0,
                        draw_measurements=output_spec["draw"],
                    )
                else:
                    metric_mask = longitudinal_mask
                    metric_measurement = longitudinal_measurement
                    image_overlay = annotate_longitudinal_fissure_measurement(
                        display_image,
                        metric_mask,
                        measurement=metric_measurement,
                        pixel_spacing=pixel_spacing,
                        draw_measurements=output_spec["draw"],
                    )
                    mask_display = cv2.cvtColor((metric_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    mask_overlay = annotate_longitudinal_fissure_measurement(
                        mask_display,
                        metric_mask,
                        measurement=metric_measurement,
                        pixel_spacing=pixel_spacing,
                        alpha=0.0,
                        draw_measurements=output_spec["draw"],
                    )
                image_overlay_path = relative_output_path(
                    output_dir,
                    output_spec["folder"],
                    "image_overlay",
                    input_root,
                    mask_path,
                    output_spec["suffix"],
                )
                mask_overlay_path = relative_output_path(
                    output_dir,
                    output_spec["folder"],
                    "mask_overlay",
                    input_root,
                    mask_path,
                    output_spec["suffix"],
                )
                if args.overwrite or not image_overlay_path.exists():
                    imwrite_unicode(image_overlay_path, image_overlay)
                if args.overwrite or not mask_overlay_path.exists():
                    imwrite_unicode(mask_overlay_path, mask_overlay)
                output_paths[f"{output_key}_image_overlay_path"] = str(image_overlay_path)
                output_paths[f"{output_key}_mask_overlay_path"] = str(mask_overlay_path)

            row = {
                "index": index,
                "mask_path": str(mask_path),
                "source_image_path": str(source_image_path) if source_image_path else "",
                **output_paths,
                "status": "ok",
                **measurement_to_row(lateral_measurement),
                **longitudinal_measurement_to_row(longitudinal_measurement),
            }
            rows.append(row)
            logging.info(
                "[%d/%d] measured %s lateral=%s longitudinal=%s width_px=%.3f depth_px=%.3f full_length_px=%.3f area_px=%s",
                index,
                len(mask_paths),
                mask_path,
                row["fissure_measurement_status"],
                row["longitudinal_fissure_measurement_status"],
                float(row["fissure_width_px"]),
                float(row["fissure_depth_px"]),
                float(row["longitudinal_fissure_full_length_px"]),
                row["longitudinal_fissure_area_px"],
            )
        except Exception as exc:
            row = {
                "index": index,
                "mask_path": str(mask_path),
                "source_image_path": "",
                "status": "error",
                "error": str(exc),
            }
            for output_key in MEASUREMENT_OUTPUTS:
                row[f"{output_key}_image_overlay_path"] = ""
                row[f"{output_key}_mask_overlay_path"] = ""
            rows.append(row)
            logging.exception("[%d/%d] failed %s", index, len(mask_paths), mask_path)

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "measurement_results.csv"
    write_csv(csv_path, rows)
    metric_csv_paths = write_metric_csvs(table_dir, rows)
    summary = summarize(rows, input_root, output_dir, log_path)
    summary["metric_csv_paths"] = metric_csv_paths
    summary["metric_csv_dir"] = str((table_dir / "by_metric").resolve())
    summary_path = table_dir / "summary.json"
    summary_path.write_text(
        json.dumps(to_jsonable({"summary": summary, "cases": rows}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logging.info("CSV: %s", csv_path)
    logging.info("Summary: %s", summary_path)
    print(f"masks={len(rows)}")
    print(f"lateral_measurable={summary['num_lateral_measurable']}")
    print(f"lateral_angle_measurable={summary['num_lateral_angle_measurable']}")
    print(f"longitudinal_measurable={summary['num_longitudinal_measurable']}")
    print(f"csv={csv_path}")
    print(f"summary={summary_path}")
    print(f"metric_csv_dir={table_dir / 'by_metric'}")
    for output_key, output_spec in MEASUREMENT_OUTPUTS.items():
        print(f"{output_key}_dir={output_dir / output_spec['folder']}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
