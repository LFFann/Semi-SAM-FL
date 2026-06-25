from __future__ import annotations

import math
import heapq
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import binary_closing, disk, remove_small_objects, skeletonize


Point = Tuple[float, float]


def parse_pixel_spacing(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if value is None or value == "":
        return None
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        spacing = float(parts[0])
        return spacing, spacing
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    raise ValueError("pixel spacing should be one value or row,col values")


def _length(p1: Point, p2: Point, pixel_spacing: Optional[Tuple[float, float]] = None) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    if pixel_spacing is None:
        return math.hypot(dx, dy)
    row_spacing, col_spacing = pixel_spacing
    return math.hypot(dx * col_spacing, dy * row_spacing)


def _components(mask: np.ndarray, min_area: int) -> Tuple[List[Tuple[np.ndarray, int]], int]:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((labels == label_idx, area))
    components.sort(key=lambda item: item[1], reverse=True)
    return components, int(num_labels - 1)


def keep_largest_component(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    best_label = None
    best_area = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label = label_idx
            best_area = area
    if best_label is None or best_area < min_area:
        return np.zeros_like(binary, dtype=np.uint8)
    return (labels == best_label).astype(np.uint8)


def preprocess_mask(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    binary = (mask > 0)
    binary = remove_small_objects(binary, min_size=min_area)
    try:
        binary = binary_closing(binary, footprint=disk(1))
    except TypeError:
        binary = binary_closing(binary, selem=disk(1))
    return keep_largest_component(binary.astype(np.uint8), min_area=min_area)


def _component_points(component: np.ndarray) -> np.ndarray:
    yx = np.column_stack(np.where(component))
    return yx[:, ::-1].astype(np.float32)


def _skeleton_endpoints(component: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(component).astype(np.uint8)
    if not skeleton.any():
        return np.empty((0, 2), dtype=np.float32)

    padded = np.pad(skeleton, 1, mode="constant")
    endpoints = []
    ys, xs = np.where(skeleton > 0)
    for y, x in zip(ys, xs):
        neighborhood = padded[y:y + 3, x:x + 3]
        neighbor_count = int(neighborhood.sum()) - 1
        if neighbor_count == 1:
            endpoints.append((float(x), float(y)))
    return np.asarray(endpoints, dtype=np.float32)


def _side_for_component(points: np.ndarray, image_width: int) -> str:
    return "left" if float(points[:, 0].mean()) < image_width / 2.0 else "right"


def _medial_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def _max_distance_pair(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) == 1:
        return points[0], points[0]
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sum(deltas * deltas, axis=2)
    i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return points[i], points[j]


def _fallback_opening_lips(points: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
    medial = _medial_sign(side) * points[:, 0]
    threshold = np.percentile(medial, 80)
    candidates = points[medial >= threshold]
    if len(candidates) < 2:
        candidates = points[np.argsort(medial)[-min(12, len(points)):]]
    return _max_distance_pair(candidates)


def _select_opening_lips(candidates: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
    if len(candidates) < 2:
        return _fallback_opening_lips(candidates, side)
    medial = _medial_sign(side) * candidates[:, 0]
    top = candidates[np.argsort(medial)[-min(6, len(candidates)):]]
    lip_a, lip_b = _max_distance_pair(top)
    if _length(tuple(lip_a), tuple(lip_b)) <= 1.0:
        return _fallback_opening_lips(candidates, side)
    return lip_a, lip_b


def _positive_runs(values: np.ndarray) -> List[Tuple[int, int]]:
    if len(values) == 0:
        return []
    runs = []
    start = int(values[0])
    previous = int(values[0])
    for value in values[1:]:
        current = int(value)
        if current == previous + 1:
            previous = current
            continue
        runs.append((start, previous))
        start = current
        previous = current
    runs.append((start, previous))
    return runs


def _extract_opening_width_profile(component: np.ndarray, side: str) -> List[Dict[str, object]]:
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return []

    xmin = int(xs.min())
    xmax = int(xs.max())
    span = xmax - xmin + 1
    if span <= 1:
        return []

    if side == "left":
        x_start = int(round(xmin + 0.40 * span))
        scan_xs = range(max(xmin, x_start), xmax + 1)
    else:
        x_end = int(round(xmax - 0.40 * span))
        scan_xs = range(xmin, min(xmax, x_end) + 1)

    results = []
    for x in scan_xs:
        col_y = np.where(component[:, x] > 0)[0]
        if len(col_y) < 2:
            continue

        runs = _positive_runs(col_y)
        if len(runs) >= 2:
            gaps = []
            for idx in range(len(runs) - 1):
                upper_end = runs[idx][1]
                lower_start = runs[idx + 1][0]
                gap = int(lower_start - upper_end)
                gaps.append((gap, upper_end, lower_start))
            gap, y_top, y_bottom = max(gaps, key=lambda item: item[0])
            if gap < 3:
                continue
            width = float(gap)
            mode = "inter_arm_gap"
        else:
            y_top = int(col_y.min())
            y_bottom = int(col_y.max())
            width = float(y_bottom - y_top)
            if width < 3.0:
                continue
            mode = "single_run_span"

        results.append({
            "x": int(x),
            "y_top": int(y_top),
            "y_bottom": int(y_bottom),
            "width": float(width),
            "mode": mode,
        })
    return results


def _smooth_profile(values: np.ndarray, window: int = 7) -> np.ndarray:
    if len(values) < 5:
        return values.astype(np.float32)

    window = min(int(window), len(values))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return values.astype(np.float32)

    try:
        import warnings
        from scipy.signal import savgol_filter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return savgol_filter(values.astype(np.float64), window_length=window, polyorder=2).astype(np.float32)
    except Exception:
        kernel = np.ones(window, dtype=np.float64) / float(window)
        pad = window // 2
        return np.convolve(np.pad(values.astype(np.float64), pad, mode="edge"), kernel, mode="valid").astype(np.float32)


def _choose_best_opening_width(profile: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if len(profile) < 5:
        return None

    gap_profile = [item for item in profile if item.get("mode") == "inter_arm_gap"]
    candidates = gap_profile if len(gap_profile) >= 5 else profile
    widths = np.asarray([float(item["width"]) for item in candidates], dtype=np.float32)
    smoothed = _smooth_profile(widths)

    n = len(candidates)
    lo = max(0, int(0.10 * n))
    hi = min(n, int(math.ceil(0.90 * n)))
    if hi - lo < 3:
        lo, hi = 0, n

    index = int(np.argmax(smoothed[lo:hi]) + lo)
    best = dict(candidates[index])
    best["width_smooth"] = float(smoothed[index])
    best["profile_point_count"] = int(len(profile))
    best["gap_profile_point_count"] = int(len(gap_profile))
    return best


def _split_continuous_runs(values: np.ndarray, max_gap: int = 2) -> List[np.ndarray]:
    if len(values) == 0:
        return []

    ordered = np.sort(np.asarray(values, dtype=np.int32))
    runs = []
    start = int(ordered[0])
    previous = int(ordered[0])
    for value in ordered[1:]:
        current = int(value)
        if current - previous <= max_gap:
            previous = current
            continue
        runs.append(np.arange(start, previous + 1, dtype=np.int32))
        start = current
        previous = current
    runs.append(np.arange(start, previous + 1, dtype=np.int32))
    return runs


def _measure_middle_depth(component: np.ndarray, pixel_spacing: Optional[Tuple[float, float]]) -> Dict[str, object]:
    ys, xs = np.where(component > 0)
    if len(xs) < 10:
        return {
            "depth_measurement_status": "too_few_component_points",
            "depth_px": 0.0,
            "depth_mm": None,
            "depth_line": None,
            "depth_y": None,
            "depth_point_count": 0,
            "depth_smooth_px": 0.0,
        }

    ymin = int(ys.min())
    ymax = int(ys.max())
    height = ymax - ymin + 1
    if height <= 1:
        return {
            "depth_measurement_status": "component_too_short",
            "depth_px": 0.0,
            "depth_mm": None,
            "depth_line": None,
            "depth_y": None,
            "depth_point_count": 0,
            "depth_smooth_px": 0.0,
        }

    y_start = int(round(ymin + 0.20 * height))
    y_end = int(round(ymin + 0.80 * height))
    if y_end <= y_start:
        y_start = int(round(ymin + 0.15 * height))
        y_end = int(round(ymin + 0.85 * height))

    candidates = []
    for y in range(y_start, y_end + 1):
        row_x = np.where(component[y, :] > 0)[0]
        if len(row_x) < 3:
            continue

        runs = _split_continuous_runs(row_x, max_gap=2)
        if not runs:
            continue

        main_run = max(runs, key=len)
        if len(main_run) < 3:
            continue

        x_left = float(np.percentile(main_run, 5))
        x_right = float(np.percentile(main_run, 95))
        depth = float(x_right - x_left)
        if depth <= 2.0:
            continue

        candidates.append({
            "y": int(y),
            "x_left": x_left,
            "x_right": x_right,
            "depth": depth,
        })

    if len(candidates) < 3:
        return {
            "depth_measurement_status": "too_few_middle_rows",
            "depth_px": 0.0,
            "depth_mm": None,
            "depth_line": None,
            "depth_y": None,
            "depth_point_count": int(len(candidates)),
            "depth_smooth_px": 0.0,
        }

    depths = np.asarray([item["depth"] for item in candidates], dtype=np.float32)
    smooth_depths = _smooth_profile(depths, window=5)
    n = len(candidates)
    lo = max(0, int(0.10 * n))
    hi = min(n, int(math.ceil(0.90 * n)))
    if hi - lo < 3:
        lo, hi = 0, n

    best_index = int(np.argmax(smooth_depths[lo:hi]) + lo)
    best = candidates[best_index]
    p_left = (float(best["x_left"]), float(best["y"]))
    p_right = (float(best["x_right"]), float(best["y"]))
    depth_px = _length(p_left, p_right)
    depth_mm = _length(p_left, p_right, pixel_spacing) if pixel_spacing else None

    return {
        "depth_measurement_status": "ok",
        "depth_px": float(depth_px),
        "depth_mm": float(depth_mm) if depth_mm is not None else None,
        "depth_line": (p_left, p_right),
        "depth_y": int(best["y"]),
        "depth_point_count": int(len(candidates)),
        "depth_smooth_px": float(smooth_depths[best_index]),
    }


def _measure_opening_width(component: np.ndarray, side: str) -> Dict[str, object]:
    profile = _extract_opening_width_profile(component, side)
    best = _choose_best_opening_width(profile)
    if best is None:
        return {
            "opening_width_measurement_status": "too_few_opening_columns",
            "width_px": 0.0,
            "width_line": None,
            "opening_width_point_count": int(len(profile)),
            "opening_width_gap_point_count": int(sum(1 for item in profile if item.get("mode") == "inter_arm_gap")),
            "opening_width_mode": "",
            "opening_width_smooth_px": 0.0,
        }

    x = float(best["x"])
    p_top = (x, float(best["y_top"]))
    p_bottom = (x, float(best["y_bottom"]))
    width_px = _length(p_top, p_bottom)
    return {
        "opening_width_measurement_status": "ok" if best.get("mode") == "inter_arm_gap" else "fallback_single_run_span",
        "width_px": float(width_px),
        "width_line": (p_top, p_bottom),
        "opening_width_point_count": int(best.get("profile_point_count", len(profile))),
        "opening_width_gap_point_count": int(best.get("gap_profile_point_count", 0)),
        "opening_width_mode": str(best.get("mode", "")),
        "opening_width_smooth_px": float(best.get("width_smooth", width_px)),
    }


def _select_sulcus_bottom(candidates: np.ndarray, points: np.ndarray, side: str) -> np.ndarray:
    source = candidates if len(candidates) >= 3 else points
    medial = _medial_sign(side) * source[:, 0]
    return source[int(np.argmin(medial))]


def _project_to_line(point: np.ndarray, line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray:
    line = line_b - line_a
    denom = float(np.dot(line, line))
    if denom <= 1e-6:
        return line_a.copy()
    t = float(np.dot(point - line_a, line) / denom)
    t = max(0.0, min(1.0, t))
    return line_a + line * t


def _angle_to_horizontal(p1: Point, p2: Point) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 90.0:
        angle = 180.0 - angle
    return float(angle)


def _line_from_point_angle(point: Point, angle_deg: float, length: float = 36.0) -> Tuple[Point, Point]:
    theta = math.radians(angle_deg)
    dx = math.cos(theta) * length
    dy = math.sin(theta) * length
    x, y = point
    return (x - dx, y - dy), (x + dx, y + dy)


def _skeleton_points(component: np.ndarray) -> np.ndarray:
    skeleton = skeletonize(component).astype(np.uint8)
    yx = np.column_stack(np.where(skeleton > 0))
    return yx[:, ::-1].astype(np.float32)


def _point_key(point: np.ndarray) -> Tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _skeleton_graph(component: np.ndarray):
    skeleton = skeletonize(component).astype(np.uint8)
    ys, xs = np.where(skeleton > 0)
    nodes = {(int(x), int(y)) for y, x in zip(ys, xs)}
    graph = {node: [] for node in nodes}
    for x, y in nodes:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x + dx, y + dy)
                if neighbor in nodes:
                    graph[(x, y)].append((neighbor, math.hypot(dx, dy)))
    return graph


def _shortest_path(graph, start: Tuple[int, int], end: Tuple[int, int]):
    queue = [(0.0, start)]
    distances = {start: 0.0}
    previous = {}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == end:
            break
        if distance > distances.get(node, float("inf")):
            continue
        for neighbor, weight in graph.get(node, []):
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))
    if end not in distances:
        return [], 0.0
    path = [end]
    node = end
    while node != start:
        node = previous[node]
        path.append(node)
    path.reverse()
    return [(float(x), float(y)) for x, y in path], float(distances[end])


def _longest_skeleton_path(component: np.ndarray) -> Tuple[List[Point], float, Point, Point]:
    graph = _skeleton_graph(component)
    if not graph:
        points = _component_points(component)
        p1, p2 = _max_distance_pair(points)
        line = [tuple(p1), tuple(p2)]
        return line, _length(line[0], line[1]), line[0], line[1]

    endpoints = [node for node, neighbors in graph.items() if len(neighbors) == 1]
    candidates = endpoints if len(endpoints) >= 2 else list(graph.keys())
    best_path = []
    best_length = -1.0
    best_pair = (candidates[0], candidates[0])
    for i, start in enumerate(candidates):
        for end in candidates[i + 1:]:
            path, length = _shortest_path(graph, start, end)
            if length > best_length:
                best_path = path
                best_length = length
                best_pair = (start, end)
    if not best_path:
        point = (float(best_pair[0][0]), float(best_pair[0][1]))
        return [point], 0.0, point, point
    return best_path, float(best_length), best_path[0], best_path[-1]


def _extract_medial_edge_curve(component: np.ndarray, side: str) -> np.ndarray:
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    percentile = 95.0 if side == "left" else 5.0
    points = []
    for y in np.unique(ys):
        row_x = xs[ys == y]
        if len(row_x) == 0:
            continue
        x = float(np.percentile(row_x, percentile))
        points.append((x, float(y)))
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _split_curve_by_y_gap(points: np.ndarray, max_y_gap: float = 3.0) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    ordered = points[np.argsort(points[:, 1])]
    segments = []
    start = 0
    for idx in range(1, len(ordered)):
        if float(ordered[idx, 1] - ordered[idx - 1, 1]) > max_y_gap:
            segments.append(ordered[start:idx])
            start = idx
    segments.append(ordered[start:])
    return [segment for segment in segments if len(segment) > 0]


def _remove_curve_x_spikes(points: np.ndarray, radius: int = 3) -> np.ndarray:
    if len(points) < radius * 2 + 3:
        return points

    xs = points[:, 0]
    keep = np.ones(len(points), dtype=bool)
    for idx in range(len(points)):
        lo = max(0, idx - radius)
        hi = min(len(points), idx + radius + 1)
        local = xs[lo:hi]
        median = float(np.median(local))
        mad = float(np.median(np.abs(local - median)))
        threshold = max(8.0, 4.0 * mad)
        if abs(float(xs[idx]) - median) > threshold:
            keep[idx] = False
    filtered = points[keep]
    return filtered if len(filtered) >= 5 else points


def _trim_curve_by_arc_ratio(points: np.ndarray, trim_ratio: float = 0.05) -> np.ndarray:
    if len(points) < 10:
        return points
    distances = _cumulative_arc_lengths(points)
    total = float(distances[-1]) if len(distances) else 0.0
    if total <= 1e-6:
        return points
    keep = np.where((distances >= total * trim_ratio) & (distances <= total * (1.0 - trim_ratio)))[0]
    trimmed = points[keep]
    return trimmed if len(trimmed) >= 5 else points


def _smooth_curve(points: np.ndarray, window: int = 9) -> np.ndarray:
    if len(points) < 5:
        return points.astype(np.float32)

    ordered = points[np.argsort(points[:, 1])].astype(np.float64)
    window = min(int(window), len(ordered))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return ordered.astype(np.float32)

    try:
        import warnings
        from scipy.signal import savgol_filter

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            xs = savgol_filter(ordered[:, 0], window_length=window, polyorder=2)
            ys = savgol_filter(ordered[:, 1], window_length=window, polyorder=2)
    except Exception:
        kernel = np.ones(window, dtype=np.float64) / float(window)
        pad = window // 2
        xs = np.convolve(np.pad(ordered[:, 0], pad, mode="edge"), kernel, mode="valid")
        ys = np.convolve(np.pad(ordered[:, 1], pad, mode="edge"), kernel, mode="valid")
    return np.column_stack([xs, ys]).astype(np.float32)


def _clean_medial_edge_curve(points: np.ndarray) -> np.ndarray:
    if len(points) < 5:
        return points.astype(np.float32)

    segments = _split_curve_by_y_gap(points)
    if not segments:
        return np.empty((0, 2), dtype=np.float32)
    segment = max(segments, key=lambda item: (len(item), _path_length([tuple(point) for point in item])))
    segment = _remove_curve_x_spikes(segment)
    segments = _split_curve_by_y_gap(segment)
    if segments:
        segment = max(segments, key=lambda item: (len(item), _path_length([tuple(point) for point in item])))
    segment = _trim_curve_by_arc_ratio(segment, trim_ratio=0.05)
    return _smooth_curve(segment)


def _find_directional_bulge_point(curve: np.ndarray, side: str, min_offset: float = 1.0) -> Tuple[Optional[Point], float]:
    if len(curve) < 3:
        return None, 0.0

    p1 = curve[0].astype(np.float64)
    p2 = curve[-1].astype(np.float64)
    vector = p2 - p1
    denom = float(np.dot(vector, vector))
    if denom <= 1e-6:
        return None, 0.0

    offsets = []
    for point in curve.astype(np.float64):
        t = float(np.dot(point - p1, vector) / denom)
        projection = p1 + t * vector
        dx = float(point[0] - projection[0])
        offsets.append(dx if side == "left" else -dx)

    offsets_array = np.asarray(offsets, dtype=np.float64)
    index = int(np.argmax(offsets_array))
    offset = float(offsets_array[index])
    if offset <= min_offset:
        return None, offset
    return tuple(curve[index].astype(np.float32)), offset


def _angle_between_three_points(p1: Point, pm: Point, p2: Point) -> Optional[float]:
    p1_array = np.asarray(p1, dtype=np.float64)
    pm_array = np.asarray(pm, dtype=np.float64)
    p2_array = np.asarray(p2, dtype=np.float64)
    v1 = p1_array - pm_array
    v2 = p2_array - pm_array
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 <= 1e-6 or norm2 <= 1e-6:
        return None
    cos_value = float(np.dot(v1, v2) / (norm1 * norm2))
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def _measure_medial_edge_curvature(component: np.ndarray, side: str) -> Dict[str, object]:
    raw_curve = _extract_medial_edge_curve(component, side)
    curve = _clean_medial_edge_curve(raw_curve)
    if len(curve) < 5:
        return {
            "curvature_measurement_status": "too_few_medial_edge_points",
            "curve_path": [tuple(point.astype(np.float32)) for point in curve],
            "curve_chord_line": None,
            "curve_length_px": 0.0,
            "curve_chord_px": 0.0,
            "curvature_ratio": 0.0,
            "curve_bulge_point": None,
            "curve_bulge_offset_px": 0.0,
            "curvature_angle_deg": None,
        }

    curve_path = [tuple(point.astype(np.float32)) for point in curve]
    curve_start = curve_path[0]
    curve_end = curve_path[-1]
    curve_length_px = _path_length(curve_path)
    chord_px = _length(curve_start, curve_end)
    bulge_point, bulge_offset = _find_directional_bulge_point(curve, side)
    curvature_ratio = float(curve_length_px / chord_px) if chord_px > 1e-6 and bulge_point is not None else 0.0
    curvature_angle = _angle_between_three_points(curve_start, bulge_point, curve_end) if bulge_point is not None else None

    return {
        "curvature_measurement_status": "ok" if bulge_point is not None else "no_directional_bulge",
        "curve_path": curve_path,
        "curve_chord_line": (curve_start, curve_end),
        "curve_length_px": float(curve_length_px) if bulge_point is not None else 0.0,
        "curve_chord_px": float(chord_px) if bulge_point is not None else 0.0,
        "curvature_ratio": float(curvature_ratio),
        "curve_bulge_point": bulge_point,
        "curve_bulge_offset_px": float(bulge_offset),
        "curvature_angle_deg": float(curvature_angle) if curvature_angle is not None else None,
    }


def _measure_opening_width_from_curvature(
    curvature_info: Dict[str, object],
    pixel_spacing: Optional[Tuple[float, float]],
) -> Dict[str, object]:
    chord_line = curvature_info.get("curve_chord_line")
    bulge_point = curvature_info.get("curve_bulge_point")
    if not isinstance(chord_line, Sequence):
        return {
            "opening_width_measurement_status": str(curvature_info.get("curvature_measurement_status", "failed")),
            "width_px": 0.0,
            "width_mm": None,
            "width_line": None,
            "opening_width_dash_lines": None,
            "opening_width_segment_a_px": 0.0,
            "opening_width_segment_b_px": 0.0,
            "opening_width_segment_a_mm": None,
            "opening_width_segment_b_mm": None,
            "opening_width_yellow_dashed_length_px": 0.0,
            "opening_width_yellow_dashed_length_mm": None,
            "opening_width_point_count": 0,
            "opening_width_gap_point_count": 0,
            "opening_width_mode": "curvature_yellow_dashed_unavailable",
            "opening_width_smooth_px": 0.0,
        }

    p1, p2 = chord_line
    if not isinstance(bulge_point, Sequence):
        chord_px = _length(p1, p2)
        chord_mm = _length(p1, p2, pixel_spacing) if pixel_spacing else None
        return {
            "opening_width_measurement_status": "ok",
            "width_px": float(chord_px),
            "width_mm": float(chord_mm) if chord_mm is not None else None,
            "width_line": (p1, p2),
            "opening_width_dash_lines": ((p1, p2),),
            "opening_width_segment_a_px": float(chord_px),
            "opening_width_segment_b_px": 0.0,
            "opening_width_segment_a_mm": float(chord_mm) if chord_mm is not None else None,
            "opening_width_segment_b_mm": None,
            "opening_width_yellow_dashed_length_px": float(chord_px),
            "opening_width_yellow_dashed_length_mm": float(chord_mm) if chord_mm is not None else None,
            "opening_width_point_count": int(len(curvature_info.get("curve_path", []))),
            "opening_width_gap_point_count": 0,
            "opening_width_mode": "curvature_yellow_chord_line",
            "opening_width_smooth_px": float(chord_px),
        }

    pm = tuple(np.asarray(bulge_point, dtype=np.float32))
    segment_a_px = _length(p1, pm)
    segment_b_px = _length(pm, p2)
    width_px = segment_a_px + segment_b_px
    segment_a_mm = _length(p1, pm, pixel_spacing) if pixel_spacing else None
    segment_b_mm = _length(pm, p2, pixel_spacing) if pixel_spacing else None
    width_mm = segment_a_mm + segment_b_mm if segment_a_mm is not None and segment_b_mm is not None else None

    return {
        "opening_width_measurement_status": "ok",
        "width_px": float(width_px),
        "width_mm": float(width_mm) if width_mm is not None else None,
        "width_line": (p1, p2),
        "opening_width_dash_lines": ((p1, pm), (pm, p2)),
        "opening_width_segment_a_px": float(segment_a_px),
        "opening_width_segment_b_px": float(segment_b_px),
        "opening_width_segment_a_mm": float(segment_a_mm) if segment_a_mm is not None else None,
        "opening_width_segment_b_mm": float(segment_b_mm) if segment_b_mm is not None else None,
        "opening_width_yellow_dashed_length_px": float(width_px),
        "opening_width_yellow_dashed_length_mm": float(width_mm) if width_mm is not None else None,
        "opening_width_point_count": int(len(curvature_info.get("curve_path", []))),
        "opening_width_gap_point_count": 0,
        "opening_width_mode": "curvature_yellow_dashed_polyline",
        "opening_width_smooth_px": float(width_px),
    }


def _scale_length(length_px: float, pixel_spacing: Optional[Tuple[float, float]]) -> Optional[float]:
    if pixel_spacing is None:
        return None
    row_spacing, col_spacing = pixel_spacing
    return float(length_px * (row_spacing + col_spacing) / 2.0)


def locate_third_ventricle_apex(mask: np.ndarray, apex: str = "inferior") -> Optional[Point]:
    points = _component_points(mask > 0)
    if len(points) == 0:
        return None
    if apex == "superior":
        limit = np.percentile(points[:, 1], 5)
        candidates = points[points[:, 1] <= limit]
        y_value = float(candidates[:, 1].min())
    else:
        limit = np.percentile(points[:, 1], 95)
        candidates = points[points[:, 1] >= limit]
        y_value = float(candidates[:, 1].max())
    if len(candidates) == 0:
        candidates = points
    x_value = float(np.median(candidates[:, 0]))
    return x_value, y_value


def _horizontal_line_through(point: Point, image_width: int, margin: int = 8) -> Tuple[Point, Point]:
    x0, y0 = point
    return (float(margin), float(y0)), (float(max(margin, image_width - margin - 1)), float(y0))


def _normalize_vec(vector, eps: float = 1e-8) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return None
    return vector / norm


def _angle_between_directions(v1, v2) -> Optional[float]:
    v1 = _normalize_vec(v1)
    v2 = _normalize_vec(v2)
    if v1 is None or v2 is None:
        return None
    cross = float(v1[0] * v2[1] - v1[1] * v2[0])
    dot = float(v1[0] * v2[0] + v1[1] * v2[1])
    angle = math.degrees(math.atan2(abs(cross), abs(dot)))
    return float(max(0.0, min(90.0, angle)))


def _line_through_direction(point: Point, direction, image_shape: Tuple[int, int]) -> Optional[Tuple[Point, Point]]:
    direction = _normalize_vec(direction)
    if direction is None:
        return None
    height, width = image_shape
    diagonal = math.hypot(width, height)
    point_array = np.asarray(point, dtype=np.float64)
    p1 = point_array - direction * diagonal
    p2 = point_array + direction * diagonal
    return tuple(p1.astype(np.float32)), tuple(p2.astype(np.float32))


def _long_line_endpoints(point: Point, direction, length: float = 170.0) -> Optional[Tuple[Point, Point]]:
    direction = _normalize_vec(direction)
    if direction is None:
        return None
    point_array = np.asarray(point, dtype=np.float64)
    p1 = point_array - direction * (length / 2.0)
    p2 = point_array + direction * (length / 2.0)
    return tuple(p1.astype(np.float32)), tuple(p2.astype(np.float32))


def _path_length(path: Sequence[Point]) -> float:
    if not isinstance(path, Sequence) or len(path) < 2:
        return 0.0
    return float(sum(_length(path[i - 1], path[i]) for i in range(1, len(path))))


def _trim_path_by_arc_length(path: List[Point], max_arc_length: float) -> List[Point]:
    if len(path) <= 1:
        return path
    output = [path[0]]
    total = 0.0
    for i in range(1, len(path)):
        total += _length(path[i - 1], path[i])
        output.append(path[i])
        if total >= max_arc_length:
            break
    return output


def _ensure_superior_to_inferior_path(path: Sequence[Point]) -> List[Point]:
    points = [tuple(point) for point in path]
    if len(points) >= 2 and points[0][1] > points[-1][1]:
        points.reverse()
    return points


def _cumulative_arc_lengths(path: Sequence[Point]) -> np.ndarray:
    points = np.asarray(path, dtype=np.float64)
    if len(points) == 0:
        return np.asarray([], dtype=np.float64)
    distances = np.zeros(len(points), dtype=np.float64)
    for idx in range(1, len(points)):
        distances[idx] = distances[idx - 1] + float(np.linalg.norm(points[idx] - points[idx - 1]))
    return distances


def _point_at_arc_ratio(path: Sequence[Point], ratio: float = 0.60) -> Tuple[Optional[np.ndarray], Optional[int]]:
    ordered = _ensure_superior_to_inferior_path(path)
    if not ordered:
        return None, None
    points = np.asarray(ordered, dtype=np.float64)
    if len(points) == 1:
        return points[0], 0
    distances = _cumulative_arc_lengths(points)
    total = float(distances[-1])
    if total <= 1e-8:
        return points[0], 0
    target = float(ratio) * total
    index = int(np.argmin(np.abs(distances - target)))
    return points[index], index


def _local_points_around_anchor(
    path: Sequence[Point],
    anchor_index: Optional[int],
    half_arc_length: float = 30.0,
) -> np.ndarray:
    ordered = _ensure_superior_to_inferior_path(path)
    if not ordered or anchor_index is None:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(ordered, dtype=np.float64)
    distances = _cumulative_arc_lengths(points)
    center_distance = float(distances[int(anchor_index)])
    keep = np.where(
        (distances >= center_distance - half_arc_length) &
        (distances <= center_distance + half_arc_length)
    )[0]
    return points[keep]


def _farthest_pair_path(graph) -> List[Point]:
    endpoints = [node for node, neighbors in graph.items() if len(neighbors) <= 1]
    if len(endpoints) == 0:
        return []
    if len(endpoints) == 1:
        return [(float(endpoints[0][0]), float(endpoints[0][1]))]

    best_path = []
    best_length = -1.0
    for i, start in enumerate(endpoints):
        for end in endpoints[i + 1:]:
            path, length = _shortest_path(graph, start, end)
            if path and length > best_length:
                best_path = path
                best_length = length
    return best_path


def _nearest_graph_node(graph, point: np.ndarray) -> Optional[Tuple[int, int]]:
    if not graph:
        return None
    nodes = list(graph.keys())
    node_array = np.asarray(nodes, dtype=np.float32)
    distances = np.linalg.norm(node_array - point[None, :], axis=1)
    return nodes[int(np.argmin(distances))]


def _geodesic_local_graph_points(graph, start: Optional[Tuple[int, int]], max_distance: float) -> np.ndarray:
    if start is None or start not in graph:
        return np.empty((0, 2), dtype=np.float64)

    distances = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > max_distance:
            continue
        if distance > distances.get(node, float("inf")):
            continue
        for neighbor, weight in graph.get(node, []):
            next_distance = distance + weight
            if next_distance <= max_distance and next_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = next_distance
                heapq.heappush(queue, (next_distance, neighbor))
    return np.asarray(list(distances.keys()), dtype=np.float64)


def _select_upper_arm_points(
    component: np.ndarray,
    side: str,
    image_shape: Tuple[int, int],
    min_fit_points: int = 20,
    max_upper_arm_ratio: float = 0.55,
) -> Tuple[np.ndarray, Dict[str, object]]:
    clean_mask = preprocess_mask(component, min_area=20)
    debug = {
        "side": side,
        "mask_area": int(np.sum(clean_mask > 0)),
        "skeleton_point_count": 0,
        "endpoint_count": 0,
        "branch_point_count": 0,
        "upper_arm_point_count": 0,
        "upper_arm_path_length": 0.0,
        "status": "failed",
        "warnings": [],
    }
    if debug["mask_area"] == 0:
        debug["warnings"].append("mask is empty after preprocessing")
        return np.empty((0, 2), dtype=np.float32), debug

    skeleton = skeletonize(clean_mask > 0).astype(np.uint8)
    skeleton_yx = np.column_stack(np.where(skeleton > 0))
    debug["skeleton_point_count"] = int(len(skeleton_yx))
    if len(skeleton_yx) < min_fit_points:
        debug["warnings"].append("too few skeleton points")
        return np.empty((0, 2), dtype=np.float32), debug

    graph = _skeleton_graph(clean_mask)
    if not graph:
        debug["warnings"].append("empty skeleton graph")
        return np.empty((0, 2), dtype=np.float32), debug

    endpoints = [node for node, neighbors in graph.items() if len(neighbors) <= 1]
    branch_points = [node for node, neighbors in graph.items() if len(neighbors) >= 3]
    debug["endpoint_count"] = int(len(endpoints))
    debug["branch_point_count"] = int(len(branch_points))

    if not endpoints:
        debug["warnings"].append("no endpoint found, fallback to all skeleton points")
        points = _component_points(skeleton > 0)
        debug["upper_arm_point_count"] = int(len(points))
        debug["status"] = "warning"
        return points, debug

    image_height = int(image_shape[0])
    default_arc = float(np.clip(0.12 * image_height, 45.0, 120.0))
    upper_path: List[Point] = []

    if branch_points:
        branch_array = np.asarray(branch_points, dtype=np.float32)
        junction = _nearest_graph_node(graph, branch_array.mean(axis=0))
        if junction is not None:
            best_path = []
            best_score = None
            for endpoint in sorted(endpoints, key=lambda point: (point[1], abs(point[0] - junction[0]))):
                path, length = _shortest_path(graph, junction, endpoint)
                if len(path) < 2 or length < 8.0:
                    continue
                score = float(endpoint[1]) - 0.05 * float(length)
                if best_score is None or score < best_score:
                    best_score = score
                    best_path = path
            if best_path:
                upper_path = list(reversed(best_path))

    if len(upper_path) < min_fit_points:
        main_path = _farthest_pair_path(graph)
        if not main_path:
            debug["warnings"].append("failed to find main skeleton path")
            return np.empty((0, 2), dtype=np.float32), debug
        if main_path[0][1] > main_path[-1][1]:
            main_path = list(reversed(main_path))
        total_length = _path_length(main_path)
        if branch_points and len(upper_path) > 1:
            upper_path = main_path
        else:
            arc_length = min(default_arc, max_upper_arm_ratio * total_length)
            arc_length = max(arc_length, 30.0)
            upper_path = _trim_path_by_arc_length(main_path, arc_length)
            if len(upper_path) < min_fit_points:
                upper_path = _trim_path_by_arc_length(main_path, 0.70 * total_length)

    upper_points = np.asarray(upper_path, dtype=np.float32)
    debug["upper_arm_point_count"] = int(len(upper_points))
    debug["upper_arm_path_length"] = _path_length(upper_path)
    if len(upper_points) < min_fit_points:
        debug["warnings"].append(f"upper arm points too few: {len(upper_points)} < {min_fit_points}")
        return upper_points, debug
    debug["status"] = "success"
    return upper_points, debug


def _fit_line_pca(points: np.ndarray, line_length: float = 180.0) -> Tuple[Optional[Tuple[Point, Point]], Optional[np.ndarray], Optional[np.ndarray]]:
    if points is None or len(points) < 2:
        return None, None, None
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    covariance = centered.T @ centered / max(len(pts) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return None, center, None
    direction = direction / norm
    if direction[1] < 0:
        direction = -direction
    p1 = center - direction * (line_length / 2.0)
    p2 = center + direction * (line_length / 2.0)
    return (tuple(p1.astype(np.float32)), tuple(p2.astype(np.float32))), center, direction


def _trim_path_by_ratio(path: Sequence[Point], start_ratio: float = 0.0, end_ratio: float = 1.0) -> List[Point]:
    if len(path) <= 2:
        return [tuple(point) for point in path]
    points = np.asarray(path, dtype=np.float64)
    distances = _cumulative_arc_lengths(points)
    total = float(distances[-1])
    if total <= 1e-8:
        return [tuple(point) for point in path]
    start_ratio = float(np.clip(start_ratio, 0.0, 1.0))
    end_ratio = float(np.clip(end_ratio, start_ratio, 1.0))
    keep = np.where((distances >= start_ratio * total) & (distances <= end_ratio * total))[0]
    if len(keep) == 0:
        return []
    return [tuple(points[idx].astype(np.float32)) for idx in keep]


def _terminal_points_from_upper_arc(upper_arc_path: Sequence[Point], terminal_arc_length: float = 35.0) -> np.ndarray:
    if upper_arc_path is None or len(upper_arc_path) == 0:
        return np.empty((0, 2), dtype=np.float64)
    points = np.asarray(upper_arc_path, dtype=np.float64)
    distances = _cumulative_arc_lengths(points)
    total = float(distances[-1]) if len(distances) else 0.0
    keep = np.where(distances >= max(0.0, total - terminal_arc_length))[0]
    return points[keep]


def _component_bbox(mask: np.ndarray) -> Optional[Dict[str, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": int(x_max - x_min + 1),
        "height": int(y_max - y_min + 1),
    }


def _shortest_path_to_targets(graph, start: Tuple[int, int], targets) -> Tuple[List[Point], float]:
    target_set = set(targets)
    if start in target_set:
        return [(float(start[0]), float(start[1]))], 0.0
    queue = [(0.0, start)]
    distances = {start: 0.0}
    previous = {}
    hit = None
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, float("inf")):
            continue
        if node in target_set:
            hit = node
            break
        for neighbor, weight in graph.get(node, []):
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))
    if hit is None:
        return [], 0.0
    path = [hit]
    node = hit
    while node != start:
        node = previous[node]
        path.append(node)
    path.reverse()
    return [(float(x), float(y)) for x, y in path], float(distances[hit])


def _dilate_graph_nodes(graph, seed_nodes, radius: float = 6.0):
    if not seed_nodes:
        return set()
    output = set(seed_nodes)
    distances = {}
    queue = []
    for node in seed_nodes:
        distances[node] = 0.0
        heapq.heappush(queue, (0.0, node))
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > radius:
            continue
        if distance > distances.get(node, float("inf")):
            continue
        output.add(node)
        for neighbor, weight in graph.get(node, []):
            new_distance = distance + weight
            if new_distance <= radius and new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))
    return output


def _trim_path_before_max_y(path: Sequence[Point], max_y: float) -> List[Point]:
    if not path:
        return []
    trimmed = []
    for point in path:
        if float(point[1]) > float(max_y):
            break
        trimmed.append(tuple(point))
    if len(trimmed) >= 3:
        return trimmed
    return [tuple(point) for point in path[:max(3, min(len(path), 6))]]


def _extract_upper_arc_path(
    component: np.ndarray,
    side: str,
    branch_cut_radius: float = 6.0,
    terminal_max_depth_ratio: float = 0.62,
    upper_arc_end_ratio: float = 0.55,
    min_path_length: float = 10.0,
) -> Tuple[np.ndarray, Dict[str, object]]:
    clean_mask = preprocess_mask(component, min_area=20)
    bbox = _component_bbox(clean_mask)
    debug = {
        "side": side,
        "status": "failed",
        "warning": "",
        "branch_cut_radius": float(branch_cut_radius),
        "terminal_max_depth_ratio": float(terminal_max_depth_ratio),
        "upper_arc_end_ratio": float(upper_arc_end_ratio),
        "skeleton_point_count": 0,
        "endpoint_count": 0,
        "branchpoint_count": 0,
        "upper_arc_path_point_count": 0,
        "upper_arc_path_length": 0.0,
        "upper_arc_terminal": None,
        "upper_arc_inferior_point": None,
    }
    if bbox is None or int(np.sum(clean_mask > 0)) == 0:
        debug["warning"] = "mask is empty after preprocessing"
        return np.empty((0, 2), dtype=np.float32), debug

    skeleton = skeletonize(clean_mask > 0).astype(np.uint8)
    skeleton_yx = np.column_stack(np.where(skeleton > 0))
    debug["skeleton_point_count"] = int(len(skeleton_yx))
    if len(skeleton_yx) < 5:
        debug["warning"] = "too few skeleton points"
        return np.empty((0, 2), dtype=np.float32), debug

    graph = _skeleton_graph(clean_mask)
    if not graph:
        debug["warning"] = "empty skeleton graph"
        return np.empty((0, 2), dtype=np.float32), debug

    endpoints = [node for node, neighbors in graph.items() if len(neighbors) <= 1]
    branchpoints = [node for node, neighbors in graph.items() if len(neighbors) >= 3]
    debug["endpoint_count"] = int(len(endpoints))
    debug["branchpoint_count"] = int(len(branchpoints))
    if not endpoints:
        debug["warning"] = "no endpoint found"
        return np.empty((0, 2), dtype=np.float32), debug

    max_allowed_y = float(bbox["y_min"] + float(terminal_max_depth_ratio) * (bbox["y_max"] - bbox["y_min"]))
    upper_arc_path: List[Point] = []
    if branchpoints:
        branch_zone = _dilate_graph_nodes(graph, branchpoints, radius=branch_cut_radius)
        candidates = []
        for endpoint in endpoints:
            if endpoint in branch_zone:
                continue
            path, length = _shortest_path_to_targets(graph, endpoint, branch_zone)
            if len(path) < 3 or length < min_path_length:
                continue
            # The last node is inside the branch exclusion zone; terminal is before it.
            path = path[:-1]
            if len(path) < 3:
                continue
            path = _ensure_superior_to_inferior_path(path)
            path = _trim_path_before_max_y(path, max_allowed_y)
            if len(path) < 3:
                continue
            score = float(endpoint[1]) - 0.02 * float(_path_length(path))
            candidates.append((score, endpoint, path, _path_length(path)))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            upper_arc_path = _ensure_superior_to_inferior_path(candidates[0][2])
        else:
            debug["warning"] = "no valid endpoint-to-prejunction upper arc path"

    if len(upper_arc_path) < 3:
        main_path = _farthest_pair_path(graph)
        if not main_path:
            debug["warning"] = debug["warning"] or "failed to find main skeleton path"
            return np.empty((0, 2), dtype=np.float32), debug
        main_path = _ensure_superior_to_inferior_path(main_path)
        upper_arc_path = _trim_path_by_ratio(
            main_path,
            start_ratio=0.0,
            end_ratio=float(np.clip(upper_arc_end_ratio, 0.50, 0.65)),
        )
        upper_arc_path = _trim_path_before_max_y(upper_arc_path, max_allowed_y)
        if len(upper_arc_path) < 5:
            upper_arc_path = _trim_path_before_max_y(main_path, max_allowed_y)

    upper_arc_path = _ensure_superior_to_inferior_path(upper_arc_path)
    if len(upper_arc_path) < 3:
        debug["warning"] = debug["warning"] or "upper arc path too short"
        return np.asarray(upper_arc_path, dtype=np.float32), debug

    upper_arc_points = np.asarray(upper_arc_path, dtype=np.float32)
    terminal = upper_arc_points[-1].astype(np.float64)
    debug["status"] = "success"
    debug["warning"] = ""
    debug["upper_arc_path_point_count"] = int(len(upper_arc_points))
    debug["upper_arc_path_length"] = _path_length([tuple(point) for point in upper_arc_points])
    debug["upper_arc_terminal"] = tuple(terminal.astype(np.float32))
    debug["upper_arc_inferior_point"] = tuple(terminal.astype(np.float32))
    return upper_arc_points, debug


def _estimate_reference_line(
    left_component: np.ndarray,
    right_component: np.ndarray,
    apex: Optional[Point] = None,
    baseline_left_point: Optional[Point] = None,
    baseline_right_point: Optional[Point] = None,
    mode: str = "auto",
    branch_cut_radius: float = 6.0,
    terminal_max_depth_ratio: float = 0.62,
    upper_arc_end_ratio: float = 0.55,
    pass_through_apex: bool = False,
) -> Dict[str, object]:
    left_upper_arc_path, left_debug = _extract_upper_arc_path(
        left_component,
        side="left",
        branch_cut_radius=branch_cut_radius,
        terminal_max_depth_ratio=terminal_max_depth_ratio,
        upper_arc_end_ratio=upper_arc_end_ratio,
    )
    right_upper_arc_path, right_debug = _extract_upper_arc_path(
        right_component,
        side="right",
        branch_cut_radius=branch_cut_radius,
        terminal_max_depth_ratio=terminal_max_depth_ratio,
        upper_arc_end_ratio=upper_arc_end_ratio,
    )
    if left_debug.get("status") != "success" or right_debug.get("status") != "success":
        raise ValueError("failed to extract bilateral upper arc terminals")
    left_upper_anchor = np.asarray(left_upper_arc_path[-1], dtype=np.float64)
    right_upper_anchor = np.asarray(right_upper_arc_path[-1], dtype=np.float64)

    if baseline_left_point is not None and baseline_right_point is not None:
        left_anchor = np.asarray(baseline_left_point, dtype=np.float64)
        right_anchor = np.asarray(baseline_right_point, dtype=np.float64)
        direction = _normalize_vec(right_anchor - left_anchor)
        if direction is None:
            raise ValueError("baseline points are too close to define a reference line")
        use_apex = pass_through_apex and apex is not None and mode == "apex_parallel"
        reference_point = np.asarray(apex, dtype=np.float64) if use_apex else (left_anchor + right_anchor) / 2.0
        line_mode = "apex_parallel" if use_apex else "manual"
        return {
            "mode": line_mode,
            "point": reference_point,
            "direction": direction,
            "left_anchor": left_anchor,
            "right_anchor": right_anchor,
            "left_upper_arc_path": left_upper_arc_path,
            "right_upper_arc_path": right_upper_arc_path,
            "left_upper_arc_terminal": left_upper_anchor,
            "right_upper_arc_terminal": right_upper_anchor,
            "left_upper_arc_inferior_point": left_upper_anchor,
            "right_upper_arc_inferior_point": right_upper_anchor,
            "left_upper_arc_debug": left_debug,
            "right_upper_arc_debug": right_debug,
            "branch_cut_radius": float(branch_cut_radius),
            "terminal_max_depth_ratio": float(terminal_max_depth_ratio),
            "upper_arc_end_ratio": float(np.clip(upper_arc_end_ratio, 0.50, 0.65)),
        }

    direction = _normalize_vec(right_upper_anchor - left_upper_anchor)
    if direction is None:
        raise ValueError("reference anchors are too close to define a reference line")
    reference_point = (left_upper_anchor + right_upper_anchor) / 2.0
    return {
        "mode": "upper_arc_prejunction_terminals" if mode in ("auto", "upper_arc_prejunction_terminals", "upper_arc_inferior_points") else mode,
        "point": reference_point,
        "direction": direction,
        "left_anchor": left_upper_anchor,
        "right_anchor": right_upper_anchor,
        "left_upper_arc_path": left_upper_arc_path,
        "right_upper_arc_path": right_upper_arc_path,
        "left_upper_arc_terminal": left_upper_anchor,
        "right_upper_arc_terminal": right_upper_anchor,
        "left_upper_arc_inferior_point": left_upper_anchor,
        "right_upper_arc_inferior_point": right_upper_anchor,
        "left_upper_arc_debug": left_debug,
        "right_upper_arc_debug": right_debug,
        "branch_cut_radius": float(branch_cut_radius),
        "terminal_max_depth_ratio": float(terminal_max_depth_ratio),
        "upper_arc_end_ratio": float(np.clip(upper_arc_end_ratio, 0.50, 0.65)),
    }


def _measure_local_sylvian_angle(
    upper_arc_path: Sequence[Point],
    reference_direction,
    terminal_arc_length: float = 28.0,
    min_local_points: int = 6,
) -> Dict[str, object]:
    if upper_arc_path is None or len(upper_arc_path) < 3:
        return {
            "status": "failed",
            "warning": "upper arc path too short",
            "angle_deg": None,
            "measurement_anchor": None,
            "local_points": np.empty((0, 2), dtype=np.float64),
            "tangent_center": None,
            "tangent_direction": None,
            "tangent_line": None,
            "nearest_skeleton_point": None,
        }

    ordered_path = _ensure_superior_to_inferior_path(upper_arc_path)
    anchor = np.asarray(ordered_path[-1], dtype=np.float64)
    local_points = _terminal_points_from_upper_arc(ordered_path, terminal_arc_length=terminal_arc_length)
    if len(local_points) < min_local_points:
        local_points = _terminal_points_from_upper_arc(ordered_path, terminal_arc_length=terminal_arc_length * 1.8)
    if len(local_points) < min_local_points:
        return {
            "status": "failed",
            "warning": f"local tangent points too few: {len(local_points)}",
            "angle_deg": None,
            "measurement_anchor": anchor,
            "local_points": local_points,
            "tangent_center": None,
            "tangent_direction": None,
            "tangent_line": None,
            "nearest_skeleton_point": tuple(anchor.astype(np.float32)),
        }

    _, tangent_center, tangent_direction = _fit_line_pca(local_points, line_length=1.0)
    if tangent_direction is None:
        return {
            "status": "failed",
            "warning": "PCA tangent fitting failed",
            "angle_deg": None,
            "measurement_anchor": anchor,
            "local_points": local_points,
            "tangent_center": tangent_center,
            "tangent_direction": None,
            "tangent_line": None,
            "nearest_skeleton_point": tuple(anchor.astype(np.float32)),
        }

    angle = _angle_between_directions(tangent_direction, reference_direction)
    tangent_line = _long_line_endpoints(anchor, tangent_direction, length=170.0)
    return {
        "status": "success" if angle is not None and tangent_line is not None else "failed",
        "warning": "",
        "angle_deg": angle,
        "measurement_anchor": anchor,
        "local_points": local_points,
        "tangent_center": tangent_center,
        "tangent_direction": tangent_direction,
        "tangent_line": tangent_line,
        "nearest_skeleton_point": tuple(anchor.astype(np.float32)),
    }


def _component_upper_arm_tangent(component: np.ndarray, side: str, lip_a: np.ndarray, lip_b: np.ndarray) -> Dict[str, object]:
    upper_points, debug = _select_upper_arm_points(component, side, component.shape, min_fit_points=20)
    tangent_line, fit_center, fit_direction = _fit_line_pca(upper_points, line_length=180.0)
    if tangent_line is None:
        upper_lip = lip_a if float(lip_a[1]) <= float(lip_b[1]) else lip_b
        point = tuple(upper_lip)
        tangent_line = (point, point)
    return {
        "upper_arm_path": [tuple(point) for point in upper_points],
        "upper_arm_tangent_line": tangent_line,
        "upper_arm_fit_center": tuple(fit_center.astype(np.float32)) if fit_center is not None else None,
        "upper_arm_fit_direction": tuple(fit_direction.astype(np.float32)) if fit_direction is not None else None,
        "upper_arm_debug": debug,
    }


def _measure_component(
    component: np.ndarray,
    area: int,
    side: str,
    pixel_spacing: Optional[Tuple[float, float]],
) -> Dict[str, object]:
    points = _component_points(component)
    endpoints = _skeleton_endpoints(component)
    candidates = endpoints if len(endpoints) >= 3 else points

    lip_a, lip_b = _select_opening_lips(candidates, side)
    curvature_info = _measure_medial_edge_curvature(component, side)
    opening_width_info = _measure_opening_width_from_curvature(curvature_info, pixel_spacing)
    depth_info = _measure_middle_depth(component, pixel_spacing)
    width_line = opening_width_info["width_line"]
    depth_line = depth_info["depth_line"]
    width_px = float(opening_width_info["width_px"])
    depth_px = float(depth_info["depth_px"])
    width_mm = opening_width_info["width_mm"]
    depth_mm = depth_info["depth_mm"]
    curve_length_px = float(curvature_info["curve_length_px"])
    chord_px = float(curvature_info["curve_chord_px"])
    curve_length_mm = _scale_length(curve_length_px, pixel_spacing) if curve_length_px > 0.0 else None
    curve_chord_line = curvature_info.get("curve_chord_line")
    if pixel_spacing and isinstance(curve_chord_line, Sequence):
        curve_start, curve_end = curve_chord_line
        chord_mm = _length(curve_start, curve_end, pixel_spacing)
    else:
        chord_mm = None
    angle_info = _component_upper_arm_tangent(component, side, lip_a, lip_b)

    return {
        "status": "ok",
        "side": side,
        "area_px": int(area),
        "depth_px": float(depth_px),
        "width_px": float(width_px),
        "depth_mm": float(depth_mm) if depth_mm is not None else None,
        "width_mm": float(width_mm) if width_mm is not None else None,
        "depth_line": depth_line,
        "depth_measurement_status": depth_info["depth_measurement_status"],
        "depth_y": depth_info["depth_y"],
        "depth_point_count": depth_info["depth_point_count"],
        "depth_smooth_px": depth_info["depth_smooth_px"],
        "width_line": width_line,
        "opening_width_measurement_status": opening_width_info["opening_width_measurement_status"],
        "opening_width_dash_lines": opening_width_info["opening_width_dash_lines"],
        "opening_width_segment_a_px": opening_width_info["opening_width_segment_a_px"],
        "opening_width_segment_b_px": opening_width_info["opening_width_segment_b_px"],
        "opening_width_segment_a_mm": opening_width_info["opening_width_segment_a_mm"],
        "opening_width_segment_b_mm": opening_width_info["opening_width_segment_b_mm"],
        "opening_width_yellow_dashed_length_px": opening_width_info["opening_width_yellow_dashed_length_px"],
        "opening_width_yellow_dashed_length_mm": opening_width_info["opening_width_yellow_dashed_length_mm"],
        "opening_width_point_count": opening_width_info["opening_width_point_count"],
        "opening_width_gap_point_count": opening_width_info["opening_width_gap_point_count"],
        "opening_width_mode": opening_width_info["opening_width_mode"],
        "opening_width_smooth_px": opening_width_info["opening_width_smooth_px"],
        "curve_path": curvature_info["curve_path"],
        "curve_chord_line": curvature_info["curve_chord_line"],
        "curve_length_px": float(curve_length_px),
        "curve_chord_px": float(chord_px),
        "curvature_ratio": float(curvature_info["curvature_ratio"]),
        "curvature_measurement_status": curvature_info["curvature_measurement_status"],
        "curve_bulge_point": curvature_info["curve_bulge_point"],
        "curve_bulge_offset_px": curvature_info["curve_bulge_offset_px"],
        "curvature_angle_deg": curvature_info["curvature_angle_deg"],
        "curve_length_mm": float(curve_length_mm) if curve_length_mm is not None else None,
        "curve_chord_mm": float(chord_mm) if chord_mm is not None else None,
        "angle_measurement_status": "pending",
        "angle_deg": None,
        "angle_line": None,
        "angle_horizontal_line": None,
        **angle_info,
        "component_mask": component,
    }


def _empty_result(component_count: int, area: int, shape: Tuple[int, int]) -> Dict[str, object]:
    return {
        "status": "empty",
        "component_count": component_count,
        "measured_component_count": 0,
        "area_px": int(area),
        "depth_px": 0.0,
        "depth_measurement_status": "empty",
        "depth_y": None,
        "depth_point_count": 0,
        "depth_smooth_px": 0.0,
        "width_px": 0.0,
        "mean_width_px": 0.0,
        "opening_width_measurement_status": "empty",
        "opening_width_dash_lines": None,
        "opening_width_segment_a_px": 0.0,
        "opening_width_segment_b_px": 0.0,
        "opening_width_segment_a_mm": None,
        "opening_width_segment_b_mm": None,
        "opening_width_yellow_dashed_length_px": 0.0,
        "opening_width_yellow_dashed_length_mm": None,
        "opening_width_point_count": 0,
        "opening_width_gap_point_count": 0,
        "opening_width_mode": "",
        "opening_width_smooth_px": 0.0,
        "curvature_ratio": 0.0,
        "curve_length_px": 0.0,
        "curve_chord_px": 0.0,
        "curvature_measurement_status": "empty",
        "curve_bulge_point": None,
        "curve_bulge_offset_px": 0.0,
        "curvature_angle_deg": None,
        "angle_measurement_status": "empty",
        "angle_deg": None,
        "mean_angle_deg": None,
        "third_ventricle_apex": None,
        "reference_line_mode": "",
        "reference_point": None,
        "reference_direction": None,
        "left_reference_anchor": None,
        "right_reference_anchor": None,
        "depth_mm": None,
        "width_mm": None,
        "mean_width_mm": None,
        "curve_length_mm": None,
        "curve_chord_mm": None,
        "orientation_deg": 0.0,
        "depth_line": None,
        "width_line": None,
        "curve_path": [],
        "curve_chord_line": None,
        "upper_arm_path": [],
        "angle_line": None,
        "angle_horizontal_line": None,
        "component_mask": np.zeros(shape, dtype=bool),
        "component_measurements": [],
    }


def _apply_bilateral_angle_measurement(
    measured: List[Dict[str, object]],
    reference_point: Optional[Point],
    image_shape: Tuple[int, int],
    baseline_left_point: Optional[Point] = None,
    baseline_right_point: Optional[Point] = None,
    reference_line_mode: str = "auto",
    branch_cut_radius: float = 6.0,
    terminal_max_depth_ratio: float = 0.62,
    upper_arc_end_ratio: float = 0.55,
    local_arc_length: float = 28.0,
    reference_pass_through_apex: bool = False,
):
    side_items = {}
    for item in measured:
        side = item.get("side")
        if side not in ("left", "right"):
            continue
        previous = side_items.get(side)
        if previous is None or int(item["area_px"]) > int(previous["area_px"]):
            side_items[side] = item

    if "left" not in side_items or "right" not in side_items:
        for item in measured:
            item["angle_measurement_status"] = "not_bilateral"
            item["angle_deg"] = None
            item["angle_line"] = None
            item["angle_horizontal_line"] = None
        return "not_bilateral", {}, {}

    has_manual_baseline = baseline_left_point is not None and baseline_right_point is not None
    needs_apex_reference = reference_pass_through_apex and reference_line_mode == "apex_parallel" and has_manual_baseline
    if reference_point is None and needs_apex_reference:
        for item in measured:
            item["angle_measurement_status"] = "missing_reference_point"
            item["angle_deg"] = None
            item["angle_line"] = None
            item["angle_horizontal_line"] = None
        return "missing_reference_point", {}, {}

    try:
        reference_line = _estimate_reference_line(
            side_items["left"]["component_mask"],
            side_items["right"]["component_mask"],
            apex=reference_point,
            baseline_left_point=baseline_left_point,
            baseline_right_point=baseline_right_point,
            mode=reference_line_mode,
            branch_cut_radius=branch_cut_radius,
            terminal_max_depth_ratio=terminal_max_depth_ratio,
            upper_arc_end_ratio=upper_arc_end_ratio,
            pass_through_apex=reference_pass_through_apex,
        )
    except ValueError:
        for item in measured:
            item["angle_measurement_status"] = "reference_line_failed"
            item["angle_deg"] = None
            item["angle_line"] = None
            item["angle_horizontal_line"] = None
        return "reference_line_failed", {}, {}

    reference_visual_line = _line_through_direction(
        tuple(reference_line["point"]),
        reference_line["direction"],
        image_shape,
    )
    angle_values = {}
    side_measures = {}
    for side, item in side_items.items():
        item["angle_horizontal_line"] = reference_visual_line
        item["reference_line_mode"] = reference_line["mode"]
        item["reference_point"] = tuple(np.asarray(reference_line["point"], dtype=np.float32))
        item["reference_direction"] = tuple(np.asarray(reference_line["direction"], dtype=np.float32))
        item["reference_anchor"] = tuple(np.asarray(reference_line[f"{side}_anchor"], dtype=np.float32))
        item["upper_arc_path"] = [tuple(point.astype(np.float32)) for point in reference_line[f"{side}_upper_arc_path"]]
        item["upper_arc_debug"] = reference_line.get(f"{side}_upper_arc_debug", {})
        item["upper_arc_terminal"] = tuple(np.asarray(reference_line[f"{side}_upper_arc_terminal"], dtype=np.float32))
        item["upper_arc_inferior_point"] = tuple(np.asarray(reference_line[f"{side}_upper_arc_inferior_point"], dtype=np.float32))
        item["upper_arc_path_point_count"] = int(len(reference_line[f"{side}_upper_arc_path"]))
        item["branch_cut_radius"] = reference_line.get("branch_cut_radius")
        item["terminal_max_depth_ratio"] = reference_line.get("terminal_max_depth_ratio")
        item["upper_arc_end_ratio"] = reference_line.get("upper_arc_end_ratio")
        measure = _measure_local_sylvian_angle(
            reference_line[f"{side}_upper_arc_path"],
            reference_line["direction"],
            terminal_arc_length=local_arc_length,
        )
        side_measures[side] = measure
        if measure["status"] != "success":
            item["angle_measurement_status"] = "upper_arm_tangent_failed"
            item["angle_deg"] = None
            item["angle_line"] = None
            item["measurement_anchor"] = tuple(np.asarray(measure["measurement_anchor"], dtype=np.float32)) if measure["measurement_anchor"] is not None else None
            item["local_tangent_points"] = [tuple(point.astype(np.float32)) for point in measure["local_points"]]
            item["local_tangent_point_count"] = int(len(measure["local_points"]))
            item["local_tangent_status"] = measure["status"]
            continue
        item["angle_measurement_status"] = "ok"
        item["angle_deg"] = float(measure["angle_deg"])
        item["angle_line"] = measure["tangent_line"]
        item["measurement_anchor"] = tuple(np.asarray(measure["measurement_anchor"], dtype=np.float32))
        item["local_tangent_points"] = [tuple(point.astype(np.float32)) for point in measure["local_points"]]
        item["local_tangent_point_count"] = int(len(measure["local_points"]))
        item["local_tangent_status"] = measure["status"]
        item["nearest_skeleton_point"] = measure.get("nearest_skeleton_point")
        angle_values[side] = float(measure["angle_deg"])

    selected_item_ids = {id(item) for item in side_items.values()}
    for item in measured:
        if id(item) not in selected_item_ids:
            item["angle_measurement_status"] = "not_selected_for_bilateral_angle"
            item["angle_deg"] = None
            item["angle_line"] = None
            item["angle_horizontal_line"] = None

    if "left" in angle_values and "right" in angle_values:
        return "ok", angle_values, reference_line
    return "upper_arm_tangent_failed", angle_values, reference_line


def measure_lateral_fissure(
    mask: np.ndarray,
    pixel_spacing: Optional[Tuple[float, float]] = None,
    min_area: int = 8,
    max_components: int = 2,
    angle_reference_point: Optional[Point] = None,
    baseline_left_point: Optional[Point] = None,
    baseline_right_point: Optional[Point] = None,
    reference_line_mode: str = "auto",
    reference_anchor_ratio: float = 0.60,
    local_tangent_ratio: float = 0.60,
    local_tangent_half_arc_length: float = 30.0,
    local_arc_length: float = 28.0,
    branch_cut_radius: float = 6.0,
    terminal_max_depth_ratio: float = 0.62,
    upper_arc_end_ratio: float = 0.55,
    reference_pass_through_apex: bool = False,
) -> Dict[str, object]:
    components, component_count = _components(mask, min_area=min_area)
    if not components:
        return _empty_result(component_count, 0, mask.shape)

    selected = components[:max_components]
    measured = []
    union_mask = np.zeros(mask.shape, dtype=bool)
    image_width = int(mask.shape[1])
    for component, area in selected:
        union_mask |= component
        points = _component_points(component)
        side = _side_for_component(points, image_width)
        measured.append(_measure_component(component, area, side, pixel_spacing))

    if not measured:
        return _empty_result(component_count, int(sum(area for _, area in selected)), mask.shape)

    max_depth = max(measured, key=lambda item: float(item["depth_px"]))
    max_width = max(measured, key=lambda item: float(item["width_px"]))
    max_curve = max(measured, key=lambda item: float(item["curvature_ratio"]))
    angle_status, angle_values, reference_line = _apply_bilateral_angle_measurement(
        measured,
        angle_reference_point,
        mask.shape,
        baseline_left_point=baseline_left_point,
        baseline_right_point=baseline_right_point,
        reference_line_mode=reference_line_mode,
        branch_cut_radius=branch_cut_radius,
        terminal_max_depth_ratio=terminal_max_depth_ratio,
        upper_arc_end_ratio=upper_arc_end_ratio,
        local_arc_length=local_arc_length,
        reference_pass_through_apex=reference_pass_through_apex,
    )
    widths = [float(item["width_px"]) for item in measured]
    depths = [float(item["depth_px"]) for item in measured]
    curvatures = [float(item["curvature_ratio"]) for item in measured]
    angles = list(angle_values.values())
    result = {
        "status": "ok",
        "component_count": component_count,
        "measured_component_count": len(measured),
        "area_px": int(sum(item["area_px"] for item in measured)),
        "depth_px": float(max(depths)),
        "depth_measurement_status": max_depth.get("depth_measurement_status", ""),
        "depth_y": max_depth.get("depth_y"),
        "depth_point_count": max_depth.get("depth_point_count", 0),
        "depth_smooth_px": max_depth.get("depth_smooth_px", 0.0),
        "width_px": float(max(widths)),
        "mean_width_px": float(np.mean(widths)),
        "opening_width_measurement_status": max_width.get("opening_width_measurement_status", ""),
        "opening_width_dash_lines": max_width.get("opening_width_dash_lines"),
        "opening_width_segment_a_px": max_width.get("opening_width_segment_a_px", 0.0),
        "opening_width_segment_b_px": max_width.get("opening_width_segment_b_px", 0.0),
        "opening_width_segment_a_mm": max_width.get("opening_width_segment_a_mm"),
        "opening_width_segment_b_mm": max_width.get("opening_width_segment_b_mm"),
        "opening_width_yellow_dashed_length_px": max_width.get("opening_width_yellow_dashed_length_px", 0.0),
        "opening_width_yellow_dashed_length_mm": max_width.get("opening_width_yellow_dashed_length_mm"),
        "opening_width_point_count": max_width.get("opening_width_point_count", 0),
        "opening_width_gap_point_count": max_width.get("opening_width_gap_point_count", 0),
        "opening_width_mode": max_width.get("opening_width_mode", ""),
        "opening_width_smooth_px": max_width.get("opening_width_smooth_px", 0.0),
        "curvature_ratio": float(max(curvatures)),
        "mean_curvature_ratio": float(np.mean(curvatures)),
        "curve_length_px": float(max_curve["curve_length_px"]),
        "curve_chord_px": float(max_curve["curve_chord_px"]),
        "curvature_measurement_status": max_curve.get("curvature_measurement_status", ""),
        "curve_bulge_point": max_curve.get("curve_bulge_point"),
        "curve_bulge_offset_px": max_curve.get("curve_bulge_offset_px", 0.0),
        "curvature_angle_deg": max_curve.get("curvature_angle_deg"),
        "angle_measurement_status": angle_status,
        "angle_deg": float(max(angles)) if angles else None,
        "mean_angle_deg": float(np.mean(angles)) if angles else None,
        "third_ventricle_apex": angle_reference_point,
        "reference_line_mode": reference_line.get("mode") if reference_line else "",
        "reference_point": tuple(np.asarray(reference_line["point"], dtype=np.float32)) if reference_line else None,
        "reference_direction": tuple(np.asarray(reference_line["direction"], dtype=np.float32)) if reference_line else None,
        "left_reference_anchor": tuple(np.asarray(reference_line["left_anchor"], dtype=np.float32)) if reference_line else None,
        "right_reference_anchor": tuple(np.asarray(reference_line["right_anchor"], dtype=np.float32)) if reference_line else None,
        "left_upper_arc_terminal": tuple(np.asarray(reference_line["left_upper_arc_terminal"], dtype=np.float32)) if reference_line else None,
        "right_upper_arc_terminal": tuple(np.asarray(reference_line["right_upper_arc_terminal"], dtype=np.float32)) if reference_line else None,
        "left_upper_arc_inferior_point": tuple(np.asarray(reference_line["left_upper_arc_inferior_point"], dtype=np.float32)) if reference_line else None,
        "right_upper_arc_inferior_point": tuple(np.asarray(reference_line["right_upper_arc_inferior_point"], dtype=np.float32)) if reference_line else None,
        "branch_cut_radius": reference_line.get("branch_cut_radius") if reference_line else None,
        "terminal_max_depth_ratio": reference_line.get("terminal_max_depth_ratio") if reference_line else None,
        "upper_arc_end_ratio": reference_line.get("upper_arc_end_ratio") if reference_line else None,
        "depth_mm": None,
        "width_mm": None,
        "mean_width_mm": None,
        "curve_length_mm": None,
        "curve_chord_mm": None,
        "orientation_deg": 0.0,
        "depth_line": max_depth["depth_line"],
        "width_line": max_width["width_line"],
        "curve_path": max_curve["curve_path"],
        "curve_chord_line": max_curve["curve_chord_line"],
        "angle_line": None,
        "angle_horizontal_line": _line_through_direction(
            tuple(reference_line["point"]),
            reference_line["direction"],
            mask.shape,
        ) if reference_line else None,
        "component_mask": union_mask,
        "component_measurements": measured,
    }

    if pixel_spacing:
        result["depth_mm"] = float(max(item["depth_mm"] for item in measured if item["depth_mm"] is not None))
        result["width_mm"] = float(max(item["width_mm"] for item in measured if item["width_mm"] is not None))
        result["mean_width_mm"] = float(np.mean([item["width_mm"] for item in measured if item["width_mm"] is not None]))
        curve_lengths = [item["curve_length_mm"] for item in measured if item["curve_length_mm"] is not None]
        curve_chords = [item["curve_chord_mm"] for item in measured if item["curve_chord_mm"] is not None]
        result["curve_length_mm"] = float(max(curve_lengths)) if curve_lengths else None
        result["curve_chord_mm"] = float(max(curve_chords)) if curve_chords else None

    for item in measured:
        side = item["side"]
        result[f"{side}_depth_px"] = item["depth_px"]
        result[f"{side}_depth_measurement_status"] = item.get("depth_measurement_status", "")
        result[f"{side}_depth_y"] = item.get("depth_y")
        result[f"{side}_depth_point_count"] = item.get("depth_point_count", 0)
        result[f"{side}_depth_smooth_px"] = item.get("depth_smooth_px", 0.0)
        result[f"{side}_width_px"] = item["width_px"]
        result[f"{side}_opening_width_measurement_status"] = item.get("opening_width_measurement_status", "")
        result[f"{side}_opening_width_segment_a_px"] = item.get("opening_width_segment_a_px", 0.0)
        result[f"{side}_opening_width_segment_b_px"] = item.get("opening_width_segment_b_px", 0.0)
        result[f"{side}_opening_width_yellow_dashed_length_px"] = item.get("opening_width_yellow_dashed_length_px", item.get("width_px", 0.0))
        result[f"{side}_opening_width_dash_lines"] = item.get("opening_width_dash_lines")
        result[f"{side}_opening_width_point_count"] = item.get("opening_width_point_count", 0)
        result[f"{side}_opening_width_gap_point_count"] = item.get("opening_width_gap_point_count", 0)
        result[f"{side}_opening_width_mode"] = item.get("opening_width_mode", "")
        result[f"{side}_opening_width_smooth_px"] = item.get("opening_width_smooth_px", 0.0)
        result[f"{side}_curvature_ratio"] = item["curvature_ratio"]
        result[f"{side}_curve_length_px"] = item["curve_length_px"]
        result[f"{side}_curve_chord_px"] = item["curve_chord_px"]
        result[f"{side}_curvature_measurement_status"] = item.get("curvature_measurement_status", "")
        result[f"{side}_curve_bulge_offset_px"] = item.get("curve_bulge_offset_px", 0.0)
        if item.get("curve_bulge_point") is not None:
            result[f"{side}_curve_bulge_point"] = item["curve_bulge_point"]
        if item.get("curvature_angle_deg") is not None:
            result[f"{side}_curvature_angle_deg"] = item["curvature_angle_deg"]
        result[f"{side}_angle_measurement_status"] = item["angle_measurement_status"]
        upper_arm_debug = item.get("upper_arm_debug", {})
        result[f"{side}_upper_arm_status"] = upper_arm_debug.get("status", "")
        result[f"{side}_upper_arm_point_count"] = upper_arm_debug.get("upper_arm_point_count", 0)
        result[f"{side}_upper_arm_path_length"] = upper_arm_debug.get("upper_arm_path_length", 0.0)
        result[f"{side}_local_tangent_status"] = item.get("local_tangent_status", "")
        result[f"{side}_local_tangent_point_count"] = item.get("local_tangent_point_count", 0)
        result[f"{side}_local_point_count"] = item.get("local_tangent_point_count", 0)
        result[f"{side}_upper_arc_debug"] = item.get("upper_arc_debug", {})
        result[f"{side}_upper_arc_path_point_count"] = item.get("upper_arc_path_point_count", 0)
        if item.get("upper_arc_terminal") is not None:
            result[f"{side}_upper_arc_terminal"] = item["upper_arc_terminal"]
        if item.get("upper_arc_inferior_point") is not None:
            result[f"{side}_upper_arc_inferior_point"] = item["upper_arc_inferior_point"]
        nearest = item.get("nearest_skeleton_point")
        if isinstance(nearest, Sequence):
            result[f"{side}_nearest_skeleton_point"] = nearest
        if item.get("measurement_anchor") is not None:
            result[f"{side}_measurement_anchor"] = item["measurement_anchor"]
        if item.get("angle_deg") is not None:
            result[f"{side}_angle_deg"] = item["angle_deg"]
        if item.get("depth_mm") is not None:
            result[f"{side}_depth_mm"] = item["depth_mm"]
            result[f"{side}_width_mm"] = item["width_mm"]
            result[f"{side}_opening_width_segment_a_mm"] = item.get("opening_width_segment_a_mm")
            result[f"{side}_opening_width_segment_b_mm"] = item.get("opening_width_segment_b_mm")
            result[f"{side}_opening_width_yellow_dashed_length_mm"] = item.get("opening_width_yellow_dashed_length_mm")
            result[f"{side}_curve_length_mm"] = item["curve_length_mm"]
            result[f"{side}_curve_chord_mm"] = item["curve_chord_mm"]
    return result


def measurement_to_row(measurement: Dict[str, object], prefix: str = "fissure") -> Dict[str, object]:
    row = {
        f"{prefix}_measurement_status": measurement["status"],
        f"{prefix}_component_count": measurement["component_count"],
        f"{prefix}_measured_component_count": measurement.get("measured_component_count", 0),
        f"{prefix}_area_px": measurement["area_px"],
        f"{prefix}_depth_px": measurement["depth_px"],
        f"{prefix}_depth_measurement_status": measurement.get("depth_measurement_status", ""),
        f"{prefix}_depth_y": measurement.get("depth_y"),
        f"{prefix}_depth_point_count": measurement.get("depth_point_count", 0),
        f"{prefix}_depth_smooth_px": measurement.get("depth_smooth_px", 0.0),
        f"{prefix}_width_px": measurement["width_px"],
        f"{prefix}_mean_width_px": measurement["mean_width_px"],
        f"{prefix}_opening_width_measurement_status": measurement.get("opening_width_measurement_status", ""),
        f"{prefix}_opening_width_segment_a_px": measurement.get("opening_width_segment_a_px", 0.0),
        f"{prefix}_opening_width_segment_b_px": measurement.get("opening_width_segment_b_px", 0.0),
        f"{prefix}_opening_width_yellow_dashed_length_px": measurement.get("opening_width_yellow_dashed_length_px", measurement.get("width_px", 0.0)),
        f"{prefix}_opening_width_point_count": measurement.get("opening_width_point_count", 0),
        f"{prefix}_opening_width_gap_point_count": measurement.get("opening_width_gap_point_count", 0),
        f"{prefix}_opening_width_mode": measurement.get("opening_width_mode", ""),
        f"{prefix}_opening_width_smooth_px": measurement.get("opening_width_smooth_px", 0.0),
        f"{prefix}_curvature_ratio": measurement.get("curvature_ratio", 0.0),
        f"{prefix}_mean_curvature_ratio": measurement.get("mean_curvature_ratio", 0.0),
        f"{prefix}_curve_length_px": measurement.get("curve_length_px", 0.0),
        f"{prefix}_curve_chord_px": measurement.get("curve_chord_px", 0.0),
        f"{prefix}_curvature_measurement_status": measurement.get("curvature_measurement_status", ""),
        f"{prefix}_curve_bulge_offset_px": measurement.get("curve_bulge_offset_px", 0.0),
        f"{prefix}_curvature_angle_deg": measurement.get("curvature_angle_deg"),
        f"{prefix}_angle_measurement_status": measurement.get("angle_measurement_status", ""),
        f"{prefix}_angle_deg": measurement.get("angle_deg"),
        f"{prefix}_mean_angle_deg": measurement.get("mean_angle_deg"),
        f"{prefix}_orientation_deg": measurement["orientation_deg"],
    }
    apex = measurement.get("third_ventricle_apex")
    if isinstance(apex, Sequence):
        row[f"{prefix}_third_ventricle_apex_x"] = apex[0]
        row[f"{prefix}_third_ventricle_apex_y"] = apex[1]
    bulge = measurement.get("curve_bulge_point")
    if isinstance(bulge, Sequence):
        row[f"{prefix}_curve_bulge_point_x"] = bulge[0]
        row[f"{prefix}_curve_bulge_point_y"] = bulge[1]
    row[f"{prefix}_reference_line_mode"] = measurement.get("reference_line_mode", "")
    row[f"{prefix}_branch_cut_radius"] = measurement.get("branch_cut_radius")
    row[f"{prefix}_terminal_max_depth_ratio"] = measurement.get("terminal_max_depth_ratio")
    row[f"{prefix}_upper_arc_end_ratio"] = measurement.get("upper_arc_end_ratio")
    for key in (
        "reference_point",
        "reference_direction",
        "left_reference_anchor",
        "right_reference_anchor",
        "left_upper_arc_terminal",
        "right_upper_arc_terminal",
        "left_upper_arc_inferior_point",
        "right_upper_arc_inferior_point",
    ):
        value = measurement.get(key)
        if isinstance(value, Sequence):
            row[f"{prefix}_{key}_x"] = value[0]
            row[f"{prefix}_{key}_y"] = value[1]
    for side in ("left", "right"):
        if f"{side}_depth_px" in measurement:
            row[f"{prefix}_{side}_depth_px"] = measurement[f"{side}_depth_px"]
            row[f"{prefix}_{side}_depth_measurement_status"] = measurement.get(f"{side}_depth_measurement_status", "")
            row[f"{prefix}_{side}_depth_y"] = measurement.get(f"{side}_depth_y")
            row[f"{prefix}_{side}_depth_point_count"] = measurement.get(f"{side}_depth_point_count", 0)
            row[f"{prefix}_{side}_depth_smooth_px"] = measurement.get(f"{side}_depth_smooth_px", 0.0)
            row[f"{prefix}_{side}_width_px"] = measurement[f"{side}_width_px"]
            row[f"{prefix}_{side}_opening_width_measurement_status"] = measurement.get(f"{side}_opening_width_measurement_status", "")
            row[f"{prefix}_{side}_opening_width_segment_a_px"] = measurement.get(f"{side}_opening_width_segment_a_px", 0.0)
            row[f"{prefix}_{side}_opening_width_segment_b_px"] = measurement.get(f"{side}_opening_width_segment_b_px", 0.0)
            row[f"{prefix}_{side}_opening_width_yellow_dashed_length_px"] = measurement.get(f"{side}_opening_width_yellow_dashed_length_px", measurement.get(f"{side}_width_px", 0.0))
            row[f"{prefix}_{side}_opening_width_point_count"] = measurement.get(f"{side}_opening_width_point_count", 0)
            row[f"{prefix}_{side}_opening_width_gap_point_count"] = measurement.get(f"{side}_opening_width_gap_point_count", 0)
            row[f"{prefix}_{side}_opening_width_mode"] = measurement.get(f"{side}_opening_width_mode", "")
            row[f"{prefix}_{side}_opening_width_smooth_px"] = measurement.get(f"{side}_opening_width_smooth_px", 0.0)
            row[f"{prefix}_{side}_curvature_ratio"] = measurement[f"{side}_curvature_ratio"]
            row[f"{prefix}_{side}_curve_length_px"] = measurement[f"{side}_curve_length_px"]
            row[f"{prefix}_{side}_curve_chord_px"] = measurement[f"{side}_curve_chord_px"]
            row[f"{prefix}_{side}_curvature_measurement_status"] = measurement.get(f"{side}_curvature_measurement_status", "")
            row[f"{prefix}_{side}_curve_bulge_offset_px"] = measurement.get(f"{side}_curve_bulge_offset_px", 0.0)
            row[f"{prefix}_{side}_curvature_angle_deg"] = measurement.get(f"{side}_curvature_angle_deg")
            bulge = measurement.get(f"{side}_curve_bulge_point")
            if isinstance(bulge, Sequence):
                row[f"{prefix}_{side}_curve_bulge_point_x"] = bulge[0]
                row[f"{prefix}_{side}_curve_bulge_point_y"] = bulge[1]
            row[f"{prefix}_{side}_angle_measurement_status"] = measurement.get(f"{side}_angle_measurement_status", "")
            row[f"{prefix}_{side}_angle_deg"] = measurement.get(f"{side}_angle_deg")
            row[f"{prefix}_{side}_upper_arm_status"] = measurement.get(f"{side}_upper_arm_status", "")
            row[f"{prefix}_{side}_upper_arm_point_count"] = measurement.get(f"{side}_upper_arm_point_count", 0)
            row[f"{prefix}_{side}_upper_arm_path_length"] = measurement.get(f"{side}_upper_arm_path_length", 0.0)
            row[f"{prefix}_{side}_local_tangent_status"] = measurement.get(f"{side}_local_tangent_status", "")
            row[f"{prefix}_{side}_local_tangent_point_count"] = measurement.get(f"{side}_local_tangent_point_count", 0)
            row[f"{prefix}_{side}_local_point_count"] = measurement.get(f"{side}_local_point_count", 0)
            row[f"{prefix}_{side}_upper_arc_path_point_count"] = measurement.get(f"{side}_upper_arc_path_point_count", 0)
            upper_arc_terminal = measurement.get(f"{side}_upper_arc_terminal")
            if isinstance(upper_arc_terminal, Sequence):
                row[f"{prefix}_{side}_upper_arc_terminal_x"] = upper_arc_terminal[0]
                row[f"{prefix}_{side}_upper_arc_terminal_y"] = upper_arc_terminal[1]
            upper_arc_inferior = measurement.get(f"{side}_upper_arc_inferior_point")
            if isinstance(upper_arc_inferior, Sequence):
                row[f"{prefix}_{side}_upper_arc_inferior_point_x"] = upper_arc_inferior[0]
                row[f"{prefix}_{side}_upper_arc_inferior_point_y"] = upper_arc_inferior[1]
            nearest = measurement.get(f"{side}_nearest_skeleton_point")
            if isinstance(nearest, Sequence):
                row[f"{prefix}_{side}_nearest_skeleton_point_x"] = nearest[0]
                row[f"{prefix}_{side}_nearest_skeleton_point_y"] = nearest[1]
            anchor = measurement.get(f"{side}_measurement_anchor")
            if isinstance(anchor, Sequence):
                row[f"{prefix}_{side}_measurement_anchor_x"] = anchor[0]
                row[f"{prefix}_{side}_measurement_anchor_y"] = anchor[1]
        if f"{side}_depth_mm" in measurement:
            row[f"{prefix}_{side}_depth_mm"] = measurement[f"{side}_depth_mm"]
            row[f"{prefix}_{side}_width_mm"] = measurement[f"{side}_width_mm"]
            row[f"{prefix}_{side}_opening_width_segment_a_mm"] = measurement.get(f"{side}_opening_width_segment_a_mm")
            row[f"{prefix}_{side}_opening_width_segment_b_mm"] = measurement.get(f"{side}_opening_width_segment_b_mm")
            row[f"{prefix}_{side}_opening_width_yellow_dashed_length_mm"] = measurement.get(f"{side}_opening_width_yellow_dashed_length_mm")
            row[f"{prefix}_{side}_curve_length_mm"] = measurement[f"{side}_curve_length_mm"]
            row[f"{prefix}_{side}_curve_chord_mm"] = measurement[f"{side}_curve_chord_mm"]
    if measurement.get("depth_mm") is not None:
        row[f"{prefix}_depth_mm"] = measurement["depth_mm"]
        row[f"{prefix}_width_mm"] = measurement["width_mm"]
        row[f"{prefix}_mean_width_mm"] = measurement["mean_width_mm"]
        row[f"{prefix}_opening_width_segment_a_mm"] = measurement.get("opening_width_segment_a_mm")
        row[f"{prefix}_opening_width_segment_b_mm"] = measurement.get("opening_width_segment_b_mm")
        row[f"{prefix}_opening_width_yellow_dashed_length_mm"] = measurement.get("opening_width_yellow_dashed_length_mm")
        row[f"{prefix}_curve_length_mm"] = measurement["curve_length_mm"]
        row[f"{prefix}_curve_chord_mm"] = measurement["curve_chord_mm"]
    return row


def _branch_points_and_endpoints(graph):
    endpoints = [node for node, neighbors in graph.items() if len(neighbors) == 1]
    branch_points = [node for node, neighbors in graph.items() if len(neighbors) >= 3]
    return branch_points, endpoints


def _select_longitudinal_component(
    mask: np.ndarray,
    min_area: int,
) -> Tuple[Optional[np.ndarray], int, int]:
    components, component_count = _components(mask, min_area=min_area)
    if not components:
        return None, 0, component_count

    image_center_x = float(mask.shape[1]) / 2.0
    best_component = None
    best_area = 0
    best_score = None
    for component, area in components:
        ys, xs = np.where(component > 0)
        if len(xs) == 0:
            continue
        height = float(ys.max() - ys.min() + 1)
        centroid_x = float(xs.mean())
        center_distance = abs(centroid_x - image_center_x)
        score = float(area) + 2.0 * height - 0.5 * center_distance
        if best_score is None or score > best_score:
            best_score = score
            best_component = component
            best_area = int(area)
    return best_component, best_area, component_count


def _preprocess_longitudinal_edge_mask(component: np.ndarray) -> np.ndarray:
    binary = (component > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    if int(closed.sum()) < max(10, int(binary.sum() * 0.25)):
        return binary
    return closed


def _contour_to_path(contour: np.ndarray, close: bool = True) -> List[Point]:
    if contour is None or len(contour) == 0:
        return []
    points = [(float(point[0][0]), float(point[0][1])) for point in contour]
    if close and points and points[0] != points[-1]:
        points.append(points[0])
    return points


def _closed_path_length(path: Sequence[Point], pixel_spacing: Optional[Tuple[float, float]] = None) -> float:
    if not isinstance(path, Sequence) or len(path) < 2:
        return 0.0
    return float(sum(_length(path[idx - 1], path[idx], pixel_spacing) for idx in range(1, len(path))))


def _measure_longitudinal_edge_length(
    component: np.ndarray,
    pixel_spacing: Optional[Tuple[float, float]],
) -> Dict[str, object]:
    clean_mask = _preprocess_longitudinal_edge_mask(component)
    contours, _ = cv2.findContours(
        (clean_mask > 0).astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return {
            "status": "no_contour",
            "full_length_px": 0.0,
            "full_length_mm": None,
            "full_length_path": [],
            "full_length_contour": None,
            "full_length_contour_point_count": 0,
        }

    main_contour = max(contours, key=lambda contour: cv2.arcLength(contour, closed=True))
    full_length_px = float(cv2.arcLength(main_contour, closed=True))
    contour_path = _contour_to_path(main_contour, close=True)
    full_length_mm = _closed_path_length(contour_path, pixel_spacing) if pixel_spacing else None
    return {
        "status": "ok",
        "full_length_px": full_length_px,
        "full_length_mm": float(full_length_mm) if full_length_mm is not None else None,
        "full_length_path": contour_path,
        "full_length_contour": main_contour,
        "full_length_contour_point_count": int(len(main_contour)),
    }


def _find_main_vertical_trunk(graph) -> List[Point]:
    if not graph:
        return []

    _, endpoints = _branch_points_and_endpoints(graph)
    candidates = endpoints if len(endpoints) >= 2 else list(graph.keys())
    best_path = []
    best_score = None
    for i, start in enumerate(candidates):
        for end in candidates[i + 1:]:
            path, length = _shortest_path(graph, start, end)
            if len(path) < 2:
                continue
            points = np.asarray(path, dtype=np.float32)
            y_span = float(abs(points[-1, 1] - points[0, 1]))
            x_span = float(abs(points[-1, 0] - points[0, 0]))
            mean_center_penalty = float(np.std(points[:, 0]))
            score = y_span + 0.05 * float(length) - 0.35 * x_span - 0.02 * mean_center_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_path = path
    return best_path


def _dijkstra_branch_path(graph, start: Tuple[int, int], targets: set, allowed_nodes: set):
    queue = [(0.0, start)]
    distances = {start: 0.0}
    previous = {}
    best_target = None
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, float("inf")):
            continue
        if node in targets:
            best_target = node
            break
        for neighbor, weight in graph.get(node, []):
            if neighbor not in allowed_nodes:
                continue
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = node
                heapq.heappush(queue, (new_distance, neighbor))
    if best_target is None:
        return [], 0.0
    path = [best_target]
    node = best_target
    while node != start:
        node = previous[node]
        path.append(node)
    path.reverse()
    return [(float(x), float(y)) for x, y in path], float(distances[best_target])


def _outside_component_from(graph, seed: Tuple[int, int], excluded_nodes: set, visited: set) -> set:
    component = set()
    stack = [seed]
    visited.add(seed)
    while stack:
        node = stack.pop()
        component.add(node)
        for neighbor, _ in graph.get(node, []):
            if neighbor in excluded_nodes or neighbor in visited:
                continue
            visited.add(neighbor)
            stack.append(neighbor)
    return component


def _measure_longitudinal_branch(component: np.ndarray) -> Dict[str, object]:
    graph = _skeleton_graph(component)
    if not graph:
        points = _component_points(component)
        return {
            "status": "empty_skeleton",
            "branch_start": None,
            "branch_tip": None,
            "branch_depth_px": 0.0,
            "branch_euclidean_px": 0.0,
            "branch_path": [],
            "main_trunk_path": [],
            "candidate_branch_count": 0,
        }

    main_trunk = _find_main_vertical_trunk(graph)
    trunk_nodes = {_point_key(np.asarray(point, dtype=np.float32)) for point in main_trunk}
    if len(trunk_nodes) < 2:
        return {
            "status": "trunk_failed",
            "branch_start": None,
            "branch_tip": None,
            "branch_depth_px": 0.0,
            "branch_euclidean_px": 0.0,
            "branch_path": [],
            "main_trunk_path": main_trunk,
            "candidate_branch_count": 0,
        }

    min_branch_length = max(5.0, 0.015 * float(component.shape[0]))
    min_lateral_offset = max(3.0, 0.004 * float(component.shape[1]))
    visited = set()
    branches = []

    for trunk_node in trunk_nodes:
        for neighbor, first_weight in graph.get(trunk_node, []):
            if neighbor in trunk_nodes or neighbor in visited:
                continue
            outside_nodes = _outside_component_from(graph, neighbor, trunk_nodes, visited)
            if not outside_nodes:
                continue

            endpoints = {
                node for node in outside_nodes
                if sum(1 for next_node, _ in graph.get(node, []) if next_node in outside_nodes) <= 1
            }
            if not endpoints:
                endpoints = set(outside_nodes)

            allowed_nodes = set(outside_nodes)
            best_for_component = None
            for endpoint in endpoints:
                path, branch_length = _dijkstra_branch_path(graph, neighbor, {endpoint}, allowed_nodes)
                if not path:
                    continue
                full_path = [(float(trunk_node[0]), float(trunk_node[1]))] + path
                path_length = float(first_weight + branch_length)
                start = full_path[0]
                tip = full_path[-1]
                dx = abs(float(tip[0] - start[0]))
                dy = abs(float(tip[1] - start[1]))
                euclidean = _length(start, tip)
                if path_length < min_branch_length:
                    continue
                if dx < min_lateral_offset:
                    continue
                if dx < dy * 0.5:
                    continue
                branch = {
                    "status": "ok",
                    "branch_start": start,
                    "branch_tip": tip,
                    "branch_depth_px": float(path_length),
                    "branch_euclidean_px": float(euclidean),
                    "branch_path": full_path,
                    "main_trunk_path": main_trunk,
                    "candidate_branch_count": 0,
                    "lateral_offset_px": float(dx),
                    "vertical_offset_px": float(dy),
                }
                if best_for_component is None or branch["branch_depth_px"] > best_for_component["branch_depth_px"]:
                    best_for_component = branch
            if best_for_component is not None:
                branches.append(best_for_component)

    if not branches:
        return {
            "status": "no_lateral_branch",
            "branch_start": None,
            "branch_tip": None,
            "branch_depth_px": 0.0,
            "branch_euclidean_px": 0.0,
            "branch_path": [],
            "main_trunk_path": main_trunk,
            "candidate_branch_count": 0,
        }

    best = max(branches, key=lambda item: item["branch_depth_px"])
    best["candidate_branch_count"] = int(len(branches))
    return best


def measure_longitudinal_fissure(
    mask: np.ndarray,
    pixel_spacing: Optional[Tuple[float, float]] = None,
    min_area: int = 8,
) -> Dict[str, object]:
    component, area, component_count = _select_longitudinal_component(mask, min_area=min_area)
    if component is None:
        return {
            "status": "empty",
            "component_count": component_count,
            "area_px": 0,
            "full_length_px": 0.0,
            "full_length_measurement_status": "empty",
            "full_length_contour_point_count": 0,
            "branch_depth_px": 0.0,
            "area_mm2": None,
            "full_length_mm": None,
            "branch_depth_mm": None,
            "full_length_path": [],
            "branch_depth_line": None,
            "branch_depth_path": [],
            "branch_euclidean_px": 0.0,
            "branch_measurement_status": "empty",
            "branch_candidate_count": 0,
            "branch_start": None,
            "branch_tip": None,
            "main_trunk_path": [],
            "component_mask": np.zeros(mask.shape, dtype=bool),
        }

    edge_info = _measure_longitudinal_edge_length(component, pixel_spacing)
    branch_info = _measure_longitudinal_branch(component)
    branch_start = branch_info["branch_start"]
    branch_tip = branch_info["branch_tip"]
    branch_depth_px = float(branch_info["branch_depth_px"])
    branch_depth_mm = _scale_length(branch_depth_px, pixel_spacing)
    area_mm2 = None
    if pixel_spacing is not None:
        row_spacing, col_spacing = pixel_spacing
        area_mm2 = float(area * row_spacing * col_spacing)

    return {
        "status": "ok",
        "component_count": component_count,
        "area_px": int(area),
        "full_length_px": float(edge_info["full_length_px"]),
        "full_length_measurement_status": edge_info["status"],
        "full_length_contour_point_count": edge_info["full_length_contour_point_count"],
        "branch_depth_px": float(branch_depth_px),
        "branch_euclidean_px": float(branch_info.get("branch_euclidean_px", 0.0)),
        "branch_measurement_status": branch_info.get("status", ""),
        "branch_candidate_count": int(branch_info.get("candidate_branch_count", 0)),
        "area_mm2": float(area_mm2) if area_mm2 is not None else None,
        "full_length_mm": edge_info["full_length_mm"],
        "branch_depth_mm": float(branch_depth_mm) if branch_depth_mm is not None else None,
        "full_length_path": edge_info["full_length_path"],
        "full_length_contour": edge_info["full_length_contour"],
        "branch_depth_line": (branch_start, branch_tip) if branch_start is not None and branch_tip is not None else None,
        "branch_depth_path": branch_info.get("branch_path", []),
        "branch_start": branch_start,
        "branch_tip": branch_tip,
        "main_trunk_path": branch_info.get("main_trunk_path", []),
        "component_mask": component,
    }


def longitudinal_measurement_to_row(
    measurement: Dict[str, object],
    prefix: str = "longitudinal_fissure",
) -> Dict[str, object]:
    row = {
        f"{prefix}_measurement_status": measurement["status"],
        f"{prefix}_component_count": measurement["component_count"],
        f"{prefix}_area_px": measurement["area_px"],
        f"{prefix}_full_length_px": measurement["full_length_px"],
        f"{prefix}_full_length_measurement_status": measurement.get("full_length_measurement_status", ""),
        f"{prefix}_full_length_contour_point_count": measurement.get("full_length_contour_point_count", 0),
        f"{prefix}_branch_depth_px": measurement["branch_depth_px"],
        f"{prefix}_branch_euclidean_px": measurement.get("branch_euclidean_px", 0.0),
        f"{prefix}_branch_measurement_status": measurement.get("branch_measurement_status", ""),
        f"{prefix}_branch_candidate_count": measurement.get("branch_candidate_count", 0),
    }
    branch_start = measurement.get("branch_start")
    if isinstance(branch_start, Sequence):
        row[f"{prefix}_branch_start_x"] = branch_start[0]
        row[f"{prefix}_branch_start_y"] = branch_start[1]
    branch_tip = measurement.get("branch_tip")
    if isinstance(branch_tip, Sequence):
        row[f"{prefix}_branch_tip_x"] = branch_tip[0]
        row[f"{prefix}_branch_tip_y"] = branch_tip[1]
    if measurement.get("area_mm2") is not None:
        row[f"{prefix}_area_mm2"] = measurement["area_mm2"]
        row[f"{prefix}_full_length_mm"] = measurement["full_length_mm"]
        row[f"{prefix}_branch_depth_mm"] = measurement["branch_depth_mm"]
    return row


def _as_int_point(point: Point) -> Tuple[int, int]:
    return int(round(float(point[0]))), int(round(float(point[1])))


def _draw_dashed_line(
    image: np.ndarray,
    p1: Point,
    p2: Point,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
    gap_length: int = 6,
):
    p1_array = np.array(p1, dtype=np.float32)
    p2_array = np.array(p2, dtype=np.float32)
    distance = float(np.linalg.norm(p2_array - p1_array))
    if distance <= 1e-6:
        return
    direction = (p2_array - p1_array) / distance
    cursor = 0.0
    while cursor < distance:
        segment_start = p1_array + direction * cursor
        segment_end = p1_array + direction * min(cursor + dash_length, distance)
        cv2.line(
            image,
            _as_int_point(tuple(segment_start)),
            _as_int_point(tuple(segment_end)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cursor += dash_length + gap_length


def _draw_endpoint(image: np.ndarray, point: Point, color: Tuple[int, int, int]):
    cv2.circle(image, _as_int_point(point), 3, color, -1, cv2.LINE_AA)


def _draw_polyline(
    image: np.ndarray,
    points: Sequence[Point],
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    if not isinstance(points, Sequence) or len(points) < 2:
        return
    int_points = np.asarray([_as_int_point(point) for point in points], dtype=np.int32)
    cv2.polylines(image, [int_points], False, color, thickness, cv2.LINE_AA)


def _draw_double_arrow(
    image: np.ndarray,
    p1: Point,
    p2: Point,
    color: Tuple[int, int, int],
    thickness: int = 2,
):
    cv2.arrowedLine(image, _as_int_point(p1), _as_int_point(p2), color, thickness, cv2.LINE_AA, tipLength=0.08)
    cv2.arrowedLine(image, _as_int_point(p2), _as_int_point(p1), color, thickness, cv2.LINE_AA, tipLength=0.08)


def _draw_points(
    image: np.ndarray,
    points: Sequence[Point],
    color: Tuple[int, int, int],
    radius: int = 2,
):
    if not isinstance(points, Sequence):
        return
    height, width = image.shape[:2]
    for point in points:
        x, y = _as_int_point(point)
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)


def _iter_measurement_items(measurement: Dict[str, object]) -> List[Dict[str, object]]:
    items = measurement.get("component_measurements")
    if isinstance(items, list) and items:
        return items
    if measurement.get("status") == "ok":
        return [measurement]
    return []


def annotate_lateral_fissure_measurement(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    measurement: Optional[Dict[str, object]] = None,
    pixel_spacing: Optional[Tuple[float, float]] = None,
    alpha: float = 0.25,
    show_labels: bool = False,
    draw_measurements: Optional[Sequence[str]] = None,
) -> np.ndarray:
    measurement = measurement or measure_lateral_fissure(mask, pixel_spacing=pixel_spacing)
    draw_set = set(draw_measurements) if draw_measurements is not None else {"depth", "width"}
    annotated = image_bgr.copy()

    component_mask = measurement.get("component_mask")
    if isinstance(component_mask, np.ndarray) and component_mask.any():
        color_layer = np.zeros_like(annotated, dtype=np.uint8)
        color_layer[component_mask] = (0, 0, 255)
        blended = cv2.addWeighted(annotated, 1.0, color_layer, alpha, 0)
        annotated[component_mask] = blended[component_mask]

    for item in _iter_measurement_items(measurement):
        depth_line = item.get("depth_line")
        width_line = item.get("width_line")
        opening_width_dash_lines = item.get("opening_width_dash_lines")
        curve_path = item.get("curve_path")
        curve_chord_line = item.get("curve_chord_line")
        curve_bulge_point = item.get("curve_bulge_point")
        measurement_anchor = item.get("measurement_anchor")
        angle_line = item.get("angle_line")
        angle_horizontal_line = item.get("angle_horizontal_line")
        if "depth" in draw_set and isinstance(depth_line, Sequence):
            p1, p2 = depth_line
            _draw_dashed_line(annotated, p1, p2, (255, 255, 0), thickness=2)
            _draw_endpoint(annotated, p1, (255, 255, 0))
            _draw_endpoint(annotated, p2, (255, 255, 0))
        if "width" in draw_set and isinstance(width_line, Sequence):
            if isinstance(opening_width_dash_lines, Sequence):
                for segment in opening_width_dash_lines:
                    if isinstance(segment, Sequence) and len(segment) == 2:
                        p1, p2 = segment
                        _draw_dashed_line(annotated, p1, p2, (0, 255, 255), thickness=2)
                        _draw_endpoint(annotated, p1, (0, 255, 255))
                        _draw_endpoint(annotated, p2, (0, 255, 255))
            else:
                p1, p2 = width_line
                _draw_dashed_line(annotated, p1, p2, (0, 255, 255), thickness=2)
                _draw_endpoint(annotated, p1, (0, 255, 255))
                _draw_endpoint(annotated, p2, (0, 255, 255))
        if "curvature" in draw_set:
            if isinstance(curve_path, Sequence):
                _draw_polyline(annotated, curve_path, (0, 255, 0), thickness=2)
            if isinstance(curve_chord_line, Sequence):
                p1, p2 = curve_chord_line
                if isinstance(curve_bulge_point, Sequence):
                    _draw_dashed_line(annotated, p1, curve_bulge_point, (0, 255, 255), thickness=1)
                    _draw_dashed_line(annotated, curve_bulge_point, p2, (0, 255, 255), thickness=1)
                    _draw_endpoint(annotated, curve_bulge_point, (0, 255, 0))
                else:
                    _draw_dashed_line(annotated, p1, p2, (0, 255, 255), thickness=1)
                _draw_endpoint(annotated, p1, (0, 255, 0))
                _draw_endpoint(annotated, p2, (0, 255, 0))
        if "angle" in draw_set:
            if isinstance(angle_horizontal_line, Sequence):
                p1, p2 = angle_horizontal_line
                _draw_dashed_line(annotated, p1, p2, (220, 220, 220), thickness=2)
            if isinstance(measurement_anchor, Sequence):
                cv2.circle(annotated, _as_int_point(measurement_anchor), 6, (0, 255, 255), -1, cv2.LINE_AA)
            if isinstance(angle_line, Sequence):
                p1, p2 = angle_line
                cv2.line(annotated, _as_int_point(p1), _as_int_point(p2), (0, 255, 0), 3, cv2.LINE_AA)

    if "angle" in draw_set and measurement.get("angle_measurement_status") == "ok":
        apex = measurement.get("third_ventricle_apex")
        if isinstance(apex, Sequence):
            _draw_endpoint(annotated, apex, (255, 255, 255))
        left_angle = measurement.get("left_angle_deg")
        right_angle = measurement.get("right_angle_deg")
        if left_angle is not None:
            cv2.putText(
                annotated,
                f"left: {float(left_angle):.1f} deg",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if right_angle is not None:
            text = f"right: {float(right_angle):.1f} deg"
            cv2.putText(
                annotated,
                text,
                (max(30, annotated.shape[1] - 320), 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    if show_labels:
        cv2.putText(
            annotated,
            "yellow: depth/opening width",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def annotate_longitudinal_fissure_measurement(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    measurement: Optional[Dict[str, object]] = None,
    pixel_spacing: Optional[Tuple[float, float]] = None,
    alpha: float = 0.25,
    draw_measurements: Optional[Sequence[str]] = None,
) -> np.ndarray:
    measurement = measurement or measure_longitudinal_fissure(mask, pixel_spacing=pixel_spacing)
    draw_set = set(draw_measurements) if draw_measurements is not None else {"branch_depth", "full_length", "area"}
    annotated = image_bgr.copy()

    component_mask = measurement.get("component_mask")
    if "area" in draw_set and isinstance(component_mask, np.ndarray) and component_mask.any():
        color_layer = np.zeros_like(annotated, dtype=np.uint8)
        color_layer[component_mask] = (255, 0, 0)
        blended = cv2.addWeighted(annotated, 1.0, color_layer, alpha, 0)
        annotated[component_mask] = blended[component_mask]

    if "full_length" in draw_set:
        contour = measurement.get("full_length_contour")
        if isinstance(contour, np.ndarray) and len(contour) > 0:
            cv2.drawContours(annotated, [contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            path = measurement.get("full_length_path")
            if isinstance(path, Sequence):
                _draw_polyline(annotated, path, (0, 255, 255), thickness=2)

    if "branch_depth" in draw_set:
        trunk_path = measurement.get("main_trunk_path")
        if isinstance(trunk_path, Sequence):
            _draw_polyline(annotated, trunk_path, (180, 180, 180), thickness=1)
        branch_path = measurement.get("branch_depth_path")
        if isinstance(branch_path, Sequence):
            _draw_polyline(annotated, branch_path, (0, 255, 255), thickness=2)
        branch_line = measurement.get("branch_depth_line")
        if isinstance(branch_line, Sequence):
            p1, p2 = branch_line
            _draw_dashed_line(annotated, p1, p2, (255, 255, 0), thickness=1)
            _draw_endpoint(annotated, p1, (255, 255, 0))
            _draw_endpoint(annotated, p2, (0, 255, 255))

    return annotated
