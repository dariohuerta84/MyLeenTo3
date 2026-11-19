"""Phase 2 Metric Scale Calibration, Passive ArUco Anchoring & Surface Caliper Geometry.

Implements uniform radial scale transformation centered at the camera optical origin O(0,0,0),
passive OpenCV ArUco fiducial detection against MoGe-2 3D point maps, and point-to-point
surface caliper calculations in Three.js world space.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Optional, Sequence

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional in minimal CPU environments
    cv2 = None  # type: ignore

SCALE_MIN: float = 0.1
SCALE_MAX: float = 10.0


def clamp_scale_factor(scale_factor: float) -> float:
    """Clamp multiplicative scene scale factor safely inside `[SCALE_MIN, SCALE_MAX]`."""
    val = float(scale_factor)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"Invalid scale_factor={scale_factor}; must be finite and > 0.")
    return float(min(SCALE_MAX, max(SCALE_MIN, val)))


def rescale_box_3d(box_3d: Sequence[float], scale_factor: float) -> list[float]:
    """Scale a 9-DOF Gemini box `[x, y, z, dx, dy, dz, roll, pitch, yaw]` radially from O(0,0,0).

    Because perspective projection (X/Y, Z/Y) is homogeneous under radial scaling from the
    camera optical center, 2D pinhole alignment and Euler orientation remain invariant while
    metric depth, extents, surface clearances, and volumes (`* alpha^3`) calibrate cleanly.
    """
    if len(box_3d) < 9:
        raise ValueError("box_3d must contain 9 floats.")
    alpha = clamp_scale_factor(scale_factor)
    x, y, z, dx, dy, dz, roll, pitch, yaw = [float(v) for v in box_3d[:9]]
    return [
        round(x * alpha, 4),
        round(max(0.04, y * alpha), 4),
        round(z * alpha, 4),
        round(max(0.005, dx * alpha), 4),
        round(max(0.005, dy * alpha), 4),
        round(max(0.005, dz * alpha), 4),
        round(roll, 1),
        round(pitch, 1),
        round(yaw, 1),
    ]


def compute_scale_factor_from_reference(
    measured_distance_m: float,
    reference_distance_m: float,
) -> dict[str, Any]:
    """Compute multiplicative calibration factor `reference / measured` with bounds check."""
    meas = float(measured_distance_m)
    ref = float(reference_distance_m)
    if not (math.isfinite(meas) and math.isfinite(ref)) or meas <= 1e-4 or ref <= 1e-4:
        raise ValueError(
            "Both measured_distance_m and reference_distance_m must be finite and > 1e-4 m."
        )
    raw = ref / meas
    clamped_val = clamp_scale_factor(raw)
    return {
        "scale_factor": round(clamped_val, 5),
        "raw_scale_factor": round(raw, 5),
        "measured_distance_m": round(meas, 5),
        "reference_distance_m": round(ref, 5),
        "delta_pct": round((clamped_val - 1.0) * 100.0, 2),
        "clamped": abs(raw - clamped_val) > 1e-7,
    }


def rescale_depth_stats(
    depth_stats: Optional[dict[str, float]],
    scale_factor: float,
) -> Optional[dict[str, float]]:
    """Rescale depth percentile summary dictionary (`min_m..max_m`)."""
    if depth_stats is None:
        return None
    alpha = clamp_scale_factor(scale_factor)
    out: dict[str, float] = {}
    for k in ("min_m", "p10_m", "median_m", "p90_m", "max_m"):
        if k in depth_stats and depth_stats[k] is not None:
            out[k] = round(float(depth_stats[k]) * alpha, 3)
    return out


def rescale_real_photo_metrics(
    real_photo_metrics: Optional[dict[str, Any]],
    scale_factor: float,
) -> Optional[dict[str, Any]]:
    """Deep-copy and rescale linear (x alpha), area (x alpha^2), and volume (x alpha^3) metrology."""
    if real_photo_metrics is None:
        return None
    alpha = clamp_scale_factor(scale_factor)
    alpha2 = alpha * alpha
    alpha3 = alpha2 * alpha
    out = copy.deepcopy(real_photo_metrics)

    plane = out.get("ransac_plane")
    if isinstance(plane, dict):
        for k in ("rmse_residual_m", "plane_height_z_m", "plane_d_gemini"):
            if isinstance(plane.get(k), (int, float)):
                plane[k] = round(float(plane[k]) * alpha, 4)

    dp = out.get("depth_profile")
    if isinstance(dp, dict) and isinstance(dp.get("scene_span_xyz_m"), list):
        dp["scene_span_xyz_m"] = [round(float(v) * alpha, 3) for v in dp["scene_span_xyz_m"]]

    fsc = out.get("free_space_consistency")
    if isinstance(fsc, dict) and isinstance(fsc.get("mean_depth_residual_m"), (int, float)):
        fsc["mean_depth_residual_m"] = round(float(fsc["mean_depth_residual_m"]) * alpha, 4)

    pom = out.get("per_object_metrology")
    if isinstance(pom, list):
        for item in pom:
            if not isinstance(item, dict):
                continue
            for k in (
                "surface_fit_rmse_m",
                "depth_iqr_m",
                "mean_relief_height_m",
                "max_relief_height_m",
                "plane_contact_gap_m",
            ):
                if isinstance(item.get(k), (int, float)):
                    item[k] = round(float(item[k]) * alpha, 4)
            if isinstance(item.get("visible_surface_area_m2"), (int, float)):
                item["visible_surface_area_m2"] = round(
                    float(item["visible_surface_area_m2"]) * alpha2, 5
                )
            if isinstance(item.get("integrated_volume_above_plane_m3"), (int, float)):
                item["integrated_volume_above_plane_m3"] = round(
                    float(item["integrated_volume_above_plane_m3"]) * alpha3, 6
                )

    return out


def compute_surface_caliper_metrics(
    point_a_three: Sequence[float],
    point_b_three: Sequence[float],
    *,
    sample_points_three: Optional[Sequence[Sequence[float]]] = None,
) -> dict[str, Any]:
    """Compute exact 3D chord distance, slope, geodesic surface arc length, and relief."""
    pa = np.asarray(point_a_three[:3], dtype=np.float64)
    pb = np.asarray(point_b_three[:3], dtype=np.float64)
    delta = pb - pa
    euclidean_m = float(np.linalg.norm(delta))

    # Three.js frame: X-Right, Y-Up, Z-Backward -> XZ is horizontal ground plane
    horiz_span = float(math.hypot(float(delta[0]), float(delta[2])))
    vert_delta = float(delta[1])
    slope_deg = (
        math.degrees(math.atan2(abs(vert_delta), max(horiz_span, 1e-7)))
        if euclidean_m > 1e-6
        else 0.0
    )

    arc_m = euclidean_m
    max_perp_m = 0.0
    if sample_points_three and len(sample_points_three) >= 2:
        pts_chain = np.vstack([pa[None, :], np.asarray(sample_points_three, dtype=np.float64)[:, :3], pb[None, :]])
        diffs = np.diff(pts_chain, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        arc_m = max(euclidean_m, float(np.sum(seg_lens)))

        if euclidean_m > 1e-6:
            rel = np.asarray(sample_points_three, dtype=np.float64)[:, :3] - pa[None, :]
            cross_norms = np.linalg.norm(np.cross(rel, delta[None, :]), axis=1)
            max_perp_m = float(np.max(cross_norms) / euclidean_m)

    return {
        "euclidean_m": round(euclidean_m, 4),
        "euclidean_cm": round(euclidean_m * 100.0, 2),
        "euclidean_mm": round(euclidean_m * 1000.0, 1),
        "delta_xyz_m": [round(float(delta[0]), 4), round(float(delta[1]), 4), round(float(delta[2]), 4)],
        "horizontal_span_m": round(horiz_span, 4),
        "vertical_delta_m": round(vert_delta, 4),
        "slope_deg": round(slope_deg, 1),
        "surface_arc_m": round(arc_m, 4),
        "surface_arc_cm": round(arc_m * 100.0, 2),
        "profile_relief_mm": round(max_perp_m * 1000.0, 1),
    }


def _sample_corner_3d(
    points_cv: np.ndarray,
    valid_mask: np.ndarray,
    cx: float,
    cy: float,
) -> Optional[np.ndarray]:
    h, w = points_cv.shape[:2]
    ix = int(np.clip(round(cx), 0, w - 1))
    iy = int(np.clip(round(cy), 0, h - 1))
    for radius in (1, 2):
        y0, y1 = max(0, iy - radius), min(h, iy + radius + 1)
        x0, x1 = max(0, ix - radius), min(w, ix + radius + 1)
        sub_m = valid_mask[y0:y1, x0:x1]
        sub_p = points_cv[y0:y1, x0:x1]
        ok = sub_m & np.isfinite(sub_p).all(axis=-1) & (sub_p[..., 2] > 0.03)
        if np.any(ok):
            return np.median(sub_p[ok], axis=0).astype(np.float64)
    return None


def detect_aruco_scale_anchor(
    rgb_u8: np.ndarray,
    points_cv: np.ndarray,
    valid_mask: np.ndarray,
    *,
    marker_size_m: float = 0.05,
) -> Optional[dict[str, Any]]:
    """Detect standard ArUco fiducial markers in RGB + MoGe-2 3D point map for automatic scale calibration."""
    if cv2 is None or not hasattr(cv2, "aruco"):
        return None
    if rgb_u8 is None or points_cv is None or valid_mask is None:
        return None
    h, w = rgb_u8.shape[:2]
    if h < 24 or w < 24 or points_cv.shape[:2] != (h, w) or valid_mask.shape[:2] != (h, w):
        return None

    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY) if rgb_u8.ndim == 3 else rgb_u8
    dict_specs = (
        ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
        ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
        ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    )

    best_cand: Optional[dict[str, Any]] = None
    best_area = 0.0
    params = cv2.aruco.DetectorParameters()

    for dict_name, dict_id in dict_specs:
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners_list, ids, _ = detector.detectMarkers(gray)
        except Exception:
            continue
        if ids is None or len(corners_list) == 0:
            continue

        flat_ids = [int(v) for v in np.asarray(ids).reshape(-1)]
        for marker_corners, marker_id in zip(corners_list, flat_ids):
            pts2d = np.asarray(marker_corners, dtype=np.float64).reshape(-1, 2)
            if pts2d.shape != (4, 2):
                continue
            # Polygon shoelace area in pixels
            x_c, y_c = pts2d[:, 0], pts2d[:, 1]
            area_px = 0.5 * abs(float(np.dot(x_c, np.roll(y_c, -1)) - np.dot(y_c, np.roll(x_c, -1))))
            if area_px < 100.0 or area_px <= best_area:
                continue

            pts3d = [_sample_corner_3d(points_cv, valid_mask, float(pt[0]), float(pt[1])) for pt in pts2d]
            if any(p is None for p in pts3d):
                continue

            side_lens = [
                float(np.linalg.norm(pts3d[(k + 1) % 4] - pts3d[k]))  # type: ignore[operator]
                for k in range(4)
            ]
            measured_side_m = float(np.median(side_lens))
            if measured_side_m <= 0.005:
                continue

            scale_info = compute_scale_factor_from_reference(measured_side_m, marker_size_m)
            best_area = area_px
            best_cand = {
                "detected": True,
                "dictionary": dict_name,
                "marker_id": marker_id,
                "corners_px": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in pts2d],
                "measured_side_m": round(measured_side_m, 4),
                "reference_side_m": round(float(marker_size_m), 4),
                "suggested_scale_factor": scale_info["scale_factor"],
            }

    return best_cand
