"""SOTA monocular real-photo 3D metrology (pure NumPy, deterministic SEED=42).

Implements paper-backed geometric verification metrics over dense metric depth,
3D point maps, surface normals, and instance masks without PyTorch runtime deps:
  - Optical sharpness (Laplacian focus variance), Immerkaer noise sigma, 35mm eq. focal.
  - MoGe-2 Normal-Depth Consistency (NDC) + Manhattan-World orthogonality score.
  - RANSAC dominant support/wall plane fitting & orthogonal point-cloud residuals.
  - Depth Pro-inspired Depth-RGB boundary alignment F1 & flying-pixel ratio.
  - Per-object 3D surface fit RMSE (analytical OBB SDF), voxel solidity, and visible area.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from PIL import Image

from .geometry_3d import euler_deg_to_quaternion, project_obb_to_2d, quaternion_to_rotation_matrix


def compute_specular_mask(rgb_u8: np.ndarray) -> tuple[np.ndarray, float]:
    """Detect clipped achromatic specular highlights (Harvard clinical endoscopy masking)."""
    arr = np.asarray(rgb_u8, dtype=np.float32)
    if arr.size == 0:
        return np.zeros(arr.shape[:2], dtype=bool), 0.0
    if float(np.max(arr)) > 1.5:
        arr = arr / 255.0
    if arr.ndim == 3 and arr.shape[2] >= 3:
        c_rgb = arr[:, :, :3]
        luma = 0.2126 * c_rgb[:, :, 0] + 0.7152 * c_rgb[:, :, 1] + 0.0722 * c_rgb[:, :, 2]
        chroma = np.max(c_rgb, axis=2) - np.min(c_rgb, axis=2)
    else:
        luma = arr.astype(np.float32)
        chroma = np.zeros_like(luma)
    spec_mask = (luma > 0.90) & (chroma < 0.14)
    return spec_mask, float(np.mean(spec_mask))


def _dilate_binary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Fast 4-connected morphological binary dilation in pure NumPy."""
    out = mask.copy()
    for _ in range(max(1, radius)):
        cur = out.copy()
        out[1:, :] |= cur[:-1, :]
        out[:-1, :] |= cur[1:, :]
        out[:, 1:] |= cur[:, :-1]
        out[:, :-1] |= cur[:, 1:]
    return out


def compute_optical_sharpness_and_noise(
    rgb_f32: np.ndarray,
    fov_h_deg: float,
    width: int,
    height: int,
) -> dict[str, Any]:
    """Compute Laplacian focus variance, Immerkaer (1996) noise sigma, Michelson contrast, and 35mm focal eq."""
    arr = np.asarray(rgb_f32, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    else:
        luma = arr.astype(np.float32)

    if float(np.max(luma)) > 1.5:
        luma = luma / 255.0

    # Pech-Pacheco discrete Laplacian focus variance on 0..255 luma
    if luma.shape[0] >= 4 and luma.shape[1] >= 4:
        luma_255 = luma * 255.0
        lap = (
            luma_255[:-2, 1:-1]
            + luma_255[2:, 1:-1]
            + luma_255[1:-1, :-2]
            + luma_255[1:-1, 2:]
            - 4.0 * luma_255[1:-1, 1:-1]
        )
        lap_var = float(np.var(lap))

        # Immerkaer (1996) fast noise estimator kernel [[1,-2,1],[-2,4,-2],[1,-2,1]]
        noise_resp = (
            luma[:-2, :-2] - 2.0 * luma[:-2, 1:-1] + luma[:-2, 2:]
            - 2.0 * luma[1:-1, :-2] + 4.0 * luma[1:-1, 1:-1] - 2.0 * luma[1:-1, 2:]
            + luma[2:, :-2] - 2.0 * luma[2:, 1:-1] + luma[2:, 2:]
        )
        h_n, w_n = noise_resp.shape
        sigma = float(np.sum(np.abs(noise_resp)) * math.sqrt(math.pi * 0.5) / max(6.0 * h_n * w_n, 1.0))
        noise_sigma_est = float(np.clip(sigma, 0.0, 1.0))
    else:
        lap_var = 0.0
        noise_sigma_est = 0.0

    if lap_var >= 180.0:
        focus_quality = "sharp"
    elif lap_var >= 45.0:
        focus_quality = "moderate"
    else:
        focus_quality = "blurry"

    p02, p98 = np.percentile(luma, [2, 98])
    contrast_ratio = float(np.clip((p98 - p02) / max(p98 + p02, 1e-4), 0.0, 1.0))

    sensor_w = 36.0 if width >= height else 24.0
    tan_half = max(math.tan(math.radians(float(np.clip(fov_h_deg, 15.0, 140.0)) * 0.5)), 1e-4)
    eq_f35 = (sensor_w * 0.5) / tan_half

    return {
        "laplacian_variance": round(lap_var, 2),
        "focus_quality": focus_quality,
        "noise_sigma_est": round(noise_sigma_est, 4),
        "dynamic_contrast_ratio": round(contrast_ratio, 4),
        "equivalent_focal_35mm_mm": round(eq_f35, 2),
    }


def _analytical_gradient_normals_cv(points_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute central spatial finite-difference normals oriented toward camera (-P)."""
    dp_du = points_cv[1:-1, 2:] - points_cv[1:-1, :-2]
    dp_dv = points_cv[2:, 1:-1] - points_cv[:-2, 1:-1]
    # OpenCV (X-Right, Y-Down, Z-Forward): +Y x +X = -Z (toward camera origin)
    raw_n = np.cross(dp_dv, dp_du)
    norms = np.linalg.norm(raw_n, axis=-1, keepdims=True)
    ok = norms[..., 0] > 1e-7
    unit_n = np.divide(raw_n, np.maximum(norms, 1e-8))

    # Enforce view-facing hemisphere dot(n, -P) >= 0
    center_pts = points_cv[1:-1, 1:-1]
    view_dot = np.sum(unit_n * (-center_pts), axis=-1, keepdims=True)
    unit_n = np.where(view_dot < 0.0, -unit_n, unit_n)
    return unit_n, ok


def compute_normal_depth_consistency(
    points_cv: np.ndarray,
    normals_cv: Optional[np.ndarray],
    valid_mask: np.ndarray,
) -> dict[str, Any]:
    """Compare predicted MoGe-2 surface normals vs depth-map gradients and score Manhattan orthogonality."""
    h, w = points_cv.shape[:2]
    if h < 4 or w < 4:
        return {
            "has_model_normals": normals_cv is not None,
            "mean_angular_error_deg": 0.0,
            "median_angular_error_deg": 0.0,
            "consistent_ratio_15deg": 1.0,
            "ndc_score_0_100": 100.0,
            "manhattan_alignment_score": 0.0,
        }

    grad_n, ok_grad = _analytical_gradient_normals_cv(points_cv)
    inner_valid = (
        valid_mask[1:-1, 1:-1]
        & valid_mask[1:-1, :-2]
        & valid_mask[1:-1, 2:]
        & valid_mask[:-2, 1:-1]
        & valid_mask[2:, 1:-1]
        & ok_grad
        & np.isfinite(grad_n).all(axis=-1)
    )

    if normals_cv is not None and normals_cv.shape[:2] == (h, w):
        pred_inner = normals_cv[1:-1, 1:-1]
        pnorm = np.linalg.norm(pred_inner, axis=-1, keepdims=True)
        valid_both = inner_valid & (pnorm[..., 0] > 1e-5) & np.isfinite(pred_inner).all(axis=-1)
        if np.any(valid_both):
            n_pred = np.divide(pred_inner[valid_both], pnorm[valid_both])
            n_grad = grad_n[valid_both]
            dots = np.clip(np.sum(n_pred * n_grad, axis=-1), -1.0, 1.0)
            angles = np.degrees(np.arccos(dots))
            mean_err = float(np.mean(angles))
            med_err = float(np.median(angles))
            ratio_15 = float(np.mean(angles < 15.0))
            ndc_score = float(np.clip(100.0 * (1.0 - med_err / 45.0) * (0.4 + 0.6 * ratio_15), 0.0, 100.0))
            norm_samples = n_pred
        else:
            mean_err, med_err, ratio_15, ndc_score = 0.0, 0.0, 0.0, 0.0
            norm_samples = grad_n[inner_valid] if np.any(inner_valid) else np.empty((0, 3), dtype=np.float32)
    else:
        # Compare gradient normals against local 3x3 smoothed normals to measure self-consistency
        if np.any(inner_valid):
            norm_samples = grad_n[inner_valid]
            # Local neighbor smoothness inside valid patch
            dots_x = np.sum(grad_n[:, :-1] * grad_n[:, 1:], axis=-1)
            mask_x = inner_valid[:, :-1] & inner_valid[:, 1:]
            angles_x = np.degrees(np.arccos(np.clip(dots_x[mask_x], -1.0, 1.0))) if np.any(mask_x) else np.array([0.0])
            mean_err = float(np.mean(angles_x))
            med_err = float(np.median(angles_x))
            ratio_15 = float(np.mean(angles_x < 15.0))
            ndc_score = float(np.clip(100.0 * (1.0 - med_err / 45.0), 0.0, 100.0))
        else:
            mean_err, med_err, ratio_15, ndc_score = 0.0, 0.0, 0.0, 0.0
            norm_samples = np.empty((0, 3), dtype=np.float32)

    # Manhattan-World principal-frame orthogonality alignment
    manhattan_score = 0.0
    if norm_samples.shape[0] >= 16:
        if norm_samples.shape[0] > 4000:
            step = max(1, norm_samples.shape[0] // 4000)
            norm_samples = norm_samples[::step]
        cov = norm_samples.T @ norm_samples
        _, eigvecs = np.linalg.eigh(cov)
        max_align = np.max(np.abs(norm_samples @ eigvecs), axis=-1)
        mean_align = float(np.mean(max_align))
        manhattan_score = float(np.clip((mean_align - 0.72) / 0.28, 0.0, 1.0))

    return {
        "has_model_normals": normals_cv is not None,
        "mean_angular_error_deg": round(mean_err, 2),
        "median_angular_error_deg": round(med_err, 2),
        "consistent_ratio_15deg": round(ratio_15, 4),
        "ndc_score_0_100": round(ndc_score, 1),
        "manhattan_alignment_score": round(manhattan_score, 4),
    }


def fit_ransac_dominant_plane(
    points_cv: np.ndarray,
    valid_mask: np.ndarray,
    *,
    seed: int = 42,
    n_iters: int = 120,
    inlier_thresh_m: Optional[float] = None,
    specular_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Fit dominant plane via RANSAC + SVD refinement in Gemini coordinates (X-Right, Y-Fwd, Z-Up)."""
    z_cv = points_cv[..., 2]
    keep = valid_mask & np.isfinite(points_cv).all(axis=-1) & (z_cv > 0.04) & (z_cv < 35.0)
    if specular_mask is not None and specular_mask.shape[:2] == keep.shape[:2]:
        keep_clean = keep & (~specular_mask)
        if int(np.sum(keep_clean)) >= 32:
            keep = keep_clean
    pts_cv = points_cv[keep]
    if pts_cv.shape[0] < 32:
        return {
            "detected": False,
            "inlier_ratio": 0.0,
            "rmse_residual_m": 0.0,
            "plane_tilt_deg": 0.0,
            "plane_height_z_m": 0.0,
            "plane_d_gemini": 0.0,
            "plane_normal_gemini": [0.0, 0.0, 1.0],
            "plane_orientation": "undetermined",
        }

    # OpenCV -> Gemini (X-right, Y-fwd, Z-up)
    pts_gem = np.column_stack([pts_cv[:, 0], pts_cv[:, 2], -pts_cv[:, 1]]).astype(np.float64)
    if pts_gem.shape[0] > 3500:
        idx_sub = np.linspace(0, pts_gem.shape[0] - 1, 3500, dtype=np.int64)
        pts_gem = pts_gem[idx_sub]

    if inlier_thresh_m is None:
        med_y = max(0.25, float(np.median(pts_gem[:, 1])))
        inlier_thresh_m = max(0.015, 0.03 * med_y)

    rng = np.random.default_rng(int(seed))
    m_pts = pts_gem.shape[0]
    best_inliers: Optional[np.ndarray] = None
    best_count = -1

    for _ in range(n_iters):
        idx = rng.choice(m_pts, size=3, replace=False)
        p0, p1, p2 = pts_gem[idx[0]], pts_gem[idx[1]], pts_gem[idx[2]]
        n = np.cross(p1 - p0, p2 - p0)
        norm_n = float(np.linalg.norm(n))
        if norm_n < 1e-6:
            continue
        n = n / norm_n
        d = -float(np.dot(n, p0))
        dists = np.abs(pts_gem @ n + d)
        mask_in = dists <= inlier_thresh_m
        cnt = int(np.sum(mask_in))
        if cnt > best_count:
            best_count = cnt
            best_inliers = mask_in

    if best_inliers is None or best_count < 16:
        return {
            "detected": False,
            "inlier_ratio": 0.0,
            "rmse_residual_m": 0.0,
            "plane_tilt_deg": 0.0,
            "plane_height_z_m": 0.0,
            "plane_d_gemini": 0.0,
            "plane_normal_gemini": [0.0, 0.0, 1.0],
            "plane_orientation": "undetermined",
        }

    inlier_pts = pts_gem[best_inliers]
    centroid = np.mean(inlier_pts, axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal = normal / max(float(np.linalg.norm(normal)), 1e-9)

    resid = centered @ normal
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    tilt_deg = float(math.degrees(math.acos(float(np.clip(abs(normal[2]), 0.0, 1.0)))))
    inlier_ratio = float(best_count) / float(m_pts)
    height_z = float(np.median(inlier_pts[:, 2]))
    d_plane = float(-np.dot(normal, centroid))

    if tilt_deg <= 25.0:
        orientation = "horizontal_support"
    elif tilt_deg >= 65.0:
        orientation = "vertical_wall"
    else:
        orientation = "oblique_surface"

    return {
        "detected": True,
        "inlier_ratio": round(inlier_ratio, 4),
        "rmse_residual_m": round(rmse, 4),
        "plane_tilt_deg": round(tilt_deg, 2),
        "plane_height_z_m": round(height_z, 3),
        "plane_d_gemini": round(d_plane, 4),
        "plane_normal_gemini": [round(float(v), 4) for v in normal],
        "plane_orientation": orientation,
    }


def compute_depth_boundary_sharpness(
    depth_m: np.ndarray,
    rgb_u8: np.ndarray,
    valid_mask: np.ndarray,
    *,
    instance_masks: Optional[list[np.ndarray]] = None,
) -> dict[str, Any]:
    """Measure Depth-RGB Boundary F1, Depth Pro multi-threshold contour sharpness & instance silhouette recall."""
    h, w = depth_m.shape[:2]
    if h < 4 or w < 4:
        return {
            "depth_rgb_boundary_f1": 0.0,
            "depth_pro_multiscale_f1": 0.0,
            "silhouette_boundary_recall": None,
            "flying_pixel_ratio": 0.0,
            "depth_smoothness_index": 1.0,
        }

    safe_d = np.where(valid_mask & np.isfinite(depth_m) & (depth_m > 0.02), depth_m, 1.0)
    log_d = np.log(safe_d)
    rel_x = np.abs(log_d[:, 1:] - log_d[:, :-1])
    rel_y = np.abs(log_d[1:, :] - log_d[:-1, :])

    jump = np.zeros((h, w), dtype=np.float32)
    jump[:-1, :-1] = np.maximum(rel_x[:-1, :], rel_y[:, :-1])

    v_int = valid_mask[:-1, :-1] & valid_mask[1:, :-1] & valid_mask[:-1, 1:]
    valid_jumps = jump[:-1, :-1][v_int]
    if valid_jumps.size == 0:
        return {
            "depth_rgb_boundary_f1": 0.0,
            "depth_pro_multiscale_f1": 0.0,
            "silhouette_boundary_recall": None,
            "flying_pixel_ratio": 0.0,
            "depth_smoothness_index": 1.0,
        }

    # Flying pixel: >8% relative neighbor depth jump
    fly_thresh = math.log(1.08)
    flying_ratio = float(np.mean(valid_jumps > fly_thresh))
    smoothness = float(np.clip(1.0 - float(np.median(valid_jumps)) * 4.0, 0.0, 1.0))

    # RGB luminance gradient edges
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    if rgb_f.shape[:2] != (h, w):
        pil_r = Image.fromarray(rgb_u8).resize((w, h), Image.Resampling.BILINEAR)
        rgb_f = np.asarray(pil_r, dtype=np.float32) / 255.0
    luma = 0.2126 * rgb_f[..., 0] + 0.7152 * rgb_f[..., 1] + 0.0722 * rgb_f[..., 2]
    gx = np.abs(luma[:, 1:] - luma[:, :-1])
    gy = np.abs(luma[1:, :] - luma[:-1, :])
    rgb_edge = np.zeros((h, w), dtype=np.float32)
    rgb_edge[:-1, :-1] = np.hypot(gx[:-1, :], gy[:, :-1])

    d_thresh = max(math.log(1.045), float(np.percentile(valid_jumps, 88)))
    depth_bnd = (jump[:-1, :-1] >= d_thresh) & v_int

    r_vals = rgb_edge[:-1, :-1][v_int]
    r_thresh = max(0.03, float(np.percentile(r_vals, 82))) if r_vals.size > 0 else 0.05
    rgb_bnd = rgb_edge[:-1, :-1] >= r_thresh

    dil_rgb = _dilate_binary(rgb_bnd, radius=1)
    dil_depth = _dilate_binary(depth_bnd, radius=1)

    n_d = float(np.sum(depth_bnd))
    n_r = float(np.sum(rgb_bnd))
    if n_d > 0.0 and n_r > 0.0:
        prec = float(np.sum(depth_bnd & dil_rgb)) / n_d
        rec = float(np.sum(rgb_bnd & dil_depth)) / n_r
        f1 = (2.0 * prec * rec) / max(prec + rec, 1e-6)
    else:
        f1 = 0.0

    # Depth Pro (Eq. 3): weighted multi-threshold occluding contours d(j)/d(i) > 1 + t/100, t in [5..25]%
    d_tl = safe_d[:-1, :-1]
    d_tr = safe_d[:-1, 1:]
    d_bl = safe_d[1:, :-1]
    rx = np.maximum(d_tr / np.maximum(d_tl, 1e-4), d_tl / np.maximum(d_tr, 1e-4))
    ry = np.maximum(d_bl / np.maximum(d_tl, 1e-4), d_tl / np.maximum(d_bl, 1e-4))
    r_max = np.maximum(rx, ry)

    thresholds_pct = (5.0, 10.0, 15.0, 20.0, 25.0)
    w_sum = sum(thresholds_pct)
    ms_f1 = 0.0
    if n_r > 0.0:
        for t_pct in thresholds_pct:
            w_t = t_pct / w_sum
            occ_t = (r_max > (1.0 + t_pct / 100.0)) & v_int
            nd_t = float(np.sum(occ_t))
            if nd_t > 0.0:
                dil_occ = _dilate_binary(occ_t, radius=1)
                p_t = float(np.sum(occ_t & dil_rgb)) / nd_t
                r_t = float(np.sum(rgb_bnd & dil_occ)) / n_r
                ms_f1 += w_t * ((2.0 * p_t * r_t) / max(p_t + r_t, 1e-6))

    sil_recall: Optional[float] = None
    if instance_masks:
        sil_bnd = np.zeros((h - 1, w - 1), dtype=bool)
        for raw_m in instance_masks:
            if raw_m.shape[:2] != (h, w):
                pil_m = Image.fromarray((raw_m > 0).astype(np.uint8) * 255, mode="L").resize(
                    (w, h), Image.Resampling.NEAREST
                )
                b = np.asarray(pil_m) > 127
            else:
                b = raw_m > 0
            sil_bnd |= (b[:-1, :-1] != b[:-1, 1:]) | (b[:-1, :-1] != b[1:, :-1])
        sil_bnd &= v_int
        n_sil = float(np.sum(sil_bnd))
        if n_sil > 0.0:
            rec_acc = 0.0
            for t_pct in thresholds_pct:
                w_t = t_pct / w_sum
                occ_t = (r_max > (1.0 + t_pct / 100.0)) & v_int
                dil2_occ = _dilate_binary(occ_t, radius=2)
                rec_acc += w_t * (float(np.sum(sil_bnd & dil2_occ)) / n_sil)
            sil_recall = round(rec_acc, 4)
        else:
            sil_recall = 0.0

    return {
        "depth_rgb_boundary_f1": round(f1, 4),
        "depth_pro_multiscale_f1": round(ms_f1, 4),
        "silhouette_boundary_recall": sil_recall,
        "flying_pixel_ratio": round(flying_ratio, 4),
        "depth_smoothness_index": round(smoothness, 4),
    }


def compute_object_point_cloud_metrology(
    points_cv: np.ndarray,
    normals_cv: Optional[np.ndarray],
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    box_3d: list[float],
    fov_h_deg: float,
    *,
    object_id: str = "obj_1",
    label: str = "object",
    ransac_plane: Optional[dict[str, Any]] = None,
    specular_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute per-object OBB point-cloud SDF RMSE, visible surface area (m^2), relief/volume over RANSAC plane & glare ratio."""
    h, w = points_cv.shape[:2]
    if binary_mask.shape != (h, w):
        pil_m = Image.fromarray((binary_mask > 0).astype(np.uint8) * 255, mode="L").resize(
            (w, h), Image.Resampling.NEAREST
        )
        mask = (np.asarray(pil_m) > 127) & valid_mask
    else:
        mask = (binary_mask > 0) & valid_mask

    mask &= np.isfinite(points_cv).all(axis=-1) & (points_cv[..., 2] > 0.04) & (points_cv[..., 2] < 45.0)
    pt_count = int(np.sum(mask))

    if pt_count < 8:
        return {
            "object_id": object_id,
            "label": label,
            "point_count": pt_count,
            "surface_fit_rmse_m": 0.0,
            "fill_solidity_ratio": 0.0,
            "visible_surface_area_m2": 0.0,
            "depth_iqr_m": 0.0,
            "angular_span_deg": [0.0, 0.0],
            "normal_consistency_deg": 0.0,
            "mean_relief_height_m": 0.0,
            "max_relief_height_m": 0.0,
            "plane_contact_gap_m": 0.0,
            "integrated_volume_above_plane_m3": 0.0,
            "specular_ratio_in_mask": 0.0,
        }

    pts_cv = points_cv[mask]
    pts_gem = np.column_stack([pts_cv[:, 0], pts_cv[:, 2], -pts_cv[:, 1]]).astype(np.float64)

    center = np.array(box_3d[:3], dtype=np.float64)
    half_ext = np.maximum(np.array(box_3d[3:6], dtype=np.float64) * 0.5, 1e-4)
    R = quaternion_to_rotation_matrix(
        euler_deg_to_quaternion(box_3d[6], box_3d[7], box_3d[8])
    )

    # Exact OBB SDF residual: distance from each 3D point to nearest cuboid surface face
    local_pts = (pts_gem - center) @ R
    q = np.abs(local_pts) - half_ext
    outside_dist = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside_dist = np.abs(np.maximum(np.maximum(q[:, 0], q[:, 1]), q[:, 2]))
    surf_dist = np.where(np.any(q > 0.0, axis=-1), outside_dist, inside_dist)
    surface_fit_rmse = float(np.sqrt(np.mean(surf_dist ** 2)))

    # Visible surface occupancy ratio in 6x6x6 normalized OBB bins (normalized to visible surface bound)
    inside_box = local_pts[np.all(np.abs(local_pts) <= half_ext * 1.15, axis=-1)]
    if inside_box.shape[0] > 0:
        norm_c = np.clip((inside_box / (half_ext * 1.15) + 1.0) * 0.5, 0.0, 0.9999)
        vox = np.floor(norm_c * 6.0).astype(np.int32)
        n_unique = np.unique(vox, axis=0).shape[0]
        # Up to ~90 surface voxels can face a monocular camera on a solid box
        solidity = float(np.clip(n_unique / 72.0, 0.0, 1.0))
    else:
        solidity = 0.0

    # Integrated visible 3D surface area sum (z / f)^2 / max(|n . v|, 0.25)
    f_px = (w * 0.5) / max(math.tan(math.radians(float(np.clip(fov_h_deg, 15.0, 140.0)) * 0.5)), 1e-4)
    z_m = pts_cv[:, 2].astype(np.float64)
    pixel_area_m2 = (z_m / max(f_px, 1.0)) ** 2

    if normals_cv is not None and normals_cv.shape[:2] == (h, w):
        obj_n = normals_cv[mask].astype(np.float64)
        obj_n_norm = np.linalg.norm(obj_n, axis=-1, keepdims=True)
        valid_n = obj_n_norm[..., 0] > 1e-5
        view_dir = -pts_cv.astype(np.float64) / np.maximum(np.linalg.norm(pts_cv, axis=-1, keepdims=True), 1e-6)
        cos_theta = np.ones(pt_count, dtype=np.float64)
        cos_theta[valid_n] = np.clip(
            np.abs(np.sum((obj_n[valid_n] / obj_n_norm[valid_n]) * view_dir[valid_n], axis=-1)),
            0.25,
            1.0,
        )
        vis_area_m2 = float(np.sum(pixel_area_m2 / cos_theta))
    else:
        vis_area_m2 = float(np.sum(pixel_area_m2))

    # Topographical relief & volume over dominant RANSAC plane (Harvard 1710.11216 & Odense NCT02423850)
    mean_relief_m = 0.0
    max_relief_m = 0.0
    contact_gap_m = 0.0
    vol_above_m3 = 0.0
    if ransac_plane is not None and ransac_plane.get("detected"):
        n_pl = np.array(ransac_plane["plane_normal_gemini"], dtype=np.float64)
        n_norm = float(np.linalg.norm(n_pl))
        if n_norm > 1e-7:
            n_pl = n_pl / n_norm
            d_pl = float(
                ransac_plane.get(
                    "plane_d_gemini",
                    -float(n_pl[2]) * float(ransac_plane.get("plane_height_z_m", 0.0)),
                )
            )
            cam_sign = 1.0 if d_pl >= 0.0 else -1.0
            h_signed = cam_sign * (pts_gem @ n_pl + d_pl)
            h_pos = np.maximum(h_signed, 0.0)
            mean_relief_m = float(np.mean(h_pos))
            max_relief_m = float(np.percentile(h_pos, 95))
            contact_gap_m = float(np.percentile(np.abs(h_signed), 5))
            vol_above_m3 = max(0.0, float(np.sum(h_pos * pixel_area_m2)))

    spec_in_mask = 0.0
    if specular_mask is not None and specular_mask.shape[:2] == (h, w):
        spec_in_mask = float(np.mean(specular_mask[mask]))

    # Depth thickness IQR (p75 - p25)
    q25, q75 = np.percentile(z_m, [25, 75])
    depth_iqr = float(q75 - q25)

    # Angular span [horizontal_deg, vertical_deg]
    ys_idx, xs_idx = np.where(mask)
    span_x = float(np.max(xs_idx) - np.min(xs_idx) + 1) / max(float(w), 1.0)
    span_y = float(np.max(ys_idx) - np.min(ys_idx) + 1) / max(float(h), 1.0)
    fov_v_deg = math.degrees(2.0 * math.atan(math.tan(math.radians(fov_h_deg) * 0.5) * (h / max(float(w), 1.0))))
    ang_h = span_x * fov_h_deg
    ang_v = span_y * fov_v_deg

    # Per-object normal-depth consistency (deg)
    obj_ndc_deg = 0.0
    if normals_cv is not None and normals_cv.shape[:2] == (h, w) and h >= 4 and w >= 4:
        grad_n, ok_grad = _analytical_gradient_normals_cv(points_cv)
        in_m = (
            mask[1:-1, 1:-1]
            & ok_grad
            & valid_mask[1:-1, :-2]
            & valid_mask[1:-1, 2:]
            & valid_mask[:-2, 1:-1]
            & valid_mask[2:, 1:-1]
        )
        pred_in = normals_cv[1:-1, 1:-1]
        pnorm = np.linalg.norm(pred_in, axis=-1, keepdims=True)
        in_m &= pnorm[..., 0] > 1e-5
        if np.any(in_m):
            np_u = pred_in[in_m] / pnorm[in_m]
            dots = np.clip(np.sum(np_u * grad_n[in_m], axis=-1), -1.0, 1.0)
            obj_ndc_deg = float(np.median(np.degrees(np.arccos(dots))))

    return {
        "object_id": object_id,
        "label": label,
        "point_count": pt_count,
        "surface_fit_rmse_m": round(surface_fit_rmse, 4),
        "fill_solidity_ratio": round(solidity, 4),
        "visible_surface_area_m2": round(vis_area_m2, 4),
        "depth_iqr_m": round(depth_iqr, 4),
        "angular_span_deg": [round(ang_h, 2), round(ang_v, 2)],
        "normal_consistency_deg": round(obj_ndc_deg, 2),
        "mean_relief_height_m": round(mean_relief_m, 4),
        "max_relief_height_m": round(max_relief_m, 4),
        "plane_contact_gap_m": round(contact_gap_m, 4),
        "integrated_volume_above_plane_m3": round(vol_above_m3, 6),
        "specular_ratio_in_mask": round(spec_in_mask, 4),
    }


def compute_free_space_consistency(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    objects: list[dict[str, Any]],
    fov_h_deg: float,
) -> dict[str, Any]:
    """Check whether projected 3D boxes respect observed metric free-space depth rays (PhySIC 2510.11649)."""
    h, w = depth_m.shape[:2]
    if h < 4 or w < 4 or not objects:
        return {
            "free_space_violation_ratio": 0.0,
            "mean_depth_residual_m": 0.0,
            "consistent_objects": 0,
            "total_objects": 0,
        }

    evaluated = 0
    violations = 0
    residuals: list[float] = []

    for obj in objects:
        box = obj.get("box_3d")
        if not isinstance(box, (list, tuple)) or len(box) < 6:
            continue
        proj = project_obb_to_2d(box, w, h, float(fov_h_deg))
        if proj.get("behind_camera"):
            continue
        verts = proj.get("vertices_2d") or []
        if not verts:
            continue
        us = [float(p[0]) for p in verts]
        vs = [float(p[1]) for p in verts]
        u0 = max(0.0, min(us))
        u1 = min(float(w - 1), max(us))
        v0 = max(0.0, min(vs))
        v1 = min(float(h - 1), max(vs))
        du = u1 - u0
        dv = v1 - v0
        if du < 4.0 or dv < 4.0:
            continue

        x0 = max(0, min(w - 1, int(round(u0 + 0.25 * du))))
        x1 = max(x0 + 1, min(w, int(round(u1 - 0.25 * du))))
        y0 = max(0, min(h - 1, int(round(v0 + 0.25 * dv))))
        y1 = max(y0 + 1, min(h, int(round(v1 - 0.25 * dv))))

        win_d = depth_m[y0:y1, x0:x1]
        win_v = valid_mask[y0:y1, x0:x1] & np.isfinite(win_d) & (win_d > 0.02)
        if int(np.sum(win_v)) < 4:
            continue

        d_obs = float(np.median(win_d[win_v]))
        y_c = max(0.05, float(box[1]))
        dy = abs(float(box[4]))
        y_front = y_c - 0.5 * dy
        y_back = y_c + 0.5 * dy

        evaluated += 1
        residuals.append(abs(d_obs - y_c))
        if (d_obs > y_back + max(0.08, 0.14 * y_c)) or (y_front > d_obs + max(0.12, 0.18 * d_obs)):
            violations += 1

    if evaluated == 0:
        return {
            "free_space_violation_ratio": 0.0,
            "mean_depth_residual_m": 0.0,
            "consistent_objects": 0,
            "total_objects": 0,
        }

    return {
        "free_space_violation_ratio": round(float(violations) / float(evaluated), 4),
        "mean_depth_residual_m": round(float(np.mean(residuals)), 4),
        "consistent_objects": int(evaluated - violations),
        "total_objects": int(evaluated),
    }


def compute_real_photo_3d_metrics(
    *,
    points_cv: np.ndarray,
    depth_m: np.ndarray,
    normals_cv: Optional[np.ndarray],
    rgb_u8: np.ndarray,
    valid_mask: np.ndarray,
    effective_fov_h_deg: float,
    exif_fov_h_deg: Optional[float] = None,
    moge_fov_h_deg: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    per_object_metrology: Optional[list[dict[str, Any]]] = None,
    instance_masks: Optional[list[np.ndarray]] = None,
    objects: Optional[list[dict[str, Any]]] = None,
    ransac_plane: Optional[dict[str, Any]] = None,
    specular_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute unified SOTA monocular 3D real-photo metrology payload."""
    h_img, w_img = depth_m.shape[:2]
    full_w = int(width or w_img)
    full_h = int(height or h_img)

    if specular_mask is None:
        spec_mask, spec_ratio = compute_specular_mask(rgb_u8)
    else:
        spec_mask = specular_mask
        spec_ratio = float(np.mean(specular_mask)) if specular_mask.size > 0 else 0.0

    ndc = compute_normal_depth_consistency(points_cv, normals_cv, valid_mask)
    ransac = (
        ransac_plane
        if ransac_plane is not None
        else fit_ransac_dominant_plane(points_cv, valid_mask, seed=42, specular_mask=spec_mask)
    )
    boundaries = compute_depth_boundary_sharpness(
        depth_m,
        rgb_u8,
        valid_mask,
        instance_masks=instance_masks,
    )
    optical = compute_optical_sharpness_and_noise(
        rgb_u8.astype(np.float32) / 255.0,
        effective_fov_h_deg,
        full_w,
        full_h,
    )
    optical["specular_reflection_ratio"] = round(spec_ratio, 4)
    free_space = compute_free_space_consistency(
        depth_m,
        valid_mask,
        objects or [],
        effective_fov_h_deg,
    )

    keep = valid_mask & np.isfinite(points_cv).all(axis=-1) & np.isfinite(depth_m) & (depth_m > 0.02)
    pts_gem = (
        np.column_stack([points_cv[keep, 0], points_cv[keep, 2], -points_cv[keep, 1]])
        if np.any(keep)
        else np.zeros((0, 3), dtype=np.float32)
    )
    if pts_gem.shape[0] > 0:
        d_vals = pts_gem[:, 1]
        d_p05, d_med, d_p95 = np.percentile(d_vals, [5, 50, 95])
        dyn_ratio = float(d_p95 / max(float(d_p05), 0.05))
        fg_ratio = float(np.mean(d_vals <= (float(d_p05) + 0.35 * max(float(d_p95 - d_p05), 0.05))))
        span_xyz = [
            round(float(np.percentile(pts_gem[:, ax], 98) - np.percentile(pts_gem[:, ax], 2)), 3)
            for ax in range(3)
        ]
        if float(d_med) < 1.2:
            scale_regime = "macro_closeup"
        elif float(d_med) < 4.5:
            scale_regime = "indoor_room"
        else:
            scale_regime = "large_environment"
    else:
        dyn_ratio, fg_ratio, span_xyz, scale_regime = 1.0, 0.0, [0.0, 0.0, 0.0], "undetermined"

    fov_diff = (
        round(abs(float(exif_fov_h_deg) - float(moge_fov_h_deg)), 2)
        if exif_fov_h_deg is not None and moge_fov_h_deg is not None
        else None
    )

    plane_bonus = max(0.0, 1.0 - min(1.0, ransac["rmse_residual_m"] / 0.12)) if ransac["detected"] else 0.45
    fly_pen = max(0.0, 1.0 - min(1.0, boundaries["flying_pixel_ratio"] / 0.25))
    overall_score = round(
        float(
            np.clip(
                100.0
                * (
                    0.35 * (ndc["ndc_score_0_100"] / 100.0)
                    + 0.25 * boundaries["depth_rgb_boundary_f1"]
                    + 0.20 * fly_pen
                    + 0.20 * plane_bonus
                ),
                0.0,
                100.0,
            )
        ),
        1,
    )

    return {
        "normal_depth_consistency": ndc,
        "manhattan_alignment_score": ndc["manhattan_alignment_score"],
        "ransac_plane": ransac,
        "depth_boundaries": boundaries,
        "depth_profile": {
            "depth_dynamic_range_ratio": round(dyn_ratio, 2),
            "foreground_ratio": round(fg_ratio, 4),
            "metric_scale_regime": scale_regime,
            "scene_span_xyz_m": span_xyz,
        },
        "optical_telemetry": optical,
        "free_space_consistency": free_space,
        "fov_exif_vs_moge_diff_deg": fov_diff,
        "per_object_metrology": per_object_metrology or [],
        "overall_3d_geometry_score_0_100": overall_score,
    }
