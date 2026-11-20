"""Phase 3 Multi-View 360-degree Registration, Surfel Fusion & Volumetric Body Metrology.

Implements:
  1. Umeyama Sim(3) closed-form SVD alignment (enforcing SO(3) det(R) = +1 chirality).
  2. ArUco-assisted + SIFT 2D-to-3D RANSAC Sim(3) coarse registration.
  3. Vectorized cKDTree Point-to-Plane Huber-trimmed ICP refinement.
  4. Azimuth orbit ICP fallback when textures are sparse across wide baselines.
  5. Voxel-hashed 360-degree surfel fusion, combined discontinuity-cut surface mesh,
     ConvexHull volumetric metrology, and 12-sector ActivePose Next-Best-View guidance.
  6. Deterministic SEED=42 calibrated 4-view 360-degree industrial benchmark generator
     plus offline depth fallback for CPU-only CI testing.
"""

from __future__ import annotations

import base64
import io
import math
import re
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional in minimal CPU environments
    cv2 = None  # type: ignore

# SO(3) reflection-free basis change: OpenCV (X-right, Y-down, Z-fwd) <-> Three.js (X-right, Y-up, Z-back)
_S_CV_TO_THREE: np.ndarray = np.diag([1.0, -1.0, -1.0]).astype(np.float64)

_SIM3_SCALE_MIN: float = 0.65
_SIM3_SCALE_MAX: float = 1.55

_VIEW_PALETTE_COLORS: list[str] = [
    "#1D4ED8",  # Cobalt (View 0 anchor)
    "#0F766E",  # Teal
    "#B45309",  # Amber
    "#BE123C",  # Crimson
    "#475569",  # Slate
    "#0369A1",  # Deep cyan
]


def _ensure_proper_rotation(R: np.ndarray) -> np.ndarray:
    """Project near-orthogonal 3x3 matrix strictly onto SO(3) with det(R) = +1."""
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    det = float(np.linalg.det(U @ Vt))
    S = np.diag([1.0, 1.0, -1.0 if det < 0.0 else 1.0])
    return U @ S @ Vt


def umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    estimate_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form Umeyama Sim(3) alignment: dst ~= s * R @ src + t with det(R) == +1."""
    P = np.asarray(src, dtype=np.float64)
    Q = np.asarray(dst, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0

    mu_p = np.mean(P, axis=0)
    mu_q = np.mean(Q, axis=0)
    Pc = P - mu_p
    Qc = Q - mu_q

    var_p = float(np.mean(np.sum(Pc * Pc, axis=1)))
    if var_p <= 1e-12:
        return np.eye(3, dtype=np.float64), mu_q - mu_p, 1.0

    cov = (Qc.T @ Pc) / float(P.shape[0])
    U, D, Vt = np.linalg.svd(cov)
    det = float(np.linalg.det(U @ Vt))
    S_det = np.diag([1.0, 1.0, -1.0 if det < 0.0 else 1.0])
    R = U @ S_det @ Vt

    if estimate_scale:
        raw_scale = float(np.sum(D * np.diag(S_det)) / max(var_p, 1e-12))
        if not math.isfinite(raw_scale) or raw_scale <= 0.0:
            scale = 1.0
        else:
            scale = float(np.clip(raw_scale, _SIM3_SCALE_MIN, _SIM3_SCALE_MAX))
    else:
        scale = 1.0

    t = mu_q - scale * (R @ mu_p)
    return R, t, scale


def _compute_normals_cv(points_cv: np.ndarray, normals_cv: Optional[np.ndarray]) -> np.ndarray:
    """Normalize provided OpenCV normals or compute camera-facing finite-difference normals."""
    h, w = points_cv.shape[:2]
    if (
        normals_cv is not None
        and normals_cv.ndim == 3
        and normals_cv.shape[:2] == (h, w)
        and normals_cv.shape[2] == 3
    ):
        n_cv = np.nan_to_num(normals_cv.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(n_cv, axis=-1, keepdims=True)
        valid_n = norms[..., 0] > 1e-6
        unit = np.divide(n_cv, np.maximum(norms, 1e-6))
        return np.where(valid_n[..., None], unit, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    if h >= 3 and w >= 3:
        dp_du = points_cv[1:-1, 2:] - points_cv[1:-1, :-2]
        dp_dv = points_cv[2:, 1:-1] - points_cv[:-2, 1:-1]
        raw_n = np.cross(dp_dv, dp_du).astype(np.float32)
        view_dot = np.sum(raw_n * (-points_cv[1:-1, 1:-1]), axis=-1, keepdims=True)
        raw_n = np.where(view_dot < 0.0, -raw_n, raw_n)
        padded = np.pad(raw_n, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(padded, axis=-1, keepdims=True)
        valid_n = norms[..., 0] > 1e-7
        unit = np.divide(padded, np.maximum(norms, 1e-7))
        return np.where(valid_n[..., None], unit, np.array([0.0, 0.0, -1.0], dtype=np.float32))

    fallback = np.zeros((h, w, 3), dtype=np.float32)
    fallback[..., 2] = -1.0
    return fallback


def _extract_valid_cloud_cv(frame: dict[str, Any], max_samples: int = 3200) -> tuple[np.ndarray, np.ndarray]:
    """Sample valid OpenCV 3D points and unit normals from a frame dictionary."""
    pts_cv = np.asarray(frame["points_cv"], dtype=np.float32)
    valid = np.asarray(frame["valid_mask"], dtype=bool)
    nrm_cv = _compute_normals_cv(pts_cv, frame.get("normals_cv"))

    fg = frame.get("foreground_mask")
    z = pts_cv[..., 2]
    mask = valid & np.isfinite(pts_cv).all(axis=-1) & (z > 0.04) & (z < 35.0)
    if fg is not None and fg.shape == mask.shape and int(np.sum(mask & fg)) >= 64:
        mask = mask & np.asarray(fg, dtype=bool)

    pts = pts_cv[mask].astype(np.float64)
    nrm = nrm_cv[mask].astype(np.float64)
    if pts.shape[0] > max_samples:
        idx = np.linspace(0, pts.shape[0] - 1, max_samples, dtype=np.int64)
        pts = pts[idx]
        nrm = nrm[idx]
    return pts, nrm


def _lookup_3d_median(
    points_cv: np.ndarray,
    valid_mask: np.ndarray,
    u: float,
    v: float,
    radius: int = 1,
) -> Optional[np.ndarray]:
    """Read robust local 3x3 median OpenCV point at pixel (u, v) to avoid edge flying pixels."""
    h, w = points_cv.shape[:2]
    cu = int(round(u))
    cv_y = int(round(v))
    if cu < 0 or cu >= w or cv_y < 0 or cv_y >= h:
        return None
    u0, u1 = max(0, cu - radius), min(w, cu + radius + 1)
    v0, v1 = max(0, cv_y - radius), min(h, cv_y + radius + 1)
    patch_p = points_cv[v0:v1, u0:u1]
    patch_m = (
        valid_mask[v0:v1, u0:u1]
        & np.isfinite(patch_p).all(axis=-1)
        & (patch_p[..., 2] > 0.04)
        & (patch_p[..., 2] < 35.0)
    )
    if not np.any(patch_m):
        return None
    pts = patch_p[patch_m]
    return np.median(pts, axis=0).astype(np.float64)


def _match_sift_3d_correspondences(
    src_frame: dict[str, Any],
    dst_frame: dict[str, Any],
    *,
    return_weights: bool = False,
) -> Any:
    """Find 3D-3D correspondences using shared ArUco corners plus SIFT Lowe-ratio matches."""
    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    w_list: list[float] = []

    src_pts_map = np.asarray(src_frame["points_cv"], dtype=np.float32)
    dst_pts_map = np.asarray(dst_frame["points_cv"], dtype=np.float32)
    src_valid = np.asarray(src_frame["valid_mask"], dtype=bool)
    dst_valid = np.asarray(dst_frame["valid_mask"], dtype=bool)

    # 1. Exact fiducial correspondence if the same ArUco marker ID is visible in both views
    aruco_a = src_frame.get("aruco_anchor")
    aruco_b = dst_frame.get("aruco_anchor")
    if (
        isinstance(aruco_a, dict)
        and isinstance(aruco_b, dict)
        and aruco_a.get("detected")
        and aruco_b.get("detected")
        and aruco_a.get("marker_id") == aruco_b.get("marker_id")
    ):
        corners_a = aruco_a.get("corners_px") or []
        corners_b = aruco_b.get("corners_px") or []
        if len(corners_a) >= 4 and len(corners_b) >= 4:
            for ca, cb in zip(corners_a[:4], corners_b[:4]):
                pa = _lookup_3d_median(src_pts_map, src_valid, float(ca[0]), float(ca[1]), radius=2)
                pb = _lookup_3d_median(dst_pts_map, dst_valid, float(cb[0]), float(cb[1]), radius=2)
                if pa is not None and pb is not None:
                    if return_weights:
                        src_list.append(pa)
                        dst_list.append(pb)
                        w_list.append(2.5)
                    else:
                        # Weight fiducial corners mildly higher for RANSAC stability
                        src_list.extend([pa, pa])
                        dst_list.extend([pb, pb])

    # 2. SIFT + BFMatcher knn Lowe ratio test lifted to metric 3D
    if cv2 is not None and hasattr(cv2, "SIFT_create"):
        gray_a = cv2.cvtColor(np.asarray(src_frame["rgb_u8"], dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(np.asarray(dst_frame["rgb_u8"], dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=1400)
        kp_a, des_a = sift.detectAndCompute(gray_a, None)
        kp_b, des_b = sift.detectAndCompute(gray_b, None)
        if des_a is not None and des_b is not None and len(kp_a) >= 6 and len(kp_b) >= 6:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            knn = matcher.knnMatch(des_a, des_b, k=2)
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair[0], pair[1]
                ratio = float(m.distance) / max(float(n.distance), 1e-6)
                if ratio >= 0.76:
                    continue
                ua, va = kp_a[m.queryIdx].pt
                ub, vb = kp_b[m.trainIdx].pt
                pa = _lookup_3d_median(src_pts_map, src_valid, ua, va, radius=1)
                pb = _lookup_3d_median(dst_pts_map, dst_valid, ub, vb, radius=1)
                if pa is not None and pb is not None:
                    src_list.append(pa)
                    dst_list.append(pb)
                    w_list.append(float(np.clip(0.5 + (1.0 - ratio), 0.55, 1.5)))

    if not src_list:
        empty_pts = np.zeros((0, 3), dtype=np.float64)
        empty_dst = np.zeros((0, 3), dtype=np.float64)
        if return_weights:
            return empty_pts, empty_dst, np.zeros(0, dtype=np.float64)
        return empty_pts, empty_dst

    src_arr = np.asarray(src_list, dtype=np.float64)
    dst_arr = np.asarray(dst_list, dtype=np.float64)
    if return_weights:
        return src_arr, dst_arr, np.asarray(w_list, dtype=np.float64)
    return src_arr, dst_arr


def ransac_sim3(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    *,
    allow_scale_alignment: bool = True,
    inlier_thresh_m: float = 0.035,
    n_iters: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Deterministic RANSAC over 3-point triplets estimating Sim(3) transform dst ~= s*R@src + t."""
    n = int(src_pts.shape[0])
    if n < 3:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0, 0

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_err = 1e9

    max_trials = min(n_iters, max(24, n * (n - 1) // 2))
    for _ in range(max_trials):
        idx = rng.choice(n, size=3, replace=False)
        p_tri = src_pts[idx]
        q_tri = dst_pts[idx]

        # Reject near-collinear source triplets
        area_p = float(np.linalg.norm(np.cross(p_tri[1] - p_tri[0], p_tri[2] - p_tri[0])))
        area_q = float(np.linalg.norm(np.cross(q_tri[1] - q_tri[0], q_tri[2] - q_tri[0])))
        if area_p < 1e-5 or area_q < 1e-5:
            continue

        # Pre-check raw scale ratio before clamping
        raw_span_p = float(np.sum(np.linalg.norm(p_tri - np.mean(p_tri, axis=0), axis=1)))
        raw_span_q = float(np.sum(np.linalg.norm(q_tri - np.mean(q_tri, axis=0), axis=1)))
        if raw_span_p < 1e-4:
            continue
        raw_ratio = raw_span_q / raw_span_p
        if raw_ratio < 0.5 or raw_ratio > 2.0:
            continue

        R_cand, t_cand, s_cand = umeyama_sim3(p_tri, q_tri, estimate_scale=allow_scale_alignment)
        transformed = (s_cand * (src_pts @ R_cand.T)) + t_cand
        res = np.linalg.norm(dst_pts - transformed, axis=1)
        inl = res <= inlier_thresh_m
        cnt = int(np.sum(inl))
        if cnt < 3:
            continue
        mean_err = float(np.mean(res[inl]))
        if cnt > best_count or (cnt == best_count and mean_err < best_err):
            best_count = cnt
            best_inliers = inl
            best_err = mean_err

    if best_count < 3:
        R_all, t_all, s_all = umeyama_sim3(src_pts, dst_pts, estimate_scale=allow_scale_alignment)
        return R_all, t_all, s_all, 0

    R_ref, t_ref, s_ref = umeyama_sim3(
        src_pts[best_inliers],
        dst_pts[best_inliers],
        estimate_scale=allow_scale_alignment,
    )
    transformed = (s_ref * (src_pts @ R_ref.T)) + t_ref
    final_cnt = int(np.sum(np.linalg.norm(dst_pts - transformed, axis=1) <= inlier_thresh_m))
    return R_ref, t_ref, s_ref, final_cnt


def refine_point_to_plane_icp(
    src_pts_cv: np.ndarray,
    src_nrm_cv: np.ndarray,
    dst_pts_cv: np.ndarray,
    dst_nrm_cv: np.ndarray,
    init_R: np.ndarray,
    init_t: np.ndarray,
    init_s: float = 1.0,
    *,
    max_iters: int = 16,
    max_corr_dist_m: float = 0.06,
    min_normal_dot: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Refine rigid+scale pose via trimmed Huber-weighted Point-to-Plane Gauss-Newton ICP."""
    R = _ensure_proper_rotation(init_R)
    t = np.asarray(init_t, dtype=np.float64).copy()
    s = float(init_s)

    if src_pts_cv.shape[0] < 16 or dst_pts_cv.shape[0] < 16:
        return R, t, s, 0.0, 0.0

    src_p = src_pts_cv[:3000] if src_pts_cv.shape[0] > 3000 else src_pts_cv
    src_n = src_nrm_cv[:3000] if src_nrm_cv.shape[0] > 3000 else src_nrm_cv
    dst_p = dst_pts_cv[:3600] if dst_pts_cv.shape[0] > 3600 else dst_pts_cv
    dst_n = dst_nrm_cv[:3600] if dst_nrm_cv.shape[0] > 3600 else dst_nrm_cv

    tree = cKDTree(dst_p)

    def _evaluate_correspondences(R_cur: np.ndarray, t_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        p_w = (s * (src_p @ R_cur.T)) + t_cur
        n_w = src_n @ R_cur.T
        dists, nn_idx = tree.query(p_w, k=1, distance_upper_bound=max_corr_dist_m)
        in_range = np.isfinite(dists) & (nn_idx < dst_p.shape[0])
        raw_overlap = float(np.mean(in_range)) if p_w.shape[0] > 0 else 0.0
        if int(np.sum(in_range)) < 12:
            return p_w[:0], n_w[:0], dst_p[:0], dst_n[:0], np.empty(0), 1e6, raw_overlap

        sel_idx = nn_idx[in_range]
        n_q = dst_n[sel_idx]
        n_p = n_w[in_range]
        dots = np.sum(n_p * n_q, axis=1)
        valid = dots >= min_normal_dot
        if int(np.sum(valid)) < 12:
            # Fallback to nearest distance-only matches if normals disagree across sharp edges
            p_sel = p_w[in_range]
            q_sel = dst_p[sel_idx]
            n_sel = n_q
            d_sel = dists[in_range]
        else:
            p_sel = p_w[in_range][valid]
            q_sel = dst_p[sel_idx][valid]
            n_sel = n_q[valid]
            d_sel = dists[in_range][valid]

        # Trim top 15% highest distance pairs for outlier immunity
        if d_sel.shape[0] >= 20:
            cutoff = float(np.percentile(d_sel, 85.0))
            keep = d_sel <= cutoff
            p_sel, q_sel, n_sel, d_sel = p_sel[keep], q_sel[keep], n_sel[keep], d_sel[keep]

        rmse = float(np.sqrt(np.mean(d_sel * d_sel))) if d_sel.size > 0 else 1e6
        return p_sel, n_w[in_range], q_sel, n_sel, d_sel, rmse, raw_overlap

    _, _, _, _, _, cur_rmse, cur_overlap = _evaluate_correspondences(R, t)
    if cur_rmse >= 1e5:
        return R, t, s, 0.0, cur_overlap

    max_step_rad = math.radians(15.0)
    max_step_m = 0.25

    for _ in range(max_iters):
        p_sel, _, q_sel, n_sel, _, rmse_before, overlap_before = _evaluate_correspondences(R, t)
        if p_sel.shape[0] < 12:
            break

        # Point-to-plane residual: (q_i - p_i) . n_i
        r = np.sum((q_sel - p_sel) * n_sel, axis=1)
        huber_k = max(0.008, 1.5 * float(np.median(np.abs(r))))
        abs_r = np.abs(r)
        w = np.where(abs_r <= huber_k, 1.0, huber_k / np.maximum(abs_r, 1e-9))

        cross_pn = np.cross(p_sel, n_sel)
        J = np.hstack([cross_pn, n_sel])
        Jw = J * w[:, None]
        H = (Jw.T @ J) + 1e-4 * np.eye(6, dtype=np.float64)
        g = Jw.T @ r

        try:
            xi = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        omega = xi[:3]
        delta_t = xi[3:]
        om_norm = float(np.linalg.norm(omega))
        dt_norm = float(np.linalg.norm(delta_t))
        if om_norm < 1e-5 and dt_norm < 1e-5:
            break
        if om_norm > max_step_rad:
            omega = omega * (max_step_rad / om_norm)
        if dt_norm > max_step_m:
            delta_t = delta_t * (max_step_m / dt_norm)

        dR = Rotation.from_rotvec(omega).as_matrix()
        R_cand = _ensure_proper_rotation(dR @ R)
        t_cand = (dR @ t) + delta_t

        _, _, _, _, _, rmse_after, overlap_after = _evaluate_correspondences(R_cand, t_cand)
        if rmse_after <= rmse_before + 1e-5:
            R = R_cand
            t = t_cand
            cur_rmse = rmse_after
            cur_overlap = overlap_after
            if om_norm < 5e-4 and dt_norm < 5e-4 and abs(rmse_before - rmse_after) < 1e-5:
                break
        else:
            break

    return R, t, s, round(float(cur_rmse), 5), round(float(cur_overlap), 4)


def _estimate_world_up_cv(points_cv: np.ndarray, normals_cv: np.ndarray) -> np.ndarray:
    """Estimate dominant world-up axis in OpenCV coordinates from top-facing surface normals."""
    if normals_cv.shape[0] >= 16:
        # Upward-facing surfaces in OpenCV have negative Y normal component
        up_mask = normals_cv[:, 1] < -0.55
        if int(np.sum(up_mask)) >= 12:
            mean_up = np.mean(normals_cv[up_mask], axis=0)
            n = float(np.linalg.norm(mean_up))
            if n > 1e-4:
                return mean_up / n
    return np.array([0.0, -1.0, 0.0], dtype=np.float64)


def _estimate_interior_pivot_cv(pts_cv: np.ndarray) -> np.ndarray:
    """Estimate volumetric interior center in OpenCV coordinates from a single partial surface cloud."""
    if pts_cv.shape[0] < 8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    p_lo = np.percentile(pts_cv, 3.0, axis=0)
    p_hi = np.percentile(pts_cv, 97.0, axis=0)
    mid = 0.5 * (p_lo + p_hi)
    ray = mid / max(float(np.linalg.norm(mid)), 1e-6)
    span_xy = float(max(p_hi[0] - p_lo[0], p_hi[1] - p_lo[1]))
    span_z = float(p_hi[2] - p_lo[2])
    # Single surface views under-observe back faces: shift behind front face by half typical depth thickness
    back_shift = max(0.0, 0.36 * span_xy - 0.35 * span_z)
    return mid + back_shift * ray


def _orbit_icp_fallback_register(
    src_pts_cv: np.ndarray,
    src_nrm_cv: np.ndarray,
    world_pts_cv: np.ndarray,
    world_nrm_cv: np.ndarray,
    pivot_cv: np.ndarray,
    *,
    up_axis_cv: Optional[np.ndarray] = None,
    yaw_hint_deg: Optional[float] = None,
    allow_scale_alignment: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Coarse-to-fine azimuth orbit search around gravity axis + interior pivot followed by Point-to-Plane ICP (+ Sim(3))."""
    if src_pts_cv.shape[0] < 16 or world_pts_cv.shape[0] < 16:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0, 0.0, 0.0

    up_ax = (
        up_axis_cv / max(float(np.linalg.norm(up_axis_cv)), 1e-8)
        if up_axis_cv is not None and float(np.linalg.norm(up_axis_cv)) > 1e-4
        else np.array([0.0, -1.0, 0.0], dtype=np.float64)
    )
    src_pivot = _estimate_interior_pivot_cv(src_pts_cv)
    if yaw_hint_deg is not None and math.isfinite(float(yaw_hint_deg)):
        base = float(yaw_hint_deg)
        candidates = [base, base - 8.0, base + 8.0]
    else:
        candidates = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0]

    src_sub = src_pts_cv[:900] if src_pts_cv.shape[0] > 900 else src_pts_cv
    world_tree = cKDTree(world_pts_cv[:4500] if world_pts_cv.shape[0] > 4500 else world_pts_cv)

    best_R = np.eye(3, dtype=np.float64)
    best_t = pivot_cv - src_pivot
    best_score = 1e12

    for deg in candidates:
        # Rotating object points by R(+deg * up_ax) aligns View k camera positioned at +deg azimuth back to View 0
        R_cand = Rotation.from_rotvec(math.radians(deg) * up_ax).as_matrix()
        t_cand = pivot_cv - (R_cand @ src_pivot)
        transformed = (src_sub @ R_cand.T) + t_cand
        dists, _ = world_tree.query(transformed, k=1)
        score = float(np.mean(np.minimum(dists, 0.14) ** 2))
        if score < best_score:
            best_score = score
            best_R = R_cand
            best_t = t_cand

    R_icp, t_icp, s_icp, rmse_icp, overlap_icp = refine_point_to_plane_icp(
        src_pts_cv,
        src_nrm_cv,
        world_pts_cv,
        world_nrm_cv,
        best_R,
        best_t,
        1.0,
        max_iters=20,
        max_corr_dist_m=0.12,
    )
    R_icp, t_icp, s_icp, rmse_icp, overlap_icp = refine_point_to_plane_icp(
        src_pts_cv,
        src_nrm_cv,
        world_pts_cv,
        world_nrm_cv,
        R_icp,
        t_icp,
        s_icp,
        max_iters=20,
        max_corr_dist_m=0.055,
    )

    if allow_scale_alignment and src_pts_cv.shape[0] >= 32 and world_pts_cv.shape[0] >= 32:
        src_eval = src_pts_cv[:2000]
        p_w = (s_icp * (src_eval @ R_icp.T)) + t_icp
        dists, nn_idx = world_tree.query(p_w, k=1, distance_upper_bound=0.06)
        in_ok = np.isfinite(dists) & (nn_idx < world_tree.n)
        if int(np.sum(in_ok)) >= 24:
            dst_matched = world_tree.data[nn_idx[in_ok]]
            src_matched = src_eval[in_ok]
            R_sim, t_sim, s_sim = umeyama_sim3(src_matched, dst_matched, estimate_scale=True)
            R_s, t_s, s_s, rmse_s, overlap_s = refine_point_to_plane_icp(
                src_pts_cv,
                src_nrm_cv,
                world_pts_cv,
                world_nrm_cv,
                R_sim,
                t_sim,
                s_sim,
                max_iters=18,
                max_corr_dist_m=0.055,
            )
            if rmse_s <= rmse_icp + 2e-4 and overlap_s >= overlap_icp - 0.04:
                return R_s, t_s, s_s, rmse_s, overlap_s

    return R_icp, t_icp, s_icp, rmse_icp, overlap_icp


def _compute_pose_three_vectors(
    R_cv: np.ndarray,
    t_cv: np.ndarray,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Map OpenCV camera-to-world (R_cv, t_cv) to Three.js center, quaternion, forward, up."""
    R_three = _ensure_proper_rotation(_S_CV_TO_THREE @ R_cv @ _S_CV_TO_THREE)
    center_three = (_S_CV_TO_THREE @ t_cv).astype(np.float64)
    quat = Rotation.from_matrix(R_three).as_quat()  # [qx, qy, qz, qw]
    fwd_three = R_three @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    up_three = R_three @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return (
        [round(float(v), 4) for v in center_three],
        [round(float(v), 5) for v in quat],
        [round(float(v), 4) for v in fwd_three],
        [round(float(v), 4) for v in up_three],
    )


def _build_local_mesh_three(
    points_cv: np.ndarray,
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray,
    normals_cv: np.ndarray,
    R_three: np.ndarray,
    t_three: np.ndarray,
    scale: float,
    *,
    target_width: int,
    max_rel_depth_jump: float = 0.06,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Triangulate one view in Three.js world space while cutting flying-pixel depth jumps."""
    h, w = points_cv.shape[:2]
    if h < 4 or w < 4:
        return None
    step = max(1, int(round(float(w) / float(max(target_width, 16)))))
    pts_sub = points_cv[::step, ::step]
    rgb_sub = rgb_uint8[::step, ::step]
    val_sub = valid_mask[::step, ::step]
    nrm_sub = normals_cv[::step, ::step]

    gh, gw = pts_sub.shape[:2]
    if gh < 2 or gw < 2:
        return None

    z_sub = pts_sub[..., 2]
    v_ok = (
        val_sub
        & np.isfinite(pts_sub).all(axis=-1)
        & np.isfinite(z_sub)
        & (z_sub > 0.04)
        & (z_sub < 35.0)
    )

    grid_idx = np.arange(gh * gw, dtype=np.uint32).reshape(gh, gw)
    tl = grid_idx[:-1, :-1].ravel()
    tr = grid_idx[:-1, 1:].ravel()
    bl = grid_idx[1:, :-1].ravel()
    br = grid_idx[1:, 1:].ravel()
    all_tris = np.concatenate([np.stack([tl, bl, tr], axis=1), np.stack([tr, bl, br], axis=1)], axis=0)

    v_flat = v_ok.ravel()
    z_flat = z_sub.ravel()
    safe_z = np.where(v_flat & (z_flat > 0.01), z_flat, 1.0)
    log_z = np.log(safe_z)

    i0, i1, i2 = all_tris[:, 0], all_tris[:, 1], all_tris[:, 2]
    ok_tri = (
        v_flat[i0]
        & v_flat[i1]
        & v_flat[i2]
        & (np.abs(log_z[i0] - log_z[i1]) < max_rel_depth_jump)
        & (np.abs(log_z[i1] - log_z[i2]) < max_rel_depth_jump)
        & (np.abs(log_z[i0] - log_z[i2]) < max_rel_depth_jump)
    )
    kept_tris = all_tris[ok_tri]
    if kept_tris.shape[0] == 0:
        return None

    used_verts, inverse = np.unique(kept_tris, return_inverse=True)
    compact_tris = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.uint32)

    pts_local_three = (pts_sub.reshape(-1, 3)[used_verts] @ _S_CV_TO_THREE.T).astype(np.float64)
    nrm_local_three = (nrm_sub.reshape(-1, 3)[used_verts] @ _S_CV_TO_THREE.T).astype(np.float64)

    pos_world = ((scale * (pts_local_three @ R_three.T)) + t_three).astype(np.float32)
    nrm_world = (nrm_local_three @ R_three.T).astype(np.float32)
    n_len = np.maximum(np.linalg.norm(nrm_world, axis=-1, keepdims=True), 1e-6)
    nrm_world = nrm_world / n_len
    nrm_i8 = np.ascontiguousarray(np.clip(np.round(nrm_world * 127.0), -127, 127).astype(np.int8))
    col_u8 = np.ascontiguousarray(rgb_sub.reshape(-1, 3)[used_verts], dtype=np.uint8)
    return pos_world, col_u8, nrm_i8, compact_tris


def compute_closed_3d_body_metrology(
    body_pts_three: np.ndarray,
    *,
    voxel_size_m: float,
    occupied_voxels: int,
    mean_icp_rmse_m: float,
    single_view_obb_volume_m3: Optional[float] = None,
    gt_volume_m3: Optional[float] = None,
    up_axis_three: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute ConvexHull 360-degree volume, tomographic slice solid volume, surface area, and GT error."""
    if body_pts_three.shape[0] < 8:
        return {
            "convex_hull_volume_m3": 0.0,
            "convex_hull_surface_area_m2": 0.0,
            "voxel_solid_volume_m3": 0.0,
            "voxel_shell_volume_m3": 0.0,
            "voxel_resolution_mm": round(voxel_size_m * 1000.0, 2),
            "occupied_voxels": int(occupied_voxels),
            "bounding_dimensions_m": [0.05, 0.05, 0.05],
            "fused_center_gemini": [0.0, 1.0, 0.0],
            "fused_box_3d": [0.0, 1.0, 0.0, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0],
            "mean_icp_rmse_mm": round(mean_icp_rmse_m * 1000.0, 2),
            "single_view_obb_volume_m3": single_view_obb_volume_m3,
            "gt_volume_m3": gt_volume_m3,
            "gt_volume_error_pct": None,
        }

    # Trim extreme 1.5% radial outliers before hull volume calculation
    med = np.median(body_pts_three, axis=0)
    radii = np.linalg.norm(body_pts_three - med, axis=1)
    r_cut = float(np.percentile(radii, 98.5))
    core = body_pts_three[radii <= max(r_cut, 1e-4)]
    if core.shape[0] < 8:
        core = body_pts_three

    # Subsample deterministically for fast exact ConvexHull
    hull_pts = core[np.linspace(0, core.shape[0] - 1, 6000, dtype=np.int64)] if core.shape[0] > 6000 else core

    hull_vol = 0.0
    hull_area = 0.0
    try:
        hull = ConvexHull(hull_pts, qhull_options="QJ")
        hull_vol = float(hull.volume)
        hull_area = float(hull.area)
    except Exception:
        hull_vol = 0.0
        hull_area = 0.0

    # Upright tomographic cross-section integration along up_axis_three (captures stepped & concave shapes)
    min_b = np.percentile(core, 1.0, axis=0)
    max_b = np.percentile(core, 99.0, axis=0)
    ext_three = np.maximum(max_b - min_b, 0.01)
    center_three = 0.5 * (min_b + max_b)

    u = (
        np.asarray(up_axis_three, dtype=np.float64)
        if up_axis_three is not None
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    u = u / max(float(np.linalg.norm(u)), 1e-8)
    ref_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref_x, u))) > 0.9:
        ref_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e1 = np.cross(u, ref_x)
    e1 /= max(float(np.linalg.norm(e1)), 1e-8)
    e2 = np.cross(u, e1)

    h_coords = core @ u
    xz_coords = np.column_stack([core @ e1, core @ e2])
    h_lo, h_hi = float(np.percentile(h_coords, 1.0)), float(np.percentile(h_coords, 99.0))
    n_slices = 20
    voxel_solid_vol = 0.0
    if h_hi - h_lo > 0.01:
        dh = (h_hi - h_lo) / float(n_slices)
        for s_idx in range(n_slices):
            s0 = h_lo + s_idx * dh
            s1 = s0 + dh
            in_sl = (h_coords >= s0) & (h_coords <= s1)
            if int(np.sum(in_sl)) < 6:
                continue
            pts_2d = xz_coords[in_sl]
            if pts_2d.shape[0] > 400:
                pts_2d = pts_2d[np.linspace(0, pts_2d.shape[0] - 1, 400, dtype=np.int64)]
            try:
                h2 = ConvexHull(pts_2d, qhull_options="QJ")
                voxel_solid_vol += float(h2.volume) * dh
            except Exception:
                span_2d = np.max(pts_2d, axis=0) - np.min(pts_2d, axis=0)
                voxel_solid_vol += float(max(span_2d[0], 0.0) * max(span_2d[1], 0.0)) * dh
    if voxel_solid_vol <= 0.0:
        voxel_solid_vol = hull_vol

    # Convert Three.js center/extents -> Gemini (X-right, Y-forward = -Z_three, Z-up = Y_three)
    cx_gem = float(center_three[0])
    cy_gem = max(0.05, float(-center_three[2]))
    cz_gem = float(center_three[1])
    dx_gem = max(0.01, float(ext_three[0]))
    dy_gem = max(0.01, float(ext_three[2]))
    dz_gem = max(0.01, float(ext_three[1]))

    vox_surf_vol = float(occupied_voxels) * (voxel_size_m ** 3)
    ref_meas_vol = voxel_solid_vol if voxel_solid_vol > 0.0 else hull_vol
    gt_err_pct: Optional[float] = None
    if gt_volume_m3 is not None and gt_volume_m3 > 1e-7 and ref_meas_vol > 0.0:
        gt_err_pct = round(abs(ref_meas_vol - float(gt_volume_m3)) / float(gt_volume_m3) * 100.0, 2)

    return {
        "convex_hull_volume_m3": round(hull_vol, 6),
        "convex_hull_surface_area_m2": round(hull_area, 5),
        "voxel_solid_volume_m3": round(voxel_solid_vol, 6),
        "voxel_shell_volume_m3": round(vox_surf_vol, 6),
        "voxel_resolution_mm": round(voxel_size_m * 1000.0, 2),
        "occupied_voxels": int(occupied_voxels),
        "bounding_dimensions_m": [round(dx_gem, 4), round(dy_gem, 4), round(dz_gem, 4)],
        "fused_center_gemini": [round(cx_gem, 4), round(cy_gem, 4), round(cz_gem, 4)],
        "fused_box_3d": [
            round(cx_gem, 4),
            round(cy_gem, 4),
            round(cz_gem, 4),
            round(dx_gem, 4),
            round(dy_gem, 4),
            round(dz_gem, 4),
            0.0,
            0.0,
            0.0,
        ],
        "mean_icp_rmse_mm": round(mean_icp_rmse_m * 1000.0, 2),
        "single_view_obb_volume_m3": round(float(single_view_obb_volume_m3), 6)
        if single_view_obb_volume_m3 is not None
        else None,
        "gt_volume_m3": round(float(gt_volume_m3), 6) if gt_volume_m3 is not None else None,
        "gt_volume_error_pct": gt_err_pct,
    }


def compute_360_coverage_and_nbv(
    camera_poses: list[dict[str, Any]],
    target_center_three: np.ndarray,
    *,
    up_axis_three: Optional[np.ndarray] = None,
    body_normals_three: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute 12-sector (30-deg) 360-degree polar coverage anchored on the horizontal object plane."""
    sector_scores = [0.0] * 12
    camera_az_list: list[float] = []

    u_raw = np.asarray(up_axis_three, dtype=np.float64) if up_axis_three is not None else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u_norm = float(np.linalg.norm(u_raw))
    u = u_raw / max(u_norm, 1e-8) if u_norm > 1e-4 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Anchor azimuth 0° to View 0 projected into plane orthogonal to u
    e0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if camera_poses:
        v0 = np.asarray(camera_poses[0]["center_three"], dtype=np.float64) - target_center_three
        if float(np.linalg.norm(v0)) < 1e-4:
            v0 = -np.asarray(camera_poses[0]["forward_three"], dtype=np.float64)
        v0_proj = v0 - float(np.dot(v0, u)) * u
        if float(np.linalg.norm(v0_proj)) > 1e-4:
            e0 = v0_proj / float(np.linalg.norm(v0_proj))
    else:
        e0_proj = e0 - float(np.dot(e0, u)) * u
        if float(np.linalg.norm(e0_proj)) > 1e-4:
            e0 = e0_proj / float(np.linalg.norm(e0_proj))

    e_right = np.cross(u, e0)
    er_norm = float(np.linalg.norm(e_right))
    e_right = e_right / max(er_norm, 1e-8)

    for pose in camera_poses:
        c = np.asarray(pose["center_three"], dtype=np.float64)
        fwd = np.asarray(pose["forward_three"], dtype=np.float64)
        vec = c - target_center_three
        vec_proj = vec - float(np.dot(vec, u)) * u
        if float(np.linalg.norm(vec_proj)) < 1e-4:
            vec = -fwd
            vec_proj = vec - float(np.dot(vec, u)) * u
        az_deg = math.degrees(math.atan2(float(np.dot(vec_proj, e_right)), float(np.dot(vec_proj, e0)))) % 360.0
        pose["azimuth_deg"] = round(az_deg, 1)
        camera_az_list.append(az_deg)

        for sec_i in range(12):
            sec_center = (sec_i * 30.0) + 15.0
            diff = abs((az_deg - sec_center + 180.0) % 360.0 - 180.0)
            if diff <= 52.0:
                score = math.cos(math.radians(min(diff, 52.0) * (90.0 / 55.0)))
                sector_scores[sec_i] = max(sector_scores[sec_i], round(float(score), 3))

    covered = [s >= 0.25 for s in sector_scores]
    coverage_pct = round((sum(1 for flag in covered if flag) / 12.0) * 100.0, 1)

    # Locate longest contiguous ring segment of uncovered sectors
    uncovered_flags = [not flag for flag in covered]
    if not any(uncovered_flags):
        largest_gap_deg = 0.0
        min_idx = int(np.argmin(sector_scores))
        rec_az_deg = round((min_idx * 30.0) + 15.0, 1)
        advice = (
            f"Cobertura 360° completa ({coverage_pct:.0f}%). Refuerzo fino opcional a "
            f"{rec_az_deg:.0f}° de azimut."
        )
    else:
        doubled = uncovered_flags + uncovered_flags
        best_len = 0
        best_start = 0
        cur_len = 0
        cur_start = 0
        for i in range(24):
            if i < 12 or cur_len > 0:
                if doubled[i]:
                    if cur_len == 0:
                        cur_start = i
                    cur_len += 1
                    if cur_len <= 12 and cur_len > best_len:
                        best_len = cur_len
                        best_start = cur_start
                else:
                    cur_len = 0
        largest_gap_deg = round(best_len * 30.0, 1)
        mid_sector = (best_start + (best_len - 1) * 0.5) % 12.0
        rec_az_deg = round(((mid_sector * 30.0) + 15.0) % 360.0, 1)
        advice = (
            f"Zona ciega de {largest_gap_deg:.0f}° detectada (cobertura {coverage_pct:.0f}%). "
            f"ActivePose NBV recomienda capturar la siguiente foto desde {rec_az_deg:.0f}° de azimut."
        )

    from app.core.quantum_qubo import plan_nbv_set_cover_qubo

    qubo_nbv_plan = plan_nbv_set_cover_qubo(
        sector_scores,
        body_normals_three=body_normals_three,
        existing_azimuths_deg=camera_az_list,
        include_hardware_export=False,
    )

    return {
        "sector_scores": sector_scores,
        "angular_coverage_pct": coverage_pct,
        "largest_blind_gap_deg": largest_gap_deg,
        "recommended_next_azimuth_deg": rec_az_deg,
        "is_watertight_360": coverage_pct >= 80.0 and len(camera_poses) >= 3,
        "next_best_view_advice_es": advice,
        "qubo_nbv_plan": qubo_nbv_plan,
    }


def fuse_multiview_frames(
    frames: list[dict[str, Any]],
    *,
    voxel_size_m: Optional[float] = None,
    max_points: int = 96000,
    allow_scale_alignment: bool = True,
    gt_volume_m3: Optional[float] = None,
    use_qubo_registration: bool = True,
) -> dict[str, Any]:
    """Register N orbital RGB-D frames into canonical world coordinates and fuse 360 surfels/mesh."""
    from app.core.quantum_qubo import plan_nbv_set_cover_qubo, qubo_mwis_sim3_registration

    if not frames:
        raise ValueError("At least 1 frame is required for multi-view fusion.")

    n_views = len(frames)
    view_clouds: list[tuple[np.ndarray, np.ndarray]] = []
    view_normals_full: list[np.ndarray] = []
    for f in frames:
        pts_c, nrm_c = _extract_valid_cloud_cv(f)
        view_clouds.append((pts_c, nrm_c))
        view_normals_full.append(_compute_normals_cv(np.asarray(f["points_cv"], dtype=np.float32), f.get("normals_cv")))

    # Reference view 0 pose
    R_list: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    t_list: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    s_list: list[float] = [1.0]
    pose_meta: list[dict[str, Any]] = [
        {
            "view_index": 0,
            "label": str(frames[0].get("label") or "Vista 1"),
            "registration_method": "anchor_view",
            "inliers": int(view_clouds[0][0].shape[0]),
            "alignment_rmse_m": 0.0,
            "overlap_ratio": 1.0,
            "estimated_scale": 1.0,
            "fov_h_deg": round(float(frames[0].get("fov_h_deg", 60.0)), 2),
            "thumbnail_base64": frames[0].get("thumbnail_base64"),
        }
    ]

    anchor_pts, anchor_nrm = view_clouds[0]
    world_pts_acc = anchor_pts.copy() if anchor_pts.shape[0] > 0 else np.zeros((0, 3), dtype=np.float64)
    world_nrm_acc = anchor_nrm.copy() if anchor_nrm.shape[0] > 0 else np.zeros((0, 3), dtype=np.float64)
    pivot_cv = _estimate_interior_pivot_cv(anchor_pts)
    up_axis_cv = _estimate_world_up_cv(anchor_pts, anchor_nrm)

    # Estimate single-view volume baseline on View 0 alone for comparison
    single_view_obb_vol: Optional[float] = None
    if anchor_pts.shape[0] >= 12:
        sv_ext = np.percentile(anchor_pts, 98, axis=0) - np.percentile(anchor_pts, 2, axis=0)
        single_view_obb_vol = float(max(sv_ext[0], 0.01) * max(sv_ext[1], 0.01) * max(sv_ext[2], 0.01))

    qubo_pair_logs: list[dict[str, Any]] = []
    views_optimized_by_qubo = 0
    best_hw_pair_args: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]] = None

    for k in range(1, n_views):
        src_pts_k, src_nrm_k = view_clouds[k]
        yaw_hint = frames[k].get("yaw_hint_deg")

        best_pair: Optional[tuple[int, np.ndarray, np.ndarray, float, int, Optional[dict[str, Any]]]] = None
        best_corr_tuple: Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = None

        for j in range(k):
            if use_qubo_registration:
                corr_s, corr_d, corr_w = _match_sift_3d_correspondences(frames[k], frames[j], return_weights=True)
            else:
                corr_s, corr_d = _match_sift_3d_correspondences(frames[k], frames[j], return_weights=False)
                corr_w = np.ones(corr_s.shape[0], dtype=np.float64)

            shared_aruco = bool(
                (frames[k].get("aruco_anchor") or {}).get("detected")
                and (frames[j].get("aruco_anchor") or {}).get("detected")
                and (frames[k].get("aruco_anchor") or {}).get("marker_id")
                == (frames[j].get("aruco_anchor") or {}).get("marker_id")
            )
            min_corr = 6 if shared_aruco else 12
            min_inl = 4 if shared_aruco else 9
            if corr_s.shape[0] >= min_corr:
                pair_seed = 42 + k * 7 + j
                if use_qubo_registration:
                    qubo_res = qubo_mwis_sim3_registration(
                        corr_s,
                        corr_d,
                        weights=corr_w,
                        allow_scale_alignment=allow_scale_alignment,
                        inlier_thresh_m=0.04,
                        seed=pair_seed,
                        include_hardware_export=False,
                    )
                    R_jk = qubo_res["R"]
                    t_jk = qubo_res["t"]
                    s_jk = float(qubo_res["scale"])
                    inl_cnt = int(qubo_res["inliers"])
                    if inl_cnt >= min_inl and (best_pair is None or inl_cnt > best_pair[4]):
                        best_pair = (j, R_jk, t_jk, s_jk, inl_cnt, qubo_res)
                        best_corr_tuple = (corr_s, corr_d, corr_w, pair_seed)
                else:
                    R_jk, t_jk, s_jk, inl_cnt = ransac_sim3(
                        corr_s,
                        corr_d,
                        allow_scale_alignment=allow_scale_alignment,
                        inlier_thresh_m=0.04,
                        seed=pair_seed,
                    )
                    if inl_cnt >= min_inl and (best_pair is None or inl_cnt > best_pair[4]):
                        best_pair = (j, R_jk, t_jk, s_jk, inl_cnt, None)

        if best_pair is not None and world_pts_acc.shape[0] >= 16 and src_pts_k.shape[0] >= 16:
            j, R_jk, t_jk, s_jk, inl_cnt, win_qubo = best_pair
            # Compose T_{0<-k} = T_{0<-j} o T_{j<-k}
            s_init = float(np.clip(s_list[j] * s_jk, _SIM3_SCALE_MIN, _SIM3_SCALE_MAX))
            R_init = _ensure_proper_rotation(R_list[j] @ R_jk)
            t_init = (s_list[j] * (R_list[j] @ t_jk)) + t_list[j]

            R_k, t_k, s_k, rmse_m, overlap = refine_point_to_plane_icp(
                src_pts_k,
                src_nrm_k,
                world_pts_acc,
                world_nrm_acc,
                R_init,
                t_init,
                s_init,
                max_corr_dist_m=0.075,
            )
            has_aruco_k = bool((frames[k].get("aruco_anchor") or {}).get("detected"))
            if win_qubo is not None and win_qubo.get("method_won") == "qubo_mwis":
                method = "qubo_mwis_aruco_sim3_icp" if has_aruco_k else "qubo_mwis_sim3_icp"
            else:
                method = "sift_aruco_sim3_icp" if has_aruco_k else "sift_sim3_icp"

            if win_qubo is not None:
                qubo_pair_logs.append(
                    {
                        "view_index": k,
                        "anchor_index": j,
                        "method_won": str(win_qubo["method_won"]),
                        "n_vars": int(win_qubo["n_vars"]),
                        "n_conflict_edges": int(win_qubo["n_conflict_edges"]),
                        "qubo_inliers": int(win_qubo["qubo_inliers"]),
                        "ransac_inliers": int(win_qubo["ransac_inliers"]),
                        "energy_gain": float(win_qubo["energy_gain"]),
                        "solver_ms": float(win_qubo["solver_ms"]),
                    }
                )
                if best_corr_tuple is not None and (
                    best_hw_pair_args is None or int(win_qubo["n_vars"]) > best_hw_pair_args[5]
                ):
                    best_hw_pair_args = (
                        best_corr_tuple[0],
                        best_corr_tuple[1],
                        best_corr_tuple[2],
                        best_corr_tuple[3],
                        k,
                        int(win_qubo["n_vars"]),
                    )
            inliers_rep = inl_cnt
            # Guard against degenerate sparse correspondences: verify dense ICP overlap & RMSE vs orbit ICP
            if rmse_m > 0.025 or overlap < 0.30:
                R_orb, t_orb, s_orb, rmse_orb, ov_orb = _orbit_icp_fallback_register(
                    src_pts_k,
                    src_nrm_k,
                    world_pts_acc,
                    world_nrm_acc,
                    pivot_cv,
                    up_axis_cv=up_axis_cv,
                    yaw_hint_deg=yaw_hint,
                    allow_scale_alignment=allow_scale_alignment,
                )
                if ov_orb > overlap + 0.05 and rmse_orb < rmse_m:
                    R_k, t_k, s_k, rmse_m, overlap = R_orb, t_orb, s_orb, rmse_orb, ov_orb
                    method = "yaw_hint_icp" if yaw_hint is not None else "orbit_icp_fallback"
                    inliers_rep = int(round(overlap * src_pts_k.shape[0]))
            if method.startswith("qubo_mwis"):
                views_optimized_by_qubo += 1
        else:
            R_k, t_k, s_k, rmse_m, overlap = _orbit_icp_fallback_register(
                src_pts_k,
                src_nrm_k,
                world_pts_acc,
                world_nrm_acc,
                pivot_cv,
                up_axis_cv=up_axis_cv,
                yaw_hint_deg=yaw_hint,
                allow_scale_alignment=allow_scale_alignment,
            )
            method = "yaw_hint_icp" if yaw_hint is not None else "orbit_icp_fallback"
            inliers_rep = int(round(overlap * src_pts_k.shape[0]))

        R_list.append(R_k)
        t_list.append(t_k)
        s_list.append(s_k)

        pose_meta.append(
            {
                "view_index": k,
                "label": str(frames[k].get("label") or f"Vista {k + 1}"),
                "registration_method": method,
                "inliers": int(inliers_rep),
                "alignment_rmse_m": round(float(rmse_m), 5),
                "overlap_ratio": round(float(overlap), 4),
                "estimated_scale": round(float(s_k), 4),
                "fov_h_deg": round(float(frames[k].get("fov_h_deg", 60.0)), 2),
                "thumbnail_base64": frames[k].get("thumbnail_base64"),
            }
        )

        if src_pts_k.shape[0] > 0:
            w_p = (s_k * (src_pts_k @ R_k.T)) + t_k
            w_n = src_nrm_k @ R_k.T
            world_pts_acc = np.vstack([world_pts_acc, w_p])
            world_nrm_acc = np.vstack([world_nrm_acc, w_n])
            if world_pts_acc.shape[0] > 6000:
                idx_acc = np.linspace(0, world_pts_acc.shape[0] - 1, 6000, dtype=np.int64)
                world_pts_acc = world_pts_acc[idx_acc]
                world_nrm_acc = world_nrm_acc[idx_acc]

    # Collect per-view full Three.js surfels + local meshes
    all_pts_three: list[np.ndarray] = []
    all_rgb_u8: list[np.ndarray] = []
    all_nrm_three: list[np.ndarray] = []
    all_view_ids: list[np.ndarray] = []
    all_body_flags: list[np.ndarray] = []

    mesh_pos_chunks: list[np.ndarray] = []
    mesh_col_chunks: list[np.ndarray] = []
    mesh_nrm_chunks: list[np.ndarray] = []
    mesh_tri_chunks: list[np.ndarray] = []
    mesh_vert_offset = 0
    mesh_target_w = max(72, int(round(144.0 / math.sqrt(max(n_views, 1)))))

    camera_poses: list[dict[str, Any]] = []

    for k, frame in enumerate(frames):
        R_cv = R_list[k]
        t_cv = t_list[k]
        s_k = s_list[k]

        center_three, quat_three, fwd_three, up_three = _compute_pose_three_vectors(R_cv, t_cv)
        R_three = _ensure_proper_rotation(_S_CV_TO_THREE @ R_cv @ _S_CV_TO_THREE)
        t_three = (_S_CV_TO_THREE @ t_cv).astype(np.float64)

        pose_entry = {
            **pose_meta[k],
            "center_three": center_three,
            "quaternion_three": quat_three,
            "forward_three": fwd_three,
            "up_three": up_three,
            "azimuth_deg": 0.0,
            "color": _VIEW_PALETTE_COLORS[k % len(_VIEW_PALETTE_COLORS)],
        }
        camera_poses.append(pose_entry)

        pts_cv = np.asarray(frame["points_cv"], dtype=np.float32)
        rgb_u8 = np.asarray(frame["rgb_u8"], dtype=np.uint8)
        valid_mask = np.asarray(frame["valid_mask"], dtype=bool)
        nrm_cv = view_normals_full[k]

        # Build aligned local surface mesh piece
        m_piece = _build_local_mesh_three(
            pts_cv,
            rgb_u8,
            valid_mask,
            nrm_cv,
            R_three,
            t_three,
            s_k,
            target_width=mesh_target_w,
        )
        if m_piece is not None:
            m_pos, m_col, m_nrm, m_tri = m_piece
            mesh_pos_chunks.append(m_pos)
            mesh_col_chunks.append(m_col)
            mesh_nrm_chunks.append(m_nrm)
            mesh_tri_chunks.append(m_tri + np.uint32(mesh_vert_offset))
            mesh_vert_offset += int(m_pos.shape[0])

        z_cv = pts_cv[..., 2]
        keep = valid_mask & np.isfinite(pts_cv).all(axis=-1) & np.isfinite(z_cv) & (z_cv > 0.04) & (z_cv < 35.0)
        fg_raw = frame.get("foreground_mask")
        fg_mask = np.asarray(fg_raw, dtype=bool) if fg_raw is not None else keep

        pts_k = pts_cv[keep]
        nrm_k = nrm_cv[keep]
        rgb_k = rgb_u8[keep]
        fg_k = fg_mask[keep]

        # Cap raw contribution per frame cleanly so voxel hash stays fast (~75k/frame)
        max_per_frame = 75000
        if pts_k.shape[0] > max_per_frame:
            sel = np.linspace(0, pts_k.shape[0] - 1, max_per_frame, dtype=np.int64)
            pts_k, nrm_k, rgb_k, fg_k = pts_k[sel], nrm_k[sel], rgb_k[sel], fg_k[sel]

        pts_k_three = pts_k.astype(np.float64) @ _S_CV_TO_THREE.T
        nrm_k_three = nrm_k.astype(np.float64) @ _S_CV_TO_THREE.T
        w_pts_three = ((s_k * (pts_k_three @ R_three.T)) + t_three).astype(np.float32)
        w_nrm_three = (nrm_k_three @ R_three.T).astype(np.float32)

        all_pts_three.append(w_pts_three)
        all_rgb_u8.append(rgb_k)
        all_nrm_three.append(w_nrm_three)
        all_view_ids.append(np.full(w_pts_three.shape[0], k + 1, dtype=np.uint8))
        all_body_flags.append(fg_k)

    pts_cat = np.concatenate(all_pts_three, axis=0) if all_pts_three else np.zeros((0, 3), dtype=np.float32)
    rgb_cat = np.concatenate(all_rgb_u8, axis=0) if all_rgb_u8 else np.zeros((0, 3), dtype=np.uint8)
    nrm_cat = np.concatenate(all_nrm_three, axis=0) if all_nrm_three else np.zeros((0, 3), dtype=np.float32)
    vid_cat = np.concatenate(all_view_ids, axis=0) if all_view_ids else np.zeros(0, dtype=np.uint8)
    fg_cat = np.concatenate(all_body_flags, axis=0) if all_body_flags else np.zeros(0, dtype=bool)

    if pts_cat.shape[0] == 0:
        raise ValueError("No valid 3D points extracted across input views.")

    median_depth = float(np.median(np.linalg.norm(pts_cat, axis=1)))
    vox_m = float(voxel_size_m) if voxel_size_m is not None else float(np.clip(0.0045 * median_depth, 0.003, 0.018))
    vox_m = max(0.002, vox_m)

    # 64-bit integer voxel hash average fusion
    q_coords = np.floor(pts_cat / vox_m).astype(np.int64)
    q_min = np.min(q_coords, axis=0)
    q_off = np.clip(q_coords - q_min, 0, (1 << 20) - 1)
    keys = (q_off[:, 0] << 40) | (q_off[:, 1] << 20) | q_off[:, 2]

    _, first_idx, inv_idx = np.unique(keys, return_index=True, return_inverse=True)
    n_vox = int(first_idx.shape[0])
    counts = np.bincount(inv_idx, minlength=n_vox).astype(np.float64)

    fused_pts = np.column_stack(
        [np.bincount(inv_idx, weights=pts_cat[:, ax], minlength=n_vox) / counts for ax in range(3)]
    ).astype(np.float32)
    fused_rgb = np.column_stack(
        [
            np.clip(np.round(np.bincount(inv_idx, weights=rgb_cat[:, ax].astype(np.float64), minlength=n_vox) / counts), 0, 255)
            for ax in range(3)
        ]
    ).astype(np.uint8)
    fused_nrm_raw = np.column_stack(
        [np.bincount(inv_idx, weights=nrm_cat[:, ax], minlength=n_vox) / counts for ax in range(3)]
    ).astype(np.float32)
    nrm_len = np.maximum(np.linalg.norm(fused_nrm_raw, axis=-1, keepdims=True), 1e-6)
    fused_nrm = fused_nrm_raw / nrm_len
    fused_vid = vid_cat[first_idx]
    fused_fg = (np.bincount(inv_idx, weights=fg_cat.astype(np.float64), minlength=n_vox) / counts) >= 0.5

    # Subsample fused surfels cleanly if above max_points budget (prioritizing foreground body points)
    if fused_pts.shape[0] > max_points:
        fg_i = np.where(fused_fg)[0]
        bg_i = np.where(~fused_fg)[0]
        if fg_i.size > 0:
            fg_t = min(int(fg_i.size), int(round(max_points * 0.78)))
            bg_t = min(int(bg_i.size), max_points - fg_t)
            rem = max_points - fg_t - bg_t
            if rem > 0 and fg_i.size > fg_t:
                fg_t = min(int(fg_i.size), fg_t + rem)
            sel_fg = fg_i[np.linspace(0, fg_i.size - 1, fg_t, dtype=np.int64)] if fg_t > 0 else np.empty(0, dtype=np.int64)
            sel_bg = bg_i[np.linspace(0, bg_i.size - 1, bg_t, dtype=np.int64)] if bg_t > 0 else np.empty(0, dtype=np.int64)
            sel_idx = np.sort(np.concatenate([sel_fg, sel_bg]))
        else:
            sel_idx = np.linspace(0, fused_pts.shape[0] - 1, max_points, dtype=np.int64)
        out_pts = np.ascontiguousarray(fused_pts[sel_idx], dtype=np.float32)
        out_rgb = np.ascontiguousarray(fused_rgb[sel_idx], dtype=np.uint8)
        out_nrm = np.ascontiguousarray(fused_nrm[sel_idx], dtype=np.float32)
        out_vid = np.ascontiguousarray(fused_vid[sel_idx], dtype=np.uint8)
        out_fg = fused_fg[sel_idx]
    else:
        out_pts = np.ascontiguousarray(fused_pts, dtype=np.float32)
        out_rgb = np.ascontiguousarray(fused_rgb, dtype=np.uint8)
        out_nrm = np.ascontiguousarray(fused_nrm, dtype=np.float32)
        out_vid = np.ascontiguousarray(fused_vid, dtype=np.uint8)
        out_fg = fused_fg

    body_pts_three = fused_pts[fused_fg] if int(np.sum(fused_fg)) >= 12 else fused_pts
    min_y_three = float(np.percentile(body_pts_three[:, 1], 2))
    out_relief_mm = np.ascontiguousarray(
        np.clip(np.round(np.maximum(out_pts[:, 1] - min_y_three, 0.0) * 1000.0), 0, 65535).astype(np.uint16)
    )
    out_nrm_i8 = np.ascontiguousarray(np.clip(np.round(out_nrm * 127.0), -127, 127).astype(np.int8))

    b_min = [round(float(v), 3) for v in np.min(out_pts, axis=0)]
    b_max = [round(float(v), 3) for v in np.max(out_pts, axis=0)]
    max_relief_mm = round(float(np.percentile(out_relief_mm, 98)), 1) if out_relief_mm.size > 0 else 0.0

    surface_mesh: Optional[dict[str, Any]] = None
    if mesh_pos_chunks and mesh_tri_chunks:
        m_pos_all = np.ascontiguousarray(np.concatenate(mesh_pos_chunks, axis=0), dtype=np.float32)
        m_col_all = np.ascontiguousarray(np.concatenate(mesh_col_chunks, axis=0), dtype=np.uint8)
        m_nrm_all = np.ascontiguousarray(np.concatenate(mesh_nrm_chunks, axis=0), dtype=np.int8)
        m_tri_all = np.ascontiguousarray(np.concatenate(mesh_tri_chunks, axis=0), dtype=np.uint32)
        surface_mesh = {
            "vertex_count": int(m_pos_all.shape[0]),
            "triangle_count": int(m_tri_all.shape[0]),
            "positions_b64": base64.b64encode(m_pos_all.tobytes()).decode("ascii"),
            "colors_b64": base64.b64encode(m_col_all.tobytes()).decode("ascii"),
            "normals_b64": base64.b64encode(m_nrm_all.tobytes()).decode("ascii"),
            "indices_b64": base64.b64encode(m_tri_all.tobytes()).decode("ascii"),
        }

    point_cloud = {
        "count": int(out_pts.shape[0]),
        "positions_b64": base64.b64encode(out_pts.tobytes()).decode("ascii"),
        "colors_b64": base64.b64encode(out_rgb.tobytes()).decode("ascii"),
        "normals_b64": base64.b64encode(out_nrm_i8.tobytes()).decode("ascii"),
        "object_indices_b64": base64.b64encode(out_vid.tobytes()).decode("ascii"),
        "relief_cm_b64": base64.b64encode(out_relief_mm.tobytes()).decode("ascii"),
        "object_id_palette": [f"view_{i}" for i in range(n_views)],
        "has_normals": True,
        "body_point_count": int(np.sum(out_fg)),
        "max_relief_mm": max_relief_mm,
        "median_depth_m": round(float(np.median(np.linalg.norm(out_pts, axis=1))), 3),
        "bounds_three": {"min": b_min, "max": b_max},
        "surface_mesh": surface_mesh,
    }

    rmses = [float(p["alignment_rmse_m"]) for p in camera_poses[1:]] if n_views > 1 else [0.0]
    mean_rmse_m = float(np.mean(rmses)) if rmses else 0.0

    target_center_three = np.median(body_pts_three, axis=0)
    up_axis_three = _S_CV_TO_THREE @ up_axis_cv

    body_metrology = compute_closed_3d_body_metrology(
        body_pts_three,
        voxel_size_m=vox_m,
        occupied_voxels=n_vox,
        mean_icp_rmse_m=mean_rmse_m,
        single_view_obb_volume_m3=single_view_obb_vol,
        gt_volume_m3=gt_volume_m3,
        up_axis_three=up_axis_three,
    )

    body_nrm_three = fused_nrm[fused_fg] if int(np.sum(fused_fg)) >= 24 else fused_nrm
    coverage_360 = compute_360_coverage_and_nbv(
        camera_poses,
        target_center_three,
        up_axis_three=up_axis_three,
        body_normals_three=body_nrm_three,
    )

    mv_hw_export: Optional[dict[str, Any]] = None
    if best_hw_pair_args is not None:
        hw_src, hw_dst, hw_w, hw_seed, _, _ = best_hw_pair_args
        hw_reg = qubo_mwis_sim3_registration(
            hw_src,
            hw_dst,
            weights=hw_w,
            allow_scale_alignment=allow_scale_alignment,
            inlier_thresh_m=0.04,
            seed=hw_seed,
            include_hardware_export=True,
        )
        mv_hw_export = hw_reg.get("hardware_export")
    else:
        nbv_with_hw = plan_nbv_set_cover_qubo(
            coverage_360["sector_scores"],
            body_normals_three=body_nrm_three,
            existing_azimuths_deg=[float(p["azimuth_deg"]) for p in camera_poses],
            include_hardware_export=True,
        )
        mv_hw_export = nbv_with_hw.get("hardware_export")

    quantum_telemetry = {
        "enabled": bool(use_qubo_registration),
        "solver_name": "Harvard-HQI Rydberg-MWIS + Simulated Bifurcation",
        "views_optimized_by_qubo": int(views_optimized_by_qubo),
        "total_qubo_vars": int(sum(p["n_vars"] for p in qubo_pair_logs)),
        "total_conflict_edges": int(sum(p["n_conflict_edges"] for p in qubo_pair_logs)),
        "total_qubo_inliers": int(sum(p["qubo_inliers"] for p in qubo_pair_logs)),
        "total_ransac_inliers": int(sum(p["ransac_inliers"] for p in qubo_pair_logs)),
        "mean_energy_gain": round(
            float(np.mean([p["energy_gain"] for p in qubo_pair_logs])), 4
        )
        if qubo_pair_logs
        else 0.0,
        "total_solver_ms": round(
            float(sum(p["solver_ms"] for p in qubo_pair_logs))
            + float((coverage_360.get("qubo_nbv_plan") or {}).get("solver_ms", 0.0)),
            2,
        ),
        "per_view_pairs": qubo_pair_logs,
        "hardware_export": mv_hw_export,
    }

    return {
        "view_count": n_views,
        "camera_poses": camera_poses,
        "coverage_360": coverage_360,
        "body_metrology": body_metrology,
        "point_cloud": point_cloud,
        "fused_box_3d": body_metrology["fused_box_3d"],
        "quantum_telemetry": quantum_telemetry,
    }


def _ray_intersect_aabb_batch(
    orig: np.ndarray,
    dirs: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized slab ray-AABB intersection returning `(hit_t, hit_normal_three)` for `(M, 3)` rays."""
    inv_d = np.where(np.abs(dirs) > 1e-8, 1.0 / np.where(np.abs(dirs) > 1e-8, dirs, 1e-8), 1e12)
    t0 = (bmin[None, :] - orig[None, :]) * inv_d
    t1 = (bmax[None, :] - orig[None, :]) * inv_d
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)

    t_enter = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)
    hit = (t_exit >= np.maximum(t_enter, 0.02)) & np.isfinite(t_enter)
    hit_t = np.where(hit, t_enter, np.inf)

    # Identify hit face normal from argmax(tmin)
    axis = np.argmax(tmin, axis=1)
    nrm = np.zeros_like(dirs)
    for ax in range(3):
        sel = hit & (axis == ax)
        if np.any(sel):
            nrm[sel, ax] = np.where(dirs[sel, ax] < 0.0, 1.0, -1.0)
    return hit_t, nrm


def _world_surface_texture(world_pts: np.ndarray, part_id: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Evaluate rich non-repeating 3D world-locked decals and smooth harmonics so SIFT matches uniquely across views."""
    x, y, z = world_pts[:, 0], world_pts[:, 1], world_pts[:, 2]
    light = np.clip(0.62 + 0.38 * (0.35 * normals[:, 0] + 0.80 * normals[:, 1] + 0.48 * normals[:, 2]), 0.42, 1.0)

    # Non-repeating irrational spatial harmonics + asymmetric calibration decals
    wave = (
        np.sin(23.7 * x + 11.3 * z) * np.cos(17.9 * z - 7.1 * x)
        + 0.7 * np.sin(37.1 * x - 29.3 * y + 19.7 * z)
        + 0.5 * np.cos(41.3 * z + 13.1 * y)
    )
    fiducial = np.zeros_like(wave)
    decals = [
        (0.05, 0.14, 0.03, 0.038, 1.2),
        (-0.08, 0.30, -0.05, 0.034, -1.4),
        (0.18, 0.24, 0.02, 0.030, 1.3),
        (-0.14, 0.14, 0.09, 0.032, -1.1),
        (0.02, 0.30, 0.06, 0.028, 1.5),
        (-0.02, 0.07, 0.17, 0.035, -1.3),
        (0.21, 0.07, -0.02, 0.035, 1.4),
        (-0.21, 0.07, 0.04, 0.035, -1.2),
        (0.03, 0.07, -0.17, 0.035, 1.1),
    ]
    for dx0, dy0, dz0, rad, sign in decals:
        dist = np.sqrt((x - dx0) ** 2 + (y - dy0) ** 2 + (z - dz0) ** 2)
        fiducial += sign * np.exp(-((dist / rad) ** 2) * 2.4) * np.cos((dist / rad) * 5.2)

    base_r = np.where(part_id == 0, 212.0, np.where(part_id == 1, 64.0, 196.0))
    base_g = np.where(part_id == 0, 204.0, np.where(part_id == 1, 138.0, 118.0))
    base_b = np.where(part_id == 0, 188.0, np.where(part_id == 1, 148.0, 68.0))

    mod = 24.0 * wave + 38.0 * fiducial
    r = np.clip((base_r + mod) * light, 18.0, 246.0)
    g = np.clip((base_g + 0.85 * mod) * light, 18.0, 246.0)
    b = np.clip((base_b - 0.65 * mod) * light, 18.0, 246.0)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _encode_png_thumbnail(rgb_u8: np.ndarray, max_width: int = 220) -> str:
    pil = Image.fromarray(rgb_u8, mode="RGB")
    w, h = pil.size
    if w > max_width:
        nh = max(16, int(round(h * (max_width / float(w)))))
        pil = pil.resize((max_width, nh), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_demo_multiview_360_frames(num_views: int = 4) -> tuple[list[dict[str, Any]], float]:
    """Deterministically render 4 calibrated orbital views of a stepped 3D industrial specimen (SEED=42)."""
    azimuths = [0.0, 80.0, 165.0, 255.0][: max(2, min(6, num_views))]
    elev_deg = 28.0
    orbit_radius = 0.84
    target_three = np.array([0.0, 0.14, 0.0], dtype=np.float64)

    # Exact non-overlapping stepped industrial assembly in Three.js world (meters):
    # Plinth: 0.42 x 0.14 x 0.34 m (vol = 0.019992 m^3)
    # Turret: 0.24 x 0.16 x 0.22 m (vol = 0.008448 m^3)
    # Side module: 0.12 x 0.10 x 0.16 m (vol = 0.001920 m^3)
    # Total analytical GT volume = 0.030360 m^3
    boxes = [
        (np.array([-0.21, 0.00, -0.17]), np.array([0.21, 0.14, 0.17]), 0),
        (np.array([-0.12, 0.14, -0.11]), np.array([0.12, 0.30, 0.11]), 1),
        (np.array([0.12, 0.14, -0.08]), np.array([0.24, 0.24, 0.08]), 2),
    ]
    gt_volume_m3 = round(sum(float(np.prod(hi - lo)) for lo, hi, _ in boxes), 6)

    w, h = 384, 288
    fov_h_deg = 56.0
    fx = (w * 0.5) / math.tan(math.radians(fov_h_deg * 0.5))
    fy = fx
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dir_x_cv = ((us - cx) / fx).ravel()
    dir_y_cv = ((vs - cy) / fy).ravel()
    dir_z_cv = np.ones_like(dir_x_cv)
    dirs_cv_unit = np.stack([dir_x_cv, dir_y_cv, dir_z_cv], axis=-1)
    dirs_cv_unit /= np.linalg.norm(dirs_cv_unit, axis=-1, keepdims=True)
    # Convert local OpenCV ray direction to local Three.js camera direction: S * dir_cv
    dirs_cam_three = dirs_cv_unit @ _S_CV_TO_THREE.T

    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    frames: list[dict[str, Any]] = []

    for idx, az_deg in enumerate(azimuths):
        az_r = math.radians(az_deg)
        el_r = math.radians(elev_deg)
        cam_pos_three = target_three + orbit_radius * np.array(
            [
                math.cos(el_r) * math.sin(az_r),
                math.sin(el_r),
                math.cos(el_r) * math.cos(az_r),
            ],
            dtype=np.float64,
        )
        # Camera looks along -Z_cam toward target_three
        z_cam = cam_pos_three - target_three
        z_cam /= np.linalg.norm(z_cam)
        x_cam = np.cross(up_world, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        R_cam_to_world_three = np.column_stack([x_cam, y_cam, z_cam])  # det = +1

        dirs_world_three = dirs_cam_three @ R_cam_to_world_three.T

        best_t = np.full(w * h, np.inf, dtype=np.float64)
        best_nrm = np.zeros((w * h, 3), dtype=np.float64)
        best_part = np.full(w * h, -1, dtype=np.int32)

        for lo, hi, pid in boxes:
            hit_t, nrm_w = _ray_intersect_aabb_batch(cam_pos_three, dirs_world_three, lo, hi)
            better = hit_t < best_t
            best_t = np.where(better, hit_t, best_t)
            best_nrm[better] = nrm_w[better]
            best_part[better] = pid

        valid_flat = np.isfinite(best_t)
        rgb_flat = np.full((w * h, 3), 240, dtype=np.uint8)
        rgb_flat[:, 1] = 242
        rgb_flat[:, 2] = 245

        world_hit = np.zeros((w * h, 3), dtype=np.float64)
        world_hit[valid_flat] = cam_pos_three[None, :] + best_t[valid_flat, None] * dirs_world_three[valid_flat]

        if np.any(valid_flat):
            rgb_flat[valid_flat] = _world_surface_texture(
                world_hit[valid_flat],
                best_part[valid_flat],
                best_nrm[valid_flat],
            )

        # Inject a realistic +4% depth scale perturbation in View 2 so Sim(3) recovery is exercised
        depth_bias = 1.04 if idx == 2 else 1.0
        pts_cam_three = np.zeros((w * h, 3), dtype=np.float64)
        pts_cam_three[valid_flat] = ((world_hit[valid_flat] - cam_pos_three[None, :]) @ R_cam_to_world_three) * depth_bias
        pts_cv = (pts_cam_three @ _S_CV_TO_THREE.T).reshape(h, w, 3).astype(np.float32)

        nrm_cam_three = np.zeros((w * h, 3), dtype=np.float64)
        nrm_cam_three[valid_flat] = best_nrm[valid_flat] @ R_cam_to_world_three
        nrm_cv = (nrm_cam_three @ _S_CV_TO_THREE.T).reshape(h, w, 3).astype(np.float32)

        valid_2d = valid_flat.reshape(h, w)
        rgb_2d = rgb_flat.reshape(h, w, 3)

        frames.append(
            {
                "label": f"Vista {idx + 1} ({int(round(az_deg))}°)",
                "rgb_u8": rgb_2d,
                "points_cv": pts_cv,
                "normals_cv": nrm_cv,
                "valid_mask": valid_2d,
                "foreground_mask": valid_2d,
                "fov_h_deg": fov_h_deg,
                "yaw_hint_deg": az_deg,
                "thumbnail_base64": _encode_png_thumbnail(rgb_2d),
            }
        )

    return frames, gt_volume_m3


def build_offline_multiview_frame(
    image_base64: str,
    *,
    label: str = "Vista",
    fov_h_deg: float = 60.0,
    yaw_hint_deg: Optional[float] = None,
) -> dict[str, Any]:
    """Lightweight deterministic RGB -> pseudo-metric 3D frame fallback when MoGe-2 GPU engine is off."""
    raw = image_base64.strip()
    m = re.match(r"^data:([^;]+);base64,(.*)$", raw, flags=re.DOTALL)
    if m:
        raw = m.group(2).strip()
    img_bytes = base64.b64decode(raw, validate=False)
    with Image.open(io.BytesIO(img_bytes)) as src:
        rgb_pil = src.convert("RGB")
    w0, h0 = rgb_pil.size
    max_w = 240
    if w0 > max_w:
        nh = max(32, int(round(h0 * (max_w / float(w0)))))
        rgb_pil = rgb_pil.resize((max_w, nh), Image.Resampling.BILINEAR)
    rgb_u8 = np.asarray(rgb_pil, dtype=np.uint8)
    h, w = rgb_u8.shape[:2]

    # Smooth central foreground depth mound (0.85 m center -> 1.25 m rim)
    nx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    ny = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    gx, gy = np.meshgrid(nx, ny)
    r2 = gx * gx + gy * gy
    gray = (0.299 * rgb_u8[..., 0] + 0.587 * rgb_u8[..., 1] + 0.114 * rgb_u8[..., 2]).astype(np.float32) / 255.0
    depth_m = 0.85 + 0.35 * np.clip(r2, 0.0, 1.0) + 0.05 * (1.0 - gray)

    fx = (w * 0.5) / math.tan(math.radians(float(fov_h_deg) * 0.5))
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    x_cv = ((us - cx) / fx) * depth_m
    y_cv = ((vs - cy) / fx) * depth_m
    pts_cv = np.stack([x_cv, y_cv, depth_m], axis=-1).astype(np.float32)
    valid_mask = np.ones((h, w), dtype=bool)
    fg_mask = r2 <= 0.72
    nrm_cv = _compute_normals_cv(pts_cv, None)

    return {
        "label": label,
        "rgb_u8": rgb_u8,
        "points_cv": pts_cv,
        "normals_cv": nrm_cv,
        "valid_mask": valid_mask,
        "foreground_mask": fg_mask,
        "fov_h_deg": float(fov_h_deg),
        "yaw_hint_deg": yaw_hint_deg,
        "thumbnail_base64": _encode_png_thumbnail(rgb_u8),
    }
