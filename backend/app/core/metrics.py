"""Quantitative 3D object detection & reprojection metrics for MyLeenTo3.

Implements Hungarian 3D OBB assignment, 3D/BEV mIoU, AP@0.25 & AP@0.50,
nuScenes-style ATE/ASE/AOE, 2D pinhole reprojection IoU, deterministic
1000-resample 95% bootstrap confidence intervals, and FOV sensitivity sweeps.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry_3d import (
    BASIS_GEMINI_TO_THREE,
    compute_3d_iou,
    compute_bev_iou,
    compute_box_2d_iou,
    euler_deg_to_quaternion,
    project_obb_to_2d,
    quaternion_to_rotation_matrix,
)

DEFAULT_FOV_SWEEP_VALUES: list[int] = [40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110]


def _aligned_scale_error(box_pred: list[float], box_gt: list[float]) -> float:
    """nuScenes Average Scale Error: 1 - IoU_3D after aligning centroids & orientations."""
    dx_p, dy_p, dz_p = abs(float(box_pred[3])), abs(float(box_pred[4])), abs(float(box_pred[5]))
    dx_g, dy_g, dz_g = abs(float(box_gt[3])), abs(float(box_gt[4])), abs(float(box_gt[5]))
    inter = min(dx_p, dx_g) * min(dy_p, dy_g) * min(dz_p, dz_g)
    vol_p = dx_p * dy_p * dz_p
    vol_g = dx_g * dy_g * dz_g
    union = vol_p + vol_g - inter
    if union <= 0.0:
        return 1.0
    return float(np.clip(1.0 - (inter / union), 0.0, 1.0))


def _geodesic_orientation_error_deg(box_pred: list[float], box_gt: list[float]) -> float:
    """Exact 3D SO(3) geodesic rotation angle between predicted and GT orientations (deg)."""
    Rp = quaternion_to_rotation_matrix(
        euler_deg_to_quaternion(box_pred[6], box_pred[7], box_pred[8])
    )
    Rg = quaternion_to_rotation_matrix(
        euler_deg_to_quaternion(box_gt[6], box_gt[7], box_gt[8])
    )
    trace = float(np.trace(Rp.T @ Rg))
    cos_theta = float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cos_theta)))


def _compute_ap(
    pred_records: list[tuple[float, float, bool]],
    gt_count: int,
    iou_threshold: float,
) -> float:
    """Compute all-point interpolated Average Precision at a given 3D IoU threshold."""
    if gt_count <= 0 or not pred_records:
        return 0.0

    # Sort descending by (confidence, matched_iou_3d)
    ordered = sorted(pred_records, key=lambda r: (r[0], r[1]), reverse=True)
    tps = np.array(
        [1.0 if (is_matched and iou >= iou_threshold) else 0.0 for (_conf, iou, is_matched) in ordered],
        dtype=float,
    )
    fps = 1.0 - tps

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / float(gt_count)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    # All-point interpolation envelope
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))
    return float(np.clip(ap, 0.0, 1.0))


def _match_predictions_to_gt(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Hungarian assignment on cost = -iou_3d + 1e-4 * center_dist."""
    n_pred = len(preds)
    n_gt = len(gts)
    if n_pred == 0 or n_gt == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.zeros((n_pred, n_gt), dtype=float),
        )

    iou_matrix = np.zeros((n_pred, n_gt), dtype=float)
    cost_matrix = np.zeros((n_pred, n_gt), dtype=float)

    for i, pred in enumerate(preds):
        pb = pred["box_3d"]
        pc = np.array(pb[:3], dtype=float)
        for j, gt in enumerate(gts):
            gb = gt["box_3d"]
            gc = np.array(gb[:3], dtype=float)
            iou = compute_3d_iou(pb, gb, seed=seed)
            dist = float(np.linalg.norm(pc - gc))
            iou_matrix[i, j] = iou
            cost_matrix[i, j] = -iou + 1e-4 * dist

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind, iou_matrix


def evaluate_scene_predictions(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    width: int,
    height: int,
    fov_h_deg: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate predicted 9-DOF 3D bounding boxes against ground-truth objects."""
    n_pred = len(preds)
    n_gt = len(gts)
    n_boot = 1000

    if n_pred == 0 or n_gt == 0:
        return {
            "mIoU_3D": 0.0,
            "mIoU_BEV": 0.0,
            "ap_25": 0.0,
            "ap_50": 0.0,
            "ate_m": 0.0,
            "ase": 1.0 if n_gt > 0 else 0.0,
            "aoe_deg": 0.0,
            "reproj_iou_2d": 0.0,
            "matched_count": 0,
            "gt_count": n_gt,
            "pred_count": n_pred,
            "bootstrap_ci_95": {
                "miou_3d_low": 0.0,
                "miou_3d_high": 0.0,
                "ate_low": 0.0,
                "ate_high": 0.0,
                "n_boot": n_boot,
            },
            "per_object_matches": [],
        }

    row_ind, col_ind, iou_matrix = _match_predictions_to_gt(preds, gts, seed=seed)
    M = BASIS_GEMINI_TO_THREE

    matched_pred_map: dict[int, tuple[int, float]] = {}
    per_object_matches: list[dict[str, Any]] = []
    ious_3d: list[float] = []
    ious_bev: list[float] = []
    ates: list[float] = []
    ases: list[float] = []
    aoes: list[float] = []
    reprojs_2d: list[float] = []

    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        pred = preds[i]
        gt = gts[j]
        pb = pred["box_3d"]
        gb = gt["box_3d"]

        iou_3d = float(iou_matrix[i, j])
        iou_bev = compute_bev_iou(pb, gb)
        pc = np.array(pb[:3], dtype=float)
        gc = np.array(gb[:3], dtype=float)
        ate = float(np.linalg.norm(pc - gc))
        ase = _aligned_scale_error(pb, gb)
        aoe = _geodesic_orientation_error_deg(pb, gb)

        proj_p = project_obb_to_2d(pb, width, height, fov_h_deg)
        if "box_2d" in gt and len(gt["box_2d"]) == 4:
            gt_box_2d = gt["box_2d"]
        else:
            gt_box_2d = project_obb_to_2d(gb, width, height, fov_h_deg)["box_2d_reproj"]
        reproj_iou = compute_box_2d_iou(proj_p["box_2d_reproj"], gt_box_2d)

        err_vec_three = (M @ (pc - gc)).tolist()

        matched_pred_map[i] = (j, iou_3d)
        ious_3d.append(iou_3d)
        ious_bev.append(iou_bev)
        ates.append(ate)
        ases.append(ase)
        aoes.append(aoe)
        reprojs_2d.append(reproj_iou)

        per_object_matches.append(
            {
                "pred_id": str(pred.get("id", f"pred_{i}")),
                "gt_id": str(gt.get("id", f"gt_{j}")),
                "label": str(gt.get("label", pred.get("label", "object"))),
                "iou_3d": iou_3d,
                "iou_bev": iou_bev,
                "ate_m": ate,
                "aoe_deg": aoe,
                "reproj_iou_2d": reproj_iou,
                "centroid_error_vec_three": [float(v) for v in err_vec_three],
            }
        )

    matched_count = len(per_object_matches)
    denom = float(max(n_pred, n_gt))
    miou_3d = float(sum(ious_3d) / denom)
    miou_bev = float(sum(ious_bev) / denom)
    ate_m = float(np.mean(ates))
    ase_mean = float(np.mean(ases))
    aoe_deg = float(np.mean(aoes))
    reproj_iou_2d = float(np.mean(reprojs_2d))

    pred_records: list[tuple[float, float, bool]] = []
    for i, pred in enumerate(preds):
        conf = float(pred.get("confidence", 1.0))
        if i in matched_pred_map:
            pred_records.append((conf, matched_pred_map[i][1], True))
        else:
            pred_records.append((conf, 0.0, False))

    ap_25 = _compute_ap(pred_records, n_gt, 0.25)
    ap_50 = _compute_ap(pred_records, n_gt, 0.50)

    # Deterministic 1000-resample 95% bootstrap confidence intervals
    rng = np.random.default_rng(int(seed))
    iou_arr = np.array(ious_3d + [0.0] * (max(n_pred, n_gt) - matched_count), dtype=float)
    ate_arr = np.array(ates, dtype=float)

    idx_iou = rng.integers(0, len(iou_arr), size=(n_boot, len(iou_arr)))
    boot_miou = np.mean(iou_arr[idx_iou], axis=1)
    miou_low, miou_high = np.percentile(boot_miou, [2.5, 97.5])

    idx_ate = rng.integers(0, len(ate_arr), size=(n_boot, len(ate_arr)))
    boot_ate = np.mean(ate_arr[idx_ate], axis=1)
    ate_low, ate_high = np.percentile(boot_ate, [2.5, 97.5])

    return {
        "mIoU_3D": miou_3d,
        "mIoU_BEV": miou_bev,
        "ap_25": ap_25,
        "ap_50": ap_50,
        "ate_m": ate_m,
        "ase": ase_mean,
        "aoe_deg": aoe_deg,
        "reproj_iou_2d": reproj_iou_2d,
        "matched_count": matched_count,
        "gt_count": n_gt,
        "pred_count": n_pred,
        "bootstrap_ci_95": {
            "miou_3d_low": float(miou_low),
            "miou_3d_high": float(miou_high),
            "ate_low": float(ate_low),
            "ate_high": float(ate_high),
            "n_boot": n_boot,
        },
        "per_object_matches": per_object_matches,
    }


def fov_sensitivity_sweep(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    width: int,
    height: int,
    fov_values: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Measure 2D reprojection IoU & pixel error across candidate horizontal FOVs.

    Compares each matched prediction reprojected at candidate `fov_h_deg` against
    the fixed image-plane ground-truth observation (`gt['box_2d']` / `gt['center_2d']`
    if present, otherwise projected at the nominal calibrated 60.0 deg FOV).
    """
    sweep_fovs = list(fov_values) if fov_values is not None else list(DEFAULT_FOV_SWEEP_VALUES)
    if not preds or not gts:
        return [
            {
                "fov_h_deg": int(fov),
                "reproj_iou_2d": 0.0,
                "mean_pixel_error": 0.0,
            }
            for fov in sweep_fovs
        ]

    row_ind, col_ind, _ = _match_predictions_to_gt(preds, gts, seed=42)
    matched_pairs: list[tuple[list[float], list[float], list[float]]] = []

    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        pb = preds[i]["box_3d"]
        gt = gts[j]
        gb = gt["box_3d"]

        if "box_2d" in gt and len(gt["box_2d"]) == 4:
            gt_box_2d = [float(v) for v in gt["box_2d"]]
            if "center_2d" in gt and len(gt["center_2d"]) == 2:
                gt_center_px = [float(gt["center_2d"][0]), float(gt["center_2d"][1])]
            else:
                gt_center_px = [
                    ((gt_box_2d[1] + gt_box_2d[3]) * 0.5 / 1000.0) * width,
                    ((gt_box_2d[0] + gt_box_2d[2]) * 0.5 / 1000.0) * height,
                ]
        else:
            ref_fov = float(gt.get("fov_h_deg", 60.0))
            ref_proj = project_obb_to_2d(gb, width, height, ref_fov)
            gt_box_2d = ref_proj["box_2d_reproj"]
            gt_center_px = ref_proj["center_2d"]

        matched_pairs.append((pb, gt_box_2d, gt_center_px))

    results: list[dict[str, Any]] = []
    for fov in sweep_fovs:
        ious: list[float] = []
        px_errors: list[float] = []
        for pb, gt_box_2d, gt_center_px in matched_pairs:
            proj = project_obb_to_2d(pb, width, height, float(fov))
            ious.append(compute_box_2d_iou(proj["box_2d_reproj"], gt_box_2d))
            du = proj["center_2d"][0] - gt_center_px[0]
            dv = proj["center_2d"][1] - gt_center_px[1]
            px_errors.append(math.hypot(du, dv))

        results.append(
            {
                "fov_h_deg": int(fov),
                "reproj_iou_2d": float(np.mean(ious)) if ious else 0.0,
                "mean_pixel_error": float(np.mean(px_errors)) if px_errors else 0.0,
            }
        )

    return results
