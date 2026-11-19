"""Comprehensive unit tests for MyLeenTo3 3D geometry, metrics, benchmarks & multi-frame audit."""

from __future__ import annotations

import base64
import io
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_BACKEND_DIR = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.core.geometry_3d import (
    BASIS_GEMINI_TO_THREE,
    OBB_EDGES,
    compute_3d_iou,
    compute_bev_iou,
    compute_box_2d_iou,
    compute_fov_v_deg,
    compute_occlusion_matrix,
    compute_pairwise_geometry,
    euler_deg_to_quaternion,
    gemini_to_threejs_pose,
    obb_corners_gemini,
    obb_intersects_sat,
    project_obb_to_2d,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from app.core.intrinsic_diagnostics import (
    compute_intrinsic_scene_diagnostics,
    extract_image_adaptive_3d_hypotheses,
)
from app.core.metrics import (
    evaluate_scene_predictions,
    fov_sensitivity_sweep,
)
from app.core.multiframe_audit import (
    audit_multiframe_consistency,
    get_demo_multiframe_sequence,
)
from app.core.synthetic_benchmark import (
    get_benchmark,
    list_benchmarks,
)


def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


# ---------------------------------------------------------------------------
# 1. Rotations, Quaternions & Basis Equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "roll,pitch,yaw",
    [
        (0.0, 0.0, 0.0),
        (90.0, 0.0, 0.0),
        (0.0, 90.0, 0.0),
        (0.0, 0.0, 180.0),
        (180.0, 0.0, 0.0),
        (0.0, 180.0, 0.0),
        (-35.5, 17.2, 64.8),
        (12.0, -14.0, -125.0),
    ],
)
def test_euler_quaternion_matrix_roundtrip(roll: float, pitch: float, yaw: float) -> None:
    q = euler_deg_to_quaternion(roll, pitch, yaw)
    assert math.isclose(math.sqrt(sum(c * c for c in q)), 1.0, abs_tol=1e-12)

    R = quaternion_to_rotation_matrix(q)
    R_expected = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)

    assert np.allclose(R, R_expected, atol=1e-11)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-11)
    assert math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-11)

    q_back = rotation_matrix_to_quaternion(R)
    R_back = quaternion_to_rotation_matrix(q_back)
    assert np.allclose(R_back, R, atol=1e-11)


def test_basis_gemini_to_three_is_chirality_preserving() -> None:
    M = BASIS_GEMINI_TO_THREE
    assert math.isclose(float(np.linalg.det(M)), 1.0, abs_tol=1e-12)
    assert np.allclose(M.T @ M, np.eye(3), atol=1e-12)
    assert np.allclose(M, _rot_x(-90.0), atol=1e-12)


@pytest.mark.parametrize(
    "box_3d",
    [
        [0.4, 2.5, -0.3, 0.6, 0.4, 0.5, 0.0, 0.0, 0.0],
        [-0.8, 3.1, 0.15, 0.72, 0.35, 0.48, 12.5, -9.0, 47.3],
        [1.1, 4.2, -0.6, 0.9, 0.55, 0.32, -18.0, 14.5, -112.0],
        [0.0, 1.8, 0.0, 0.3, 0.8, 0.2, 45.0, -30.0, 135.0],
    ],
)
def test_gemini_to_threejs_pose_corner_equivalence_to_1e9(box_3d: list[float]) -> None:
    pose = gemini_to_threejs_pose(box_3d)
    M = BASIS_GEMINI_TO_THREE

    corners_gem = obb_corners_gemini(box_3d)
    assert corners_gem.shape == (8, 3)

    expected_three = (M @ corners_gem.T).T
    assert np.allclose(np.array(pose["corners_three"]), expected_three, atol=1e-12)

    # Transform Three.js local box corners ([±dx/2, ±dz/2, ±dy/2]) via R_three + pos_three
    sx, sy, sz = pose["size"]  # [dx, dz, dy]
    local_three = np.array(
        [
            [ix, iy, iz]
            for ix in (-sx / 2.0, sx / 2.0)
            for iy in (-sy / 2.0, sy / 2.0)
            for iz in (-sz / 2.0, sz / 2.0)
        ],
        dtype=float,
    )
    R_three = quaternion_to_rotation_matrix(pose["quaternion"])
    pos_three = np.array(pose["position"], dtype=float)
    reconstructed_three = (R_three @ local_three.T).T + pos_three

    # Verify set-level match to 1e-9
    for pt in reconstructed_three:
        dists = np.linalg.norm(expected_three - pt, axis=1)
        assert float(np.min(dists)) < 1e-9

    # Forward direction must be unit vector matching M @ R_gem @ [0,1,0]
    fwd = np.array(pose["forward_dir_three"], dtype=float)
    assert math.isclose(float(np.linalg.norm(fwd)), 1.0, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 2. Pinhole Projection & FOV
# ---------------------------------------------------------------------------


def test_compute_fov_v_deg() -> None:
    # 16:9 or 3:2 exact formula
    fov_v = compute_fov_v_deg(60.0, 960, 640)
    expected = 2.0 * math.degrees(math.atan(math.tan(math.radians(30.0)) * (640 / 960)))
    assert math.isclose(fov_v, expected, abs_tol=1e-12)
    # Square sensor => fov_v == fov_h
    assert math.isclose(compute_fov_v_deg(72.0, 800, 800), 72.0, abs_tol=1e-12)


def test_project_obb_to_2d_centered_and_behind_camera() -> None:
    box = [0.0, 2.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
    proj = project_obb_to_2d(box, 960, 640, 60.0)

    assert len(proj["vertices_2d"]) == 8
    assert len(proj["vertices_norm"]) == 8
    assert proj["edges"] == OBB_EDGES
    assert proj["behind_camera"] is False
    assert math.isclose(proj["center_2d"][0], 480.0, abs_tol=1e-9)
    assert math.isclose(proj["center_2d"][1], 320.0, abs_tol=1e-9)
    assert math.isclose(proj["min_depth_m"], 1.75, abs_tol=1e-9)
    assert math.isclose(proj["max_depth_m"], 2.25, abs_tol=1e-9)

    ymin, xmin, ymax, xmax = proj["box_2d_reproj"]
    assert 0.0 <= ymin < ymax <= 1000.0
    assert 0.0 <= xmin < xmax <= 1000.0

    # Box extending behind camera triggers clamp & behind_camera=True
    behind_box = [0.0, 0.1, 0.0, 0.4, 0.6, 0.4, 0.0, 0.0, 0.0]
    proj_behind = project_obb_to_2d(behind_box, 960, 640, 60.0)
    assert proj_behind["behind_camera"] is True
    assert proj_behind["min_depth_m"] < 0.05


# ---------------------------------------------------------------------------
# 3. 2D, BEV & 3D IoU + SAT + Pairwise Clearance + Occlusion
# ---------------------------------------------------------------------------


def test_bev_and_3d_iou_exact_analytical_cases() -> None:
    unit = [0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    unit_rot45 = [0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 45.0]
    half_shift = [0.5, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    disjoint = [3.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    assert math.isclose(compute_bev_iou(unit, unit), 1.0, abs_tol=1e-9)
    assert math.isclose(compute_bev_iou(unit, disjoint), 0.0, abs_tol=1e-9)
    # Exact analytical intersection of two unit squares rotated by 45 deg: IoU = 1 / sqrt(2)
    expected_rot45_iou = (2.0 * (math.sqrt(2.0) - 1.0)) / (2.0 - 2.0 * (math.sqrt(2.0) - 1.0))
    assert math.isclose(compute_bev_iou(unit, unit_rot45), expected_rot45_iou, abs_tol=1e-9)
    assert math.isclose(compute_3d_iou(unit, unit_rot45), expected_rot45_iou, abs_tol=1e-9)

    assert math.isclose(compute_3d_iou(unit, half_shift), 1.0 / 3.0, abs_tol=1e-9)
    assert math.isclose(compute_3d_iou(unit, disjoint), 0.0, abs_tol=1e-9)

    # 2D IoU exact check
    assert math.isclose(compute_box_2d_iou([0, 0, 100, 100], [0, 50, 100, 150]), 1.0 / 3.0, abs_tol=1e-9)
    assert math.isclose(compute_box_2d_iou([0, 0, 100, 100], [200, 200, 300, 300]), 0.0, abs_tol=1e-9)


def test_3d_iou_tilted_9dof_sat_and_stratified_determinism() -> None:
    a = [0.0, 2.0, 0.0, 0.6, 0.5, 0.4, 15.0, -10.0, 30.0]
    b_far = [2.0, 2.0, 0.0, 0.6, 0.5, 0.4, -12.0, 8.0, -40.0]
    b_overlap = [0.12, 2.05, 0.02, 0.6, 0.5, 0.4, 10.0, -8.0, 35.0]

    assert obb_intersects_sat(a, b_far) is False
    assert compute_3d_iou(a, b_far) == 0.0
    assert compute_3d_iou(a, a) == 1.0

    iou1 = compute_3d_iou(a, b_overlap, seed=42)
    iou2 = compute_3d_iou(a, b_overlap, seed=42)
    assert iou1 == iou2
    assert 0.35 < iou1 < 0.85


def test_pairwise_geometry_clearance_and_collision() -> None:
    box_a = [-0.5, 2.0, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0]
    box_b = [0.5, 2.0, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0]
    pw = compute_pairwise_geometry(box_a, box_b)

    assert pw["is_colliding"] is False
    assert math.isclose(pw["center_distance_m"], 1.0, abs_tol=1e-9)
    assert math.isclose(pw["surface_clearance_m"], 0.6, abs_tol=1e-6)
    pa = np.array(pw["closest_point_a_three"])
    pb = np.array(pw["closest_point_b_three"])
    assert math.isclose(float(np.linalg.norm(pa - pb)), pw["surface_clearance_m"], abs_tol=1e-6)
    assert pw["centroid_delta_m"] == [1.0, 0.0, 0.0]

    # Colliding pair
    box_c = [-0.35, 2.0, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 15.0]
    pw_coll = compute_pairwise_geometry(box_a, box_c)
    assert pw_coll["is_colliding"] is True
    assert pw_coll["surface_clearance_m"] == 0.0
    assert pw_coll["iou_3d"] > 0.0


def test_compute_occlusion_matrix_ray_obb() -> None:
    # Front object completely blocks a smaller aligned object behind it along +Y
    front = {"id": "front", "label": "shield", "box_3d": [0.0, 1.5, 0.0, 0.8, 0.2, 0.8, 0.0, 0.0, 0.0]}
    back = {"id": "back", "label": "target", "box_3d": [0.0, 3.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0]}
    side = {"id": "side", "label": "aside", "box_3d": [1.8, 3.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0]}

    occ = compute_occlusion_matrix([front, back, side])
    assert len(occ) == 1
    entry = occ[0]
    assert entry["occluder_id"] == "front"
    assert entry["occluded_id"] == "back"
    assert math.isclose(entry["occlusion_ratio"], 1.0, abs_tol=1e-9)
    assert math.isclose(entry["depth_gap_m"], 1.5, abs_tol=1e-9)
    assert len(entry["occluded_rays_three"]) == 9
    for ray in entry["occluded_rays_three"]:
        assert ray["origin"] == [0.0, 0.0, 0.0]
        hit_d = float(np.linalg.norm(ray["hit"]))
        tgt_d = float(np.linalg.norm(ray["target"]))
        assert 0.0 < hit_d < tgt_d


# ---------------------------------------------------------------------------
# 4. Metrics & FOV Sensitivity Sweep
# ---------------------------------------------------------------------------


def test_evaluate_scene_predictions_perfect_and_empty() -> None:
    gts = [
        {"id": "gt_1", "label": "box_a", "box_3d": [-0.4, 2.2, -0.2, 0.4, 0.3, 0.3, 0.0, 0.0, 15.0]},
        {"id": "gt_2", "label": "box_b", "box_3d": [0.5, 2.8, -0.1, 0.5, 0.4, 0.35, 8.0, -6.0, -30.0]},
    ]
    res = evaluate_scene_predictions(gts, gts, 960, 640, 60.0)
    assert math.isclose(res["mIoU_3D"], 1.0, abs_tol=1e-9)
    assert math.isclose(res["mIoU_BEV"], 1.0, abs_tol=1e-9)
    assert math.isclose(res["ap_25"], 1.0, abs_tol=1e-9)
    assert math.isclose(res["ap_50"], 1.0, abs_tol=1e-9)
    assert math.isclose(res["ate_m"], 0.0, abs_tol=1e-9)
    assert math.isclose(res["ase"], 0.0, abs_tol=1e-9)
    assert math.isclose(res["aoe_deg"], 0.0, abs_tol=1e-9)
    assert math.isclose(res["reproj_iou_2d"], 1.0, abs_tol=1e-9)
    assert res["matched_count"] == 2
    assert res["bootstrap_ci_95"]["n_boot"] == 1000
    assert math.isclose(res["bootstrap_ci_95"]["miou_3d_low"], 1.0, abs_tol=1e-9)

    empty_res = evaluate_scene_predictions([], gts, 960, 640, 60.0)
    assert empty_res["matched_count"] == 0
    assert empty_res["mIoU_3D"] == 0.0


def test_fov_sensitivity_sweep_peaks_at_calibrated_fov() -> None:
    bench = get_benchmark("lab_desk_calibrated")
    gts = bench["gt_objects"]
    sweep_exact = fov_sensitivity_sweep(gts, gts, bench["width"], bench["height"], [40, 50, 60, 75, 95])
    by_fov = {row["fov_h_deg"]: row for row in sweep_exact}

    assert math.isclose(by_fov[60]["reproj_iou_2d"], 1.0, abs_tol=1e-9)
    assert math.isclose(by_fov[60]["mean_pixel_error"], 0.0, abs_tol=1e-9)
    assert by_fov[40]["reproj_iou_2d"] < 0.9
    assert by_fov[95]["reproj_iou_2d"] < 0.9
    assert by_fov[40]["mean_pixel_error"] > 10.0


# ---------------------------------------------------------------------------
# 5. Synthetic Benchmarks & Multi-Frame Audit
# ---------------------------------------------------------------------------


def test_synthetic_benchmarks_structure_png_and_model_ranking() -> None:
    benches = list_benchmarks()
    assert len(benches) == 3
    ids = {b["id"] for b in benches}
    assert ids == {"lab_desk_calibrated", "occlusion_corridor", "rotated_workshop"}

    with pytest.raises(KeyError):
        get_benchmark("non_existent_scene")

    for b in benches:
        assert b["width"] == 960
        assert b["height"] == 640
        assert b["fov_h_deg"] == 60.0

        # Decode PNG base64 and verify exact dimensions
        raw = base64.b64decode(b["image_base64"])
        img = Image.open(io.BytesIO(raw))
        assert img.size == (960, 640)

        # Verify all GT vertices stay cleanly inside frame
        for g in b["gt_objects"]:
            for u, v in g["projected_2d"]["vertices_2d"]:
                assert 0.0 <= u <= 960.0
                assert 0.0 <= v <= 640.0

        # Verify all 3 Gemini models exist and rank monotonically
        preds_by_model = b["demo_predictions_by_model"]
        assert set(preds_by_model.keys()) == {
            "gemini-4-hydrogen",
            "gemini-4-argon",
            "gemini-3.5-flash-lite",
        }
        ev_h = evaluate_scene_predictions(preds_by_model["gemini-4-hydrogen"]["objects"], b["gt_objects"], 960, 640, 60.0)
        ev_a = evaluate_scene_predictions(preds_by_model["gemini-4-argon"]["objects"], b["gt_objects"], 960, 640, 60.0)
        ev_f = evaluate_scene_predictions(preds_by_model["gemini-3.5-flash-lite"]["objects"], b["gt_objects"], 960, 640, 60.0)

        assert ev_h["mIoU_3D"] > ev_a["mIoU_3D"] > ev_f["mIoU_3D"]
        assert ev_h["ate_m"] < ev_a["ate_m"] < ev_f["ate_m"]

    # Occlusion corridor must have strong occlusions
    corridor = get_benchmark("occlusion_corridor")
    occ = compute_occlusion_matrix(corridor["gt_objects"])
    assert len(occ) >= 4
    assert any(o["occlusion_ratio"] >= 0.5 for o in occ)


def test_multiframe_audit_demo_and_static_stability() -> None:
    seq = get_demo_multiframe_sequence()
    assert seq["sequence_id"] == "world_model_corridor_audit"
    assert len(seq["frames"]) == 4

    for fr in seq["frames"]:
        raw = base64.b64decode(fr["image_base64"])
        img = Image.open(io.BytesIO(raw))
        assert img.size == (960, 640)
        assert len(fr["objects"]) == 4

    audit = audit_multiframe_consistency(seq["frames"])
    assert audit["frame_count"] == 4
    assert 0.0 < audit["geometric_stability_score"] < 90.0
    assert audit["mean_scale_drift_pct"] > 5.0
    assert audit["mean_yaw_jitter_deg"] > 2.0
    assert len(audit["object_tracks"]) == 4
    assert len(audit["per_frame_summary"]) == 4
    assert isinstance(audit["drift_diagnosis"], str) and len(audit["drift_diagnosis"]) > 20

    # Static identical frames must score exact 100.0 with zero drift/jitter
    static_frames = [
        {"frame_idx": i, "timestamp_s": i * 0.5, "objects": [{"label": "cube", "box_3d": [0.0, 2.5, 0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 15.0]}]}
        for i in range(4)
    ]
    static_audit = audit_multiframe_consistency(static_frames, sequence_id="static_test")
    assert static_audit["geometric_stability_score"] == 100.0
    assert static_audit["mean_scale_drift_pct"] == 0.0
    assert static_audit["mean_yaw_jitter_deg"] == 0.0


# ---------------------------------------------------------------------------
# 6. Universal Reference-Free Intrinsic Diagnostics & Salience Hypotheses
# ---------------------------------------------------------------------------


def test_intrinsic_diagnostics_photometric_stress_and_physical_anomalies() -> None:
    # Build a dark stylized image with a bright high-contrast central subject
    arr = np.zeros((160, 240, 3), dtype=np.uint8)
    arr[:, :] = [10, 12, 18]  # deep dark background
    arr[45:125, 95:145] = [245, 250, 255]  # bright bloom subject
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Test image-adaptive 3D hypothesis extraction from real pixels
    hyps = extract_image_adaptive_3d_hypotheses(b64, 240, 160, 60.0, max_objects=4)
    assert len(hyps) >= 1
    for h in hyps:
        assert len(h["box_3d"]) == 9
        assert h["box_3d"][1] > 0.5  # positive depth

    # Test physical weakness detection: overlapping + floating + pancake boxes
    stressed_objects = [
        {"id": "o1", "label": "floor_block", "box_3d": [0.0, 2.5, -0.4, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0]},
        {"id": "o2", "label": "colliding_floater", "box_3d": [0.1, 2.5, 0.2, 0.9, 0.4, 0.9, 12.0, -8.0, 20.0]},
        {"id": "o3", "label": "far_giant", "box_3d": [0.4, 5.5, 0.2, 2.5, 2.2, 0.3, 0.0, 0.0, 0.0]},
    ]
    diag = compute_intrinsic_scene_diagnostics(stressed_objects, b64, 240, 160, 60.0)

    assert diag["image_profile"]["domain_classification"] == "high_contrast_stylized"
    assert diag["image_profile"]["dark_shadow_ratio"] > 0.7
    assert diag["image_profile"]["bloom_saturation_ratio"] > 0.05

    lat = diag["latent_metrics"]
    assert 0.0 <= lat["physical_plausibility_score"] <= 100.0
    assert lat["volume_interpenetration_index"] > 0.01  # o1 and o2 overlap in 3D
    assert lat["gravity_floating_anomaly_m"] > 0.3  # o2 floats well above o1
    assert lat["scale_depth_ambiguity_score"] > 0.4
    assert lat["mean_depth_uncertainty_m"] > 0.1
    assert len(diag["per_object_diagnostics"]) == 3
    assert len(diag["reference_free_fov_sweep"]) == 12
    assert any(w["id"] in ("photometric_stress", "solid_interpenetration", "gravity_floating") for w in diag["exploited_weaknesses"])

    # Empty scene edge case
    empty_diag = compute_intrinsic_scene_diagnostics([], b64, 240, 160, 60.0)
    assert empty_diag["latent_metrics"]["physical_plausibility_score"] == 0.0
    assert len(empty_diag["per_object_diagnostics"]) == 0
    assert empty_diag["cascade_policy"]["operational_verdict"] == "RECHAZADO_FISICA"
    assert empty_diag["cascade_policy"]["recommended_tier"] == "REJECT_RECAPTURE_OR_SENSOR"


def test_cascade_policy_c8286_verdicts() -> None:
    # 1. Clean well-supported non-occluded scene -> APROBADO
    clean_objs = [
        {"id": "c1", "label": "desk_instrument", "box_3d": [-0.5, 2.2, -0.2, 0.4, 0.3, 0.3, 0.0, 0.0, 10.0]},
        {"id": "c2", "label": "laptop", "box_3d": [0.45, 2.5, -0.2, 0.36, 0.28, 0.22, 0.0, 0.0, -12.0]},
    ]
    clean_diag = compute_intrinsic_scene_diagnostics(clean_objs, None, 960, 640, 60.0)
    cp_clean = clean_diag["cascade_policy"]
    assert cp_clean["operational_verdict"] == "APROBADO"
    assert cp_clean["c8286_degradation_regime"] == "clean"
    assert cp_clean["recommended_tier"] == "VLM_3D_FAST_TIER"
    assert cp_clean["estimated_latency_savings_pct"] > 0.0
    assert cp_clean["min_surface_clearance_m"] > 0.0

    # 2. Occlusion corridor benchmark -> ESCALAR_MODELO (occlusion_bg)
    corridor = get_benchmark("occlusion_corridor")
    corr_objs = corridor["demo_predictions_by_model"]["gemini-4-hydrogen"]["objects"]
    corr_diag = compute_intrinsic_scene_diagnostics(
        corr_objs,
        corridor["image_base64"],
        corridor["width"],
        corridor["height"],
        corridor["fov_h_deg"],
    )
    cp_corr = corr_diag["cascade_policy"]
    assert cp_corr["operational_verdict"] == "ESCALAR_MODELO"
    assert cp_corr["c8286_degradation_regime"] == "occlusion_bg"
    assert cp_corr["recommended_model"] == "gemini-4-hydrogen"
    assert cp_corr["occlusion_pairs_count"] >= 1

    # 3. Severe solid interpenetration & levitation -> RECHAZADO_FISICA
    broken_objs = [
        {"id": "b1", "label": "base_crate", "box_3d": [0.0, 2.5, -0.4, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0]},
        {"id": "b2", "label": "penetrating_crate", "box_3d": [0.05, 2.52, -0.35, 0.8, 0.8, 0.8, 0.0, 0.0, 5.0]},
        {"id": "b3", "label": "floating_crate", "box_3d": [1.0, 3.0, 1.4, 0.5, 0.5, 0.5, 25.0, -15.0, 10.0]},
    ]
    broken_diag = compute_intrinsic_scene_diagnostics(broken_objs, None, 960, 640, 60.0)
    cp_broken = broken_diag["cascade_policy"]
    assert cp_broken["operational_verdict"] == "RECHAZADO_FISICA"
    assert cp_broken["recommended_tier"] == "REJECT_RECAPTURE_OR_SENSOR"
    assert cp_broken["colliding_pairs_count"] >= 1


def test_local_3d_coordinate_transforms_and_chirality() -> None:
    from app.services.local_metric_3d import opencv_to_gemini_points, opencv_to_threejs_points

    pts_cv = np.array([[0.25, -0.10, 0.72], [-0.15, 0.30, 1.45]], dtype=np.float32)
    pts_gem = opencv_to_gemini_points(pts_cv)
    pts_three = opencv_to_threejs_points(pts_cv)

    # OpenCV (X-right, Y-down, Z-fwd) -> Gemini (X-right, Y-fwd, Z-up)
    assert np.allclose(pts_gem, [[0.25, 0.72, 0.10], [-0.15, 1.45, -0.30]], atol=1e-6)
    # OpenCV (X-right, Y-down, Z-fwd) -> Three.js (X-right, Y-up, -Z-fwd)
    assert np.allclose(pts_three, [[0.25, 0.10, -0.72], [-0.15, -0.30, -1.45]], atol=1e-6)

    # Both coordinate transforms must match BASIS_GEMINI_TO_THREE
    mapped_three = pts_gem @ BASIS_GEMINI_TO_THREE.T
    assert np.allclose(mapped_three, pts_three, atol=1e-6)


def test_local_3d_exif_35mm_landscape_and_portrait_fov() -> None:
    from app.services.local_metric_3d import extract_exif_image_info

    for (w, h), expected_sensor_mm in [((320, 240), 36.0), ((240, 320), 24.0)]:
        img = Image.new("RGB", (w, h), color=(140, 145, 150))
        exif = Image.Exif()
        exif[41989] = 26  # 26mm equivalent phone main camera
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif.tobytes())
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        rgb, rw, rh, fov_deg = extract_exif_image_info(f"data:image/jpeg;base64,{b64}")
        expected_fov = round(math.degrees(2.0 * math.atan((expected_sensor_mm * 0.5) / 26.0)), 2)
        assert (rw, rh) == (w, h)
        assert rgb.mode == "RGB"
        assert fov_deg == pytest.approx(expected_fov, abs=0.02)


def test_local_3d_lift_mask_and_point_cloud_packing() -> None:
    from app.services.local_metric_3d import (
        lift_mask_to_gemini_obb,
        pack_point_cloud_threejs,
        render_depth_map_png,
    )

    h, w = 80, 100
    xs = np.linspace(-0.08, 0.08, w, dtype=np.float32)[None, :]
    ys = np.linspace(-0.06, 0.06, h, dtype=np.float32)[:, None]
    points_cv = np.zeros((h, w, 3), dtype=np.float32)
    points_cv[..., 0] = np.broadcast_to(xs, (h, w))
    points_cv[..., 1] = np.broadcast_to(ys, (h, w))
    points_cv[..., 2] = 0.85

    mask = np.zeros((h, w), dtype=bool)
    mask[25:55, 35:65] = True
    points_cv[mask, 2] = 0.38  # Close-up foreground object at 38 cm

    valid_mask = np.ones((h, w), dtype=bool)
    box = lift_mask_to_gemini_obb(points_cv, mask, valid_mask)
    assert box is not None
    assert 0.34 <= box[1] <= 0.42  # y_fwd around 0.38 m
    assert 0.01 <= box[3] <= 0.12  # physical width dx in cm-dm scale
    assert 0.01 <= box[5] <= 0.12  # physical height dz in cm-dm scale

    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    pc = pack_point_cloud_threejs(points_cv, rgb, valid_mask, max_points=500)
    assert pc["count"] == 500
    pts_decoded = np.frombuffer(base64.b64decode(pc["positions_b64"]), dtype=np.float32).reshape(-1, 3)
    cols_decoded = np.frombuffer(base64.b64decode(pc["colors_b64"]), dtype=np.uint8).reshape(-1, 3)
    assert pts_decoded.shape == (500, 3)
    assert cols_decoded.shape == (500, 3)
    assert pc["median_depth_m"] > 0.35

    depth_b64, stats = render_depth_map_png(points_cv[..., 2], valid_mask, max_width=80)
    assert stats["min_m"] == pytest.approx(0.38, abs=0.02)
    assert stats["max_m"] == pytest.approx(0.85, abs=0.02)
    with Image.open(io.BytesIO(base64.b64decode(depth_b64))) as depth_img:
        assert depth_img.format == "PNG"
        assert depth_img.width <= 80


def test_metrics_nds_static_and_depth_bins() -> None:
    bm = get_benchmark("lab_desk_calibrated")
    preds = bm["demo_predictions_by_model"]["gemini-4-hydrogen"]["objects"]
    ev = evaluate_scene_predictions(
        preds,
        bm["gt_objects"],
        int(bm["width"]),
        int(bm["height"]),
        float(bm["fov_h_deg"]),
        seed=42,
    )
    assert 0.5 <= ev["nds_static"] <= 1.0
    bins = ev["depth_bins"]
    assert len(bins) == 3
    assert [b["bin"] for b in bins] == ["near_0_2m", "mid_2_3.5m", "far_3.5m_plus"]
    assert sum(b["gt_count"] for b in bins) == len(bm["gt_objects"])


def test_real_photo_optical_sharpness_and_noise() -> None:
    from PIL import ImageFilter
    from app.core.real_photo_metrics import compute_optical_sharpness_and_noise

    h, w = 96, 128
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    checker = (((xx // 4) + (yy // 4)) % 2).astype(np.float32)
    rgb_sharp = np.stack([checker, checker, checker], axis=-1)
    sharp_res = compute_optical_sharpness_and_noise(rgb_sharp, fov_h_deg=60.0, width=w, height=h)
    assert sharp_res["laplacian_variance"] >= 180.0
    assert sharp_res["focus_quality"] == "sharp"
    assert sharp_res["equivalent_focal_35mm_mm"] == pytest.approx(31.18, abs=0.2)

    pil_sharp = Image.fromarray((rgb_sharp * 255).astype(np.uint8), mode="RGB")
    rgb_blur = np.asarray(pil_sharp.filter(ImageFilter.GaussianBlur(radius=6.0)), dtype=np.float32) / 255.0
    blur_res = compute_optical_sharpness_and_noise(rgb_blur, fov_h_deg=60.0, width=w, height=h)
    assert blur_res["laplacian_variance"] < 45.0
    assert blur_res["focus_quality"] == "blurry"


def test_real_photo_normal_depth_consistency_and_manhattan() -> None:
    from app.core.real_photo_metrics import compute_normal_depth_consistency

    h, w = 64, 64
    xs = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :]
    ys = np.linspace(-0.5, 0.5, h, dtype=np.float32)[:, None]
    pts_cv = np.zeros((h, w, 3), dtype=np.float32)
    pts_cv[..., 0] = np.broadcast_to(xs, (h, w))
    pts_cv[..., 1] = np.broadcast_to(ys, (h, w))
    # Gentle planar slope z = 1.8 - 0.15*x + 0.10*y
    pts_cv[..., 2] = 1.8 - 0.15 * pts_cv[..., 0] + 0.10 * pts_cv[..., 1]
    valid = np.ones((h, w), dtype=bool)

    # Normal to z + 0.15*x - 0.10*y = 1.8 pointing toward -Z camera
    n_exact = np.array([-0.15, 0.10, -1.0], dtype=np.float32)
    n_exact /= np.linalg.norm(n_exact)
    normals_ok = np.broadcast_to(n_exact, (h, w, 3)).copy()

    res_ok = compute_normal_depth_consistency(pts_cv, normals_ok, valid)
    assert res_ok["has_model_normals"] is True
    assert res_ok["median_angular_error_deg"] < 2.0
    assert res_ok["consistent_ratio_15deg"] > 0.99
    assert res_ok["ndc_score_0_100"] > 95.0
    assert res_ok["manhattan_alignment_score"] > 0.85

    # Tilted inconsistent normals (>45 deg off)
    n_bad = np.array([0.8, 0.0, -0.2], dtype=np.float32)
    n_bad /= np.linalg.norm(n_bad)
    normals_bad = np.broadcast_to(n_bad, (h, w, 3)).copy()
    res_bad = compute_normal_depth_consistency(pts_cv, normals_bad, valid)
    assert res_bad["median_angular_error_deg"] > 35.0
    assert res_bad["ndc_score_0_100"] < 30.0


def test_real_photo_ransac_boundaries_and_object_metrology() -> None:
    from app.core.real_photo_metrics import (
        compute_depth_boundary_sharpness,
        compute_object_point_cloud_metrology,
        fit_ransac_dominant_plane,
    )

    h, w = 80, 100
    xs = np.linspace(-0.4, 0.4, w, dtype=np.float32)[None, :]
    ys = np.linspace(-0.3, 0.3, h, dtype=np.float32)[:, None]
    pts_cv = np.zeros((h, w, 3), dtype=np.float32)
    pts_cv[..., 0] = np.broadcast_to(xs, (h, w))
    # Horizontal support plane in Gemini space means Z_gem = -Y_cv = -0.50 => Y_cv = +0.50
    pts_cv[..., 1] = 0.50
    pts_cv[..., 2] = 1.60 + np.broadcast_to(ys, (h, w)) * 1.2

    valid = np.ones((h, w), dtype=bool)
    # Add 15% floating outlier box above the horizontal plane
    obj_mask = np.zeros((h, w), dtype=bool)
    obj_mask[24:56, 36:64] = True
    pts_cv[obj_mask, 0] = np.broadcast_to(xs, (h, w))[obj_mask] * 0.6
    pts_cv[obj_mask, 1] = np.broadcast_to(ys, (h, w))[obj_mask] * 0.6
    pts_cv[obj_mask, 2] = 1.15

    ransac = fit_ransac_dominant_plane(pts_cv, valid, seed=42)
    assert ransac["detected"] is True
    assert ransac["plane_orientation"] == "horizontal_support"
    assert ransac["plane_tilt_deg"] < 5.0
    assert ransac["rmse_residual_m"] < 0.015
    assert ransac["plane_height_z_m"] == pytest.approx(-0.50, abs=0.03)

    rgb_u8 = np.full((h, w, 3), 210, dtype=np.uint8)
    rgb_u8[obj_mask] = [30, 80, 190]
    bnd = compute_depth_boundary_sharpness(pts_cv[..., 2], rgb_u8, valid)
    assert bnd["depth_rgb_boundary_f1"] > 0.55
    assert 0.0 <= bnd["flying_pixel_ratio"] < 0.25

    box_3d = [0.0, 1.15, 0.0, 0.14, 0.04, 0.12, 0.0, 0.0, 0.0]
    obj_met = compute_object_point_cloud_metrology(
        pts_cv,
        None,
        obj_mask,
        valid,
        box_3d,
        fov_h_deg=60.0,
        object_id="moge_test_1",
        label="test_panel",
    )
    assert obj_met["point_count"] == int(np.sum(obj_mask))
    assert obj_met["surface_fit_rmse_m"] < 0.03
    assert obj_met["visible_surface_area_m2"] > 0.01
    assert obj_met["fill_solidity_ratio"] > 0.10


def test_clinical_specular_masking_and_modal_foreground_clustering() -> None:
    from app.core.real_photo_metrics import compute_specular_mask
    from app.services.local_metric_3d import lift_mask_to_gemini_obb

    h, w = 80, 100
    xs = np.linspace(-0.22, 0.22, w, dtype=np.float32)[None, :]
    ys = np.linspace(-0.18, 0.18, h, dtype=np.float32)[:, None]
    pts_cv = np.zeros((h, w, 3), dtype=np.float32)
    pts_cv[..., 0] = np.broadcast_to(xs, (h, w))
    pts_cv[..., 1] = np.broadcast_to(ys, (h, w))
    pts_cv[..., 2] = 1.85

    mask = np.zeros((h, w), dtype=bool)
    mask[20:60, 25:75] = True
    # 65% true foreground mode around 0.55m; inner hollow rings expose 1.85m background; corner has glare spike
    pts_cv[mask, 2] = 0.55
    pts_cv[32:48, 38:52] = [0.0, 0.0, 1.85]  # background bleed inside hollow hole (about 10%)
    pts_cv[20:28, 60:75, 2] = 1.85  # edge bleed
    pts_cv[22:29, 27:36, 2] = 4.20  # specular spike

    rgb_u8 = np.full((h, w, 3), 90, dtype=np.uint8)
    rgb_u8[22:29, 27:36] = 250  # bright achromatic specular highlight

    spec_mask, spec_ratio = compute_specular_mask(rgb_u8)
    assert spec_ratio > 0.005
    assert bool(np.all(spec_mask[22:29, 27:36]))

    valid = np.ones((h, w), dtype=bool)
    box = lift_mask_to_gemini_obb(points_cv=pts_cv, binary_mask=mask, valid_mask=valid, rgb_u8=rgb_u8, label="cable")
    assert box is not None
    # Modal foreground clustering keeps centroid locked near 0.55 m instead of dragging to 1.0+ m
    assert 0.48 <= box[1] <= 0.64


def test_lshape_bev_orientation_and_amodal_front_anchor() -> None:
    from app.services.local_metric_3d import _fit_lshape_bev_yaw, lift_mask_to_gemini_obb

    # Asymmetric L-shape point cluster in BEV rotated by +28 degrees
    rng = np.random.default_rng(42)
    t_front = np.linspace(-0.28, 0.28, 120)
    front_pts = np.column_stack([t_front, np.full_like(t_front, -0.10) + rng.normal(0, 0.002, t_front.size)])
    t_side = np.linspace(-0.10, 0.14, 70)
    side_pts = np.column_stack([np.full_like(t_side, 0.28) + rng.normal(0, 0.002, t_side.size), t_side])
    local_l = np.vstack([front_pts, side_pts])

    theta_true = 28.0
    rad = math.radians(theta_true)
    c, s = math.cos(rad), math.sin(rad)
    rot_xy = np.column_stack([local_l[:, 0] * c - local_l[:, 1] * s, local_l[:, 0] * s + local_l[:, 1] * c])
    yaw_est = _fit_lshape_bev_yaw(rot_xy - np.mean(rot_xy, axis=0))
    # Box orientation matches modulo 90 degrees orthogonal axes
    ang_err = min(abs(yaw_est - theta_true), abs(yaw_est + (90.0 - theta_true)))
    assert ang_err <= 5.0

    # Front-face amodal anchoring for a shallow observed monitor surface at y_front = 1.20m
    h, w = 80, 100
    xs = np.linspace(-0.60, 0.60, w, dtype=np.float32)[None, :]
    ys = np.linspace(-0.40, 0.40, h, dtype=np.float32)[:, None]
    pts_cv = np.zeros((h, w, 3), dtype=np.float32)
    pts_cv[..., 0] = np.broadcast_to(xs, (h, w))
    pts_cv[..., 1] = np.broadcast_to(ys, (h, w))
    pts_cv[..., 2] = 2.50

    mon_mask = np.zeros((h, w), dtype=bool)
    mon_mask[18:62, 22:78] = True
    pts_cv[mon_mask, 2] = 1.20  # pure planar front skin (0 cm observed thickness)

    box = lift_mask_to_gemini_obb(pts_cv, mon_mask, np.ones((h, w), dtype=bool), label="monitor")
    assert box is not None
    # Amodal prior enforces minimum depth thickness while anchoring front face near 1.20m
    assert box[4] >= 0.06
    assert (box[1] - 0.5 * box[4]) == pytest.approx(1.20, abs=0.03)


def test_depth_pro_multiscale_silhouette_recall_and_topographical_volume() -> None:
    from app.core.real_photo_metrics import (
        compute_depth_boundary_sharpness,
        compute_object_point_cloud_metrology,
        fit_ransac_dominant_plane,
    )

    h, w = 80, 100
    xs = np.linspace(-0.4, 0.4, w, dtype=np.float32)[None, :]
    ys = np.linspace(-0.3, 0.3, h, dtype=np.float32)[:, None]
    pts_cv = np.zeros((h, w, 3), dtype=np.float32)
    pts_cv[..., 0] = np.broadcast_to(xs, (h, w))
    pts_cv[..., 1] = np.broadcast_to(ys, (h, w))
    pts_cv[..., 2] = 1.50  # vertical flat plane at Y_gem = 1.50 m

    obj_mask = np.zeros((h, w), dtype=bool)
    obj_mask[24:56, 35:65] = True
    pts_cv[obj_mask, 2] = 1.12  # 38 cm protruding object (>25% ratio across Depth Pro thresholds)

    valid = np.ones((h, w), dtype=bool)
    rgb_u8 = np.full((h, w, 3), 180, dtype=np.uint8)
    rgb_u8[obj_mask] = [40, 70, 160]

    ransac = fit_ransac_dominant_plane(pts_cv, valid, seed=42)
    assert ransac["detected"] is True
    assert ransac["plane_orientation"] == "vertical_wall"
    assert "plane_d_gemini" in ransac

    bnd = compute_depth_boundary_sharpness(pts_cv[..., 2], rgb_u8, valid, instance_masks=[obj_mask])
    assert bnd["depth_pro_multiscale_f1"] > 0.70
    assert bnd["silhouette_boundary_recall"] is not None
    assert bnd["silhouette_boundary_recall"] > 0.85

    obj_met = compute_object_point_cloud_metrology(
        pts_cv,
        None,
        obj_mask,
        valid,
        [0.0, 1.12, 0.0, 0.24, 0.04, 0.24, 0.0, 0.0, 0.0],
        fov_h_deg=60.0,
        object_id="moge_relief",
        label="electrical_switch",
        ransac_plane=ransac,
    )
    assert obj_met["max_relief_height_m"] == pytest.approx(0.38, abs=0.03)
    assert obj_met["mean_relief_height_m"] >= 0.34
    assert obj_met["integrated_volume_above_plane_m3"] > 0.0
    assert obj_met["specular_ratio_in_mask"] == pytest.approx(0.0, abs=0.01)


def test_ransac_vertical_wall_gfi_and_free_space_consistency() -> None:
    from app.core.intrinsic_diagnostics import compute_intrinsic_scene_diagnostics
    from app.core.real_photo_metrics import compute_free_space_consistency, fit_ransac_dominant_plane

    h, w = 64, 80
    xs = np.linspace(-0.5, 0.5, w, dtype=np.float32)[None, :]
    ys = np.linspace(-0.4, 0.4, h, dtype=np.float32)[:, None]
    pts_cv = np.zeros((h, w, 3), dtype=np.float32)
    pts_cv[..., 0] = np.broadcast_to(xs, (h, w))
    pts_cv[..., 1] = np.broadcast_to(ys, (h, w))
    pts_cv[..., 2] = 1.60  # vertical wall at y = 1.60 m

    valid = np.ones((h, w), dtype=bool)
    ransac = fit_ransac_dominant_plane(pts_cv, valid, seed=42)
    assert ransac["plane_orientation"] == "vertical_wall"

    # Two boxes mounted on the vertical wall at z = -0.35m and z = +0.35m
    wall_objects = [
        {"id": "w1", "label": "outlet_low", "box_3d": [-0.15, 1.56, -0.35, 0.12, 0.08, 0.12, 0.0, 0.0, 0.0]},
        {"id": "w2", "label": "outlet_high", "box_3d": [0.15, 1.56, 0.35, 0.12, 0.08, 0.12, 0.0, 0.0, 0.0]},
    ]
    fs_ok = compute_free_space_consistency(pts_cv[..., 2], valid, wall_objects, fov_h_deg=60.0)
    assert fs_ok["total_objects"] == 2
    assert fs_ok["free_space_violation_ratio"] == 0.0

    # Floating outlier placed far ahead at 0.45m when depth rays clearly show 1.60m wall
    bad_objects = wall_objects + [
        {"id": "bad_float", "label": "hallucination", "box_3d": [0.0, 0.45, 0.0, 0.25, 0.10, 0.25, 0.0, 0.0, 0.0]},
    ]
    fs_bad = compute_free_space_consistency(pts_cv[..., 2], valid, bad_objects, fov_h_deg=60.0)
    assert fs_bad["free_space_violation_ratio"] > 0.2

    rpm = {
        "ransac_plane": ransac,
        "free_space_consistency": fs_ok,
        "depth_boundaries": {"flying_pixel_ratio": 0.01},
        "optical_telemetry": {"specular_reflection_ratio": 0.0},
        "normal_depth_consistency": {"ndc_score_0_100": 90.0},
    }
    diag = compute_intrinsic_scene_diagnostics(
        wall_objects, None, w, h, 60.0, real_photo_metrics=rpm
    )
    assert diag["latent_metrics"]["support_plane_mode"] == "vertical_wall"
    # Vertical wall attachment should have near-zero GFI instead of ~0.70m false floor gap
    assert diag["latent_metrics"]["gravity_floating_anomaly_m"] < 0.06
    assert "next_best_view_advice" in diag["cascade_policy"]
    assert len(diag["scene_graph_3d"]) == 1


def test_reproject_canonical_fov_depth_rescaling_and_scene_graph() -> None:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.post(
        "/api/reproject",
        json={
            "objects": [
                {"id": "o1", "label": "left_box", "box_3d": [-0.30, 2.00, 0.0, 0.20, 0.20, 0.20, 0.0, 0.0, 0.0]},
                {"id": "o2", "label": "right_box", "box_3d": [0.40, 2.60, 0.0, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0]},
            ],
            "width": 960,
            "height": 640,
            "fov_h_deg": 45.0,
            "reference_fov_h_deg": 60.0,
            "rescale_depth_with_fov": True,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    expected_scale = math.tan(math.radians(30.0)) / math.tan(math.radians(22.5))
    assert payload["objects"][0]["box_3d"][1] == pytest.approx(2.00 * expected_scale, abs=0.01)
    assert payload["objects"][0]["box_3d"][3] == pytest.approx(0.20 * expected_scale, abs=0.01)

    sg = payload["intrinsic_diagnostics"]["scene_graph_3d"]
    assert len(sg) == 1
    assert sg[0]["relation_horizontal"] == "left"
    assert sg[0]["relation_depth"] == "in_front"
