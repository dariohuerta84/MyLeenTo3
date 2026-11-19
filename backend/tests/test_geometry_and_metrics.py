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
