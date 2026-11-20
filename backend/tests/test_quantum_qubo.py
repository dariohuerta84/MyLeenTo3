"""Comprehensive tests for Phase 4 Harvard-HQI QUBO-MWIS Quantum-Inspired Engine."""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from app.core.quantum_qubo import (
    _demo_nms_greedy_trap_objects,
    evaluate_qubo_energy,
    export_qubo_hardware_payload,
    normalize_qubo,
    plan_nbv_set_cover_qubo,
    qubo_mwis_sim3_registration,
    qubo_to_ising,
    resolve_3d_objects_mwis_qubo,
    solve_mwis_rydberg,
)
from app.main import app

client = TestClient(app)


def test_qubo_to_ising_exact_energy_identity() -> None:
    rng = np.random.default_rng(42)
    Q_raw = rng.uniform(-4.5, 4.5, size=(7, 7))
    Q_sym = normalize_qubo(Q_raw)
    J, h, offset = qubo_to_ising(Q_sym)

    for bits in range(1 << 7):
        x = np.array([(bits >> k) & 1 for k in range(7)], dtype=np.float64)
        s = 2.0 * x - 1.0
        e_qubo_raw = float(x @ Q_raw @ x)
        e_qubo_sym = evaluate_qubo_energy(Q_sym, x)
        e_ising = float(s @ J @ s + h @ s + offset)
        assert abs(e_qubo_raw - e_qubo_sym) < 1e-10
        assert abs(e_qubo_sym - e_ising) < 1e-10


def test_solve_mwis_rydberg_beats_greedy_on_star_trap() -> None:
    # Node 0 (hub, weight=0.95) conflicts with leaves 1, 2, 3 (weight=0.70 each).
    # Greedy picks hub -> weight 0.95; Global MWIS picks {1,2,3} -> weight 2.10.
    weights = np.array([0.95, 0.70, 0.70, 0.70], dtype=np.float64)
    conflicts = np.zeros((4, 4), dtype=np.float64)
    for j in (1, 2, 3):
        conflicts[0, j] = 1.0
        conflicts[j, 0] = 1.0

    res = solve_mwis_rydberg(weights, conflicts, seed=42)
    assert res["conflict_violations"] == 0
    assert res["exact_optimum_verified"] is True
    assert res["selected_indices"] == [1, 2, 3]
    assert abs(res["selected_weight_sum"] - 2.10) < 1e-5
    assert abs(res["greedy_weight_sum"] - 0.95) < 1e-5
    assert abs(res["energy_gain"] - 1.15) < 1e-5


def test_qubo_mwis_sim3_registration_75pct_adversarial_outliers() -> None:
    rng = np.random.default_rng(42)
    n = 72
    n_in = 18
    n_out = n - n_in

    gt_R = Rotation.from_euler("zyx", [24.0, -16.0, 33.0], degrees=True).as_matrix()
    gt_t = np.array([0.28, -0.15, 0.42], dtype=np.float64)
    gt_s = 1.12

    src_in = rng.uniform([-0.45, -0.35, 0.55], [0.45, 0.35, 1.35], size=(n_in, 3))
    dst_in = (gt_s * (src_in @ gt_R.T)) + gt_t + rng.normal(0.0, 0.004, size=(n_in, 3))

    src_out = rng.uniform([-0.45, -0.35, 0.55], [0.45, 0.35, 1.35], size=(n_out, 3))
    r_decoy = Rotation.from_euler("zyx", [-95.0, 22.0, 140.0], degrees=True).as_matrix()
    dst_decoy = (gt_s * (src_out[:10] @ r_decoy.T)) + np.array([-0.22, 0.19, 0.35])
    dst_rand = rng.uniform([-0.70, -0.55, 0.45], [0.80, 0.60, 1.75], size=(n_out - 10, 3))
    dst_out = np.vstack([dst_decoy, dst_rand])

    src = np.vstack([src_in, src_out])
    dst = np.vstack([dst_in, dst_out])
    perm = rng.permutation(n)
    src = src[perm]
    dst = dst[perm]

    res = qubo_mwis_sim3_registration(
        src,
        dst,
        allow_scale_alignment=True,
        inlier_thresh_m=0.04,
        seed=42,
        include_hardware_export=True,
    )
    rot_err_deg = float(np.degrees(np.linalg.norm(Rotation.from_matrix(res["R"] @ gt_R.T).as_rotvec())))
    trans_err_m = float(np.linalg.norm(res["t"] - gt_t))
    scale_err_pct = float(abs(res["scale"] - gt_s) / gt_s * 100.0)

    assert res["method_won"] == "qubo_mwis"
    assert res["qubo_inliers"] >= 17
    assert res["inliers"] >= res["ransac_inliers"]
    assert rot_err_deg < 2.0
    assert trans_err_m < 0.02
    assert scale_err_pct < 3.0
    assert "hardware_export" in res


def test_resolve_3d_objects_mwis_qubo_trap_and_support_immunity() -> None:
    objs = _demo_nms_greedy_trap_objects()
    res = resolve_3d_objects_mwis_qubo(objs, seed=42, include_hardware_export=True)
    qs = res["qubo_stats"]
    kept_ids = {o["id"] for o in res["kept_objects"]}

    assert qs["exact_optimum_verified"] is True
    assert qs["conflict_violations"] == 0
    assert qs["qubo_weight_sum"] > qs["greedy_weight_sum"]
    assert qs["energy_gain"] > 1.0
    # Giant greedy trap must be suppressed while keeping all 3 true sub-objects + table + mug on table
    assert "trap_mega_box" not in kept_ids
    assert {"true_oscilloscope", "true_power_supply", "true_multimeter_unit"} <= kept_ids
    assert {"lab_table", "coffee_mug_on_table"} <= kept_ids


def test_plan_nbv_set_cover_qubo_reaches_high_coverage() -> None:
    plan = plan_nbv_set_cover_qubo(
        [0.0] * 12,
        budget_k=2,
        existing_azimuths_deg=[0.0, 40.0],
        seed=42,
        include_hardware_export=True,
    )
    assert plan["budget_k"] == 2
    assert len(plan["optimal_shot_plan_deg"]) == 2
    assert len(plan["optimal_sector_indices"]) == 2
    assert plan["predicted_coverage_after_plan_pct"] >= 90.0
    assert len(plan["shadow_uncertainty_by_sector"]) == 12


def test_export_qubo_hardware_payload_braket_and_aquila() -> None:
    weights = np.array([0.85, 0.80, 0.78, 0.72], dtype=np.float64)
    conflicts = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    mwis = solve_mwis_rydberg(weights, conflicts, seed=42)
    hw = export_qubo_hardware_payload(
        mwis["Q_sym"],
        problem_name="test_aquila_4",
        labels=["a", "b", "c", "d"],
        conflict_matrix=conflicts,
        selected_indices=mwis["selected_indices"],
        seed=42,
    )
    brk = hw["bqm_braket"]
    aq = hw["quera_aquila_register"]

    assert brk["format"] == "bqm_qubo_v1"
    assert brk["vartype"] == "BINARY"
    assert brk["num_variables"] == 4
    assert len(brk["linear"]) == 4
    assert len(brk["quadratic"]) == 3

    assert aq["rydberg_blockade_radius_um"] == pytest.approx(8.37)
    assert aq["observed_min_spacing_um"] >= 3.99
    assert 0.0 <= aq["unit_disk_fidelity"] <= 1.0
    assert aq["hardware_ready"] is True
    assert len(aq["atoms_um"]) == 4


def test_quantum_api_multiview_and_copilot_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYLEENTO3_LOCAL_3D_ENGINE", "0")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # 1. GET /api/quantum/demo
    demo_resp = client.get("/api/quantum/demo")
    assert demo_resp.status_code == 200
    demo_data = demo_resp.json()
    assert demo_data["seed"] == 42
    assert set(demo_data["scenarios"].keys()) == {"registration_stress", "nms_3d", "nbv_plan"}
    assert demo_data["scenarios"]["registration_stress"]["baseline_comparison"]["rotation_error_deg"] < 2.0
    assert demo_data["scenarios"]["nms_3d"]["baseline_comparison"]["energy_gain"] > 1.0

    # 2. POST /api/quantum/optimize for all 3 tasks + custom scene objects
    opt_reg = client.post(
        "/api/quantum/optimize",
        json={"task": "registration_stress", "n_correspondences": 60, "outlier_ratio": 0.70},
    )
    assert opt_reg.status_code == 200
    assert opt_reg.json()["baseline_comparison"]["rotation_error_deg"] < 2.0

    trap_objs = _demo_nms_greedy_trap_objects()[:5]
    opt_nms = client.post(
        "/api/quantum/optimize",
        json={"task": "nms_3d", "objects": trap_objs, "conflict_iou_threshold": 0.28},
    )
    assert opt_nms.status_code == 200
    assert opt_nms.json()["details"]["input_count"] == 5
    assert "trap_mega_box" not in set(opt_nms.json()["details"]["kept_ids"])

    opt_nbv = client.post(
        "/api/quantum/optimize",
        json={"task": "nbv_plan", "existing_azimuths_deg": [0.0, 40.0], "budget_k": 2},
    )
    assert opt_nbv.status_code == 200
    assert opt_nbv.json()["baseline_comparison"]["qubo_predicted_coverage_pct"] >= 90.0

    # 3. Multiview QUBO telemetry + toggle comparison
    mv_demo = client.get("/api/multiview-demo")
    assert mv_demo.status_code == 200
    mv_data = mv_demo.json()
    assert mv_data["quantum_telemetry"]["enabled"] is True
    assert mv_data["coverage_360"]["qubo_nbv_plan"]["budget_k"] >= 1
    assert mv_data["body_metrology"]["gt_volume_error_pct"] < 20.0

    mv_off = client.post(
        "/api/multiview-reconstruct",
        json={"use_demo": True, "use_qubo_optimizer": False},
    )
    assert mv_off.status_code == 200
    assert mv_off.json()["quantum_telemetry"]["enabled"] is False
    assert mv_off.json()["quantum_telemetry"]["views_optimized_by_qubo"] == 0

    # 4. Copilot tool #9 offline route
    cop_resp = client.post(
        "/api/copilot",
        json={
            "message": "Optimiza la escena con QUBO MWIS para eliminar colisiones",
            "objects": trap_objs,
            "width": 960,
            "height": 640,
            "fov_h_deg": 60.0,
        },
    )
    assert cop_resp.status_code == 200
    cop_data = cop_resp.json()
    assert any(t["name"] == "optimize_scene_qubo_mwis" for t in cop_data["tool_trace"])
    assert any(a["type"] == "highlight_subset" for a in cop_data["scene_actions"])
