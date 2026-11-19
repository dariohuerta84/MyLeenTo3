"""Comprehensive tests for Phase 3 multi-view 360-degree reconstruction & volumetric fusion."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from scipy.spatial.transform import Rotation

from app.core.multiview_fusion import (
    build_demo_multiview_360_frames,
    compute_360_coverage_and_nbv,
    fuse_multiview_frames,
    refine_point_to_plane_icp,
    umeyama_sim3,
)
from app.main import app

client = TestClient(app)


def _sample_textured_png_b64(seed_offset: int = 0) -> str:
    rng = np.random.default_rng(42 + seed_offset)
    arr = rng.integers(30, 220, size=(96, 128, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_umeyama_sim3_exact_recovery_and_proper_rotation() -> None:
    rng = np.random.default_rng(42)
    src = rng.uniform(-1.0, 1.0, size=(60, 3))
    true_R = Rotation.from_euler("zyx", [22.0, -14.0, 31.0], degrees=True).as_matrix()
    true_t = np.array([0.35, -0.18, 0.72], dtype=np.float64)
    true_s = 1.24
    dst = (true_s * (src @ true_R.T)) + true_t

    R_est, t_est, s_est = umeyama_sim3(src, dst, estimate_scale=True)
    assert abs(np.linalg.det(R_est) - 1.0) < 1e-6
    assert abs(s_est - true_s) < 1e-5
    assert np.allclose(R_est, true_R, atol=1e-5)
    assert np.allclose(t_est, true_t, atol=1e-5)

    # Without scale estimation s remains strictly 1.0 on rigid input
    dst_rigid = (src @ true_R.T) + true_t
    R_rig, t_rig, s_rig = umeyama_sim3(src, dst_rigid, estimate_scale=False)
    assert s_rig == 1.0
    assert np.allclose(R_rig, true_R, atol=1e-5)
    assert np.allclose(t_rig, true_t, atol=1e-5)


def test_refine_point_to_plane_icp_reduces_error() -> None:
    gx, gy = np.meshgrid(np.linspace(-0.3, 0.3, 28), np.linspace(-0.25, 0.25, 24))
    gz = 0.8 + 0.35 * (gx * gx + gy * gy) + 0.05 * np.sin(8.0 * gx) + 0.05 * np.cos(9.0 * gy)
    dst_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

    # Analytical normals for z = f(x,y) facing toward camera (-Z)
    dz_dx = 0.70 * gx.ravel() + 0.40 * np.cos(8.0 * gx.ravel())
    dz_dy = 0.70 * gy.ravel() - 0.45 * np.sin(9.0 * gy.ravel())
    nrm_raw = np.stack([dz_dx, dz_dy, -np.ones_like(dz_dx)], axis=-1)
    dst_nrm = nrm_raw / np.linalg.norm(nrm_raw, axis=-1, keepdims=True)

    # Perturb source pose by 6 degrees + 1.5 cm translation
    delta_R = Rotation.from_euler("xyz", [3.0, -5.0, 2.5], degrees=True).as_matrix()
    delta_t = np.array([0.014, -0.010, 0.012], dtype=np.float64)
    src_pts = (dst_pts - delta_t) @ delta_R
    src_nrm = dst_nrm @ delta_R

    R_out, t_out, s_out, rmse_m, overlap = refine_point_to_plane_icp(
        src_pts,
        src_nrm,
        dst_pts,
        dst_nrm,
        np.eye(3),
        np.zeros(3),
        1.0,
        max_iters=20,
        max_corr_dist_m=0.06,
    )
    assert s_out == 1.0
    assert abs(np.linalg.det(R_out) - 1.0) < 1e-6
    assert rmse_m < 0.008
    assert overlap > 0.70
    assert np.allclose(R_out, delta_R, atol=0.04)
    assert np.allclose(t_out, delta_t, atol=0.015)


def test_calibrated_4view_demo_360_fusion_and_buffers() -> None:
    frames, gt_vol = build_demo_multiview_360_frames(4)
    assert len(frames) == 4
    assert 0.025 < gt_vol < 0.035

    res = fuse_multiview_frames(frames, gt_volume_m3=gt_vol)
    assert res["view_count"] == 4
    assert len(res["camera_poses"]) == 4

    for pose in res["camera_poses"]:
        rot = Rotation.from_quat(pose["quaternion_three"]).as_matrix()
        assert abs(np.linalg.det(rot) - 1.0) < 1e-4

    met = res["body_metrology"]
    assert met["mean_icp_rmse_mm"] < 20.0
    assert met["gt_volume_error_pct"] is not None
    assert met["gt_volume_error_pct"] < 20.0
    assert met["convex_hull_volume_m3"] > gt_vol

    cov = res["coverage_360"]
    assert len(cov["sector_scores"]) == 12
    assert cov["angular_coverage_pct"] >= 75.0
    assert cov["is_watertight_360"] is True

    pc = res["point_cloud"]
    cnt = pc["count"]
    assert cnt > 5000
    assert len(base64.b64decode(pc["positions_b64"])) == cnt * 12
    assert len(base64.b64decode(pc["colors_b64"])) == cnt * 3
    assert len(base64.b64decode(pc["normals_b64"])) == cnt * 3
    assert len(base64.b64decode(pc["object_indices_b64"])) == cnt
    assert len(base64.b64decode(pc["relief_cm_b64"])) == cnt * 2

    mesh = pc["surface_mesh"]
    assert mesh is not None
    assert mesh["triangle_count"] > 1000
    assert len(base64.b64decode(mesh["positions_b64"])) == mesh["vertex_count"] * 12
    assert len(base64.b64decode(mesh["indices_b64"])) == mesh["triangle_count"] * 12


def test_coverage_360_identifies_wide_blind_gap() -> None:
    poses = [
        {"center_three": [0.0, 0.2, 1.0], "forward_three": [0.0, -0.2, -1.0]},
        {"center_three": [0.64, 0.2, 0.77], "forward_three": [-0.64, -0.2, -0.77]},
    ]
    cov = compute_360_coverage_and_nbv(poses, np.array([0.0, 0.0, 0.0]))
    assert cov["angular_coverage_pct"] < 65.0
    assert cov["largest_blind_gap_deg"] >= 180.0
    assert cov["is_watertight_360"] is False
    assert 90.0 <= cov["recommended_next_azimuth_deg"] <= 330.0
    assert "ActivePose NBV" in cov["next_best_view_advice_es"]


def test_multiview_api_endpoints(monkeypatch) -> None:
    monkeypatch.setenv("MYLEENTO3_LOCAL_3D_ENGINE", "0")

    demo_resp = client.get("/api/multiview-demo")
    assert demo_resp.status_code == 200
    demo_data = demo_resp.json()
    assert demo_data["mode"] == "multiview_calibrated_demo"
    assert demo_data["view_count"] == 4
    assert len(demo_data["objects"]) == 1
    assert demo_data["coverage_360"]["angular_coverage_pct"] >= 75.0

    post_demo = client.post("/api/multiview-reconstruct", json={"use_demo": True})
    assert post_demo.status_code == 200
    assert post_demo.json()["view_count"] == 4

    too_few = client.post(
        "/api/multiview-reconstruct",
        json={"images": [{"image_base64": _sample_textured_png_b64(0)}]},
    )
    assert too_few.status_code == 400

    too_many = client.post(
        "/api/multiview-reconstruct",
        json={"images": [{"image_base64": _sample_textured_png_b64(i)} for i in range(7)]},
    )
    assert too_many.status_code == 400

    offline_resp = client.post(
        "/api/multiview-reconstruct",
        json={
            "images": [
                {"image_base64": _sample_textured_png_b64(1), "label": "Frente"},
                {"image_base64": _sample_textured_png_b64(2), "label": "Derecha", "yaw_hint_deg": 65.0},
            ],
            "fov_h_deg": 60.0,
        },
    )
    assert offline_resp.status_code == 200
    off_data = offline_resp.json()
    assert off_data["mode"] == "multiview_offline_fallback"
    assert off_data["view_count"] == 2
    assert off_data["point_cloud"]["count"] > 500
    assert "recommended_next_azimuth_deg" in off_data["coverage_360"]
