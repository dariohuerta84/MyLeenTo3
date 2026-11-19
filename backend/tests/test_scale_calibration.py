"""Phase 2 tests: scale calibration, 2D pinhole invariance, ArUco anchoring, surface caliper & MoGe backbone."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.geometry_3d import project_obb_to_2d
from app.core.scale_calibration import (
    SCALE_MAX,
    SCALE_MIN,
    compute_scale_factor_from_reference,
    compute_surface_caliper_metrics,
    detect_aruco_scale_anchor,
    rescale_box_3d,
)
from app.main import app

client = TestClient(app)


def _make_sample_png_base64(width: int = 320, height: int = 240) -> str:
    img = Image.new("RGB", (width, height), color=(215, 220, 226))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_rescale_box_3d_preserves_orientation_and_pinhole_projection() -> None:
    box = [0.24, 1.60, -0.12, 0.36, 0.28, 0.18, 2.5, -1.5, 22.0]
    proj_orig = project_obb_to_2d(box, 960, 640, 60.0)

    scaled = rescale_box_3d(box, 1.75)
    assert scaled[0] == pytest.approx(0.24 * 1.75, abs=1e-4)
    assert scaled[1] == pytest.approx(1.60 * 1.75, abs=1e-4)
    assert scaled[2] == pytest.approx(-0.12 * 1.75, abs=1e-4)
    assert scaled[3] == pytest.approx(0.36 * 1.75, abs=1e-4)
    assert scaled[4] == pytest.approx(0.28 * 1.75, abs=1e-4)
    assert scaled[5] == pytest.approx(0.18 * 1.75, abs=1e-4)
    assert scaled[6:] == [2.5, -1.5, 22.0]

    # Radial scaling from camera origin must preserve 2D pinhole center & projected silhouette invariant
    proj_scaled = project_obb_to_2d(scaled, 960, 640, 60.0)
    assert proj_scaled["center_2d"][0] == pytest.approx(proj_orig["center_2d"][0], abs=0.1)
    assert proj_scaled["center_2d"][1] == pytest.approx(proj_orig["center_2d"][1], abs=0.1)
    for k in range(4):
        assert proj_scaled["box_2d_reproj"][k] == pytest.approx(proj_orig["box_2d_reproj"][k], abs=0.2)


def test_compute_scale_factor_from_reference_and_clamping() -> None:
    res = compute_scale_factor_from_reference(0.10, 0.0856)
    assert res["scale_factor"] == pytest.approx(0.856, abs=1e-4)
    assert res["delta_pct"] == pytest.approx(-14.4, abs=0.1)
    assert res["clamped"] is False

    hi = compute_scale_factor_from_reference(0.02, 1.0)
    assert hi["scale_factor"] == pytest.approx(SCALE_MAX, abs=1e-5)
    assert hi["clamped"] is True

    lo = compute_scale_factor_from_reference(5.0, 0.1)
    assert lo["scale_factor"] == pytest.approx(SCALE_MIN, abs=1e-5)
    assert lo["clamped"] is True

    with pytest.raises(ValueError):
        compute_scale_factor_from_reference(0.0, 0.1)


def test_compute_surface_caliper_metrics_arc_and_profile_relief() -> None:
    pa = [0.0, 0.0, -1.0]
    pb = [0.20, 0.0, -1.0]
    # Raised ridge peaking 15mm toward camera (+Z in Three.js)
    samples = [
        [0.05, 0.0, -0.992],
        [0.10, 0.0, -0.985],
        [0.15, 0.0, -0.992],
    ]
    out = compute_surface_caliper_metrics(pa, pb, sample_points_three=samples)
    assert out["euclidean_m"] == pytest.approx(0.20, abs=1e-4)
    assert out["euclidean_cm"] == pytest.approx(20.0, abs=0.05)
    assert out["euclidean_mm"] == pytest.approx(200.0, abs=0.5)
    assert out["surface_arc_m"] > out["euclidean_m"]
    assert out["profile_relief_mm"] == pytest.approx(15.0, abs=0.5)
    assert out["slope_deg"] == pytest.approx(0.0, abs=0.1)


def test_detect_aruco_scale_anchor_synthetic_marker() -> None:
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_gray = cv2.aruco.generateImageMarker(aruco_dict, 7, 140)
    canvas = np.full((260, 260, 3), 245, dtype=np.uint8)
    canvas[60:200, 60:200] = np.stack([marker_gray] * 3, axis=-1)

    # Synthetic pinhole point map where 140px marker spans 0.10m side (while true marker = 0.05m -> factor ~0.50)
    h, w = 260, 260
    xs = (np.arange(w, dtype=np.float32) - 60.0) * (0.10 / 139.0)
    ys = (np.arange(h, dtype=np.float32) - 60.0) * (0.10 / 139.0)
    gx, gy = np.meshgrid(xs, ys)
    pts_cv = np.stack([gx, gy, np.full((h, w), 0.70, dtype=np.float32)], axis=-1)
    valid_mask = np.ones((h, w), dtype=bool)

    anchor = detect_aruco_scale_anchor(canvas, pts_cv, valid_mask, marker_size_m=0.05)
    assert anchor is not None
    assert anchor["detected"] is True
    assert anchor["dictionary"] == "DICT_4X4_50"
    assert anchor["marker_id"] == 7
    assert anchor["measured_side_m"] == pytest.approx(0.10, abs=0.008)
    assert anchor["suggested_scale_factor"] == pytest.approx(0.50, abs=0.04)


def test_calibrate_scale_endpoint_updates_objects_priors_and_depth_stats() -> None:
    # Overestimated laptop (measured 0.70m instead of 0.35m) -> calibrate with scale 0.50 or reference 0.0856/0.1712
    objs = [
        {
            "id": "moge_1",
            "label": "laptop",
            "box_3d": [0.20, 1.80, -0.10, 0.70, 0.50, 0.06, 0.0, 0.0, 12.0],
            "confidence": 0.91,
        }
    ]
    resp = client.post(
        "/api/calibrate-scale",
        json={
            "objects": objs,
            "width": 960,
            "height": 640,
            "fov_h_deg": 60.0,
            "measured_distance_m": 0.10,
            "reference_distance_m": 0.05,
            "reference_label": "ArUco 5 cm",
            "depth_stats": {"min_m": 1.0, "p10_m": 1.2, "median_m": 1.8, "p90_m": 2.4, "max_m": 3.0},
            "real_photo_metrics": {
                "ransac_plane": {"detected": True, "rmse_residual_m": 0.02, "plane_height_z_m": -0.30, "plane_orientation": "horizontal_support", "plane_tilt_deg": 2.0, "inlier_ratio": 0.7, "plane_normal_gemini": [0.0, 0.0, 1.0]},
                "per_object_metrology": [{"object_id": "moge_1", "label": "laptop", "visible_surface_area_m2": 0.40, "integrated_volume_above_plane_m3": 0.08, "max_relief_height_m": 0.12}],
            },
        },
    )
    assert resp.status_code == 200
    d = resp.json()
    assert d["scale_factor"] == pytest.approx(0.5, abs=1e-4)
    assert d["reference_label"] == "ArUco 5 cm"
    assert d["objects"][0]["box_3d"][1] == pytest.approx(0.90, abs=1e-3)
    assert d["objects"][0]["box_3d"][3] == pytest.approx(0.35, abs=1e-3)
    assert d["depth_stats"]["median_m"] == pytest.approx(0.9, abs=1e-3)
    assert d["real_photo_metrics"]["ransac_plane"]["plane_height_z_m"] == pytest.approx(-0.15, abs=1e-3)
    assert d["real_photo_metrics"]["per_object_metrology"][0]["visible_surface_area_m2"] == pytest.approx(0.10, abs=1e-4)
    assert d["real_photo_metrics"]["per_object_metrology"][0]["integrated_volume_above_plane_m3"] == pytest.approx(0.01, abs=1e-5)

    si = d["scene_intelligence"]
    assert si is not None
    checks = si["diagnosis"]["scale_prior_checks"]
    assert len(checks) == 1
    assert checks[0]["status"] == "consistente"


def test_analyze_accepts_moge_backbone_parameter_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    b64 = _make_sample_png_base64(320, 240)
    mock_local = {
        "mode": "moge2_yoloe_gpu",
        "model_desc": "moge-2-vitb-normal + YOLOE-OV (Mock)",
        "moge_backbone": "vitb",
        "aruco_anchor": None,
        "width": 320,
        "height": 240,
        "image_base64": b64,
        "mime_type": "image/png",
        "effective_fov_h_deg": 55.0,
        "fov_source": "moge2_intrinsics",
        "exif_fov_h_deg": None,
        "estimated_moge_fov_h_deg": 55.0,
        "objects": [
            {"id": "moge_1", "label": "laptop", "box_3d": [0.0, 0.9, 0.0, 0.35, 0.25, 0.04, 0.0, 0.0, 0.0], "confidence": 0.85}
        ],
        "scene_summary": "MoGe-2 ViT-B mock.",
        "depth_map_base64": b64,
        "depth_stats": {"min_m": 0.5, "p10_m": 0.6, "median_m": 0.9, "p90_m": 1.2, "max_m": 1.6},
        "point_cloud": None,
        "real_photo_metrics": None,
        "latency_ms": 220.0,
    }
    with (
        patch("app.main.is_local_3d_available", return_value=True),
        patch("app.main.run_local_metric_3d", return_value=mock_local) as mock_run,
    ):
        resp = client.post(
            "/api/analyze",
            json={"image_base64": b64, "engine": "moge2_local", "moge_backbone": "vitb", "aruco_marker_size_cm": 5.0},
        )
    assert resp.status_code == 200
    d = resp.json()
    assert mock_run.call_args.kwargs["moge_backbone"] == "vitb"
    assert mock_run.call_args.kwargs["aruco_marker_size_cm"] == 5.0
    assert d["moge_backbone"] == "vitb"
    assert d["scale_factor"] == 1.0
