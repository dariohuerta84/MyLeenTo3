"""Phase 1 tests: Hybrid VLM / Prompt-Free discovery, 2D-to-3D foreground lifting & 3D diagnosis."""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.local_metric_3d import (
    fuse_discovery_and_segmentation_candidates,
    lift_box2d_to_foreground_mask,
)
from app.services.scene_intelligence import (
    evaluate_scale_priors,
    infer_local_scene_discovery,
)

client = TestClient(app)


def _make_sample_png_base64(width: int = 320, height: int = 240) -> str:
    img = Image.new("RGB", (width, height), color=(215, 220, 226))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_infer_local_scene_discovery_domains() -> None:
    elec = infer_local_scene_discovery(
        [
            {"label": "electrical_switch", "conf": 0.62},
            {"label": "wire", "conf": 0.44},
        ],
        depth_stats={"min_m": 0.35, "p10_m": 0.45, "median_m": 0.68, "p90_m": 0.95, "max_m": 1.2},
    )
    assert elec["domain"] == "electrical_industrial"
    assert "electrical switch" in [c.lower() for c in elec["active_vocabulary"]]

    clin = infer_local_scene_discovery(
        [
            {"label": "skin", "conf": 0.72},
            {"label": "wound", "conf": 0.55},
        ],
        depth_stats={"min_m": 0.30, "p10_m": 0.40, "median_m": 0.52, "p90_m": 0.80, "max_m": 1.1},
    )
    assert clin["domain"] == "clinical_anatomical"

    desk = infer_local_scene_discovery(
        [
            {"label": "laptop", "conf": 0.81},
            {"label": "keyboard", "conf": 0.74},
        ],
        depth_stats={"min_m": 0.7, "p10_m": 0.9, "median_m": 1.4, "p90_m": 2.2, "max_m": 3.0},
    )
    assert desk["domain"] == "desk_electronics"

    logi = infer_local_scene_discovery(
        [
            {"label": "cardboard_box", "conf": 0.79},
            {"label": "pallet", "conf": 0.66},
        ],
    )
    assert logi["domain"] == "logistics_packaging"

    custom = infer_local_scene_discovery(
        [{"label": "cardboard_box", "conf": 0.30}],
        custom_prompt="circuit breaker, braided wire, multimeter",
    )
    assert custom["active_vocabulary"][0] == "circuit breaker"
    assert custom["active_vocabulary"][1] == "braided wire"
    assert custom["domain"] == "electrical_industrial"


def test_evaluate_scale_priors_consistent_and_biased() -> None:
    consistent_objs = [
        {"id": "o1", "label": "human_hand", "box_3d": [0.0, 0.55, 0.0, 0.18, 0.06, 0.10, 0.0, 0.0, 0.0]},
        {"id": "o2", "label": "laptop", "box_3d": [0.25, 0.90, -0.1, 0.36, 0.25, 0.03, 0.0, 0.0, 0.0]},
    ]
    checks_ok, note_ok = evaluate_scale_priors(consistent_objs)
    assert len(checks_ok) == 2
    assert all(c["status"] == "consistente" for c in checks_ok)
    assert "consistente" in note_ok.lower()

    biased_objs = [
        {"id": "o1", "label": "human_hand", "box_3d": [0.0, 1.5, 0.0, 0.58, 0.22, 0.30, 0.0, 0.0, 0.0]},
    ]
    checks_bad, note_bad = evaluate_scale_priors(biased_objs)
    assert len(checks_bad) == 1
    assert checks_bad[0]["status"] == "sobreestimada"
    assert checks_bad[0]["ratio"] > 2.0
    assert "sobreestimada" in note_bad.lower()

    closeup_partial_person = [
        {"id": "o_hand_crop", "label": "person", "box_3d": [0.0, 0.65, 0.0, 0.22, 0.14, 0.24, 0.0, 0.0, 0.0]},
    ]
    checks_closeup, note_closeup = evaluate_scale_priors(closeup_partial_person)
    assert checks_closeup == []
    assert "subestimada" not in note_closeup.lower()


def test_lift_box2d_to_foreground_mask_isolates_foreground_depth() -> None:
    pw, ph = 120, 90
    xs = np.linspace(-0.5, 0.5, pw, dtype=np.float32)
    ys = np.linspace(-0.4, 0.4, ph, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)

    # Background wall at Z = 3.0 m
    pts_cv = np.stack([gx * 3.0, gy * 3.0, np.full((ph, pw), 3.0, dtype=np.float32)], axis=-1)
    # Place foreground object in center patch at Z = 0.6 m
    pts_cv[24:64, 36:84, 0] = gx[24:64, 36:84] * 0.6
    pts_cv[24:64, 36:84, 1] = gy[24:64, 36:84] * 0.6
    pts_cv[24:64, 36:84, 2] = 0.6
    valid_mask = np.ones((ph, pw), dtype=bool)

    # Slightly generous normalized VLM bbox around center foreground [ymin, xmin, ymax, xmax] in 0..1000
    mask = lift_box2d_to_foreground_mask(pts_cv, valid_mask, [180, 220, 800, 780], pw, ph)
    assert mask is not None
    assert mask.shape == (ph, pw)
    assert int(np.sum(mask)) > 100
    # Foreground pixels inside ROI must be kept while Z=3.0m background pixels in the outer ring are rejected
    assert np.all(pts_cv[mask, 2] < 1.0)
    assert float(np.mean(pts_cv[mask, 2])) == pytest.approx(0.6, abs=0.05)


def test_fuse_discovery_and_segmentation_candidates_hybrid_and_local() -> None:
    pw, ph = 100, 100
    pts_cv = np.zeros((ph, pw, 3), dtype=np.float32)
    pts_cv[..., 2] = 2.8
    pts_cv[10:40, 10:40, 2] = 0.55
    pts_cv[55:85, 55:85, 2] = 0.72
    pts_cv[15:45, 55:85, 2] = 0.64
    valid_mask = np.ones((ph, pw), dtype=bool)

    ov_mask = np.zeros((ph, pw), dtype=bool)
    ov_mask[10:40, 10:40] = True
    ov_cands = [
        {"label": "hand", "conf": 0.65, "mask": ov_mask, "xyxy": [10.0, 10.0, 40.0, 40.0], "source": "yoloe_ov"}
    ]
    pf_dup_mask = np.zeros((ph, pw), dtype=bool)
    pf_dup_mask[11:39, 11:39] = True
    pf_novel_mask = np.zeros((ph, pw), dtype=bool)
    pf_novel_mask[58:82, 58:82] = True
    pf_cands = [
        {"label": "person", "conf": 0.55, "mask": pf_dup_mask, "xyxy": [11.0, 11.0, 39.0, 39.0], "source": "yoloe_pf_4585"},
        {"label": "wall_outlet", "conf": 0.34, "mask": pf_novel_mask, "xyxy": [58.0, 58.0, 82.0, 82.0], "source": "yoloe_pf_4585"},
    ]

    # Hybrid VLM mode: pairs VLM labels onto overlapping OV/PF masks, lifts unmatched third VLM box from depth
    vlm_objs = [
        {"label": "human hand", "box_2d": [100, 100, 400, 400], "confidence": 0.92},
        {"label": "wall outlet", "box_2d": [550, 550, 850, 850], "confidence": 0.88},
        {"label": "braided wire", "box_2d": [150, 550, 450, 850], "confidence": 0.84},
    ]
    hybrid = fuse_discovery_and_segmentation_candidates(
        ov_candidates=ov_cands,
        pf_candidates=pf_cands,
        vlm_objects_2d=vlm_objs,
        points_cv=pts_cv,
        valid_mask=valid_mask,
        proc_width=pw,
        proc_height=ph,
        max_objects=10,
    )
    assert len(hybrid) == 3
    assert hybrid[0]["label"] == "human_hand"
    assert hybrid[0]["source"] == "vlm_yoloe_fused"
    assert hybrid[1]["label"] == "wall_outlet"
    assert hybrid[1]["source"] == "vlm_yoloe_fused"
    assert hybrid[2]["label"] == "braided_wire"
    assert hybrid[2]["source"] == "vlm_depth_lifted"

    # Autonomous local mode: keeps OV, drops overlapping PF dup, appends non-overlapping PF novel
    local_fused = fuse_discovery_and_segmentation_candidates(
        ov_candidates=ov_cands,
        pf_candidates=pf_cands,
        vlm_objects_2d=[],
        points_cv=pts_cv,
        valid_mask=valid_mask,
        proc_width=pw,
        proc_height=ph,
        max_objects=10,
    )
    assert [c["label"] for c in local_fused] == ["hand", "wall_outlet"]
    assert [c["source"] for c in local_fused] == ["yoloe_ov", "yoloe_pf_4585"]


def test_analyze_custom_image_hybrid_vlm_moge2_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    b64 = _make_sample_png_base64(320, 240)

    vlm_disc_json = json.dumps(
        {
            "domain": "electrical_industrial",
            "domain_label_es": "Inspección Eléctrica e Industrial",
            "scene_context_es": "Manos manipulando un interruptor de pared y cables aislados.",
            "detected_classes": ["human hand", "electrical switch", "electrical wire"],
            "objects_2d": [
                {"label": "electrical switch", "box_2d": [200, 200, 700, 650], "typical_size_m": 0.12, "confidence": 0.91}
            ],
            "scale_priors": [
                {"label": "electrical switch", "typical_extent_m": 0.12, "rationale": "Caja estándar"}
            ],
            "inspection_focus_es": ["Medir holgura entre dedos y bornera."],
        }
    )
    diag_json = json.dumps(
        {
            "domain": "electrical_industrial",
            "domain_label_es": "Inspección Eléctrica e Industrial",
            "diagnosis_title": "Diagnóstico Eléctrico 3D Validado",
            "executive_summary": "Interruptor a 0.62 m con envolvente de 12.0x4.0x8.0 cm.",
            "model_predictions_es": "Escala métrica consistente (1.00x) respecto a caja eléctrica estándar.",
            "confidence_level": "alta",
            "scale_consistency_note": "Escala física consistente en 1/1 ancla verificada.",
            "key_findings": [
                {
                    "title": "Distancia de trabajo",
                    "metric_evidence": "0.62 m",
                    "severity": "info",
                    "interpretation": "Proximidad compatible con inspección manual.",
                }
            ],
            "recommended_actions": ["Verificar aislamiento con el cortador X/Y/Z."],
        }
    )

    inter_disc = SimpleNamespace(
        id="int_disc_1",
        output_text=vlm_disc_json,
        steps=[],
        usage=SimpleNamespace(total_input_tokens=300, total_output_tokens=110, total_thought_tokens=40, total_tokens=450),
    )
    inter_diag = SimpleNamespace(
        id="int_diag_1",
        output_text=diag_json,
        steps=[],
        usage=SimpleNamespace(total_input_tokens=420, total_output_tokens=160, total_thought_tokens=90, total_tokens=670),
    )

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = [inter_disc, inter_diag]

    mock_local_ret = {
        "mode": "hybrid_vlm_moge2_gpu",
        "model_desc": "moge-2-vits-normal + YOLOE-OV (Mock)",
        "width": 320,
        "height": 240,
        "image_base64": b64,
        "mime_type": "image/png",
        "effective_fov_h_deg": 52.0,
        "fov_source": "moge2_intrinsics",
        "exif_fov_h_deg": None,
        "estimated_moge_fov_h_deg": 52.0,
        "objects": [
            {
                "id": "moge_1",
                "label": "electrical_switch",
                "box_3d": [0.02, 0.62, -0.01, 0.12, 0.04, 0.08, 0.0, 0.0, 0.0],
                "confidence": 0.91,
            }
        ],
        "discovery_meta": {
            "discovery_mode": "vlm_gemini",
            "domain": "electrical_industrial",
            "domain_label_es": "Inspección Eléctrica e Industrial",
            "scene_context_es": "Manos manipulando un interruptor de pared.",
            "active_vocabulary": ["human hand", "electrical switch", "electrical wire"],
            "prompt_free_discoveries": [],
            "vlm_guided_count": 1,
            "detection_sources": ["vlm_yoloe_fused"],
            "scale_priors": [{"label": "electrical switch", "typical_extent_m": 0.12, "rationale": "Caja estándar"}],
            "inspection_focus_es": ["Medir holgura."],
        },
        "scene_summary": "Híbrido VLM + MoGe-2 mock.",
        "depth_map_base64": b64,
        "depth_stats": {"min_m": 0.45, "p10_m": 0.52, "median_m": 0.62, "p90_m": 0.95, "max_m": 1.40},
        "point_cloud": None,
        "real_photo_metrics": None,
        "latency_ms": 210.0,
    }

    with (
        patch("app.main.is_local_3d_available", return_value=True),
        patch("app.main.run_local_metric_3d", return_value=mock_local_ret) as mock_run_local,
        patch("app.services.scene_intelligence.genai.Client", return_value=mock_client),
    ):
        resp = client.post(
            "/api/analyze",
            headers={"X-Gemini-API-Key": "hybrid-secret-key"},
            json={"image_base64": b64, "fov_h_deg": 60.0, "engine": "hybrid_vlm_3d"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "hybrid_vlm_moge2_gpu"
    assert mock_run_local.call_args.kwargs["vlm_discovery"] is not None
    assert mock_run_local.call_args.kwargs["vlm_discovery"]["domain"] == "electrical_industrial"
    si = data["scene_intelligence"]
    assert si is not None
    assert si["discovery_mode"] == "vlm_gemini"
    assert si["vlm_guided_count"] == 1
    assert si["diagnosis"]["diagnosis_source"] == "vlm_3d_grounded"
    assert len(si["diagnosis"]["scale_prior_checks"]) == 1
    assert data["usage"]["total_tokens"] == 450 + 670


def test_analyze_hybrid_vlm_failure_no_key_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    b64 = _make_sample_png_base64(320, 240)
    secret = "hybrid-secret-should-never-leak-777"

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = RuntimeError(f"Auth failure with {secret}")

    mock_local_fallback = {
        "mode": "moge2_yoloe_gpu",
        "model_desc": "moge-2-vits-normal + YOLOE-OV (Mock)",
        "width": 320,
        "height": 240,
        "image_base64": b64,
        "mime_type": "image/png",
        "effective_fov_h_deg": 55.0,
        "fov_source": "moge2_intrinsics",
        "exif_fov_h_deg": None,
        "estimated_moge_fov_h_deg": 55.0,
        "objects": [
            {"id": "moge_1", "label": "laptop", "box_3d": [0.0, 0.9, 0.0, 0.35, 0.25, 0.04, 0.0, 0.0, 0.0], "confidence": 0.72}
        ],
        "scene_summary": "Fallback local.",
        "depth_map_base64": b64,
        "depth_stats": {"min_m": 0.6, "p10_m": 0.7, "median_m": 0.9, "p90_m": 1.2, "max_m": 1.6},
        "point_cloud": None,
        "real_photo_metrics": None,
        "latency_ms": 180.0,
    }

    with (
        patch("app.main.is_local_3d_available", return_value=True),
        patch("app.main.run_local_metric_3d", return_value=mock_local_fallback),
        patch("app.services.scene_intelligence.genai.Client", return_value=mock_client),
    ):
        resp = client.post(
            "/api/analyze",
            headers={"X-Gemini-API-Key": secret},
            json={"image_base64": b64, "engine": "auto"},
        )

    assert resp.status_code == 200
    body_text = json.dumps(resp.json())
    assert secret not in body_text
    si = resp.json()["scene_intelligence"]
    assert si is not None
    assert si["diagnosis"]["diagnosis_source"] == "deterministic_3d_grounded"
    assert si["vlm_fallback_reason"] == "RuntimeError"


def test_copilot_generate_domain_diagnosis_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    objs = [
        {"id": "obj_1", "label": "human_hand", "box_3d": [-0.05, 0.55, 0.0, 0.18, 0.06, 0.10, 0.0, 0.0, 5.0], "confidence": 0.88},
        {"id": "obj_2", "label": "electrical_switch", "box_3d": [0.04, 0.68, 0.02, 0.12, 0.04, 0.09, 0.0, 0.0, 0.0], "confidence": 0.84},
    ]
    resp = client.post(
        "/api/copilot",
        json={
            "message": "Genera diagnóstico 3D y verifica escala de esta imagen",
            "objects": objs,
            "width": 640,
            "height": 480,
            "fov_h_deg": 58.0,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "offline_demo"
    traces = [t for t in data["tool_trace"] if t["name"] == "generate_domain_diagnosis"]
    assert len(traces) == 1
    res = traces[0]["result"]
    assert res["domain"] == "electrical_industrial"
    assert len(res["scale_prior_checks"]) == 2
    assert len(res["diagnosis"]["key_findings"]) >= 1
    assert "Inspección Eléctrica e Industrial" in data["reply"]
