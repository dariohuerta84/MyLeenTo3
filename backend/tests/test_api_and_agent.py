"""Comprehensive API & Spatial Copilot tests (all 9 endpoints + mocked GenAI Interactions API)."""

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

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.schemas import RawDetectionList, SUPPORTED_MODELS

client = TestClient(app)


def _make_sample_png_base64(width: int = 320, height: int = 240) -> str:
    img = Image.new("RGB", (width, height), color=(240, 242, 245))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sample_objects() -> list[dict]:
    return [
        {
            "id": "obj_1",
            "label": "oscilloscope",
            "box_3d": [-0.6, 2.0, -0.2, 0.4, 0.3, 0.3, 0.0, 0.0, 15.0],
            "confidence": 0.94,
        },
        {
            "id": "obj_2",
            "label": "laptop",
            "box_3d": [0.4, 2.6, -0.25, 0.36, 0.26, 0.22, 0.0, 0.0, -10.0],
            "confidence": 0.91,
        },
        {
            "id": "obj_3",
            "label": "spectrometer",
            "box_3d": [-0.58, 3.6, -0.15, 0.5, 0.4, 0.45, 0.0, 0.0, 5.0],
            "confidence": 0.88,
        },
    ]


def test_health_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["gemini_env_configured"] is False
    assert data["supported_models"] == SUPPORTED_MODELS
    assert data["sdk_version"] != "unknown"

    monkeypatch.setenv("GEMINI_API_KEY", "test-env-key")
    resp2 = client.get("/api/health")
    assert resp2.status_code == 200
    assert resp2.json()["gemini_env_configured"] is True


def test_benchmarks_endpoint() -> None:
    resp = client.get("/api/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    bms = data["benchmarks"]
    assert len(bms) == 3
    ids = {b["id"] for b in bms}
    assert ids == {"lab_desk_calibrated", "occlusion_corridor", "rotated_workshop"}
    for b in bms:
        assert b["width"] == 960
        assert b["height"] == 640
        assert b["fov_h_deg"] == 60.0
        assert len(b["image_base64"]) > 100
        assert len(b["gt_objects"]) >= 5


def test_analyze_benchmark_offline_calibrated_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    resp = client.post(
        "/api/analyze",
        json={"benchmark_id": "lab_desk_calibrated", "model": "gemini-4-hydrogen", "fov_h_deg": 60.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "calibrated_demo"
    assert data["model"] == "gemini-4-hydrogen"
    assert data["thinking_level"] == "high"
    assert data["notice"]
    assert len(data["objects"]) == 5
    assert data["gt_objects"] is not None and len(data["gt_objects"]) == 5
    assert data["metrics"] is not None
    assert "mIoU_3D" in data["metrics"]
    assert "bootstrap_ci_95" in data["metrics"]
    assert data["fov_sweep"] is not None and len(data["fov_sweep"]) > 0
    assert isinstance(data["occlusion_matrix"], list)
    first = data["objects"][0]
    assert "projected_2d" in first and len(first["projected_2d"]["vertices_2d"]) == 8
    assert "threejs_pose" in first and len(first["threejs_pose"]["quaternion"]) == 4
    assert first["volume_m3"] > 0
    assert first["distance_m"] > 0


def test_analyze_custom_image_offline_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    b64 = _make_sample_png_base64(480, 320)
    resp = client.post(
        "/api/analyze",
        json={"image_base64": f"data:image/png;base64,{b64}", "fov_h_deg": 65.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "deterministic_fallback"
    assert data["width"] == 480
    assert data["height"] == 320
    assert len(data["objects"]) >= 1
    assert data["metrics"] is None
    assert data["fov_sweep"] is not None and len(data["fov_sweep"]) == 12
    diag = data["intrinsic_diagnostics"]
    assert diag is not None
    assert "image_profile" in diag and "latent_metrics" in diag
    assert len(diag["per_object_diagnostics"]) == len(data["objects"])
    assert len(diag["exploited_weaknesses"]) >= 1
    assert "cascade_policy" in diag
    assert diag["cascade_policy"]["operational_verdict"] in {
        "APROBADO",
        "ESCALAR_MODELO",
        "RECHAZADO_FISICA",
    }


def test_analyze_validation_errors() -> None:
    assert client.post("/api/analyze", json={}).status_code == 400
    assert client.post("/api/analyze", json={"benchmark_id": "does_not_exist"}).status_code == 404
    assert (
        client.post(
            "/api/analyze",
            json={"benchmark_id": "lab_desk_calibrated", "model": "unsupported-model-x"},
        ).status_code
        == 422
    )


def test_analyze_live_mocked_interactions_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key-should-be-overridden")
    payload_json = json.dumps(
        {
            "objects": [
                {
                    "label": "oscilloscope",
                    "box_3d": [-0.62, 2.35, -0.29, 0.42, 0.28, 0.32, 0.0, 0.0, 18.0],
                    "confidence": 0.96,
                },
                {
                    "label": "laptop",
                    "box_3d": [0.15, 2.15, -0.35, 0.36, 0.26, 0.22, 0.0, 0.0, -8.0],
                    "confidence": 0.92,
                },
            ],
            "scene_summary": "Two tabletop lab instruments at 2.15m and 2.35m depth.",
        }
    )
    mock_interaction = SimpleNamespace(
        id="inter_vision_001",
        output_text=payload_json,
        steps=[],
        usage=SimpleNamespace(
            total_input_tokens=1100,
            total_output_tokens=210,
            total_thought_tokens=540,
            total_tokens=1850,
        ),
    )

    mock_client_instance = MagicMock()
    mock_client_instance.interactions.create.return_value = mock_interaction

    with patch("app.services.gemini_vision.genai.Client", return_value=mock_client_instance) as mock_ctor:
        resp = client.post(
            "/api/analyze",
            headers={"X-Gemini-API-Key": "header-priority-secret-key"},
            json={
                "benchmark_id": "lab_desk_calibrated",
                "model": "gemini-4-argon",
                "fov_h_deg": 60.0,
                "max_objects": 5,
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "live_gemini"
    assert data["notice"] is None
    assert data["model"] == "gemini-4-argon"
    assert data["thinking_level"] == "medium"
    assert len(data["objects"]) == 2
    assert data["usage"]["total_tokens"] == 1850

    # Verify header key took precedence over env var
    mock_ctor.assert_called_once_with(api_key="header-priority-secret-key")

    # Verify exact Interactions API v1 arguments
    call_kwargs = mock_client_instance.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-4-argon"
    assert call_kwargs["generation_config"] == {"temperature": 0.2, "thinking_level": "medium"}
    assert call_kwargs["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": RawDetectionList.model_json_schema(),
    }
    image_inputs = [item for item in call_kwargs["input"] if item.get("type") == "image"]
    assert len(image_inputs) == 1
    assert image_inputs[0]["mime_type"] == "image/png"
    assert len(image_inputs[0]["data"]) > 100


def test_analyze_live_failure_graceful_fallback_no_key_leak() -> None:
    secret = "super-secret-api-key-999"
    mock_client_instance = MagicMock()
    mock_client_instance.interactions.create.side_effect = RuntimeError(f"Auth failed for {secret}")

    with patch("app.services.gemini_vision.genai.Client", return_value=mock_client_instance):
        resp = client.post(
            "/api/analyze",
            headers={"X-Gemini-API-Key": secret},
            json={"benchmark_id": "lab_desk_calibrated", "model": "gemini-4-hydrogen"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "deterministic_fallback"
    assert len(data["objects"]) == 5
    assert secret not in (data["notice"] or "")


def test_reproject_endpoint_instant_and_fov_sensitive() -> None:
    objs = _sample_objects()
    r60 = client.post(
        "/api/reproject",
        json={"objects": objs, "gt_objects": objs, "width": 960, "height": 640, "fov_h_deg": 60.0},
    )
    r90 = client.post(
        "/api/reproject",
        json={"objects": objs, "gt_objects": objs, "width": 960, "height": 640, "fov_h_deg": 90.0},
    )
    assert r60.status_code == 200
    assert r90.status_code == 200
    d60, d90 = r60.json(), r90.json()
    assert d90["fov_v_deg"] > d60["fov_v_deg"]
    assert d60["metrics"]["mIoU_3D"] == pytest.approx(1.0, abs=1e-4)
    # Wider FOV pulls off-center points closer to principal point
    u60 = d60["objects"][0]["projected_2d"]["center_2d"][0]
    u90 = d90["objects"][0]["projected_2d"]["center_2d"][0]
    assert abs(u90 - 480.0) < abs(u60 - 480.0)
    assert isinstance(d60["occlusion_matrix"], list)


def test_measure_endpoint_separated_and_colliding() -> None:
    objs = _sample_objects()
    resp = client.post("/api/measure", json={"object_a": objs[0], "object_b": objs[1]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_colliding"] is False
    assert data["center_distance_m"] > 1.0
    assert data["surface_clearance_m"] > 0.0
    assert set(data["ruler_endpoints_three"].keys()) == {"center_a", "center_b", "surface_a", "surface_b"}

    # Overlapping pair -> colliding
    colliding_b = {
        "id": "obj_overlap",
        "label": "overlapping_box",
        "box_3d": [-0.55, 2.05, -0.2, 0.4, 0.3, 0.3, 0.0, 0.0, 15.0],
    }
    resp2 = client.post("/api/measure", json={"object_a": objs[0], "object_b": colliding_b})
    assert resp2.status_code == 200
    d2 = resp2.json()
    assert d2["is_colliding"] is True
    assert d2["iou_3d"] > 0.3


def test_compare_models_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    resp = client.post(
        "/api/compare-models",
        json={"benchmark_id": "occlusion_corridor", "fov_h_deg": 60.0, "use_live": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "calibrated_demo"
    assert len(data["results"]) == 3
    models = [r["model"] for r in data["results"]]
    assert models == SUPPORTED_MODELS
    assert len(data["fov_sweep"]) > 0
    # Flagship hydrogen should achieve higher mIoU_3D than flash-lite on calibrated benchmark
    miou_by_model = {r["model"]: r["metrics"]["mIoU_3D"] for r in data["results"]}
    assert miou_by_model["gemini-4-hydrogen"] >= miou_by_model["gemini-3.5-flash-lite"]


def test_multiframe_demo_and_custom_audit() -> None:
    demo = client.get("/api/multiframe-demo")
    assert demo.status_code == 200
    d = demo.json()
    assert len(d["frames"]) == 4
    audit = d["audit"]
    assert 0.0 <= audit["geometric_stability_score"] <= 100.0
    assert len(audit["object_tracks"]) >= 1
    assert len(audit["per_frame_summary"]) == 4

    # Custom audit POST
    post_resp = client.post(
        "/api/multiframe-audit",
        json={"sequence_id": "test_seq_42", "frames": d["frames"], "use_live": False},
    )
    assert post_resp.status_code == 200
    pd = post_resp.json()
    assert pd["sequence_id"] == "test_seq_42"
    assert pd["frame_count"] == 4
    assert pd["geometric_stability_score"] == pytest.approx(audit["geometric_stability_score"], abs=1e-4)


def test_copilot_offline_all_six_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    objs = _sample_objects()

    # 1. Measure nearest objects
    r1 = client.post(
        "/api/copilot",
        json={"message": "Mide la distancia entre los 2 objetos más cercanos", "objects": objs},
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["mode"] == "offline_demo"
    assert any(t["name"] == "measure_distance" for t in d1["tool_trace"])
    assert any(a["type"] == "show_measurement" for a in d1["scene_actions"])

    # 2. Occlusions & collisions
    r2 = client.post(
        "/api/copilot",
        json={"message": "Analiza oclusiones y colisiones en la escena", "objects": objs},
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert any(t["name"] == "analyze_occlusions_and_collisions" for t in d2["tool_trace"])
    assert any(a["type"] == "toggle_occlusion_rays" and a["enabled"] is True for a in d2["scene_actions"])

    # 3. Compound command: BEV camera view + focus farthest object
    r3 = client.post(
        "/api/copilot",
        json={"message": "Cambia a vista BEV y enfoca el objeto más lejano", "objects": objs},
    )
    assert r3.status_code == 200
    d3 = r3.json()
    tool_names = [t["name"] for t in d3["tool_trace"]]
    assert "set_camera_view" in tool_names
    assert "focus_object" in tool_names
    assert any(a == {"type": "set_camera_view", "view_mode": "bev"} for a in d3["scene_actions"])
    assert any(a == {"type": "focus_object", "object_id": "obj_3"} for a in d3["scene_actions"])

    # 4. Filter objects by depth
    r4 = client.post(
        "/api/copilot",
        json={"message": "Filtra objetos a menos de 2.7 m", "objects": objs},
    )
    assert r4.status_code == 200
    d4 = r4.json()
    assert any(t["name"] == "filter_objects" for t in d4["tool_trace"])
    hl = [a for a in d4["scene_actions"] if a["type"] == "highlight_subset"]
    assert len(hl) == 1
    assert set(hl[0]["object_ids"]) == {"obj_1", "obj_2"}

    # 5. FOV sensitivity sweep
    r5 = client.post(
        "/api/copilot",
        json={
            "message": "Ejecuta el barrido de sensibilidad FOV",
            "objects": objs,
            "gt_objects": objs,
            "width": 960,
            "height": 640,
            "fov_h_deg": 60.0,
        },
    )
    assert r5.status_code == 200
    d5 = r5.json()
    assert any(t["name"] == "run_fov_sensitivity_sweep" for t in d5["tool_trace"])
    assert any(a == {"type": "open_tab", "tab": "metrics"} for a in d5["scene_actions"])

    # 6. Escalation policy & Physical Reliability Gate (C8286 lineage)
    r6 = client.post(
        "/api/copilot",
        json={
            "message": "¿Conviene escalar el modelo o es confiable la escena?",
            "objects": objs,
            "width": 960,
            "height": 640,
            "fov_h_deg": 60.0,
        },
    )
    assert r6.status_code == 200
    d6 = r6.json()
    esc_traces = [t for t in d6["tool_trace"] if t["name"] == "evaluate_escalation_policy"]
    assert len(esc_traces) == 1
    assert esc_traces[0]["result"]["operational_verdict"] == "APROBADO"
    assert "APROBADO" in d6["reply"]


def test_copilot_live_multiturn_function_calling_loop() -> None:
    objs = _sample_objects()

    turn_1 = SimpleNamespace(
        id="int_turn_1",
        output_text=None,
        steps=[
            SimpleNamespace(
                type="function_call",
                id="call_measure_1",
                name="measure_distance",
                arguments={"object_a_id": "obj_1", "object_b_id": "obj_2"},
            ),
            SimpleNamespace(
                type="function_call",
                id="call_view_2",
                name="set_camera_view",
                arguments={"view_mode": "isometric"},
            ),
        ],
        usage=SimpleNamespace(
            total_input_tokens=400,
            total_output_tokens=60,
            total_thought_tokens=120,
            total_tokens=580,
        ),
    )
    turn_2 = SimpleNamespace(
        id="int_turn_2",
        output_text="La distancia exacta entre oscilloscope y laptop es 1.167 m y cambié a vista isométrica.",
        steps=[
            SimpleNamespace(
                type="model_output",
                content=[
                    SimpleNamespace(
                        type="text",
                        text="La distancia exacta entre oscilloscope y laptop es 1.167 m y cambié a vista isométrica.",
                    )
                ],
            )
        ],
        usage=SimpleNamespace(
            total_input_tokens=520,
            total_output_tokens=45,
            total_thought_tokens=80,
            total_tokens=645,
        ),
    )

    mock_client_instance = MagicMock()
    mock_client_instance.interactions.create.side_effect = [turn_1, turn_2]

    with patch("app.services.spatial_agent.genai.Client", return_value=mock_client_instance):
        resp = client.post(
            "/api/copilot",
            headers={"X-Gemini-API-Key": "live-test-key"},
            json={
                "message": "Mide oscilloscope vs laptop y pon vista isométrica",
                "objects": objs,
                "model": "gemini-4-hydrogen",
                "previous_interaction_id": "int_prev_0",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "live_gemini"
    assert data["interaction_id"] == "int_turn_2"
    assert "1.167" in data["reply"]
    assert len(data["tool_trace"]) == 2
    assert data["tool_trace"][0]["name"] == "measure_distance"
    assert data["tool_trace"][0]["result"]["center_distance_m"] > 1.0
    assert any(a["type"] == "show_measurement" for a in data["scene_actions"])
    assert any(a == {"type": "set_camera_view", "view_mode": "isometric"} for a in data["scene_actions"])
    assert data["usage"]["total_tokens"] == 580 + 645

    # Verify both Interactions API calls re-specified tools, system_instruction, generation_config, model
    assert mock_client_instance.interactions.create.call_count == 2
    call1_kw = mock_client_instance.interactions.create.call_args_list[0].kwargs
    call2_kw = mock_client_instance.interactions.create.call_args_list[1].kwargs

    assert call1_kw["previous_interaction_id"] == "int_prev_0"
    assert call2_kw["previous_interaction_id"] == "int_turn_1"
    for kw in (call1_kw, call2_kw):
        assert kw["model"] == "gemini-4-hydrogen"
        assert "tools" in kw and len(kw["tools"]) == 7
        assert "system_instruction" in kw
        assert kw["generation_config"]["thinking_level"] == "high"

    # Verify function_result payload format in Turn 2
    fn_results = call2_kw["input"]
    assert len(fn_results) == 2
    assert fn_results[0]["type"] == "function_result"
    assert fn_results[0]["call_id"] == "call_measure_1"
    assert fn_results[0]["name"] == "measure_distance"
    parsed_res = json.loads(fn_results[0]["result"][0]["text"])
    assert parsed_res["found"] is True
    assert fn_results[1]["call_id"] == "call_view_2"
