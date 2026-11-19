"""FastAPI application exposing MyLeenTo3 3D vision, calibration, audit, and copilot endpoints."""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from app.core.geometry_3d import (
    compute_fov_v_deg,
    compute_occlusion_matrix,
    compute_pairwise_geometry,
    gemini_to_threejs_pose,
    project_obb_to_2d,
)
from app.core.metrics import evaluate_scene_predictions, fov_sensitivity_sweep
from app.core.multiframe_audit import (
    audit_multiframe_consistency,
    get_demo_multiframe_sequence,
)
from app.core.synthetic_benchmark import get_benchmark, list_benchmarks
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BenchmarksResponse,
    BenchmarkSummary,
    CompareModelsRequest,
    CompareModelsResponse,
    CopilotRequest,
    CopilotResponse,
    EnrichedObject,
    HealthResponse,
    MeasureRequest,
    MeasureResponse,
    ModelComparisonEntry,
    MultiFrameAuditRequest,
    ReprojectRequest,
    ReprojectResponse,
    SUPPORTED_MODELS,
)
from app.services.gemini_vision import (
    build_deterministic_fallback_detections,
    detect_3d_objects_live,
    infer_image_dimensions,
    resolve_api_key,
    resolve_thinking_level,
    strip_data_uri,
)
from app.services.spatial_agent import run_spatial_copilot

app = FastAPI(
    title="MyLeenTo3 — Monocular 3D Spatial Perception & Verification Lab",
    version="1.0.0",
    description="Calibrated monocular 9-DOF 3D vision API powered by Google AI Studio Interactions API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _enrich_object(obj: dict[str, Any], width: int, height: int, fov_h_deg: float) -> EnrichedObject:
    b = [float(v) for v in obj["box_3d"]]
    volume_m3 = round(b[3] * b[4] * b[5], 5)
    distance_m = round(math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2), 4)
    projected_2d = project_obb_to_2d(b, width, height, fov_h_deg)
    threejs_pose = gemini_to_threejs_pose(b)
    return EnrichedObject(
        id=str(obj["id"]),
        label=str(obj["label"]),
        box_3d=b,
        confidence=float(obj.get("confidence", 0.9)),
        volume_m3=volume_m3,
        distance_m=distance_m,
        projected_2d=projected_2d,
        threejs_pose=threejs_pose,
    )


def _extract_demo_prediction(benchmark: dict[str, Any], model: str) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    """Extract normalized (objects, latency_ms, usage) from benchmark demo_predictions_by_model."""
    demo_map = benchmark.get("demo_predictions_by_model", {})
    entry = demo_map.get(model) or demo_map.get("gemini-4-hydrogen")
    if isinstance(entry, dict):
        objs = entry.get("objects") or entry.get("predictions") or []
        latency_ms = float(entry.get("latency_ms", 120.0))
        raw_usage = dict(entry.get("usage", {}))
        inp = int(raw_usage.get("input_tokens", 1150) or 0)
        out = int(raw_usage.get("output_tokens", 420) or 0)
        thought = int(raw_usage.get("thought_tokens", raw_usage.get("thinking_tokens", 280)) or 0)
        total = int(raw_usage.get("total_tokens", inp + out + thought) or 0)
        usage = {
            "input_tokens": inp,
            "output_tokens": out,
            "thought_tokens": thought,
            "total_tokens": total,
        }
        return [dict(o) for o in objs], latency_ms, usage
    if isinstance(entry, list):
        default_latency = {
            "gemini-4-hydrogen": 1420.0,
            "gemini-4-argon": 780.0,
            "gemini-3.5-flash-lite": 310.0,
        }.get(model, 800.0)
        return (
            [dict(o) for o in entry],
            default_latency,
            {"input_tokens": 1120, "output_tokens": 390, "thought_tokens": 240, "total_tokens": 1750},
        )
    return [], 0.0, {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0}


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        gemini_env_configured=bool(os.getenv("GEMINI_API_KEY", "").strip()),
        supported_models=SUPPORTED_MODELS,
        sdk_version=getattr(genai, "__version__", "unknown"),
    )


@app.get("/api/benchmarks", response_model=BenchmarksResponse)
def get_benchmarks() -> BenchmarksResponse:
    items = list_benchmarks()
    summaries = [
        BenchmarkSummary(
            id=item["id"],
            title=item["title"],
            subtitle=item["subtitle"],
            description=item["description"],
            width=int(item["width"]),
            height=int(item["height"]),
            fov_h_deg=float(item["fov_h_deg"]),
            image_base64=item["image_base64"],
            gt_objects=item["gt_objects"],
        )
        for item in items
    ]
    return BenchmarksResponse(benchmarks=summaries)


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_scene(
    req: AnalyzeRequest,
    x_gemini_api_key: Optional[str] = Header(default=None, alias="X-Gemini-API-Key"),
) -> AnalyzeResponse:
    api_key = resolve_api_key(x_gemini_api_key)
    thinking_level = resolve_thinking_level(req.model, req.thinking_level)

    benchmark: Optional[dict[str, Any]] = None
    gt_raw: Optional[list[dict[str, Any]]] = None

    if req.benchmark_id:
        try:
            benchmark = get_benchmark(req.benchmark_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown benchmark_id: {req.benchmark_id}") from exc
        image_b64 = benchmark["image_base64"]
        mime_type = "image/png"
        width = int(req.width or benchmark["width"])
        height = int(req.height or benchmark["height"])
        gt_raw = benchmark["gt_objects"]
    elif req.image_base64:
        image_b64, mime_type = strip_data_uri(req.image_base64, default_mime=req.mime_type)
        inferred_w, inferred_h = infer_image_dimensions(image_b64)
        width = int(req.width or inferred_w)
        height = int(req.height or inferred_h)
    else:
        raise HTTPException(status_code=400, detail="Provide either 'benchmark_id' or 'image_base64'.")

    mode = "calibrated_demo"
    notice: Optional[str] = None
    scene_summary = ""
    raw_objects: list[dict[str, Any]] = []
    latency_ms = 0.0
    usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0}

    if api_key and not req.force_demo:
        try:
            live = detect_3d_objects_live(
                api_key=api_key,
                image_base64=image_b64,
                mime_type=mime_type,
                model=req.model,
                thinking_level=thinking_level,
                fov_h_deg=req.fov_h_deg,
                max_objects=req.max_objects,
                custom_prompt=req.custom_prompt,
            )
            raw_objects = live["objects"]
            scene_summary = live["scene_summary"]
            latency_ms = live["latency_ms"]
            usage = live["usage"]
            mode = "live_gemini"
            notice = None
        except Exception as exc:
            mode = "deterministic_fallback"
            notice = (
                f"Inferencia live con Google AI Studio falló ({type(exc).__name__}); "
                "se activó el modo de respaldo calibrado/determinista."
            )
            if benchmark is not None:
                raw_objects, latency_ms, usage = _extract_demo_prediction(benchmark, req.model)
                raw_objects = raw_objects[: req.max_objects]
                scene_summary = f"Respaldo calibrado para {benchmark['title']} ({req.model})."
            else:
                fb = build_deterministic_fallback_detections(
                    width=width,
                    height=height,
                    fov_h_deg=req.fov_h_deg,
                    model=req.model,
                    thinking_level=thinking_level,
                    max_objects=req.max_objects,
                )
                raw_objects = fb["objects"]
                scene_summary = fb["scene_summary"]
                latency_ms = fb["latency_ms"]
                usage = fb["usage"]
    else:
        if benchmark is not None:
            raw_objects, latency_ms, usage = _extract_demo_prediction(benchmark, req.model)
            raw_objects = raw_objects[: req.max_objects]
            mode = "calibrated_demo"
            scene_summary = f"Benchmark calibrado '{benchmark['title']}' evaluado con perfil {req.model}."
            if req.force_demo:
                notice = "Modo demo calibrado solicitado explícitamente (sin llamada externa a Google AI Studio)."
            else:
                notice = (
                    "Sin API Key de Google AI Studio configurada: ejecutando benchmark calibrado reproducible "
                    "(SEED=42). Ingresa tu clave en el botón API Key para inferencia live."
                )
        else:
            fb = build_deterministic_fallback_detections(
                width=width,
                height=height,
                fov_h_deg=req.fov_h_deg,
                model=req.model,
                thinking_level=thinking_level,
                max_objects=req.max_objects,
            )
            raw_objects = fb["objects"]
            scene_summary = fb["scene_summary"]
            latency_ms = fb["latency_ms"]
            usage = fb["usage"]
            mode = "deterministic_fallback"
            notice = (
                "Imagen personalizada analizada en modo determinista offline porque no se detectó "
                "X-Gemini-API-Key ni GEMINI_API_KEY."
            )

    fov_v_deg = round(compute_fov_v_deg(req.fov_h_deg, width, height), 4)
    enriched_objects = [_enrich_object(obj, width, height, req.fov_h_deg) for obj in raw_objects]
    enriched_gt = (
        [_enrich_object(obj, width, height, req.fov_h_deg) for obj in gt_raw]
        if gt_raw is not None
        else None
    )
    occlusion_matrix = compute_occlusion_matrix(raw_objects)

    metrics = None
    fov_sweep = None
    if gt_raw is not None:
        metrics = evaluate_scene_predictions(raw_objects, gt_raw, width, height, req.fov_h_deg)
        fov_sweep = fov_sensitivity_sweep(raw_objects, gt_raw, width, height)

    return AnalyzeResponse(
        benchmark_id=req.benchmark_id,
        model=req.model,
        thinking_level=thinking_level,
        mode=mode,
        notice=notice,
        width=width,
        height=height,
        fov_h_deg=req.fov_h_deg,
        fov_v_deg=fov_v_deg,
        image_base64=image_b64,
        mime_type=mime_type,
        scene_summary=scene_summary,
        objects=enriched_objects,
        gt_objects=enriched_gt,
        occlusion_matrix=occlusion_matrix,
        metrics=metrics,
        fov_sweep=fov_sweep,
        latency_ms=latency_ms,
        usage=usage,
    )


@app.post("/api/reproject", response_model=ReprojectResponse)
def reproject_scene(req: ReprojectRequest) -> ReprojectResponse:
    t0 = time.perf_counter()
    raw_objects = [obj.model_dump(exclude_none=True) for obj in req.objects]
    raw_gt = [obj.model_dump(exclude_none=True) for obj in req.gt_objects] if req.gt_objects is not None else None

    fov_v_deg = round(compute_fov_v_deg(req.fov_h_deg, req.width, req.height), 4)
    enriched_objects = [_enrich_object(obj, req.width, req.height, req.fov_h_deg) for obj in raw_objects]
    enriched_gt = (
        [_enrich_object(obj, req.width, req.height, req.fov_h_deg) for obj in raw_gt]
        if raw_gt is not None
        else None
    )
    occlusion_matrix = compute_occlusion_matrix(raw_objects)
    metrics = (
        evaluate_scene_predictions(raw_objects, raw_gt, req.width, req.height, req.fov_h_deg)
        if raw_gt is not None
        else None
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return ReprojectResponse(
        width=req.width,
        height=req.height,
        fov_h_deg=req.fov_h_deg,
        fov_v_deg=fov_v_deg,
        objects=enriched_objects,
        gt_objects=enriched_gt,
        occlusion_matrix=occlusion_matrix,
        metrics=metrics,
        latency_ms=latency_ms,
    )


@app.post("/api/measure", response_model=MeasureResponse)
def measure_pair(req: MeasureRequest) -> MeasureResponse:
    pair = compute_pairwise_geometry(req.object_a.box_3d, req.object_b.box_3d)
    pose_a = gemini_to_threejs_pose(req.object_a.box_3d)
    pose_b = gemini_to_threejs_pose(req.object_b.box_3d)
    return MeasureResponse(
        object_a_id=req.object_a.id,
        object_a_label=req.object_a.label,
        object_b_id=req.object_b.id,
        object_b_label=req.object_b.label,
        center_distance_m=pair["center_distance_m"],
        surface_clearance_m=pair["surface_clearance_m"],
        is_colliding=pair["is_colliding"],
        iou_3d=pair["iou_3d"],
        iou_bev=pair["iou_bev"],
        closest_point_a_three=pair["closest_point_a_three"],
        closest_point_b_three=pair["closest_point_b_three"],
        centroid_delta_m=pair["centroid_delta_m"],
        ruler_endpoints_three={
            "center_a": pose_a["position"],
            "center_b": pose_b["position"],
            "surface_a": pair["closest_point_a_three"],
            "surface_b": pair["closest_point_b_three"],
        },
    )


@app.post("/api/compare-models", response_model=CompareModelsResponse)
def compare_models(
    req: CompareModelsRequest,
    x_gemini_api_key: Optional[str] = Header(default=None, alias="X-Gemini-API-Key"),
) -> CompareModelsResponse:
    try:
        benchmark = get_benchmark(req.benchmark_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown benchmark_id: {req.benchmark_id}") from exc

    api_key = resolve_api_key(x_gemini_api_key)
    width = int(benchmark["width"])
    height = int(benchmark["height"])
    gt_raw = benchmark["gt_objects"]

    entries: list[ModelComparisonEntry] = []
    any_live = False
    any_fallback = False

    for model_name in SUPPORTED_MODELS:
        thinking_level = resolve_thinking_level(model_name, None)
        entry_mode = "calibrated_demo"
        entry_notice: Optional[str] = None

        if req.use_live and api_key:
            try:
                live = detect_3d_objects_live(
                    api_key=api_key,
                    image_base64=benchmark["image_base64"],
                    mime_type="image/png",
                    model=model_name,
                    thinking_level=thinking_level,
                    fov_h_deg=req.fov_h_deg,
                )
                preds = live["objects"]
                latency_ms = live["latency_ms"]
                usage = live["usage"]
                entry_mode = "live_gemini"
                any_live = True
            except Exception as exc:
                preds, latency_ms, usage = _extract_demo_prediction(benchmark, model_name)
                entry_mode = "deterministic_fallback"
                entry_notice = f"Fallback calibrado tras error live ({type(exc).__name__})."
                any_fallback = True
        else:
            preds, latency_ms, usage = _extract_demo_prediction(benchmark, model_name)

        metrics = evaluate_scene_predictions(preds, gt_raw, width, height, req.fov_h_deg)
        sweep = fov_sensitivity_sweep(preds, gt_raw, width, height)
        entries.append(
            ModelComparisonEntry(
                model=model_name,
                thinking_level=thinking_level,
                mode=entry_mode,
                notice=entry_notice,
                latency_ms=latency_ms,
                usage=usage,
                object_count=len(preds),
                metrics=metrics,
                fov_sweep=sweep,
            )
        )

    overall_mode = "live_gemini" if any_live and not any_fallback else (
        "deterministic_fallback" if any_fallback else "calibrated_demo"
    )
    overall_notice = None
    if req.use_live and not api_key:
        overall_notice = "use_live=True solicitado sin API Key; ejecutado sobre perfiles calibrados reproducibles."
    elif overall_mode == "calibrated_demo":
        overall_notice = "Comparativa reproducible (SEED=42) sobre los 3 perfiles oficiales de Gemini 4 / 3.5."

    primary_sweep = entries[0].fov_sweep if entries else []
    return CompareModelsResponse(
        benchmark_id=req.benchmark_id,
        fov_h_deg=req.fov_h_deg,
        mode=overall_mode,
        notice=overall_notice,
        results=entries,
        fov_sweep=primary_sweep,
    )


@app.get("/api/multiframe-demo")
def multiframe_demo() -> dict[str, Any]:
    sequence = get_demo_multiframe_sequence()
    audit = sequence.get("audit") or audit_multiframe_consistency(
        sequence["frames"],
        sequence_id=sequence.get("sequence_id", "world_model_corridor_audit"),
    )
    return {
        **sequence,
        "audit": audit,
        "mode": "calibrated_demo",
    }


@app.post("/api/multiframe-audit")
def multiframe_audit(
    req: MultiFrameAuditRequest,
    x_gemini_api_key: Optional[str] = Header(default=None, alias="X-Gemini-API-Key"),
) -> dict[str, Any]:
    api_key = resolve_api_key(x_gemini_api_key)
    processed_frames: list[dict[str, Any]] = []
    mode = "precomputed_frames"
    notice: Optional[str] = None

    for idx, frame in enumerate(req.frames):
        frame_copy = dict(frame)
        frame_copy.setdefault("frame_idx", idx)
        frame_copy.setdefault("timestamp_s", round(idx * 0.5, 2))
        if not frame_copy.get("objects") and frame_copy.get("image_base64"):
            if req.use_live and api_key:
                try:
                    live = detect_3d_objects_live(
                        api_key=api_key,
                        image_base64=frame_copy["image_base64"],
                        mime_type=frame_copy.get("mime_type", "image/png"),
                        model=req.model,
                        fov_h_deg=req.fov_h_deg,
                    )
                    frame_copy["objects"] = live["objects"]
                    mode = "live_gemini"
                except Exception as exc:
                    w, h = infer_image_dimensions(frame_copy["image_base64"])
                    fb = build_deterministic_fallback_detections(
                        width=w,
                        height=h,
                        fov_h_deg=req.fov_h_deg,
                        model=req.model,
                        thinking_level=resolve_thinking_level(req.model, None),
                    )
                    frame_copy["objects"] = fb["objects"]
                    mode = "deterministic_fallback"
                    notice = f"Frame {idx}: respaldo determinista ({type(exc).__name__})."
            else:
                w, h = infer_image_dimensions(frame_copy["image_base64"])
                fb = build_deterministic_fallback_detections(
                    width=w,
                    height=h,
                    fov_h_deg=req.fov_h_deg,
                    model=req.model,
                    thinking_level=resolve_thinking_level(req.model, None),
                )
                frame_copy["objects"] = fb["objects"]
                mode = "deterministic_fallback"
                notice = "Frames con imagen sin API Key analizados con respaldo determinista."
        processed_frames.append(frame_copy)

    seq_id = req.sequence_id or "custom_multiframe_audit"
    audit = audit_multiframe_consistency(processed_frames, sequence_id=seq_id)
    return {
        **audit,
        "audit": audit,
        "frames": processed_frames,
        "mode": mode,
        "notice": notice,
    }


@app.post("/api/copilot", response_model=CopilotResponse)
def spatial_copilot(
    req: CopilotRequest,
    x_gemini_api_key: Optional[str] = Header(default=None, alias="X-Gemini-API-Key"),
) -> CopilotResponse:
    api_key = resolve_api_key(x_gemini_api_key)
    raw_objects = [obj.model_dump(exclude_none=True) for obj in req.objects]
    raw_gt = [obj.model_dump(exclude_none=True) for obj in req.gt_objects] if req.gt_objects is not None else None

    result = run_spatial_copilot(
        message=req.message,
        objects=raw_objects,
        gt_objects=raw_gt,
        width=req.width,
        height=req.height,
        fov_h_deg=req.fov_h_deg,
        model=req.model,
        previous_interaction_id=req.previous_interaction_id,
        api_key=api_key,
    )
    return CopilotResponse(**result)
