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
from app.core.intrinsic_diagnostics import compute_intrinsic_scene_diagnostics
from app.core.metrics import evaluate_scene_predictions, fov_sensitivity_sweep
from app.core.multiframe_audit import (
    audit_multiframe_consistency,
    get_demo_multiframe_sequence,
)
from app.core.scale_calibration import (
    clamp_scale_factor,
    compute_scale_factor_from_reference,
    rescale_box_3d,
    rescale_depth_stats,
    rescale_real_photo_metrics,
)
from app.core.synthetic_benchmark import get_benchmark, list_benchmarks
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BenchmarksResponse,
    BenchmarkSummary,
    CalibrateScaleRequest,
    CalibrateScaleResponse,
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
from app.services.local_metric_3d import (
    extract_depth_and_point_cloud,
    get_local_3d_status,
    is_local_3d_available,
    run_local_metric_3d,
)
from app.services.scene_intelligence import (
    build_scene_intelligence_payload,
    discover_scene_vocabulary_vlm,
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
    local_status = get_local_3d_status()
    return HealthResponse(
        status="ok",
        gemini_env_configured=bool(os.getenv("GEMINI_API_KEY", "").strip()),
        supported_models=SUPPORTED_MODELS,
        sdk_version=getattr(genai, "__version__", "unknown"),
        local_metric_3d_available=bool(local_status["available"]),
        local_3d_device=str(local_status["device"]),
        local_3d_models=list(local_status["models"]),
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


def _merge_token_usage(acc: dict[str, Any], extra: Optional[dict[str, Any]]) -> dict[str, int]:
    """Accumulate token counters across discovery, 3D detection, and grounded diagnosis calls."""
    if not extra:
        return {
            "input_tokens": int(acc.get("input_tokens", 0)),
            "output_tokens": int(acc.get("output_tokens", 0)),
            "thought_tokens": int(acc.get("thought_tokens", 0)),
            "total_tokens": int(acc.get("total_tokens", 0)),
        }
    return {
        "input_tokens": int(acc.get("input_tokens", 0)) + int(extra.get("input_tokens", 0)),
        "output_tokens": int(acc.get("output_tokens", 0)) + int(extra.get("output_tokens", 0)),
        "thought_tokens": int(acc.get("thought_tokens", 0)) + int(extra.get("thought_tokens", 0)),
        "total_tokens": int(acc.get("total_tokens", 0)) + int(extra.get("total_tokens", 0)),
    }


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
    effective_fov_h_deg = float(req.fov_h_deg)
    fov_source = "manual"
    exif_fov_h_deg: Optional[float] = None
    estimated_moge_fov_h_deg: Optional[float] = None
    depth_map_base64: Optional[str] = None
    depth_stats: Optional[dict[str, float]] = None
    point_cloud: Optional[dict[str, Any]] = None
    real_photo_metrics: Optional[dict[str, Any]] = None
    discovery_meta: Optional[dict[str, Any]] = None
    active_moge_backbone: Optional[str] = None
    aruco_anchor: Optional[dict[str, Any]] = None

    use_local_depth = req.engine != "salience_fallback" and is_local_3d_available()

    def _apply_local_result(local_res: dict[str, Any]) -> None:
        nonlocal image_b64, mime_type, width, height, effective_fov_h_deg, fov_source
        nonlocal exif_fov_h_deg, estimated_moge_fov_h_deg, raw_objects, scene_summary
        nonlocal depth_map_base64, depth_stats, point_cloud, real_photo_metrics
        nonlocal latency_ms, mode, discovery_meta, active_moge_backbone, aruco_anchor

        image_b64 = local_res["image_base64"]
        mime_type = local_res["mime_type"]
        width = int(local_res["width"])
        height = int(local_res["height"])
        effective_fov_h_deg = float(local_res["effective_fov_h_deg"])
        fov_source = str(local_res["fov_source"])
        exif_fov_h_deg = local_res["exif_fov_h_deg"]
        estimated_moge_fov_h_deg = local_res["estimated_moge_fov_h_deg"]
        raw_objects = local_res["objects"]
        scene_summary = local_res["scene_summary"]
        depth_map_base64 = local_res["depth_map_base64"]
        depth_stats = local_res["depth_stats"]
        point_cloud = local_res["point_cloud"]
        real_photo_metrics = local_res.get("real_photo_metrics")
        discovery_meta = local_res.get("discovery_meta")
        active_moge_backbone = local_res.get("moge_backbone")
        aruco_anchor = local_res.get("aruco_anchor")
        latency_ms = local_res["latency_ms"]
        mode = local_res["mode"]

    def _run_salience_fallback(reason_notice: str) -> None:
        nonlocal raw_objects, scene_summary, latency_ms, usage, mode, notice
        fb = build_deterministic_fallback_detections(
            width=width,
            height=height,
            fov_h_deg=effective_fov_h_deg,
            model=req.model,
            thinking_level=thinking_level,
            max_objects=req.max_objects,
            image_base64=image_b64,
        )
        raw_objects = fb["objects"]
        scene_summary = fb["scene_summary"]
        latency_ms = fb["latency_ms"]
        usage = fb["usage"]
        mode = "deterministic_fallback"
        notice = reason_notice

    if benchmark is not None:
        # Calibrated benchmarks preserve exact single-call Gemini Live behavior or deterministic demo
        if api_key and req.engine in ("auto", "gemini_live", "hybrid_vlm_3d") and not req.force_demo:
            try:
                live = detect_3d_objects_live(
                    api_key=api_key,
                    image_base64=image_b64,
                    mime_type=mime_type,
                    model=req.model,
                    thinking_level=thinking_level,
                    fov_h_deg=effective_fov_h_deg,
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
                raw_objects, latency_ms, usage = _extract_demo_prediction(benchmark, req.model)
                raw_objects = raw_objects[: req.max_objects]
                scene_summary = f"Respaldo calibrado para {benchmark['title']} ({req.model})."
        else:
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
    elif api_key and use_local_depth and req.engine in ("auto", "hybrid_vlm_3d") and not req.force_demo:
        # Case A: Hybrid VLM Scene Discovery -> Open-Vocab YOLOE-OV -> Metric MoGe-2 3D Lifting
        vlm_discovery: Optional[dict[str, Any]] = None
        vlm_notice = ""
        try:
            vlm_discovery = discover_scene_vocabulary_vlm(
                api_key=api_key,
                image_base64=image_b64,
                mime_type=mime_type,
                model=req.model,
                custom_prompt=req.custom_prompt,
            )
            usage = _merge_token_usage(usage, vlm_discovery.get("usage"))
        except Exception as exc:
            vlm_discovery = None
            vlm_notice = f"Descubrimiento VLM omitido ({type(exc).__name__}); "

        try:
            local_res = run_local_metric_3d(
                image_b64,
                fov_override_deg=effective_fov_h_deg,
                auto_fov=req.auto_fov,
                max_objects=req.max_objects,
                custom_prompt=req.custom_prompt,
                vlm_discovery=vlm_discovery,
                moge_backbone=req.moge_backbone,
                aruco_marker_size_cm=req.aruco_marker_size_cm,
            )
            _apply_local_result(local_res)
            d_min = float((depth_stats or {}).get("min_m", 0.0))
            d_max = float((depth_stats or {}).get("max_m", 0.0))
            if vlm_discovery is not None:
                latency_ms = round(float(latency_ms) + float(vlm_discovery.get("latency_ms", 0.0)), 2)
                notice = (
                    f"Pipeline Híbrido VLM → YOLOE-OV → MoGe-2 activo ({local_res['model_desc']}): "
                    f"rango {d_min:.2f}–{d_max:.2f} m "
                    f"(FOV_h={effective_fov_h_deg:.1f}° vía {fov_source})."
                )
            else:
                notice = (
                    f"{vlm_notice}Reconstrucción métrica local real en {local_res['model_desc']}: "
                    f"rango {d_min:.2f}–{d_max:.2f} m "
                    f"(FOV_h={effective_fov_h_deg:.1f}° vía {fov_source})."
                )
        except Exception as exc:
            _run_salience_fallback(
                f"Motor híbrido VLM + MoGe-2 falló ({type(exc).__name__}); usando extractor de saliencia sin referencia."
            )
    elif api_key and not req.force_demo and (
        req.engine == "gemini_live" or (req.engine in ("auto", "hybrid_vlm_3d") and not use_local_depth)
    ):
        # Case B: Gemini Live 9-DOF inference (optionally enriched with MoGe-2 depth/surfels)
        if use_local_depth:
            try:
                depth_ctx = extract_depth_and_point_cloud(
                    image_b64,
                    fov_override_deg=effective_fov_h_deg,
                    auto_fov=req.auto_fov,
                    moge_backbone=req.moge_backbone,
                    aruco_marker_size_cm=req.aruco_marker_size_cm,
                )
                effective_fov_h_deg = float(depth_ctx["effective_fov_h_deg"])
                fov_source = str(depth_ctx["fov_source"])
                exif_fov_h_deg = depth_ctx["exif_fov_h_deg"]
                estimated_moge_fov_h_deg = depth_ctx["estimated_moge_fov_h_deg"]
                depth_map_base64 = depth_ctx["depth_map_base64"]
                depth_stats = depth_ctx["depth_stats"]
                point_cloud = depth_ctx["point_cloud"]
                real_photo_metrics = depth_ctx.get("real_photo_metrics")
                active_moge_backbone = depth_ctx.get("moge_backbone")
                aruco_anchor = depth_ctx.get("aruco_anchor")
            except Exception:
                pass
        try:
            live = detect_3d_objects_live(
                api_key=api_key,
                image_base64=image_b64,
                mime_type=mime_type,
                model=req.model,
                thinking_level=thinking_level,
                fov_h_deg=effective_fov_h_deg,
                max_objects=req.max_objects,
                custom_prompt=req.custom_prompt,
            )
            raw_objects = live["objects"]
            scene_summary = live["scene_summary"]
            latency_ms = live["latency_ms"]
            usage = _merge_token_usage(usage, live["usage"])
            mode = "live_gemini"
            notice = None
        except Exception as exc:
            if use_local_depth:
                try:
                    local_res = run_local_metric_3d(
                        image_b64,
                        fov_override_deg=effective_fov_h_deg,
                        auto_fov=req.auto_fov,
                        max_objects=req.max_objects,
                        custom_prompt=req.custom_prompt,
                        moge_backbone=req.moge_backbone,
                        aruco_marker_size_cm=req.aruco_marker_size_cm,
                    )
                    _apply_local_result(local_res)
                    notice = (
                        f"Google AI Studio live no disponible ({type(exc).__name__}); "
                        f"reconstrucción métrica local activa vía {local_res['model_desc']}."
                    )
                except Exception:
                    _run_salience_fallback(
                        f"Inferencia live con Google AI Studio falló ({type(exc).__name__}); se activó el modo de respaldo determinista."
                    )
            else:
                _run_salience_fallback(
                    f"Inferencia live con Google AI Studio falló ({type(exc).__name__}); se activó el modo de respaldo determinista."
                )
    elif use_local_depth and not req.force_demo and req.engine in ("auto", "hybrid_vlm_3d", "moge2_local"):
        # Case C: Autonomous local dual-pass GPU/CPU (YOLOE-PF 4585 -> Domain -> YOLOE-OV -> MoGe-2)
        try:
            local_res = run_local_metric_3d(
                image_b64,
                fov_override_deg=effective_fov_h_deg,
                auto_fov=req.auto_fov,
                max_objects=req.max_objects,
                custom_prompt=req.custom_prompt,
                moge_backbone=req.moge_backbone,
                aruco_marker_size_cm=req.aruco_marker_size_cm,
            )
            _apply_local_result(local_res)
            d_min = float((depth_stats or {}).get("min_m", 0.0))
            d_max = float((depth_stats or {}).get("max_m", 0.0))
            notice = (
                f"Reconstrucción métrica local real en {local_res['model_desc']}: "
                f"rango {d_min:.2f}–{d_max:.2f} m "
                f"(FOV_h={effective_fov_h_deg:.1f}° vía {fov_source})."
            )
        except Exception as exc:
            _run_salience_fallback(
                f"Motor métrico local MoGe-2 falló ({type(exc).__name__}); usando extractor de saliencia sin referencia."
            )
    else:
        # Case D: Deterministic salience fallback without reference
        _run_salience_fallback(
            "Imagen personalizada analizada con extracción adaptativa de saliencia + diagnóstico intrínseco "
            "sin referencia (sin X-Gemini-API-Key / GEMINI_API_KEY)."
        )

    fov_v_deg = round(compute_fov_v_deg(effective_fov_h_deg, width, height), 4)
    enriched_objects = [_enrich_object(obj, width, height, effective_fov_h_deg) for obj in raw_objects]
    enriched_gt = (
        [_enrich_object(obj, width, height, effective_fov_h_deg) for obj in gt_raw]
        if gt_raw is not None
        else None
    )
    occlusion_matrix = compute_occlusion_matrix(raw_objects)
    intrinsic_diagnostics = compute_intrinsic_scene_diagnostics(
        raw_objects,
        image_b64,
        width,
        height,
        effective_fov_h_deg,
        occlusion_matrix=occlusion_matrix,
        real_photo_metrics=real_photo_metrics,
    )

    scene_intelligence: Optional[dict[str, Any]] = None
    if req.enable_diagnosis:
        allow_vlm_diag = (
            benchmark is None
            and bool(api_key)
            and not req.force_demo
            and req.engine in ("auto", "hybrid_vlm_3d", "gemini_live")
        )
        scene_intelligence, diag_usage = build_scene_intelligence_payload(
            discovery=discovery_meta,
            objects=raw_objects,
            occlusion_matrix=occlusion_matrix,
            intrinsic_diagnostics=intrinsic_diagnostics,
            real_photo_metrics=real_photo_metrics,
            depth_stats=depth_stats,
            point_cloud=point_cloud,
            custom_prompt=req.custom_prompt,
            image_base64=image_b64,
            mime_type=mime_type,
            model=req.model,
            api_key=api_key,
            allow_live_vlm=allow_vlm_diag,
        )
        usage = _merge_token_usage(usage, diag_usage)

    metrics = None
    fov_sweep = intrinsic_diagnostics["reference_free_fov_sweep"]
    if gt_raw is not None:
        metrics = evaluate_scene_predictions(raw_objects, gt_raw, width, height, effective_fov_h_deg)
        fov_sweep = fov_sensitivity_sweep(raw_objects, gt_raw, width, height)

    return AnalyzeResponse(
        benchmark_id=req.benchmark_id,
        model=req.model,
        thinking_level=thinking_level,
        mode=mode,
        notice=notice,
        width=width,
        height=height,
        fov_h_deg=effective_fov_h_deg,
        fov_v_deg=fov_v_deg,
        fov_source=fov_source,
        exif_fov_h_deg=exif_fov_h_deg,
        estimated_moge_fov_h_deg=estimated_moge_fov_h_deg,
        moge_backbone=active_moge_backbone,
        scale_factor=1.0,
        aruco_anchor=aruco_anchor,
        image_base64=image_b64,
        mime_type=mime_type,
        scene_summary=scene_summary,
        objects=enriched_objects,
        gt_objects=enriched_gt,
        occlusion_matrix=occlusion_matrix,
        metrics=metrics,
        fov_sweep=fov_sweep,
        intrinsic_diagnostics=intrinsic_diagnostics,
        depth_map_base64=depth_map_base64,
        depth_stats=depth_stats,
        point_cloud=point_cloud,
        real_photo_metrics=real_photo_metrics,
        scene_intelligence=scene_intelligence,
        latency_ms=latency_ms,
        usage=usage,
    )


@app.post("/api/reproject", response_model=ReprojectResponse)
def reproject_scene(req: ReprojectRequest) -> ReprojectResponse:
    t0 = time.perf_counter()
    raw_objects = [obj.model_dump(exclude_none=True) for obj in req.objects]
    if req.rescale_depth_with_fov and req.reference_fov_h_deg is not None:
        ref_tan = math.tan(math.radians(float(req.reference_fov_h_deg)) * 0.5)
        cur_tan = max(math.tan(math.radians(float(req.fov_h_deg)) * 0.5), 1e-4)
        depth_scale = ref_tan / cur_tan
        for obj in raw_objects:
            box = list(obj.get("box_3d", []))
            if len(box) >= 6:
                for k in range(6):
                    box[k] = round(float(box[k]) * depth_scale, 4)
                obj["box_3d"] = box
    raw_gt = [obj.model_dump(exclude_none=True) for obj in req.gt_objects] if req.gt_objects is not None else None

    fov_v_deg = round(compute_fov_v_deg(req.fov_h_deg, req.width, req.height), 4)
    enriched_objects = [_enrich_object(obj, req.width, req.height, req.fov_h_deg) for obj in raw_objects]
    enriched_gt = (
        [_enrich_object(obj, req.width, req.height, req.fov_h_deg) for obj in raw_gt]
        if raw_gt is not None
        else None
    )
    occlusion_matrix = compute_occlusion_matrix(raw_objects)
    intrinsic_diagnostics = compute_intrinsic_scene_diagnostics(
        raw_objects,
        req.image_base64,
        req.width,
        req.height,
        req.fov_h_deg,
        occlusion_matrix=occlusion_matrix,
    )
    metrics = (
        evaluate_scene_predictions(raw_objects, raw_gt, req.width, req.height, req.fov_h_deg)
        if raw_gt is not None
        else None
    )
    fov_sweep = (
        fov_sensitivity_sweep(raw_objects, raw_gt, req.width, req.height)
        if raw_gt is not None
        else intrinsic_diagnostics["reference_free_fov_sweep"]
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
        fov_sweep=fov_sweep,
        intrinsic_diagnostics=intrinsic_diagnostics,
        latency_ms=latency_ms,
    )


@app.post("/api/calibrate-scale", response_model=CalibrateScaleResponse)
def calibrate_scene_scale(req: CalibrateScaleRequest) -> CalibrateScaleResponse:
    t0 = time.perf_counter()
    if req.scale_factor is not None:
        factor = round(clamp_scale_factor(req.scale_factor), 5)
    elif req.measured_distance_m is not None and req.reference_distance_m is not None:
        try:
            factor = compute_scale_factor_from_reference(
                req.measured_distance_m, req.reference_distance_m
            )["scale_factor"]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'scale_factor' or both 'measured_distance_m' and 'reference_distance_m'.",
        )

    raw_objects = []
    for obj in req.objects:
        d = obj.model_dump(exclude_none=True)
        d["box_3d"] = rescale_box_3d(d["box_3d"], factor)
        raw_objects.append(d)

    raw_gt = (
        [obj.model_dump(exclude_none=True) for obj in req.gt_objects]
        if req.gt_objects is not None
        else None
    )

    scaled_depth_stats = rescale_depth_stats(req.depth_stats, factor)
    scaled_real_metrics = rescale_real_photo_metrics(req.real_photo_metrics, factor)

    fov_v_deg = round(compute_fov_v_deg(req.fov_h_deg, req.width, req.height), 4)
    enriched_objects = [_enrich_object(obj, req.width, req.height, req.fov_h_deg) for obj in raw_objects]
    enriched_gt = (
        [_enrich_object(obj, req.width, req.height, req.fov_h_deg) for obj in raw_gt]
        if raw_gt is not None
        else None
    )
    occlusion_matrix = compute_occlusion_matrix(raw_objects)
    intrinsic_diagnostics = compute_intrinsic_scene_diagnostics(
        raw_objects,
        req.image_base64,
        req.width,
        req.height,
        req.fov_h_deg,
        occlusion_matrix=occlusion_matrix,
        real_photo_metrics=scaled_real_metrics,
    )
    metrics = (
        evaluate_scene_predictions(raw_objects, raw_gt, req.width, req.height, req.fov_h_deg)
        if raw_gt is not None
        else None
    )

    discovery_hint: Optional[dict[str, Any]] = None
    if req.scene_intelligence and isinstance(req.scene_intelligence, dict):
        si = req.scene_intelligence
        discovery_hint = {
            "discovery_mode": si.get("discovery_mode", "local_yoloe_pf_adaptive"),
            "domain": si.get("domain", "general_physical"),
            "domain_label_es": si.get("domain_label_es", "Inspección Física General"),
            "scene_context_es": si.get("scene_context_es", ""),
            "detected_classes": list(si.get("active_vocabulary") or []),
            "active_vocabulary": list(si.get("active_vocabulary") or []),
            "prompt_free_discoveries": list(si.get("prompt_free_discoveries") or []),
            "vlm_guided_count": int(si.get("vlm_guided_count", 0) or 0),
            "detection_sources": list(si.get("detection_sources") or []),
            "inspection_focus_es": list(si.get("inspection_focus_es") or []),
        }

    scene_intelligence, _ = build_scene_intelligence_payload(
        discovery=discovery_hint,
        objects=raw_objects,
        occlusion_matrix=occlusion_matrix,
        intrinsic_diagnostics=intrinsic_diagnostics,
        real_photo_metrics=scaled_real_metrics,
        depth_stats=scaled_depth_stats,
        allow_live_vlm=False,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return CalibrateScaleResponse(
        scale_factor=factor,
        reference_label=req.reference_label,
        width=req.width,
        height=req.height,
        fov_h_deg=req.fov_h_deg,
        fov_v_deg=fov_v_deg,
        objects=enriched_objects,
        gt_objects=enriched_gt,
        occlusion_matrix=occlusion_matrix,
        metrics=metrics,
        intrinsic_diagnostics=intrinsic_diagnostics,
        depth_stats=scaled_depth_stats,
        real_photo_metrics=scaled_real_metrics,
        scene_intelligence=scene_intelligence,
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
        image_base64=req.image_base64,
        width=req.width,
        height=req.height,
        fov_h_deg=req.fov_h_deg,
        model=req.model,
        previous_interaction_id=req.previous_interaction_id,
        api_key=api_key,
        real_photo_metrics=req.real_photo_metrics,
        scene_intelligence=req.scene_intelligence,
    )
    return CopilotResponse(**result)
