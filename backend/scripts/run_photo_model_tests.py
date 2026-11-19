#!/usr/bin/env python3
"""Live HTTP evaluation of MyLeenTo3 across all 3 Gemini models and custom RGB photos.

Executes real HTTP requests against the running FastAPI backend (:8085) to benchmark:
  - Suite A: /api/compare-models across all 3 calibrated benchmarks (960x640, FOV_h=60)
  - Suite B: /api/analyze on 3 distinct custom RGB photo stress profiles (dark/bloom,
    clean tabletop indoor, and cluttered wide-angle hallway), plus /api/copilot policy query.

Also saves reproducible results JSON at docs/assets/_live_photo_eval_results.json so
docs/generate_readme_assets.py can render photo_stress_tests.png from 100% live data.
"""

from __future__ import annotations

import base64
import io
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ASSETS = REPO_ROOT / "docs" / "assets"
BASE_URL = os.getenv("MYLEENTO3_API_URL", "http://127.0.0.1:8085/api")


def _encode_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_custom_test_photos() -> list[dict[str, Any]]:
    """Build 3 reproducible custom RGB photos representing diverse real visual stress domains."""
    photos: list[dict[str, Any]] = []

    # 1. High-contrast stylized / dark low-light scene with luminous focal core (like rendered/gaming scenes)
    w, h = 720, 420
    arr1 = np.zeros((h, w, 3), dtype=np.uint8)
    y_grad = np.linspace(10, 26, h, dtype=np.float32)[:, None]
    arr1[:, :, 0] = y_grad.astype(np.uint8)
    arr1[:, :, 1] = (y_grad + 4).astype(np.uint8)
    arr1[:, :, 2] = (y_grad + 16).astype(np.uint8)
    # Misty glow column in center + dark silhouette figure
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot((xx - w * 0.50) / 85.0, (yy - h * 0.62) / 68.0)
    glow = np.exp(-dist * dist * 1.4)
    arr1[:, :, 0] = np.clip(arr1[:, :, 0] + glow * 195, 0, 255).astype(np.uint8)
    arr1[:, :, 1] = np.clip(arr1[:, :, 1] + glow * 218, 0, 255).astype(np.uint8)
    arr1[:, :, 2] = np.clip(arr1[:, :, 2] + glow * 245, 0, 255).astype(np.uint8)
    img1 = Image.fromarray(arr1, mode="RGB")
    d1 = ImageDraw.Draw(img1)
    # Horizon bridge & silhouettes
    d1.rectangle([0, int(h * 0.68), w, h], fill=(18, 20, 28))
    d1.rectangle([int(w * 0.48), int(h * 0.44), int(w * 0.52), int(h * 0.68)], fill=(14, 16, 22))
    d1.ellipse([int(w * 0.485), int(h * 0.39), int(w * 0.515), int(h * 0.445)], fill=(14, 16, 22))
    d1.polygon([(50, 410), (160, 270), (240, 410)], fill=(22, 26, 36))
    d1.polygon([(520, 410), (610, 250), (690, 410)], fill=(22, 26, 36))
    photos.append(
        {
            "id": "photo_dark_bloom",
            "title": "Foto Oscura / Alto Contraste",
            "fov_h_deg": 68.0,
            "width": w,
            "height": h,
            "image_base64": _encode_png(img1),
        }
    )

    # 2. Clean indoor wooden desk tabletop with sharp everyday geometry (well-lit photography)
    w2, h2 = 720, 480
    img2 = Image.new("RGB", (w2, h2), color=(234, 232, 226))
    d2 = ImageDraw.Draw(img2)
    # Wall & table horizon
    d2.rectangle([0, 0, w2, 240], fill=(241, 240, 235))
    d2.polygon([(0, 240), (w2, 240), (w2, h2), (0, h2)], fill=(214, 198, 174))
    d2.line([(0, 240), (w2, 240)], fill=(185, 168, 145), width=2)
    # Laptop stand + screen + chassis (left-center)
    d2.rectangle([120, 150, 320, 280], fill=(42, 48, 60), outline=(24, 28, 36), width=2)
    d2.rectangle([130, 160, 310, 268], fill=(224, 236, 250))
    d2.polygon([(115, 280), (325, 280), (352, 326), (92, 326)], fill=(148, 156, 168), outline=(55, 62, 74))
    # Espresso mug (center-right)
    d2.rectangle([388, 252, 444, 320], fill=(29, 78, 216), outline=(17, 24, 39), width=2)
    # Thick textbook stack (right)
    d2.rectangle([490, 218, 636, 268], fill=(15, 118, 110), outline=(17, 24, 39), width=2)
    d2.rectangle([484, 268, 642, 318], fill=(180, 83, 9), outline=(17, 24, 39), width=2)
    photos.append(
        {
            "id": "photo_clean_desk",
            "title": "Foto Interior Iluminada (Escritorio)",
            "fov_h_deg": 60.0,
            "width": w2,
            "height": h2,
            "image_base64": _encode_png(img2),
        }
    )

    # 3. Wide-angle shelf / corridor workspace with multiple separated objects at varied depths
    w3, h3 = 720, 480
    img3 = Image.new("RGB", (w3, h3), color=(240, 241, 244))
    d3 = ImageDraw.Draw(img3)
    # Perspective hallway guide lines (rectilinear architecture)
    d3.line([(0, 0), (260, 175)], fill=(190, 196, 206), width=2)
    d3.line([(w3, 0), (460, 175)], fill=(190, 196, 206), width=2)
    d3.line([(0, h3), (260, 305)], fill=(180, 186, 196), width=2)
    d3.line([(w3, h3), (460, 305)], fill=(180, 186, 196), width=2)
    d3.rectangle([260, 175, 460, 305], outline=(180, 186, 196), width=2)
    # Distinct well-separated foreground & background containers for multi-blob salience
    d3.rectangle([52, 230, 195, 420], fill=(162, 108, 64), outline=(55, 35, 18), width=3)
    d3.rectangle([298, 210, 412, 318], fill=(29, 78, 216), outline=(17, 24, 39), width=3)
    d3.rectangle([505, 195, 668, 410], fill=(15, 118, 110), outline=(17, 24, 39), width=3)
    photos.append(
        {
            "id": "photo_corridor_clutter",
            "title": "Taller con 3 Cajas en Profundidad",
            "fov_h_deg": 72.0,
            "width": w3,
            "height": h3,
            "image_base64": _encode_png(img3),
        }
    )

    return photos


def main() -> None:
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=BASE_URL, timeout=15.0) as client:
        health = client.get("/health").raise_for_status().json()
        print("=== HEALTH ===", json.dumps(health))

        # Suite A: Compare 3 models across all 3 benchmarks
        benchmark_ids = ["lab_desk_calibrated", "occlusion_corridor", "rotated_workshop"]
        compare_summary: dict[str, Any] = {}
        model_agg: dict[str, list[dict[str, Any]]] = {}

        print("\n=== SUITE A: COMPARATIVA 3 MODELOS GEMINI ===")
        for bid in benchmark_ids:
            comp = client.post(
                "/compare-models",
                json={"benchmark_id": bid, "fov_h_deg": 60.0, "use_live": False},
            ).raise_for_status().json()
            # Also query /analyze for hydrogen to capture cascade_policy
            an = client.post(
                "/analyze",
                json={"benchmark_id": bid, "model": "gemini-4-hydrogen", "fov_h_deg": 60.0},
            ).raise_for_status().json()
            cp = an["intrinsic_diagnostics"]["cascade_policy"]
            compare_summary[bid] = {"results": comp["results"], "cascade_policy": cp}

            print(
                f"\n[{bid}] Compuerta={cp['operational_verdict']} | Regimen={cp['c8286_degradation_regime']} "
                f"| RecModel={cp['recommended_model']} | Savings=-{cp['estimated_latency_savings_pct']:.0f}%"
            )
            for row in comp["results"]:
                m = row["metrics"]
                model_agg.setdefault(row["model"], []).append(
                    {
                        "mIoU_3D": m["mIoU_3D"],
                        "mIoU_BEV": m["mIoU_BEV"],
                        "ap_25": m["ap_25"],
                        "ate_m": m["ate_m"],
                        "aoe_deg": m["aoe_deg"],
                        "latency_ms": row["latency_ms"],
                    }
                )
                print(
                    f"  - {row['model']:22s} | think={row['thinking_level']:6s} "
                    f"| mIoU_3D={m['mIoU_3D']*100:5.1f}% | BEV={m['mIoU_BEV']*100:5.1f}% "
                    f"| ATE={m['ate_m']*100:5.1f}cm | AOE={m['aoe_deg']:4.1f}° | lat={row['latency_ms']:.0f}ms"
                )

        print("\n--- AGREGADO 3 ESCENAS ---")
        for mdl, rows in model_agg.items():
            print(
                f"  * {mdl:22s}: mIoU_3D={statistics.mean(r['mIoU_3D'] for r in rows)*100:5.1f}% "
                f"| BEV={statistics.mean(r['mIoU_BEV'] for r in rows)*100:5.1f}% "
                f"| AP@0.25={statistics.mean(r['ap_25'] for r in rows)*100:5.1f}% "
                f"| ATE={statistics.mean(r['ate_m'] for r in rows)*100:5.1f}cm "
                f"| AOE={statistics.mean(r['aoe_deg'] for r in rows):4.1f}°"
            )

        # Suite B: Custom RGB photo stress tests across the 3 Gemini models + diagnostics
        photos = make_custom_test_photos()
        photo_results: list[dict[str, Any]] = []

        print("\n=== SUITE B: PRUEBAS CON FOTOS PERSONALIZADAS (/api/analyze + /api/copilot) ===")
        for p in photos:
            an = client.post(
                "/analyze",
                json={
                    "image_base64": p["image_base64"],
                    "mime_type": "image/png",
                    "width": p["width"],
                    "height": p["height"],
                    "model": "gemini-4-hydrogen",
                    "fov_h_deg": p["fov_h_deg"],
                    "max_objects": 8,
                },
            ).raise_for_status().json()

            diag = an["intrinsic_diagnostics"]
            prof = diag["image_profile"]
            lat = diag["latent_metrics"]
            cp = diag["cascade_policy"]

            cop = client.post(
                "/copilot",
                json={
                    "message": "¿Conviene escalar el modelo 3D o es confiable esta escena?",
                    "objects": [
                        {
                            "id": o["id"],
                            "label": o["label"],
                            "box_3d": o["box_3d"],
                            "confidence": o["confidence"],
                        }
                        for o in an["objects"]
                    ],
                    "image_base64": p["image_base64"],
                    "width": p["width"],
                    "height": p["height"],
                    "fov_h_deg": p["fov_h_deg"],
                    "model": "gemini-4-hydrogen",
                },
            ).raise_for_status().json()

            print(
                f"\n[{p['title']}] mode={an['mode']} | objs={len(an['objects'])} | domain={prof['domain_classification']} "
                f"| reg={cp['c8286_degradation_regime']} | verdict={cp['operational_verdict']} ({cp['recommended_model']})"
            )
            print(
                f"  Plausibilidad={lat['physical_plausibility_score']:.1f}/100 | PECS={lat['photometric_edge_coherence']*100:.1f}% "
                f"| PVI={lat['volume_interpenetration_index']*100:.1f}% | GFI={lat['gravity_floating_anomaly_m']:.2f}m "
                f"| ±σ_y={lat['mean_depth_uncertainty_m']:.2f}m | FOV_opt={diag['estimated_optimal_fov_h_deg']}°"
            )
            print(f"  Motivos: {cp['trigger_reasons']}")
            print(f"  Copilot tool_trace: {[t['name'] for t in cop['tool_trace']]}")

            photo_results.append(
                {
                    "id": p["id"],
                    "title": p["title"],
                    "width": p["width"],
                    "height": p["height"],
                    "fov_h_deg": p["fov_h_deg"],
                    "image_base64": p["image_base64"],
                    "mode": an["mode"],
                    "objects": an["objects"],
                    "image_profile": prof,
                    "latent_metrics": lat,
                    "cascade_policy": cp,
                    "estimated_optimal_fov_h_deg": diag["estimated_optimal_fov_h_deg"],
                    "exploited_weaknesses": diag["exploited_weaknesses"],
                    "copilot_reply": cop["reply"],
                }
            )

        out_path = DOCS_ASSETS / "_live_photo_eval_results.json"
        out_path.write_text(
            json.dumps(
                {
                    "health": health,
                    "compare_summary": compare_summary,
                    "photo_results": photo_results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved live results JSON -> {out_path}")


if __name__ == "__main__":
    main()
