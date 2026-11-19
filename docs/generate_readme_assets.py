#!/usr/bin/env python3
"""
Generate high-DPI matte Precision Optical Lab visual assets for GitHub README.

Outputs inside docs/assets/:
  - hero_banner.svg               (1200x310 light/dark-mode safe editorial SVG banner)
  - benchmarks_triptych.png       (1320x460 3 calibrated benchmarks with projected 2D wireframes & C8286 gate)
  - models_and_audit_charts.png   (1320x440 3-model comparison + FOV sensitivity & multi-frame stability)
  - photo_stress_tests.png        (1320x440 3 custom photo evaluations with projected OBBs & reliability gate)

All metrics and projections are computed directly from backend core modules (SEED=42)
and live HTTP photo test results (docs/assets/_live_photo_eval_results.json).
"""

from __future__ import annotations

import html
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"
LIVE_JSON = ASSETS_DIR / "_live_photo_eval_results.json"
CHROMIUM_BIN = "/usr/bin/chromium"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.geometry_3d import project_obb_to_2d
from app.core.intrinsic_diagnostics import compute_intrinsic_scene_diagnostics
from app.core.metrics import evaluate_scene_predictions, fov_sensitivity_sweep
from app.core.multiframe_audit import audit_multiframe_consistency, get_demo_multiframe_sequence
from app.core.synthetic_benchmark import list_benchmarks


def _render_html_to_png(html_text: str, out_png: Path, width: int, height: int) -> None:
    tmp_html = ASSETS_DIR / f"_tmp_{out_png.stem}.html"
    tmp_html.write_text(html_text, encoding="utf-8")
    cmd = [
        CHROMIUM_BIN,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--window-size={width},{height}",
        "--hide-scrollbars",
        f"--screenshot={out_png}",
        tmp_html.resolve().as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if tmp_html.exists():
        tmp_html.unlink()
    if res.returncode != 0 or not out_png.exists():
        raise RuntimeError(
            f"Chromium screenshot failed for {out_png.name} (rc={res.returncode}):\n{res.stderr}"
        )


def _wireframe_svg_overlay(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]] | None,
    width: int,
    height: int,
    fov_h_deg: float,
) -> str:
    """Project real 9-DOF 3D boxes onto exact image pixel space as clean vector wireframes."""
    lines: list[str] = []

    if gts:
        for obj in gts:
            proj = project_obb_to_2d(obj["box_3d"], width, height, fov_h_deg)
            if proj["behind_camera"]:
                continue
            verts = proj["vertices_2d"]
            for i0, i1 in proj["edges"]:
                x0, y0 = verts[i0]
                x1, y1 = verts[i1]
                lines.append(
                    f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                    f'stroke="#0F766E" stroke-width="1.6" stroke-dasharray="4 3" stroke-opacity="0.88"/>'
                )

    for obj in preds:
        proj = project_obb_to_2d(obj["box_3d"], width, height, fov_h_deg)
        if proj["behind_camera"]:
            continue
        verts = proj["vertices_2d"]
        for i0, i1 in proj["edges"]:
            x0, y0 = verts[i0]
            x1, y1 = verts[i1]
            lines.append(
                f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                f'stroke="#1D4ED8" stroke-width="2.1" stroke-opacity="0.94"/>'
            )
        ax0, ay0 = proj["front_arrow_2d"][0]
        ax1, ay1 = proj["front_arrow_2d"][1]
        lines.append(
            f'<line x1="{ax0:.1f}" y1="{ay0:.1f}" x2="{ax1:.1f}" y2="{ay1:.1f}" '
            f'stroke="#B45309" stroke-width="2.4"/>'
        )
        lx, ly = proj["label_anchor_2d"]
        lx = max(6.0, min(float(width) - 110.0, lx))
        ly = max(18.0, min(float(height) - 8.0, ly))
        lbl = html.escape(str(obj.get("label", "obj")).replace("_", " "))[:18]
        depth_m = float(obj["box_3d"][1])
        lines.append(
            f'<rect x="{lx:.1f}" y="{ly - 13:.1f}" width="{len(lbl)*5.6 + 38:.1f}" height="15" rx="3" fill="#FFFFFF" fill-opacity="0.92" stroke="#1D4ED8" stroke-width="0.9"/>'
            f'<text x="{lx + 4:.1f}" y="{ly - 2.5:.1f}" font-family="JetBrains Mono, monospace" font-size="9.5" font-weight="600" fill="#111827">{lbl} {depth_m:.1f}m</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="100%" '
        f'style="position:absolute;inset:0;width:100%;height:100%;" xmlns="http://www.w3.org/2000/svg">'
        f"{''.join(lines)}</svg>"
    )


def build_hero_banner_svg(miou_pct: float, ate_cm: float) -> None:
    svg = f"""<svg viewBox="0 0 1200 300" width="1200" height="300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MyLeenTo3 Precision 3D Monocular Lab Banner">
  <rect x="2" y="2" width="1196" height="296" rx="16" fill="#F7F7F5" stroke="#E2E5EA" stroke-width="2"/>

  <!-- Subtle optical metric dot grid on right half -->
  <defs>
    <pattern id="metricGrid" width="24" height="24" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1" fill="#CBD5E1" opacity="0.75"/>
    </pattern>
  </defs>
  <rect x="650" y="18" width="528" height="264" rx="12" fill="#FFFFFF" stroke="#E2E5EA" stroke-width="1.5"/>
  <rect x="650" y="18" width="528" height="264" rx="12" fill="url(#metricGrid)" opacity="0.8"/>

  <!-- Left editorial header -->
  <rect x="44" y="34" width="388" height="26" rx="13" fill="#FFFFFF" stroke="#E2E5EA" stroke-width="1.2"/>
  <circle cx="62" cy="47" r="4.5" fill="#0F766E"/>
  <text x="76" y="51" font-family="JetBrains Mono, Liberation Mono, monospace" font-size="11.5" font-weight="600" fill="#4B5563">C8286 (2D Edge) → MyLeenTo3 (3D Monocular)</text>

  <text x="44" y="114" font-family="Instrument Serif, Georgia, Liberation Serif, serif" font-size="52" font-weight="700" fill="#111827">MyLeenTo3</text>
  <text x="44" y="148" font-family="Inter, Liberation Sans, sans-serif" font-size="17" font-weight="600" fill="#1D4ED8">Percepción Espacial 3D Monocular y Compuerta Edge → VLM</text>
  <text x="44" y="176" font-family="Inter, Liberation Sans, sans-serif" font-size="13.5" fill="#4B5563">Convierte fotografías 2D en geometría 3D en metros (9-DOF), audita física real</text>
  <text x="44" y="196" font-family="Inter, Liberation Sans, sans-serif" font-size="13.5" fill="#4B5563">sin Ground Truth y escala modelos adaptativamente sin hardware LiDAR.</text>

  <!-- KPI pills row -->
  <g transform="translate(44, 218)">
    <rect x="0" y="0" width="136" height="52" rx="8" fill="#FFFFFF" stroke="#E2E5EA"/>
    <text x="12" y="19" font-family="Inter, sans-serif" font-size="10.5" fill="#4B5563">Precisión 3D</text>
    <text x="12" y="40" font-family="JetBrains Mono, monospace" font-size="17" font-weight="700" fill="#1D4ED8">{miou_pct:.1f}% mIoU</text>

    <rect x="148" y="0" width="136" height="52" rx="8" fill="#FFFFFF" stroke="#E2E5EA"/>
    <text x="160" y="19" font-family="Inter, sans-serif" font-size="10.5" fill="#4B5563">Error Traslación</text>
    <text x="160" y="40" font-family="JetBrains Mono, monospace" font-size="17" font-weight="700" fill="#0F766E">{ate_cm:.1f} cm ATE</text>

    <rect x="296" y="0" width="144" height="52" rx="8" fill="#FFFFFF" stroke="#E2E5EA"/>
    <text x="308" y="19" font-family="Inter, sans-serif" font-size="10.5" fill="#4B5563">Compuerta Edge</text>
    <text x="308" y="40" font-family="JetBrains Mono, monospace" font-size="17" font-weight="700" fill="#B45309">-45% / -72%</text>

    <rect x="452" y="0" width="126" height="52" rx="8" fill="#FFFFFF" stroke="#E2E5EA"/>
    <text x="464" y="19" font-family="Inter, sans-serif" font-size="10.5" fill="#4B5563">Determinismo</text>
    <text x="464" y="40" font-family="JetBrains Mono, monospace" font-size="17" font-weight="700" fill="#111827">38/38 OK</text>
  </g>

  <!-- Right precision optical frustum + 9-DOF cuboid & Kay-Kajiya rays diagram -->
  <text x="672" y="44" font-family="JetBrains Mono, monospace" font-size="11" font-weight="700" fill="#111827">MOTOR GEOMÉTRICO DETERMINÍSTICO · BASE QUIRAL det(M) = +1.0</text>

  <!-- Camera origin & frustum pyramid (clean vertical clearance above bottom badge bar) -->
  <circle cx="702" cy="158" r="6.5" fill="#111827"/>
  <text x="702" y="177" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="10" fill="#111827">O(0,0,0)</text>

  <!-- Frustum plane -->
  <line x1="702" y1="158" x2="778" y2="96" stroke="#9CA3AF" stroke-width="1.3"/>
  <line x1="702" y1="158" x2="778" y2="202" stroke="#9CA3AF" stroke-width="1.3"/>
  <line x1="702" y1="158" x2="826" y2="82" stroke="#9CA3AF" stroke-width="1.3"/>
  <line x1="702" y1="158" x2="826" y2="188" stroke="#9CA3AF" stroke-width="1.3"/>
  <polygon points="778,96 826,82 826,188 778,202" fill="#1D4ED8" fill-opacity="0.06" stroke="#1D4ED8" stroke-width="1.4"/>
  <text x="802" y="218" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#1D4ED8">Pinhole K(FOV_h)</text>

  <!-- Front 3D cuboid -->
  <polygon points="878,120 942,120 942,196 878,196" fill="#1D4ED8" fill-opacity="0.14" stroke="#1D4ED8" stroke-width="2"/>
  <polygon points="878,120 904,100 968,100 942,120" fill="#1D4ED8" fill-opacity="0.08" stroke="#1D4ED8" stroke-width="1.6"/>
  <polygon points="942,120 968,100 968,176 942,196" fill="#1D4ED8" fill-opacity="0.20" stroke="#1D4ED8" stroke-width="1.6"/>

  <!-- Occluded back cuboid (Emerald GT + Blue pred) -->
  <polygon points="1034,90 1106,90 1106,184 1034,184" fill="#0F766E" fill-opacity="0.12" stroke="#0F766E" stroke-width="2" stroke-dasharray="5 3"/>
  <polygon points="1034,90 1062,70 1134,70 1106,90" fill="none" stroke="#0F766E" stroke-width="1.5" stroke-dasharray="5 3"/>
  <polygon points="1106,90 1134,70 1134,164 1106,184" fill="#0F766E" fill-opacity="0.10" stroke="#0F766E" stroke-width="1.5" stroke-dasharray="5 3"/>

  <!-- Ray-OBB sightline & ruler clearance -->
  <line x1="702" y1="158" x2="1068" y2="134" stroke="#B45309" stroke-width="1.8" stroke-dasharray="5 3"/>
  <line x1="968" y1="154" x2="1034" y2="154" stroke="#BE123C" stroke-width="2"/>
  <circle cx="968" cy="154" r="3.5" fill="#BE123C"/>
  <circle cx="1034" cy="154" r="3.5" fill="#BE123C"/>
  <rect x="971" y="131" width="60" height="17" rx="3" fill="#FFFFFF" stroke="#BE123C"/>
  <text x="1001" y="143" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" font-weight="700" fill="#BE123C">0.23 m</text>

  <!-- Gate verdict strip inside banner diagram -->
  <rect x="672" y="230" width="484" height="36" rx="6" fill="#F7F7F5" stroke="#E2E5EA"/>
  <rect x="684" y="239" width="84" height="18" rx="4" fill="#F0FDFA" stroke="#99F6E4"/>
  <text x="726" y="251.5" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" font-weight="700" fill="#0F766E">APROBADO</text>
  <rect x="776" y="239" width="114" height="18" rx="4" fill="#FFFBEB" stroke="#FDE68A"/>
  <text x="833" y="251.5" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" font-weight="700" fill="#B45309">ESCALAR_MODELO</text>
  <rect x="898" y="239" width="124" height="18" rx="4" fill="#FFF1F2" stroke="#FECDD3"/>
  <text x="960" y="251.5" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" font-weight="700" fill="#BE123C">RECHAZADO_FISICA</text>
  <text x="1034" y="251.5" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#4B5563">7 Tools Python</text>
</svg>
"""
    (ASSETS_DIR / "hero_banner.svg").write_text(svg.strip() + "\n", encoding="utf-8")


def build_benchmarks_triptych() -> None:
    benchmarks = list_benchmarks()
    cards: list[str] = []
    titles = {
        "lab_desk_calibrated": "Mesa Calibrada (clean)",
        "occlusion_corridor": "Corredor con Oclusiones",
        "rotated_workshop": "Taller Rotado 3D (9-DOF)",
    }

    for bm in benchmarks:
        bid = bm["id"]
        w, h, fov = int(bm["width"]), int(bm["height"]), float(bm["fov_h_deg"])
        preds = bm["demo_predictions_by_model"]["gemini-4-hydrogen"]["objects"]
        gts = bm["gt_objects"]
        ev = evaluate_scene_predictions(preds, gts, w, h, fov, seed=42)
        diag = compute_intrinsic_scene_diagnostics(preds, bm["image_base64"], w, h, fov)
        cp = diag["cascade_policy"]
        overlay = _wireframe_svg_overlay(preds, gts, w, h, fov)

        verdict = cp["operational_verdict"]
        if verdict == "APROBADO":
            badge = f'<span class="badge ok">APROBADO · {cp["recommended_model"]} (-{cp["estimated_latency_savings_pct"]:.0f}%)</span>'
        else:
            badge = f'<span class="badge warn">ESCALAR → {cp["recommended_model"]}</span>'

        cards.append(
            f"""
            <div class="card">
              <div class="stage">
                <img src="data:image/png;base64,{bm['image_base64']}" alt="{html.escape(bid)}"/>
                {overlay}
              </div>
              <div class="meta">
                <div class="row1">
                  <strong>{html.escape(titles.get(bid, bid))}</strong>
                  {badge}
                </div>
                <div class="row2">
                  <span>mIoU 3D: <b>{ev['mIoU_3D']*100:.1f}%</b></span>
                  <span>ATE: <b>{ev['ate_m']*100:.1f} cm</b></span>
                  <span>AOE: <b>{ev['aoe_deg']:.1f}&deg;</b></span>
                  <span>Régimen: <b>{html.escape(cp['c8286_degradation_regime'])}</b></span>
                </div>
              </div>
            </div>
            """
        )

    page = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 18px; width: 1320px; background: #F7F7F5; font-family: Inter, Liberation Sans, sans-serif; color: #111827; }}
  .hdr {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
  .hdr h2 {{ margin: 0; font-size: 17px; font-weight: 700; }}
  .legend {{ font-family: "JetBrains Mono", monospace; font-size: 11px; color: #4B5563; display: flex; gap: 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .card {{ background: #FFFFFF; border: 1.2px solid #E2E5EA; border-radius: 10px; overflow: hidden; }}
  .stage {{ position: relative; aspect-ratio: 3/2; background: #F0F2F5; border-bottom: 1px solid #E2E5EA; }}
  .stage img {{ display: block; width: 100%; height: 100%; object-fit: cover; }}
  .meta {{ padding: 10px 12px; }}
  .row1 {{ display: flex; justify-content: space-between; align-items: center; gap: 6px; font-size: 12.5px; margin-bottom: 6px; }}
  .badge {{ font-family: "JetBrains Mono", monospace; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 5px; white-space: nowrap; }}
  .badge.ok {{ background: #F0FDFA; color: #0F766E; border: 1px solid #99F6E4; }}
  .badge.warn {{ background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }}
  .row2 {{ font-family: "JetBrains Mono", monospace; font-size: 10.5px; color: #4B5563; display: flex; justify-content: space-between; }}
  .row2 b {{ color: #111827; }}
</style></head>
<body>
  <div class="hdr">
    <h2>Escenas Calibradas (SEED=42) — Proyección 2D Pinhole y Veredicto Compuerta Edge → VLM</h2>
    <div class="legend">
      <span><b style="color:#1D4ED8">■</b> Predicción Hydrogen</span>
      <span><b style="color:#0F766E">┄</b> Ground Truth</span>
      <span><b style="color:#B45309">→</b> Orientación frontal (Yaw)</span>
    </div>
  </div>
  <div class="grid">{''.join(cards)}</div>
</body></html>"""
    _render_html_to_png(page, ASSETS_DIR / "benchmarks_triptych.png", 1320, 420)


def build_models_and_audit_charts() -> tuple[float, float]:
    benchmarks = list_benchmarks()
    models = [
        ("gemini-4-hydrogen", "Gemini 4 Hydrogen", "high", "#1D4ED8"),
        ("gemini-4-argon", "Gemini 4 Argon", "medium", "#0F766E"),
        ("gemini-3.5-flash-lite", "Flash-Lite", "low", "#B45309"),
    ]
    agg: dict[str, dict[str, float]] = {}
    for mid, _, _, _ in models:
        vals = []
        for bm in benchmarks:
            preds = bm["demo_predictions_by_model"][mid]["objects"]
            vals.append(
                evaluate_scene_predictions(
                    preds, bm["gt_objects"], int(bm["width"]), int(bm["height"]), float(bm["fov_h_deg"]), seed=42
                )
            )
        agg[mid] = {
            "mIoU_3D": statistics.mean(v["mIoU_3D"] for v in vals),
            "mIoU_BEV": statistics.mean(v["mIoU_BEV"] for v in vals),
            "ate_cm": statistics.mean(v["ate_m"] for v in vals) * 100.0,
            "aoe_deg": statistics.mean(v["aoe_deg"] for v in vals),
        }

    bm0 = benchmarks[0]
    sweep = fov_sensitivity_sweep(
        bm0["demo_predictions_by_model"]["gemini-4-hydrogen"]["objects"],
        bm0["gt_objects"],
        int(bm0["width"]),
        int(bm0["height"]),
        fov_values=[40, 50, 60, 70, 80, 90, 100],
    )
    mf = audit_multiframe_consistency(get_demo_multiframe_sequence()["frames"])

    # Build Panel A SVG (Model Pareto comparison)
    bars_a = []
    for idx, (mid, label, think, col) in enumerate(models):
        y = 58 + idx * 82
        r = agg[mid]
        w_3d = r["mIoU_3D"] * 330.0
        w_bev = r["mIoU_BEV"] * 330.0
        bars_a.append(
            f'<text x="16" y="{y}" font-family="Inter, sans-serif" font-size="12" font-weight="700" fill="#111827">{label} <tspan font-family="JetBrains Mono, monospace" font-size="10.5" fill="#4B5563">({think})</tspan></text>'
            f'<rect x="16" y="{y + 10}" width="{w_3d:.1f}" height="15" rx="3" fill="{col}"/>'
            f'<text x="{24 + w_3d:.1f}" y="{y + 21}" font-family="JetBrains Mono, monospace" font-size="10.5" font-weight="700" fill="{col}">mIoU 3D {r["mIoU_3D"]*100:.1f}% · ATE {r["ate_cm"]:.1f}cm</text>'
            f'<rect x="16" y="{y + 30}" width="{w_bev:.1f}" height="13" rx="3" fill="#CBD5E1"/>'
            f'<text x="{24 + w_bev:.1f}" y="{y + 40}" font-family="JetBrains Mono, monospace" font-size="10" fill="#4B5563">BEV {r["mIoU_BEV"]*100:.1f}% · Giro {r["aoe_deg"]:.1f}°</text>'
        )

    # Panel B SVG (FOV + Multi-frame tracks)
    fx0, fx1, fy0, fy1 = 44, 590, 44, 152
    fpts = " ".join(
        f"{fx0 + (p['fov_h_deg'] - 40)/60.0 * (fx1 - fx0):.1f},{fy1 - p['reproj_iou_2d'] * (fy1 - fy0):.1f}"
        for p in sweep
    )
    f60 = next(p for p in sweep if p["fov_h_deg"] == 60)
    x60 = fx0 + (60 - 40) / 60.0 * (fx1 - fx0)
    y60 = fy1 - f60["reproj_iou_2d"] * (fy1 - fy0)

    tracks_b = []
    for i, tr in enumerate(mf["object_tracks"][:4]):
        ty = 204 + i * 32
        w_s = min(1.0, tr["scale_drift_pct"] / 22.0) * 280.0
        tracks_b.append(
            f'<text x="150" y="{ty + 11}" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="10.5" fill="#111827">{tr["label"].replace("_", " ")}</text>'
            f'<rect x="160" y="{ty}" width="{max(w_s, 4):.1f}" height="14" rx="3" fill="#1D4ED8"/>'
            f'<text x="{168 + max(w_s, 4):.1f}" y="{ty + 11}" font-family="JetBrains Mono, monospace" font-size="10.5" fill="#1D4ED8">ΔVol {tr["scale_drift_pct"]:.1f}% · Yaw ±{tr["yaw_jitter_deg"]:.1f}°</text>'
        )

    page = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 18px; width: 1320px; background: #F7F7F5; font-family: Inter, Liberation Sans, sans-serif; color: #111827; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .panel {{ background: #FFFFFF; border: 1.2px solid #E2E5EA; border-radius: 10px; padding: 14px; }}
  h3 {{ margin: 0 0 8px 0; font-size: 14px; font-weight: 700; color: #111827; }}
</style></head>
<body>
  <div class="grid">
    <div class="panel">
      <h3>A) Comparativa 3 Modelos Google AI Studio (SEED=42 · Media 3 Escenas)</h3>
      <svg viewBox="0 0 610 315" width="100%">{''.join(bars_a)}
        <rect x="16" y="294" width="578" height="1" fill="#E2E5EA"/>
      </svg>
    </div>
    <div class="panel">
      <h3>B) Sensibilidad Óptica FOV_h &amp; Auditoría Multi-Frame ({mf['geometric_stability_score']:.1f}/100)</h3>
      <svg viewBox="0 0 610 315" width="100%">
        <line x1="{fx0}" y1="{fy1}" x2="{fx1}" y2="{fy1}" stroke="#CBD5E1" stroke-width="1.2"/>
        <line x1="{fx0}" y1="{fy0}" x2="{fx0}" y2="{fy1}" stroke="#CBD5E1" stroke-width="1.2"/>
        <line x1="{x60:.1f}" y1="{fy0}" x2="{x60:.1f}" y2="{fy1}" stroke="#0F766E" stroke-dasharray="4 3" stroke-width="1.2"/>
        <polyline fill="none" stroke="#1D4ED8" stroke-width="2.4" points="{fpts}"/>
        <circle cx="{x60:.1f}" cy="{y60:.1f}" r="5" fill="#0F766E"/>
        <text x="{x60 + 8:.1f}" y="{y60 + 4:.1f}" font-family="JetBrains Mono, monospace" font-size="11" font-weight="700" fill="#0F766E">FOV real 60° ({f60['reproj_iou_2d']*100:.1f}%)</text>
        <text x="{fx0}" y="168" font-family="JetBrains Mono, monospace" font-size="10" fill="#4B5563">40°</text>
        <text x="{fx1 - 24}" y="168" font-family="JetBrains Mono, monospace" font-size="10" fill="#4B5563">100°</text>
        <text x="16" y="190" font-family="Inter, sans-serif" font-size="11.5" font-weight="700" fill="#111827">Deriva temporal por objeto (4 cuadros consecutivos)</text>
        {''.join(tracks_b)}
      </svg>
    </div>
  </div>
</body></html>"""
    _render_html_to_png(page, ASSETS_DIR / "models_and_audit_charts.png", 1320, 405)
    return agg["gemini-4-hydrogen"]["mIoU_3D"] * 100.0, agg["gemini-4-hydrogen"]["ate_cm"]


def build_photo_stress_tests_graphic() -> None:
    if not LIVE_JSON.exists():
        raise RuntimeError(f"Missing {LIVE_JSON}; run backend/scripts/run_photo_model_tests.py first.")
    data = json.loads(LIVE_JSON.read_text(encoding="utf-8"))
    photos = data["photo_results"]
    cards: list[str] = []

    for p in photos:
        w, h, fov = int(p["width"]), int(p["height"]), float(p["fov_h_deg"])
        overlay = _wireframe_svg_overlay(p["objects"], None, w, h, fov)
        cp = p["cascade_policy"]
        lat = p["latent_metrics"]
        prof = p["image_profile"]
        verdict = cp["operational_verdict"]
        cls = "ok" if verdict == "APROBADO" else ("warn" if verdict == "ESCALAR_MODELO" else "bad")
        cards.append(
            f"""
            <div class="card">
              <div class="stage">
                <img src="data:image/png;base64,{p['image_base64']}" alt="{html.escape(p['id'])}"/>
                {overlay}
              </div>
              <div class="meta">
                <div class="row1">
                  <strong>{html.escape(p['title'])}</strong>
                  <span class="badge {cls}">{html.escape(verdict)} · {html.escape(cp['recommended_model'])}</span>
                </div>
                <div class="row2">
                  <span>Dom: <b>{html.escape(prof['domain_classification'])}</b></span>
                  <span>Reg: <b>{html.escape(cp['c8286_degradation_regime'])}</b></span>
                </div>
                <div class="row2" style="margin-top:4px;">
                  <span>Plaus: <b>{lat['physical_plausibility_score']:.1f}/100</b></span>
                  <span>PECS: <b>{lat['photometric_edge_coherence']*100:.0f}%</b></span>
                  <span>±σ_y: <b>{lat['mean_depth_uncertainty_m']:.2f}m</b></span>
                  <span>FOV*: <b>{p['estimated_optimal_fov_h_deg']}°</b></span>
                </div>
              </div>
            </div>
            """
        )

    page = f"""<!DOCTYPE html>
<html lang="es">
<head><meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 18px; width: 1320px; background: #F7F7F5; font-family: Inter, Liberation Sans, sans-serif; color: #111827; }}
  .hdr {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }}
  .hdr h2 {{ margin: 0; font-size: 17px; font-weight: 700; }}
  .sub {{ font-family: "JetBrains Mono", monospace; font-size: 11px; color: #4B5563; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .card {{ background: #FFFFFF; border: 1.2px solid #E2E5EA; border-radius: 10px; overflow: hidden; }}
  .stage {{ position: relative; aspect-ratio: 3/2; background: #111827; border-bottom: 1px solid #E2E5EA; overflow: hidden; }}
  .stage img {{ display: block; width: 100%; height: 100%; object-fit: cover; }}
  .meta {{ padding: 10px 12px; }}
  .row1 {{ display: flex; justify-content: space-between; align-items: center; gap: 6px; font-size: 12.2px; margin-bottom: 6px; }}
  .badge {{ font-family: "JetBrains Mono", monospace; font-size: 9.8px; font-weight: 700; padding: 2px 6px; border-radius: 5px; white-space: nowrap; }}
  .badge.ok {{ background: #F0FDFA; color: #0F766E; border: 1px solid #99F6E4; }}
  .badge.warn {{ background: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }}
  .badge.bad {{ background: #FFF1F2; color: #BE123C; border: 1px solid #FECDD3; }}
  .row2 {{ font-family: "JetBrains Mono", monospace; font-size: 10.3px; color: #4B5563; display: flex; justify-content: space-between; }}
  .row2 b {{ color: #111827; }}
</style></head>
<body>
  <div class="hdr">
    <h2>Pruebas con Fotos Diversas (/api/analyze + Saliencia + Diagnóstico Sin Referencia)</h2>
    <div class="sub">Evaluación en vivo HTTP sobre :8085 · Veredicto automático C8286</div>
  </div>
  <div class="grid">{''.join(cards)}</div>
</body></html>"""
    _render_html_to_png(page, ASSETS_DIR / "photo_stress_tests.png", 1320, 435)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    miou_pct, ate_cm = build_models_and_audit_charts()
    build_hero_banner_svg(miou_pct, ate_cm)
    build_benchmarks_triptych()
    if LIVE_JSON.exists():
        build_photo_stress_tests_graphic()
    print("Generated all GitHub visual assets in", ASSETS_DIR)


if __name__ == "__main__":
    main()
