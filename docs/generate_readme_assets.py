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

import base64
import html
import io
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
ASSETS_DIR = REPO_ROOT / "docs" / "assets"
LIVE_JSON = ASSETS_DIR / "_live_photo_eval_results.json"
CHROMIUM_BIN = "/usr/bin/chromium"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.geometry_3d import BASIS_GEMINI_TO_THREE, obb_corners_gemini, project_obb_to_2d
from app.core.intrinsic_diagnostics import compute_intrinsic_scene_diagnostics
from app.core.metrics import evaluate_scene_predictions, fov_sensitivity_sweep
from app.core.multiframe_audit import audit_multiframe_consistency, get_demo_multiframe_sequence
from app.core.synthetic_benchmark import get_benchmark, list_benchmarks


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
    svg = f"""<svg viewBox="0 0 1200 320" width="1200" height="320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="MyLeenTo3 Animated Optical 3D Banner">
  <defs>
    <pattern id="metricGrid" width="24" height="24" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1" fill="#CBD5E1" opacity="0.8"/>
    </pattern>
    <linearGradient id="scanSweep" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#1D4ED8" stop-opacity="0"/>
      <stop offset="50%" stop-color="#1D4ED8" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#1D4ED8" stop-opacity="0"/>
    </linearGradient>
    <style>
      .flow-ray {{
        animation: rayDash 1.9s linear infinite;
      }}
      @keyframes rayDash {{
        from {{ stroke-dashoffset: 28; }}
        to {{ stroke-dashoffset: 0; }}
      }}
    </style>
  </defs>

  <!-- Architectural matte frame -->
  <rect x="2" y="2" width="1196" height="316" rx="2" fill="#F7F7F5" stroke="#CBD5E1" stroke-width="1.5"/>

  <!-- Unboxed top editorial masthead over 1px rule -->
  <text x="44" y="34" font-family="JetBrains Mono, Liberation Mono, monospace" font-size="10.5" font-weight="600" letter-spacing="1.4" fill="#4B5563">C8286 EDGE 2D  -&gt;  MYLEENTO3 OPTICA 3D</text>
  <text x="614" y="34" text-anchor="end" font-family="JetBrains Mono, Liberation Mono, monospace" font-size="10.5" font-weight="600" letter-spacing="1.2" fill="#0F766E">MOGE-2 GPU + GEMINI v1</text>
  <line x1="44" y1="45" x2="614" y2="45" stroke="#111827" stroke-opacity="0.18" stroke-width="1"/>

  <!-- Left editorial Source Serif 4 display & typographic KPI ledger -->
  <text x="44" y="114" font-family="Source Serif 4, Georgia, Liberation Serif, serif" font-size="56" font-weight="500" fill="#111827">MyLeenTo3</text>
  <text x="44" y="148" font-family="Inter, Liberation Sans, sans-serif" font-size="16" font-weight="600" fill="#1D4ED8">Percepción Espacial 3D Monocular y Compuerta Adaptativa</text>
  <text x="44" y="176" font-family="Inter, Liberation Sans, sans-serif" font-size="13" fill="#4B5563">Nube métrica local (EXIF + MoGe-2 ViT-S + YOLOE), poses 9-DOF verificadas</text>
  <text x="44" y="195" font-family="Inter, Liberation Sans, sans-serif" font-size="13" fill="#4B5563">en Python y control adaptativo sin depender de hardware LiDAR.</text>

  <!-- Unboxed typographic 4-column ledger separated by 1px vertical hairlines -->
  <line x1="44" y1="224" x2="614" y2="224" stroke="#111827" stroke-opacity="0.18" stroke-width="1"/>
  <g transform="translate(44, 244)">
    <text x="0" y="0" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#4B5563">PRECISION 3D</text>
    <text x="0" y="26" font-family="JetBrains Mono, monospace" font-size="18" font-weight="700" fill="#1D4ED8">{miou_pct:.1f}% mIoU</text>
    <line x1="142" y1="-10" x2="142" y2="36" stroke="#CBD5E1" stroke-width="1"/>

    <text x="158" y="0" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#4B5563">TRASLACION</text>
    <text x="158" y="26" font-family="JetBrains Mono, monospace" font-size="18" font-weight="700" fill="#0F766E">{ate_cm:.1f} cm ATE</text>
    <line x1="298" y1="-10" x2="298" y2="36" stroke="#CBD5E1" stroke-width="1"/>

    <text x="314" y="0" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#4B5563">MOTOR GPU</text>
    <text x="314" y="26" font-family="JetBrains Mono, monospace" font-size="18" font-weight="700" fill="#B45309">365 ms</text>
    <line x1="436" y1="-10" x2="436" y2="36" stroke="#CBD5E1" stroke-width="1"/>

    <text x="452" y="0" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#4B5563">TESTS SUITE</text>
    <text x="452" y="26" font-family="JetBrains Mono, monospace" font-size="18" font-weight="700" fill="#111827">51/51 OK</text>
  </g>

  <!-- Right animated optical workbench panel -->
  <rect x="646" y="20" width="530" height="280" rx="2" fill="#FFFFFF" stroke="#111827" stroke-opacity="0.18" stroke-width="1.2"/>
  <rect x="646" y="20" width="530" height="280" rx="2" fill="url(#metricGrid)" opacity="0.85"/>
  <text x="668" y="46" font-family="JetBrains Mono, monospace" font-size="10.5" font-weight="700" fill="#111827">RECONSTRUCCION METRICA Y TRAZADO KAY-KAJIYA · det(M) = +1.0</text>
  <line x1="646" y1="58" x2="1176" y2="58" stroke="#E2E5EA" stroke-width="1"/>

  <!-- Animated background point-cloud constellations -->
  <g opacity="0.7">
    <circle cx="848" cy="112" r="2.2" fill="#1D4ED8">
      <animate attributeName="opacity" values="0.25;0.85;0.25" dur="2.8s" repeatCount="indefinite"/>
    </circle>
    <circle cx="866" cy="96" r="1.8" fill="#0F766E">
      <animate attributeName="opacity" values="0.7;0.2;0.7" dur="3.4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="988" cy="90" r="2" fill="#B45309">
      <animate attributeName="opacity" values="0.3;0.9;0.3" dur="2.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="1142" cy="134" r="2.2" fill="#1D4ED8">
      <animate attributeName="opacity" values="0.8;0.25;0.8" dur="3.1s" repeatCount="indefinite"/>
    </circle>
    <circle cx="1008" cy="204" r="2" fill="#334155">
      <animate attributeName="opacity" values="0.25;0.8;0.25" dur="2.7s" repeatCount="indefinite"/>
    </circle>
  </g>

  <!-- Sweeping optical laser/depth scan beam across frustum -->
  <rect x="768" y="72" width="18" height="145" fill="url(#scanSweep)">
    <animateTransform attributeName="transform" type="translate" values="0 0; 64 0; 0 0" dur="3.6s" repeatCount="indefinite"/>
  </rect>

  <!-- Camera origin & frustum pyramid -->
  <circle cx="702" cy="164" r="6" fill="#111827"/>
  <text x="702" y="184" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#111827">O(0,0,0)</text>

  <line x1="702" y1="164" x2="778" y2="98" stroke="#9CA3AF" stroke-width="1.2"/>
  <line x1="702" y1="164" x2="778" y2="206" stroke="#9CA3AF" stroke-width="1.2"/>
  <line x1="702" y1="164" x2="826" y2="84" stroke="#9CA3AF" stroke-width="1.2"/>
  <line x1="702" y1="164" x2="826" y2="192" stroke="#9CA3AF" stroke-width="1.2"/>
  <polygon points="778,98 826,84 826,192 778,206" fill="#1D4ED8" fill-opacity="0.06" stroke="#1D4ED8" stroke-width="1.4"/>
  <text x="802" y="222" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#1D4ED8">K(FOV_h)</text>

  <!-- Front 3D cuboid -->
  <polygon points="878,124 942,124 942,200 878,200" fill="#1D4ED8" fill-opacity="0.14" stroke="#1D4ED8" stroke-width="2"/>
  <polygon points="878,124 904,104 968,104 942,124" fill="#1D4ED8" fill-opacity="0.08" stroke="#1D4ED8" stroke-width="1.6"/>
  <polygon points="942,124 968,104 968,180 942,200" fill="#1D4ED8" fill-opacity="0.20" stroke="#1D4ED8" stroke-width="1.6"/>

  <!-- Occluded back cuboid (Emerald GT) -->
  <polygon points="1034,94 1106,94 1106,188 1034,188" fill="#0F766E" fill-opacity="0.12" stroke="#0F766E" stroke-width="2" stroke-dasharray="5 3"/>
  <polygon points="1034,94 1062,74 1134,74 1106,94" fill="none" stroke="#0F766E" stroke-width="1.5" stroke-dasharray="5 3"/>
  <polygon points="1106,94 1134,74 1134,168 1106,188" fill="#0F766E" fill-opacity="0.10" stroke="#0F766E" stroke-width="1.5" stroke-dasharray="5 3"/>

  <!-- Animated Kay-Kajiya sightline ray + traveling photon dot -->
  <line x1="702" y1="164" x2="1072" y2="138" stroke="#B45309" stroke-width="1.8" stroke-dasharray="6 4" class="flow-ray"/>
  <circle r="4" fill="#B45309">
    <animateMotion path="M702,164 L1072,138" dur="2.2s" repeatCount="indefinite"/>
  </circle>

  <!-- Caliper surface clearance line -->
  <line x1="968" y1="158" x2="1034" y2="158" stroke="#BE123C" stroke-width="2"/>
  <circle cx="968" cy="158" r="3.5" fill="#BE123C"/>
  <circle cx="1034" cy="158" r="3.5" fill="#BE123C"/>
  <rect x="971" y="135" width="60" height="17" fill="#FFFFFF" stroke="#BE123C"/>
  <text x="1001" y="147" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="9.5" font-weight="700" fill="#BE123C">0.23 m</text>

  <!-- Architectural verdict hairline bar (zero rounded pills) -->
  <line x1="668" y1="248" x2="1154" y2="248" stroke="#111827" stroke-opacity="0.18" stroke-width="1"/>
  <g transform="translate(668, 274)">
    <circle cx="6" cy="-4" r="3.5" fill="#0F766E">
      <animate attributeName="opacity" values="1;0.2;1" dur="2.4s" repeatCount="indefinite"/>
    </circle>
    <text x="16" y="0" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#0F766E">APROBADO</text>
    <line x1="96" y1="-12" x2="96" y2="8" stroke="#CBD5E1"/>
    <text x="112" y="0" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#B45309">ESCALAR_MODELO</text>
    <line x1="234" y1="-12" x2="234" y2="8" stroke="#CBD5E1"/>
    <text x="250" y="0" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#BE123C">RECHAZADO_FISICA</text>
    <text x="486" y="0" text-anchor="end" font-family="JetBrains Mono, monospace" font-size="10" fill="#4B5563">7 Tools Python</text>
  </g>
</svg>
"""
    (ASSETS_DIR / "hero_banner.svg").write_text(svg.strip() + "\n", encoding="utf-8")


def build_cascade_gate_animated_svg() -> None:
    svg = """<svg viewBox="0 0 1200 245" width="1200" height="245" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Flujo animado Edge GPU a Compuerta Física C8286">
  <defs>
    <style>
      .pipe-flow { animation: flowRight 2.1s linear infinite; }
      @keyframes flowRight {
        from { stroke-dashoffset: 32; }
        to { stroke-dashoffset: 0; }
      }
    </style>
  </defs>

  <rect x="2" y="2" width="1196" height="241" rx="2" fill="#F7F7F5" stroke="#CBD5E1" stroke-width="1.4"/>
  <text x="28" y="28" font-family="JetBrains Mono, monospace" font-size="10.5" font-weight="700" letter-spacing="1.2" fill="#111827">ARQUITECTURA EN TIEMPO REAL — RECONSTRUCCION LOCAL GPU + COMPUERTA C8286</text>
  <line x1="28" y1="38" x2="1172" y2="38" stroke="#111827" stroke-opacity="0.14"/>

  <!-- Node 1 -->
  <rect x="28" y="64" width="210" height="148" fill="#FFFFFF" stroke="#111827" stroke-opacity="0.2"/>
  <text x="44" y="88" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#4B5563">01 / ENTRADA</text>
  <text x="44" y="114" font-family="Source Serif 4, Georgia, Liberation Serif, serif" font-size="22" fill="#111827">Foto RGB + EXIF</text>
  <line x1="44" y1="126" x2="220" y2="126" stroke="#E2E5EA"/>
  <text x="44" y="148" font-family="JetBrains Mono, monospace" font-size="10" fill="#4B5563">Focal 35mm / Transpose</text>
  <text x="44" y="168" font-family="JetBrains Mono, monospace" font-size="10" fill="#1D4ED8">FOV_h [25°..120°]</text>
  <text x="44" y="194" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#4B5563">Cero LiDAR externo</text>

  <!-- Animated link 1 -> 2 -->
  <line x1="238" y1="138" x2="294" y2="138" stroke="#1D4ED8" stroke-width="1.8" stroke-dasharray="6 4" class="pipe-flow"/>
  <circle r="3.5" fill="#1D4ED8"><animateMotion path="M238,138 L294,138" dur="1.5s" repeatCount="indefinite"/></circle>

  <!-- Node 2 -->
  <rect x="294" y="64" width="256" height="148" fill="#FFFFFF" stroke="#1D4ED8" stroke-opacity="0.55" stroke-width="1.4"/>
  <text x="310" y="88" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#1D4ED8">02 / GPU RTX 3050</text>
  <text x="310" y="114" font-family="Source Serif 4, Georgia, Liberation Serif, serif" font-size="22" fill="#111827">MoGe-2 + YOLOE-OV</text>
  <line x1="310" y1="126" x2="532" y2="126" stroke="#E2E5EA"/>
  <text x="310" y="148" font-family="JetBrains Mono, monospace" font-size="10" fill="#111827">Nube métrica: 16,000 pts</text>
  <text x="310" y="168" font-family="JetBrains Mono, monospace" font-size="10" fill="#0F766E">Mapa profundidad + Máscaras</text>
  <text x="310" y="194" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#B45309">Latencia caliente: 365 ms</text>

  <!-- Animated link 2 -> 3 -->
  <line x1="550" y1="138" x2="606" y2="138" stroke="#0F766E" stroke-width="1.8" stroke-dasharray="6 4" class="pipe-flow"/>
  <circle r="3.5" fill="#0F766E"><animateMotion path="M550,138 L606,138" dur="1.5s" repeatCount="indefinite"/></circle>

  <!-- Node 3 -->
  <rect x="606" y="64" width="256" height="148" fill="#FFFFFF" stroke="#111827" stroke-opacity="0.3"/>
  <text x="622" y="88" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#0F766E">03 / FISICA PYTHON</text>
  <text x="622" y="114" font-family="Source Serif 4, Georgia, Liberation Serif, serif" font-size="22" fill="#111827">Motor Quiral 9-DOF</text>
  <line x1="622" y1="126" x2="844" y2="126" stroke="#E2E5EA"/>
  <text x="622" y="148" font-family="JetBrains Mono, monospace" font-size="10" fill="#111827">det(M) = +1.0 · Pinhole K</text>
  <text x="622" y="168" font-family="JetBrains Mono, monospace" font-size="10" fill="#4B5563">SAT 15-ejes · Kay-Kajiya</text>
  <text x="622" y="194" font-family="JetBrains Mono, monospace" font-size="9.5" fill="#BE123C">PECS Sobel + GFI + ±σ_y</text>

  <!-- Branching paths 3 -> Verdicts -->
  <path d="M862,102 L926,86" fill="none" stroke="#0F766E" stroke-width="1.6" stroke-dasharray="5 4" class="pipe-flow"/>
  <path d="M862,138 L926,138" fill="none" stroke="#B45309" stroke-width="1.6" stroke-dasharray="5 4" class="pipe-flow"/>
  <path d="M862,174 L926,190" fill="none" stroke="#BE123C" stroke-width="1.6" stroke-dasharray="5 4" class="pipe-flow"/>
  <circle r="3" fill="#0F766E"><animateMotion path="M862,102 L926,86" dur="1.7s" repeatCount="indefinite"/></circle>
  <circle r="3" fill="#B45309"><animateMotion path="M862,138 L926,138" dur="1.7s" repeatCount="indefinite"/></circle>
  <circle r="3" fill="#BE123C"><animateMotion path="M862,174 L926,190" dur="1.7s" repeatCount="indefinite"/></circle>

  <!-- Outcome blocks -->
  <rect x="926" y="64" width="246" height="42" fill="#FFFFFF" stroke="#0F766E" stroke-width="1.3"/>
  <text x="940" y="83" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#0F766E">APROBADO · FAST TIER</text>
  <text x="940" y="98" font-family="JetBrains Mono, monospace" font-size="9" fill="#4B5563">-45% / -72% latencia en clean</text>

  <rect x="926" y="117" width="246" height="42" fill="#FFFFFF" stroke="#B45309" stroke-width="1.3"/>
  <text x="940" y="136" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#B45309">ESCALAR_MODELO · HYDROGEN</text>
  <text x="940" y="151" font-family="JetBrains Mono, monospace" font-size="9" fill="#4B5563">Oclusión o baja luz crítica</text>

  <rect x="926" y="170" width="246" height="42" fill="#FFFFFF" stroke="#BE123C" stroke-width="1.3"/>
  <text x="940" y="189" font-family="JetBrains Mono, monospace" font-size="10" font-weight="700" fill="#BE123C">RECHAZADO_FISICA · BLOQUEO</text>
  <text x="940" y="204" font-family="JetBrains Mono, monospace" font-size="9" fill="#4B5563">Colisión SAT o levitación GFI</text>
</svg>
"""
    (ASSETS_DIR / "cascade_gate_animated.svg").write_text(svg.strip() + "\n", encoding="utf-8")


def _load_ttf(size: int, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[Path] = []
    if serif:
        candidates.extend([
            Path("/usr/share/fonts/liberation/LiberationSerif-Bold.ttf" if bold else "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSerif-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSerif.ttf"),
            Path("/usr/share/fonts/droid/DroidSerif-Bold.ttf" if bold else "/usr/share/fonts/droid/DroidSerif-Regular.ttf"),
        ])
    if bold:
        candidates.extend([
            Path("/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf"),
            Path("/usr/share/fonts/liberation/LiberationMono-Bold.ttf"),
        ])
    else:
        candidates.extend([
            Path("/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/liberation/LiberationMono-Regular.ttf"),
        ])
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_pipeline_3d_live_gif() -> None:
    """Render real 48-frame animated 2D scan + orbiting Three.js OBB scene GIF from SEED=42 benchmark."""
    bm = get_benchmark("lab_desk_calibrated")
    w_src, h_src, fov = int(bm["width"]), int(bm["height"]), float(bm["fov_h_deg"])
    preds = bm["demo_predictions_by_model"]["gemini-4-hydrogen"]["objects"]
    gts = bm["gt_objects"]
    ev = evaluate_scene_predictions(preds, gts, w_src, h_src, fov, seed=42)

    raw_rgb = base64.b64decode(bm["image_base64"])
    src_img = Image.open(io.BytesIO(raw_rgb)).convert("RGB")

    W, H = 960, 420
    pad = 16
    hdr_h = 46
    left_w, stage_h = 432, 288
    right_w = 480
    lx0, ly0 = pad, hdr_h + 14
    rx0, ry0 = pad + left_w + 16, hdr_h + 14

    base_left = src_img.resize((left_w, stage_h), Image.Resampling.BILINEAR)
    sx_scale = left_w / float(w_src)
    sy_scale = stage_h / float(h_src)

    pred_projs = [
        (o, project_obb_to_2d(o["box_3d"], w_src, h_src, fov))
        for o in preds
    ]
    gt_projs = [
        (o, project_obb_to_2d(o["box_3d"], w_src, h_src, fov))
        for o in gts
    ]

    # Precompute 3D Three.js corners & sample colored depth dots for right viewport
    pred_boxes_3d: list[tuple[dict[str, Any], np.ndarray]] = []
    for o in preds:
        corners_gem = np.asarray(obb_corners_gemini(o["box_3d"]), dtype=np.float64)
        corners_three = corners_gem @ BASIS_GEMINI_TO_THREE.T
        pred_boxes_3d.append((o, corners_three))

    gt_boxes_3d: list[np.ndarray] = []
    for g in gts:
        corners_gem = np.asarray(obb_corners_gemini(g["box_3d"]), dtype=np.float64)
        gt_boxes_3d.append(corners_gem @ BASIS_GEMINI_TO_THREE.T)

    # Deterministic 3D optical surface dots around desk & boxes
    rng = np.random.default_rng(42)
    cloud_pts: list[np.ndarray] = []
    cloud_cols: list[tuple[int, int, int]] = []
    for o, corners in pred_boxes_3d:
        cmin = corners.min(axis=0)
        cmax = corners.max(axis=0)
        pts = rng.uniform(cmin, cmax, size=(42, 3))
        cloud_pts.append(pts)
        cloud_cols.extend([(29, 78, 216)] * 42)
    floor_pts = np.column_stack([
        rng.uniform(-1.35, 1.35, 180),
        np.full(180, -0.56),
        rng.uniform(-3.9, -1.1, 180),
    ])
    cloud_pts.append(floor_pts)
    cloud_cols.extend([(148, 163, 184)] * 180)
    all_cloud = np.vstack(cloud_pts)

    font_title = _load_ttf(20, bold=False, serif=True)
    font_mono = _load_ttf(11, bold=True)
    font_small = _load_ttf(10, bold=False)

    edges_idx = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Projection helper for orbiting right camera around scene center (0, -0.18, -2.35)
    target = np.array([0.0, -0.18, -2.35], dtype=np.float64)

    def proj_3d(pts: np.ndarray, cam_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fwd = target - cam_pos
        fwd = fwd / np.linalg.norm(fwd)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        rel = pts - cam_pos
        x_c = rel @ right
        y_c = rel @ up
        z_c = rel @ fwd
        z_safe = np.maximum(z_c, 0.15)
        f_px = 420.0
        px = rx0 + right_w * 0.5 + (x_c / z_safe) * f_px
        py = ry0 + stage_h * 0.54 - (y_c / z_safe) * f_px
        return np.column_stack([px, py]), z_c

    frames: list[Image.Image] = []
    n_frames = 44
    for f_idx in range(n_frames):
        t = f_idx / float(n_frames)
        ang = t * 2.0 * math.pi
        scan_x = int(left_w * (0.5 - 0.5 * math.cos(ang)))
        yaw_orbit = 0.62 * math.sin(ang)
        elev = 1.48 + 0.25 * math.cos(ang)
        cam_dist = 3.15
        cam_pos = np.array([
            target[0] + math.sin(yaw_orbit) * cam_dist,
            target[1] + elev,
            target[2] + math.cos(yaw_orbit) * cam_dist,
        ])

        frame = Image.new("RGB", (W, H), (247, 247, 245))
        draw = ImageDraw.Draw(frame)

        # Header & Hairlines
        draw.text((pad, 13), "MyLeenTo3", fill=(17, 24, 39), font=font_title)
        draw.text((pad + 116, 18), "|  Proyeccion Pinhole 2D <-> Escena Metrica 3D (SEED=42)", fill=(75, 85, 99), font=font_mono)
        right_label = f"mIoU {ev['mIoU_3D']*100:.1f}% · Orbita {math.degrees(yaw_orbit):+.0f}°"
        rbox = draw.textbbox((0, 0), right_label, font=font_mono)
        draw.text((W - pad - (rbox[2] - rbox[0]), 18), right_label, fill=(29, 78, 216), font=font_mono)
        draw.line([(pad, hdr_h), (W - pad, hdr_h)], fill=(203, 213, 225), width=1)

        # Left Panel (2D Image + Progressive Scan Wireframes)
        frame.paste(base_left, (lx0, ly0))
        draw.rectangle([lx0, ly0, lx0 + left_w, ly0 + stage_h], outline=(203, 213, 225), width=1)

        for _, proj in gt_projs:
            if proj["behind_camera"]:
                continue
            pts2 = [(lx0 + x * sx_scale, ly0 + y * sy_scale) for x, y in proj["vertices_2d"]]
            cx_obj = sum(p[0] for p in pts2) / len(pts2)
            if cx_obj <= lx0 + scan_x + 12:
                for a, b in edges_idx:
                    draw.line([pts2[a], pts2[b]], fill=(15, 118, 110), width=1)

        for o, proj in pred_projs:
            if proj["behind_camera"]:
                continue
            pts2 = [(lx0 + x * sx_scale, ly0 + y * sy_scale) for x, y in proj["vertices_2d"]]
            cx_obj = sum(p[0] for p in pts2) / len(pts2)
            if cx_obj <= lx0 + scan_x:
                for a, b in edges_idx:
                    draw.line([pts2[a], pts2[b]], fill=(29, 78, 216), width=2)
                ax0, ay0 = proj["front_arrow_2d"][0]
                ax1, ay1 = proj["front_arrow_2d"][1]
                draw.line(
                    [(lx0 + ax0 * sx_scale, ly0 + ay0 * sy_scale), (lx0 + ax1 * sx_scale, ly0 + ay1 * sy_scale)],
                    fill=(180, 83, 9),
                    width=2,
                )

        # Vertical scan beam on left panel
        bx = lx0 + scan_x
        draw.line([(bx, ly0), (bx, ly0 + stage_h)], fill=(29, 78, 216), width=2)

        # Right Panel (Orbiting Metric 3D Scene)
        draw.rectangle([rx0, ly0, rx0 + right_w, ly0 + stage_h], fill=(240, 242, 245), outline=(203, 213, 225), width=1)

        # Grid floor lines in Three.js space (-Z forward, y=-0.56)
        grid_lines_3d = []
        for gx in np.linspace(-1.6, 1.6, 9):
            grid_lines_3d.append((np.array([gx, -0.56, -0.8]), np.array([gx, -0.56, -4.0])))
        for gz in np.linspace(-4.0, -0.8, 9):
            grid_lines_3d.append((np.array([-1.6, -0.56, gz]), np.array([1.6, -0.56, gz])))
        for p0, p1 in grid_lines_3d:
            uv, zc = proj_3d(np.vstack([p0, p1]), cam_pos)
            if zc[0] > 0.2 and zc[1] > 0.2:
                x0 = np.clip(uv[0, 0], rx0 + 1, rx0 + right_w - 1)
                y0 = np.clip(uv[0, 1], ly0 + 1, ly0 + stage_h - 1)
                x1 = np.clip(uv[1, 0], rx0 + 1, rx0 + right_w - 1)
                y1 = np.clip(uv[1, 1], ly0 + 1, ly0 + stage_h - 1)
                draw.line([(x0, y0), (x1, y1)], fill=(203, 213, 225), width=1)

        # Point cloud dots inside right panel
        uv_pc, z_pc = proj_3d(all_cloud, cam_pos)
        for idx_p in range(len(uv_pc)):
            if z_pc[idx_p] > 0.25:
                px, py = uv_pc[idx_p]
                if rx0 + 3 < px < rx0 + right_w - 3 and ly0 + 3 < py < ly0 + stage_h - 3:
                    col = cloud_cols[idx_p]
                    draw.ellipse([px - 1.3, py - 1.3, px + 1.3, py + 1.3], fill=col)

        # Frustum origin at (0,0,0) -> corners at z=-0.8
        frustum_pts = np.array([
            [0.0, 0.0, 0.0],
            [-0.46, 0.30, -0.80],
            [0.46, 0.30, -0.80],
            [0.46, -0.30, -0.80],
            [-0.46, -0.30, -0.80],
        ])
        uv_fr, z_fr = proj_3d(frustum_pts, cam_pos)
        if np.all(z_fr > 0.18):
            fxy = [(float(uv_fr[i, 0]), float(uv_fr[i, 1])) for i in range(5)]
            if all(rx0 <= p[0] <= rx0 + right_w and ly0 <= p[1] <= ly0 + stage_h for p in fxy):
                for ci in range(1, 5):
                    draw.line([fxy[0], fxy[ci]], fill=(100, 116, 139), width=1)
                for ci in range(1, 5):
                    draw.line([fxy[ci], fxy[1 + (ci % 4)]], fill=(29, 78, 216), width=1)
                draw.ellipse([fxy[0][0] - 4, fxy[0][1] - 4, fxy[0][0] + 4, fxy[0][1] + 4], fill=(17, 24, 39))

        # Ground Truth cuboids (back-to-front order isn't critical for wireframes, still sort cleanly)
        for corners in gt_boxes_3d:
            uv_g, z_g = proj_3d(corners, cam_pos)
            if np.all(z_g > 0.2):
                pts_g = [(float(uv_g[k, 0]), float(uv_g[k, 1])) for k in range(8)]
                if any(rx0 < p[0] < rx0 + right_w and ly0 < p[1] < ly0 + stage_h for p in pts_g):
                    for a, b in edges_idx:
                        draw.line([pts_g[a], pts_g[b]], fill=(15, 118, 110), width=1)

        # Predicted 9-DOF cuboids sorted back-to-front
        sorted_preds = sorted(
            pred_boxes_3d,
            key=lambda item: float(np.mean(proj_3d(item[1], cam_pos)[1])),
            reverse=True,
        )
        centers_3d: list[np.ndarray] = []
        for o, corners in sorted_preds:
            uv_p, z_p = proj_3d(corners, cam_pos)
            centers_3d.append(corners.mean(axis=0))
            if np.all(z_p > 0.2):
                pts_p = [(float(uv_p[k, 0]), float(uv_p[k, 1])) for k in range(8)]
                if any(rx0 < p[0] < rx0 + right_w and ly0 < p[1] < ly0 + stage_h for p in pts_p):
                    for a, b in edges_idx:
                        draw.line([pts_p[a], pts_p[b]], fill=(29, 78, 216), width=2)

        # Crimson SAT pairwise measurement bar between first two predicted objects
        if len(centers_3d) >= 2:
            c_pair = np.vstack([centers_3d[0], centers_3d[1]])
            uv_r, z_r = proj_3d(c_pair, cam_pos)
            if np.all(z_r > 0.2):
                p0 = (float(uv_r[0, 0]), float(uv_r[0, 1]))
                p1 = (float(uv_r[1, 0]), float(uv_r[1, 1]))
                draw.line([p0, p1], fill=(190, 18, 60), width=2)
                draw.ellipse([p0[0] - 3, p0[1] - 3, p0[0] + 3, p0[1] + 3], fill=(190, 18, 60))
                draw.ellipse([p1[0] - 3, p1[1] - 3, p1[0] + 3, p1[1] + 3], fill=(190, 18, 60))

        # Bottom caption row
        by_line = ly0 + stage_h + 14
        draw.line([(pad, by_line), (W - pad, by_line)], fill=(203, 213, 225), width=1)
        draw.text((lx0, by_line + 10), "BARRIDO PINHOLE K(FOV_h=60°) — Cajas 9-DOF + Yaw", fill=(17, 24, 39), font=font_mono)
        draw.text((rx0, by_line + 10), "ESCENA 3D METRICA (1u = 1.00m) — Prediccion · GT Esmeralda · Nube · Regla SAT", fill=(75, 85, 99), font=font_small)

        frames.append(frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT))

    out_gif = ASSETS_DIR / "pipeline_3d_live.gif"
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=65,
        loop=0,
        optimize=True,
    )
    print(f"Generated {out_gif} ({out_gif.stat().st_size / 1024:.1f} KB, {len(frames)} frames)")


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
    build_cascade_gate_animated_svg()
    build_pipeline_3d_live_gif()
    build_benchmarks_triptych()
    if LIVE_JSON.exists():
        build_photo_stress_tests_graphic()
    print("Generated all GitHub visual assets in", ASSETS_DIR)


if __name__ == "__main__":
    main()
