"""Deterministic synthetic optical/architectural 3D benchmarks for MyLeenTo3.

Generates 3 calibrated 960x640 (FOV_h = 60 deg) lab scenes rendered via exact
pinhole perspective projection with Lambertian-shaded polyhedra, metric floor
grids, and contact shadows, plus realistic offline model prediction profiles for
gemini-4-hydrogen, gemini-4-argon, and gemini-3.5-flash-lite.
"""

from __future__ import annotations

import base64
import functools
import io
import math
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .geometry_3d import (
    R_VIEW_GEMINI_TO_OPENCV,
    gemini_to_threejs_pose,
    obb_corners_gemini,
    project_obb_to_2d,
)

SEED: int = 42
DEFAULT_WIDTH: int = 960
DEFAULT_HEIGHT: int = 640
DEFAULT_FOV_H_DEG: float = 60.0

# Outward-facing quad index definitions matching obb_corners_gemini ordering
_OBB_FACES: list[tuple[int, int, int, int]] = [
    (0, 3, 2, 1),  # +Z top
    (4, 5, 6, 7),  # -Z bottom
    (4, 7, 3, 0),  # -Y near
    (5, 1, 2, 6),  # +Y far
    (4, 0, 1, 5),  # -X left
    (7, 6, 2, 3),  # +X right
]

# Editorial matte architectural & optical-lab palette (zero neon)
_PALETTE: list[tuple[int, int, int]] = [
    (68, 96, 128),    # Slate cobalt
    (168, 108, 82),   # Matte terracotta
    (96, 128, 108),   # Architectural sage
    (92, 98, 108),    # Brushed graphite
    (184, 152, 98),   # Muted ochre
    (124, 106, 132),  # Dusty plum slate
]

_LIGHT_DIR: np.ndarray = np.array([-0.38, -0.52, 0.765], dtype=float)
_LIGHT_DIR /= float(np.linalg.norm(_LIGHT_DIR))


def _project_pt(
    pt_gem: np.ndarray, width: int, height: int, fov_h_deg: float
) -> tuple[float, float, float]:
    f = float(width) / (2.0 * math.tan(math.radians(float(fov_h_deg)) / 2.0))
    cx, cy = width * 0.5, height * 0.5
    p_cv = R_VIEW_GEMINI_TO_OPENCV @ pt_gem
    z = max(float(p_cv[2]), 0.05)
    u = f * (float(p_cv[0]) / z) + cx
    v = f * (float(p_cv[1]) / z) + cy
    return u, v, float(p_cv[2])


def render_scene_png_base64(
    objects: list[dict[str, Any]],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
    floor_z: float = -0.55,
    max_grid_y: float = 6.5,
    scene_seed: int = SEED,
) -> str:
    """Render 3D OBB objects into a crisp 2x-supersampled matte architectural PNG (base64)."""
    scale = 2
    rw, rh = width * scale, height * scale

    img = Image.new("RGB", (rw, rh), (242, 240, 236))
    draw = ImageDraw.Draw(img, "RGBA")

    # Soft studio vertical gradient wall (warm ivory -> cool stone near horizon)
    horizon_y = int(rh * 0.5)
    for row in range(0, rh, 2):
        t = row / float(rh)
        if row <= horizon_y:
            u = row / float(max(horizon_y, 1))
            r = int(246 - 14 * u)
            g = int(245 - 13 * u)
            b = int(241 - 10 * u)
        else:
            u = (row - horizon_y) / float(max(rh - horizon_y, 1))
            r = int(224 + 10 * u)
            g = int(223 + 10 * u)
            b = int(219 + 9 * u)
        draw.rectangle([(0, row), (rw, row + 2)], fill=(r, g, b))

    # Metric checkerboard floor plane (0.5m tiles) at z = floor_z
    tile = 0.5
    x_min_grid, x_max_grid = -4.0, 4.0
    y_min_grid = 0.8
    nx = int((x_max_grid - x_min_grid) / tile)
    ny = int((max_grid_y - y_min_grid) / tile)

    for iy in range(ny - 1, -1, -1):
        y0 = y_min_grid + iy * tile
        y1 = y0 + tile
        depth_fade = min(1.0, max(0.0, (y0 - y_min_grid) / (max_grid_y - y_min_grid)))
        for ix in range(nx):
            x0 = x_min_grid + ix * tile
            x1 = x0 + tile
            corners_tile = [
                np.array([x0, y0, floor_z]),
                np.array([x1, y0, floor_z]),
                np.array([x1, y1, floor_z]),
                np.array([x0, y1, floor_z]),
            ]
            poly = [_project_pt(c, rw, rh, fov_h_deg)[:2] for c in corners_tile]
            is_alt = (ix + iy) % 2 == 0
            base_c = 232 if is_alt else 222
            # Subtle atmospheric perspective blend toward horizon tone
            c_r = int(base_c * (1.0 - 0.12 * depth_fade) + 232 * 0.12 * depth_fade)
            c_g = int((base_c - 1) * (1.0 - 0.12 * depth_fade) + 232 * 0.12 * depth_fade)
            c_b = int((base_c - 4) * (1.0 - 0.12 * depth_fade) + 230 * 0.12 * depth_fade)
            draw.polygon(poly, fill=(c_r, c_g, c_b, 255), outline=(206, 204, 198, 190))

    # Horizon architectural baseboard line
    hz_l = _project_pt(np.array([-6.0, max_grid_y, floor_z]), rw, rh, fov_h_deg)[:2]
    hz_r = _project_pt(np.array([6.0, max_grid_y, floor_z]), rw, rh, fov_h_deg)[:2]
    draw.line([hz_l, hz_r], fill=(178, 176, 170, 255), width=2)

    # Soft contact shadows projected onto floor_z beneath each object
    for obj in objects:
        corners = obb_corners_gemini(obj["box_3d"])
        # Project bottom 4 corners slightly offset away from light onto floor_z
        shadow_pts = []
        for idx in (4, 5, 6, 7):
            pt = corners[idx]
            dz_above = max(0.0, float(pt[2] - floor_z))
            sx = float(pt[0]) + 0.14 * (dz_above + 0.08)
            sy = float(pt[1]) + 0.12 * (dz_above + 0.08)
            shadow_pts.append(_project_pt(np.array([sx, sy, floor_z + 0.002]), rw, rh, fov_h_deg)[:2])
        draw.polygon(shadow_pts, fill=(45, 50, 58, 38))

    # Gather all visible faces across all objects for global painter's depth sort
    face_records: list[tuple[float, float, list[tuple[float, float]], tuple[int, int, int], tuple[int, int, int]]] = []
    for obj_idx, obj in enumerate(objects):
        corners = obb_corners_gemini(obj["box_3d"])
        color_rgb = obj.get("color_rgb")
        if color_rgb is None:
            color_rgb = _PALETTE[(obj_idx + scene_seed) % len(_PALETTE)]
        br, bg, bb = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])

        for face in _OBB_FACES:
            v0, v1, v2, v3 = [corners[k] for k in face]
            normal = np.cross(v1 - v0, v2 - v0)
            n_norm = float(np.linalg.norm(normal))
            if n_norm < 1e-12:
                continue
            normal = normal / n_norm
            face_center = (v0 + v1 + v2 + v3) * 0.25

            # Back-face culling in camera space (camera at origin)
            if float(np.dot(normal, -face_center)) <= 0.0:
                continue

            # Lambertian diffuse + hemisphere ambient illumination
            ndotl = max(0.0, float(np.dot(normal, _LIGHT_DIR)))
            hemi = 0.5 + 0.5 * float(normal[2])
            shade = 0.44 + 0.18 * hemi + 0.46 * ndotl

            # Slight warm specular-free contrast for top vs side
            r = int(np.clip(br * shade, 24, 248))
            g = int(np.clip(bg * shade, 24, 248))
            b = int(np.clip(bb * shade, 24, 248))
            edge_rgb = (max(20, r - 48), max(20, g - 48), max(20, b - 48))

            poly_2d = [_project_pt(v, rw, rh, fov_h_deg)[:2] for v in (v0, v1, v2, v3)]
            mean_depth = float(face_center[1])
            max_depth = float(max(v[1] for v in (v0, v1, v2, v3)))
            face_records.append((mean_depth, max_depth, poly_2d, (r, g, b), edge_rgb))

    # Draw far-to-near (painter's algorithm)
    face_records.sort(key=lambda item: (-item[0], -item[1]))
    for _mean_d, _max_d, poly_2d, fill_rgb, edge_rgb in face_records:
        draw.polygon(poly_2d, fill=(*fill_rgb, 255))
        # Crisp anti-aliased edge outline
        closed = poly_2d + [poly_2d[0]]
        draw.line(closed, fill=(*edge_rgb, 235), width=2, joint="curve")

    # Subtle optical crosshair calibration marks in corners
    mark = (130, 134, 142, 160)
    margin = 28 * scale
    arm = 14 * scale
    for cx_m, cy_m, sx, sy in (
        (margin, margin, 1, 1),
        (rw - margin, margin, -1, 1),
        (margin, rh - margin, 1, -1),
        (rw - margin, rh - margin, -1, -1),
    ):
        draw.line([(cx_m, cy_m), (cx_m + sx * arm, cy_m)], fill=mark, width=2)
        draw.line([(cx_m, cy_m), (cx_m, cy_m + sy * arm)], fill=mark, width=2)

    img_final = img.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img_final.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _enrich_gt_object(
    obj_id: str,
    label: str,
    box_3d: Sequence[float],
    color_rgb: tuple[int, int, int],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
) -> dict[str, Any]:
    box = [round(float(v), 4) for v in box_3d]
    proj = project_obb_to_2d(box, width, height, fov_h_deg)
    pose = gemini_to_threejs_pose(box)
    vol = float(abs(box[3]) * abs(box[4]) * abs(box[5]))
    return {
        "id": obj_id,
        "label": label,
        "box_3d": box,
        "box_2d": proj["box_2d_reproj"],
        "center_2d": proj["center_2d"],
        "projected_2d": proj,
        "threejs_pose": pose,
        "volume_m3": round(vol, 5),
        "distance_m": round(float(math.sqrt(box[0] ** 2 + box[1] ** 2 + box[2] ** 2)), 4),
        "color_rgb": list(color_rgb),
    }


def _build_model_demo_predictions(
    gt_objects: list[dict[str, Any]],
    scene_offset: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov_h_deg: float = DEFAULT_FOV_H_DEG,
) -> dict[str, dict[str, Any]]:
    """Generate deterministic, calibrated perturbation profiles for the 3 Gemini models."""
    profiles: dict[str, dict[str, Any]] = {
        "gemini-4-hydrogen": {
            "seed_delta": 101,
            "xy_sigma": 0.022,
            "z_sigma": 0.015,
            "scale_sigma": 0.028,
            "rp_sigma": 1.2,
            "yaw_sigma": 3.2,
            "depth_bias": 0.008,
            "latency_ms": 1920 + scene_offset * 145,
            "thinking_level": "high",
            "usage": {
                "input_tokens": 1248,
                "output_tokens": 412 + scene_offset * 38,
                "thinking_tokens": 1640 + scene_offset * 190,
            },
            "conf_base": 0.94,
        },
        "gemini-4-argon": {
            "seed_delta": 202,
            "xy_sigma": 0.052,
            "z_sigma": 0.032,
            "scale_sigma": 0.068,
            "rp_sigma": 2.4,
            "yaw_sigma": 7.5,
            "depth_bias": 0.025,
            "latency_ms": 980 + scene_offset * 85,
            "thinking_level": "medium",
            "usage": {
                "input_tokens": 1248,
                "output_tokens": 376 + scene_offset * 30,
                "thinking_tokens": 720 + scene_offset * 95,
            },
            "conf_base": 0.87,
        },
        "gemini-3.5-flash-lite": {
            "seed_delta": 303,
            "xy_sigma": 0.098,
            "z_sigma": 0.055,
            "scale_sigma": 0.125,
            "rp_sigma": 4.2,
            "yaw_sigma": 14.5,
            "depth_bias": 0.055,
            "latency_ms": 465 + scene_offset * 42,
            "thinking_level": "low",
            "usage": {
                "input_tokens": 1248,
                "output_tokens": 318 + scene_offset * 24,
                "thinking_tokens": 180 + scene_offset * 35,
            },
            "conf_base": 0.78,
        },
    }

    output: dict[str, dict[str, Any]] = {}
    for model_name, cfg in profiles.items():
        rng = np.random.default_rng(SEED + scene_offset * 37 + cfg["seed_delta"])
        preds: list[dict[str, Any]] = []
        for idx, gt in enumerate(gt_objects):
            gx, gy, gz, gdx, gdy, gdz, gr, gp, gyaw = gt["box_3d"]
            depth_factor = max(0.8, gy / 2.5)

            px = gx + float(rng.normal(0.0, cfg["xy_sigma"] * depth_factor))
            py = gy + cfg["depth_bias"] * depth_factor + float(rng.normal(0.0, cfg["xy_sigma"] * 1.15 * depth_factor))
            pz = gz + float(rng.normal(0.0, cfg["z_sigma"]))

            pdx = max(0.08, gdx * (1.0 + float(rng.normal(0.0, cfg["scale_sigma"]))))
            pdy = max(0.08, gdy * (1.0 + float(rng.normal(0.0, cfg["scale_sigma"]))))
            pdz = max(0.06, gdz * (1.0 + float(rng.normal(0.0, cfg["scale_sigma"]))))

            # Keep upright boxes nearly upright unless tilted in GT
            rp_scale = 1.0 if (abs(gr) > 1.0 or abs(gp) > 1.0) else 0.35
            pr = gr + float(rng.normal(0.0, cfg["rp_sigma"] * rp_scale))
            pp = gp + float(rng.normal(0.0, cfg["rp_sigma"] * rp_scale))
            pyaw = gyaw + float(rng.normal(0.0, cfg["yaw_sigma"]))

            pred_box = [round(v, 4) for v in (px, py, pz, pdx, pdy, pdz, pr, pp, pyaw)]
            proj = project_obb_to_2d(pred_box, width, height, fov_h_deg)
            pose = gemini_to_threejs_pose(pred_box)
            conf = float(np.clip(cfg["conf_base"] - 0.015 * idx + rng.normal(0.0, 0.015), 0.55, 0.99))

            preds.append(
                {
                    "id": f"pred_{idx + 1}",
                    "label": gt["label"],
                    "box_3d": pred_box,
                    "confidence": round(conf, 3),
                    "projected_2d": proj,
                    "threejs_pose": pose,
                    "volume_m3": round(float(pdx * pdy * pdz), 5),
                    "distance_m": round(float(math.sqrt(px * px + py * py + pz * pz)), 4),
                }
            )

        output[model_name] = {
            "model": model_name,
            "thinking_level": cfg["thinking_level"],
            "latency_ms": cfg["latency_ms"],
            "usage": cfg["usage"],
            "objects": preds,
        }
    return output


@functools.lru_cache(maxsize=1)
def _build_all_benchmarks() -> dict[str, dict[str, Any]]:
    width, height, fov_h = DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FOV_H_DEG

    # 1) Calibrated optical lab desk (5 objects, 1.6m..3.4m, desk at z = -0.45m)
    desk_z = -0.45
    scene1_specs = [
        ("gt_1", "oscilloscope", [-0.72, 2.35, desk_z + 0.16, 0.42, 0.28, 0.32, 0.0, 0.0, 18.0], (68, 96, 128)),
        ("gt_2", "sample_case", [-0.14, 1.72, desk_z + 0.10, 0.34, 0.26, 0.20, 0.0, 0.0, -12.0], (168, 108, 82)),
        ("gt_3", "laptop", [0.42, 2.02, desk_z + 0.09, 0.36, 0.25, 0.18, 0.0, 0.0, -24.0], (92, 98, 108)),
        ("gt_4", "spectrometer", [0.88, 3.10, desk_z + 0.21, 0.48, 0.36, 0.42, 0.0, 0.0, 10.0], (96, 128, 108)),
        ("gt_5", "calibration_cube", [-0.18, 3.30, desk_z + 0.15, 0.30, 0.30, 0.30, 0.0, 0.0, 32.0], (184, 152, 98)),
    ]
    gt1 = [_enrich_gt_object(oid, lbl, box, col, width, height, fov_h) for oid, lbl, box, col in scene1_specs]
    img1 = render_scene_png_base64(gt1, width, height, fov_h, floor_z=desk_z, max_grid_y=5.5, scene_seed=1)

    # 2) Occlusion corridor (6 obstacles, 1.6m..5.1m, deliberate partial & strong occlusions)
    corr_z = -0.50
    scene2_specs = [
        ("gt_1", "front_barrier", [-0.30, 1.75, corr_z + 0.25, 0.46, 0.26, 0.50, 0.0, 0.0, 6.0], (168, 108, 82)),
        ("gt_2", "sensor_pedestal", [0.34, 2.00, corr_z + 0.29, 0.40, 0.34, 0.58, 0.0, 0.0, -8.0], (68, 96, 128)),
        ("gt_3", "mid_cabinet", [-0.52, 3.10, corr_z + 0.38, 0.62, 0.44, 0.76, 0.0, 0.0, 4.0], (96, 128, 108)),
        ("gt_4", "mobile_cart", [0.56, 3.45, corr_z + 0.35, 0.58, 0.48, 0.70, 0.0, 0.0, -14.0], (184, 152, 98)),
        ("gt_5", "far_storage_crate", [-0.74, 4.55, corr_z + 0.44, 0.76, 0.54, 0.88, 0.0, 0.0, 8.0], (92, 98, 108)),
        ("gt_6", "corridor_beacon", [0.80, 5.05, corr_z + 0.48, 0.68, 0.56, 0.96, 0.0, 0.0, -5.0], (124, 106, 132)),
    ]
    gt2 = [_enrich_gt_object(oid, lbl, box, col, width, height, fov_h) for oid, lbl, box, col in scene2_specs]
    img2 = render_scene_png_base64(gt2, width, height, fov_h, floor_z=corr_z, max_grid_y=7.0, scene_seed=2)

    # 3) Rotated 9-DOF workshop (5 objects, yaw -65..+70 deg, roll/pitch ±10 deg)
    work_z = -0.60
    scene3_specs = [
        ("gt_1", "tilted_chassis", [-0.62, 2.20, work_z + 0.24, 0.44, 0.30, 0.28, 9.0, -7.0, -62.0], (68, 96, 128)),
        ("gt_2", "angled_motor_block", [0.05, 1.85, work_z + 0.21, 0.32, 0.28, 0.26, -8.5, 9.5, 42.0], (168, 108, 82)),
        ("gt_3", "optical_jig", [0.64, 2.45, work_z + 0.26, 0.46, 0.32, 0.30, 7.5, 8.0, 68.0], (96, 128, 108)),
        ("gt_4", "diagnostic_console", [-0.38, 3.35, work_z + 0.30, 0.56, 0.38, 0.36, -10.0, -6.5, -38.0], (184, 152, 98)),
        ("gt_5", "calibration_prism", [0.42, 3.55, work_z + 0.32, 0.48, 0.40, 0.38, 6.0, -9.5, 55.0], (124, 106, 132)),
    ]
    gt3 = [_enrich_gt_object(oid, lbl, box, col, width, height, fov_h) for oid, lbl, box, col in scene3_specs]
    img3 = render_scene_png_base64(gt3, width, height, fov_h, floor_z=work_z, max_grid_y=6.0, scene_seed=3)

    scenes = {
        "lab_desk_calibrated": {
            "id": "lab_desk_calibrated",
            "title": "Laboratorio Óptico Calibrado",
            "subtitle": "5 instrumentos sobre mesa métrica (1.6m – 3.4m)",
            "description": (
                "Escena de referencia calibrada con 5 objetos de laboratorio "
                "sin oclusión severa para validar precisión métrica, ATE y proyección pinhole."
            ),
            "width": width,
            "height": height,
            "fov_h_deg": fov_h,
            "image_base64": img1,
            "gt_objects": gt1,
            "demo_predictions_by_model": _build_model_demo_predictions(gt1, 0, width, height, fov_h),
        },
        "occlusion_corridor": {
            "id": "occlusion_corridor",
            "title": "Corredor con Oclusión Estructurada",
            "subtitle": "6 obstáculos alineados con oclusión parcial y fuerte (1.5m – 5.2m)",
            "description": (
                "Escena de profundidad extendida diseñada para auditar rayos de oclusión "
                "Kay-Kajiya desde la cámara (0,0,0) y degradación de profundidad monocular."
            ),
            "width": width,
            "height": height,
            "fov_h_deg": fov_h,
            "image_base64": img2,
            "gt_objects": gt2,
            "demo_predictions_by_model": _build_model_demo_predictions(gt2, 1, width, height, fov_h),
        },
        "rotated_workshop": {
            "id": "rotated_workshop",
            "title": "Taller Multi-Orientación (9-DOF)",
            "subtitle": "5 volúmenes rotados (yaw -65°..+70°, roll/pitch ±10°)",
            "description": (
                "Escena de prueba 9-DOF completa que exige intersección SAT de 15 ejes, "
                "muestreo estratificado 3D y error de orientación geodésico SO(3)."
            ),
            "width": width,
            "height": height,
            "fov_h_deg": fov_h,
            "image_base64": img3,
            "gt_objects": gt3,
            "demo_predictions_by_model": _build_model_demo_predictions(gt3, 2, width, height, fov_h),
        },
    }
    return scenes


def list_benchmarks() -> list[dict[str, Any]]:
    """Return all 3 deterministic calibrated benchmark scenes."""
    return list(_build_all_benchmarks().values())


def get_benchmark(benchmark_id: str) -> dict[str, Any]:
    """Retrieve a single calibrated benchmark scene by ID (raises KeyError if unknown)."""
    scenes = _build_all_benchmarks()
    if benchmark_id not in scenes:
        raise KeyError(f"Unknown benchmark_id: {benchmark_id!r}. Valid IDs: {list(scenes.keys())}")
    return scenes[benchmark_id]
