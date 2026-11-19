"""Local GPU-accelerated monocular metric 3D engine (EXIF + MoGe-2 ViT-S + YOLOE).

Provides:
  1. EXIF orientation & 35mm-equivalent horizontal FOV extraction.
  2. Lazy GPU/CPU inference via Microsoft MoGe-2 (`Ruicheng/moge-2-vits-normal`) for
     dense metric 3D points (meters), metric depth map, and automatic camera intrinsics.
  3. Instance mask extraction via Ultralytics YOLOE (`yoloe-11s-seg-pf.pt` prompt-free or
     text-prompted open-vocab; graceful fallback to `yolo11n-seg.pt`), plus depth-clustered
     foreground fallback if 0 masks are detected.
  4. Robust 3D OBB lifting into Gemini 9-DOF coordinates (X-Right, Y-Forward, Z-Up)
     and chirality-preserving compact binary point cloud / matte depth map export for Three.js.
"""

from __future__ import annotations

import base64
import inspect
import io
import math
import os
import re
import threading
import time
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

from ..core.real_photo_metrics import (
    compute_object_point_cloud_metrology,
    compute_real_photo_3d_metrics,
    compute_specular_mask,
    fit_ransac_dominant_plane,
)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional in CPU-only CI
    cv2 = None  # type: ignore

MOGE_MODEL_ID = os.getenv("MYLEENTO3_MOGE_MODEL", "Ruicheng/moge-2-vits-normal")
YOLOE_PF_WEIGHTS = os.getenv("MYLEENTO3_YOLOE_PF_WEIGHTS", "yoloe-11s-seg-pf.pt")
YOLOE_OV_WEIGHTS = os.getenv("MYLEENTO3_YOLOE_OV_WEIGHTS", "yoloe-11s-seg.pt")
YOLO_FALLBACK_WEIGHTS = os.getenv("MYLEENTO3_YOLO_FALLBACK", "yolo11n-seg.pt")

DEFAULT_OPEN_VOCAB_CLASSES: list[str] = [
    "person",
    "hand",
    "wire",
    "cable",
    "cord",
    "wall outlet",
    "power outlet",
    "electrical switch",
    "socket",
    "power plug",
    "laptop",
    "monitor",
    "keyboard",
    "computer mouse",
    "phone",
    "cup",
    "mug",
    "bottle",
    "book",
    "cardboard box",
    "storage crate",
    "leather bag",
    "bag",
    "backpack",
    "desk",
    "table",
    "chair",
    "tool",
    "oscilloscope",
    "electronic device",
]

_MODEL_LOCK = threading.Lock()
_MOGE_MODEL: Any = None
_SEG_MODEL: Any = None
_SEG_MODEL_KIND: str = "unloaded"
_OV_MODEL: Any = None
_OV_CLASSES_CACHE: Optional[tuple[str, ...]] = None
_LOCAL_DEVICE: str = "cpu"
_DEVICE_LABEL: str = "CPU"


def _strip_b64_header(image_base64: str) -> bytes:
    raw = image_base64.strip()
    match = re.match(r"^data:([^;]+);base64,(.*)$", raw, flags=re.DOTALL)
    if match:
        raw = match.group(2).strip()
    return base64.b64decode(raw, validate=False)


def _extract_35mm_focal_mm(img: Image.Image) -> Optional[float]:
    """Read FocalLengthIn35mmFilm (EXIF tag 41989) from top-level or Exif Sub-IFD."""
    try:
        exif = img.getexif()
        if not exif:
            return None
        val = exif.get(41989)
        if val is None and hasattr(exif, "get_ifd"):
            exif_sub = exif.get_ifd(0x8769)
            if exif_sub:
                val = exif_sub.get(41989)
        if val is None:
            return None
        f35 = float(val[0] / val[1]) if isinstance(val, tuple) and len(val) == 2 and val[1] else float(val)
        if 8.0 <= f35 <= 400.0:
            return f35
    except Exception:
        return None
    return None


def extract_exif_image_info(image_base64: str) -> tuple[Image.Image, int, int, Optional[float]]:
    """Decode base64 image, apply EXIF transpose, and compute horizontal FOV if 35mm tag exists."""
    raw_bytes = _strip_b64_header(image_base64)
    with Image.open(io.BytesIO(raw_bytes)) as src:
        f35_mm = _extract_35mm_focal_mm(src)
        transposed = ImageOps.exif_transpose(src)
        rgb = transposed.convert("RGB")
        w, h = rgb.size

    exif_fov_h_deg: Optional[float] = None
    if f35_mm is not None and w > 0 and h > 0:
        # Standard 35mm full-frame equivalent width is 36mm landscape, 24mm portrait
        sensor_w_mm = 36.0 if w >= h else 24.0
        fov_rad = 2.0 * math.atan((sensor_w_mm * 0.5) / f35_mm)
        fov_deg = math.degrees(fov_rad)
        if 20.0 <= fov_deg <= 130.0:
            exif_fov_h_deg = round(fov_deg, 2)

    return rgb, int(w), int(h), exif_fov_h_deg


def opencv_to_gemini_points(points_cv: np.ndarray) -> np.ndarray:
    """Convert OpenCV points (X-Right, Y-Down, Z-Forward) -> Gemini (X-Right, Y-Forward, Z-Up)."""
    out = np.empty_like(points_cv, dtype=np.float32)
    out[..., 0] = points_cv[..., 0]
    out[..., 1] = points_cv[..., 2]
    out[..., 2] = -points_cv[..., 1]
    return out


def opencv_to_threejs_points(points_cv: np.ndarray) -> np.ndarray:
    """Convert OpenCV points (X-Right, Y-Down, Z-Forward) -> Three.js (X-Right, Y-Up, -Z-Forward)."""
    out = np.empty_like(points_cv, dtype=np.float32)
    out[..., 0] = points_cv[..., 0]
    out[..., 1] = -points_cv[..., 1]
    out[..., 2] = -points_cv[..., 2]
    return out


def _torch_imports_available() -> bool:
    if os.getenv("MYLEENTO3_LOCAL_3D_ENGINE", "1").strip() == "0":
        return False
    try:
        import torch  # noqa: F401
        from moge.model.v2 import MoGeModel  # noqa: F401
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def is_local_3d_available() -> bool:
    """Return True when local GPU/CPU metric 3D runtime is enabled and importable."""
    return _torch_imports_available()


def get_local_3d_status() -> dict[str, Any]:
    """Inspect availability, device name, and loaded/configured local 3D models without blocking."""
    avail = _torch_imports_available()
    if not avail:
        return {
            "available": False,
            "device": "disabled",
            "models": [],
        }

    dev_name = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            dev_name = f"cuda:{torch.cuda.get_device_name(0)}"
        else:
            dev_name = "cpu"
    except Exception:
        dev_name = "cpu"

    seg_name = _SEG_MODEL_KIND if _SEG_MODEL_KIND != "unloaded" else YOLOE_OV_WEIGHTS
    return {
        "available": True,
        "device": dev_name,
        "models": [MOGE_MODEL_ID, seg_name],
    }


def _load_moge2_and_segmenter(
    custom_classes: Optional[list[str]] = None,
) -> tuple[Any, Any, str, str]:
    """Thread-safe lazy loader for MoGe-2 ViT-S and Ultralytics YOLOE/YOLO-seg."""
    global _MOGE_MODEL, _SEG_MODEL, _SEG_MODEL_KIND, _OV_MODEL, _OV_CLASSES_CACHE, _LOCAL_DEVICE, _DEVICE_LABEL

    import torch
    from moge.model.v2 import MoGeModel
    from ultralytics import YOLO

    with _MODEL_LOCK:
        if _MOGE_MODEL is None:
            if torch.cuda.is_available():
                _LOCAL_DEVICE = "cuda"
                _DEVICE_LABEL = torch.cuda.get_device_name(0)
            else:
                _LOCAL_DEVICE = "cpu"
                _DEVICE_LABEL = "CPU"

            model = MoGeModel.from_pretrained(MOGE_MODEL_ID).to(_LOCAL_DEVICE).eval()
            _MOGE_MODEL = model

        target_classes = custom_classes if custom_classes else DEFAULT_OPEN_VOCAB_CLASSES
        try:
            from ultralytics import YOLOE  # type: ignore

            if _OV_MODEL is None:
                _OV_MODEL = YOLOE(YOLOE_OV_WEIGHTS)
            target_tuple = tuple(target_classes)
            if _OV_CLASSES_CACHE != target_tuple:
                _OV_MODEL.set_classes(target_classes)
                _OV_CLASSES_CACHE = target_tuple
            _SEG_MODEL_KIND = f"YOLOE-OV ({YOLOE_OV_WEIGHTS})"
            return (
                _MOGE_MODEL,
                _OV_MODEL,
                _LOCAL_DEVICE,
                f"{MOGE_MODEL_ID.split('/')[-1]} + {_SEG_MODEL_KIND} ({_DEVICE_LABEL})",
            )
        except Exception:
            pass

        if _SEG_MODEL is None:
            try:
                from ultralytics import YOLOE  # type: ignore

                _SEG_MODEL = YOLOE(YOLOE_PF_WEIGHTS)
                _SEG_MODEL_KIND = f"YOLOE-pf ({YOLOE_PF_WEIGHTS})"
            except Exception:
                _SEG_MODEL = YOLO(YOLO_FALLBACK_WEIGHTS)
                _SEG_MODEL_KIND = f"YOLO-seg ({YOLO_FALLBACK_WEIGHTS})"

        return _MOGE_MODEL, _SEG_MODEL, _LOCAL_DEVICE, f"{MOGE_MODEL_ID.split('/')[-1]} + {_SEG_MODEL_KIND} ({_DEVICE_LABEL})"


def compute_depth_stats(depth_m: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    """Compute metric summary percentiles in meters from valid finite depths."""
    valid = depth_m[valid_mask & np.isfinite(depth_m) & (depth_m > 0.02)]
    if valid.size == 0:
        return {"min_m": 0.0, "p10_m": 0.0, "median_m": 0.0, "p90_m": 0.0, "max_m": 0.0}
    return {
        "min_m": round(float(np.percentile(valid, 1)), 3),
        "p10_m": round(float(np.percentile(valid, 10)), 3),
        "median_m": round(float(np.median(valid)), 3),
        "p90_m": round(float(np.percentile(valid, 90)), 3),
        "max_m": round(float(np.percentile(valid, 99)), 3),
    }


def render_depth_map_png(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    max_width: int = 480,
) -> tuple[str, dict[str, float]]:
    """Render metric depth array into a matte perceptual PNG colormap (amber near -> teal mid -> slate far)."""
    stats = compute_depth_stats(depth_m, valid_mask)
    h, w = depth_m.shape
    mask = valid_mask & np.isfinite(depth_m) & (depth_m > 0.02)

    if not np.any(mask):
        canvas = np.full((h, w, 3), 229, dtype=np.uint8)
    else:
        vals = depth_m[mask]
        d_lo = float(np.percentile(vals, 2))
        d_hi = max(float(np.percentile(vals, 98)), d_lo + 0.05)
        # Logarithmic compression highlights close-range structure alongside room range
        norm = (np.log1p(np.clip(depth_m, d_lo, d_hi)) - math.log1p(d_lo)) / max(
            math.log1p(d_hi) - math.log1p(d_lo), 1e-5
        )
        t = np.clip(norm, 0.0, 1.0).astype(np.float32)

        # Matte Precision Optical Lab stops (zero neon):
        # t=0.0 (near): Warm Amber #B45309 -> (180, 83, 9)
        # t=0.5 (mid):  Emerald Teal #0F766E -> (15, 118, 110)
        # t=1.0 (far):  Deep Slate #1E293B -> (30, 41, 59)
        c0 = np.array([180.0, 83.0, 9.0], dtype=np.float32)
        c1 = np.array([15.0, 118.0, 110.0], dtype=np.float32)
        c2 = np.array([30.0, 41.0, 59.0], dtype=np.float32)

        t_lo = np.clip(t * 2.0, 0.0, 1.0)[..., None]
        t_hi = np.clip((t - 0.5) * 2.0, 0.0, 1.0)[..., None]
        rgb_lo = c0 * (1.0 - t_lo) + c1 * t_lo
        rgb_hi = c1 * (1.0 - t_hi) + c2 * t_hi
        rgb_float = np.where((t <= 0.5)[..., None], rgb_lo, rgb_hi)

        canvas = np.full((h, w, 3), 229, dtype=np.uint8)
        canvas[mask] = np.clip(rgb_float[mask], 0, 255).astype(np.uint8)

    pil_depth = Image.fromarray(canvas, mode="RGB")
    if w > max_width:
        scale = max_width / float(w)
        pil_depth = pil_depth.resize((max_width, max(16, int(round(h * scale)))), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    pil_depth.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), stats


def pack_point_cloud_threejs(
    points_cv: np.ndarray,
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray,
    max_points: int = 16000,
) -> dict[str, Any]:
    """Subsample valid metric points into compact float32/uint8 base64 buffers in Three.js coordinates."""
    pts_three = opencv_to_threejs_points(points_cv)
    z_cv = points_cv[..., 2]
    keep = (
        valid_mask
        & np.isfinite(pts_three).all(axis=-1)
        & np.isfinite(z_cv)
        & (z_cv > 0.04)
        & (z_cv < 35.0)
    )
    flat_pts = pts_three[keep]
    flat_rgb = rgb_uint8[keep]
    flat_z = z_cv[keep]

    total = int(flat_pts.shape[0])
    if total == 0:
        return {
            "count": 0,
            "positions_b64": "",
            "colors_b64": "",
            "median_depth_m": 0.0,
            "bounds_three": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
        }

    if total > max_points:
        indices = np.linspace(0, total - 1, max_points, dtype=np.int64)
        flat_pts = flat_pts[indices]
        flat_rgb = flat_rgb[indices]
        flat_z = flat_z[indices]

    pts_f32 = np.ascontiguousarray(flat_pts, dtype=np.float32)
    cols_u8 = np.ascontiguousarray(flat_rgb, dtype=np.uint8)
    b_min = [round(float(v), 3) for v in np.min(pts_f32, axis=0)]
    b_max = [round(float(v), 3) for v in np.max(pts_f32, axis=0)]

    return {
        "count": int(pts_f32.shape[0]),
        "positions_b64": base64.b64encode(pts_f32.tobytes()).decode("ascii"),
        "colors_b64": base64.b64encode(cols_u8.tobytes()).decode("ascii"),
        "median_depth_m": round(float(np.median(flat_z)), 3),
        "bounds_three": {"min": b_min, "max": b_max},
    }


_AMODAL_THICKNESS_RATIO: dict[str, float] = {
    "cardboard_box": 0.65,
    "storage_crate": 0.65,
    "cup": 0.85,
    "mug": 0.85,
    "bottle": 0.85,
    "oscilloscope": 0.55,
    "monitor": 0.18,
    "laptop": 0.60,
    "book": 0.25,
    "electronic_device": 0.30,
    "electrical_switch": 0.25,
    "wall_outlet": 0.22,
    "power_outlet": 0.22,
    "socket": 0.22,
    "chair": 0.75,
    "desk": 0.55,
    "table": 0.60,
    "person": 0.45,
    "hand": 0.30,
    "bag": 0.50,
    "backpack": 0.55,
    "tool": 0.30,
    "phone": 0.15,
    "keyboard": 0.35,
    "computer_mouse": 0.65,
}


def _filter_foreground_depth_mode(pts_cv: np.ndarray) -> np.ndarray:
    """Isolate dominant foreground depth cluster to prevent background bleed in hollow/edge masks."""
    if pts_cv.shape[0] < 24:
        return pts_cv
    z = pts_cv[:, 2]
    if float(np.ptp(z)) < 0.03:
        return pts_cv

    z02, z98 = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    if z98 - z02 < 0.03:
        return pts_cv

    counts, edges = np.histogram(z, bins=24, range=(z02, z98))
    max_cnt = int(counts.max())
    min_sig = max(8, int(round(0.28 * max_cnt)), int(round(0.14 * len(z))))
    sig_bins = np.where(counts >= min_sig)[0]
    idx_mode = int(sig_bins[0]) if sig_bins.size > 0 else int(np.argmax(counts))
    z_mode = 0.5 * float(edges[idx_mode] + edges[idx_mode + 1])

    nb_win = max(0.10, 0.25 * z_mode)
    local_z = z[np.abs(z - z_mode) <= nb_win]
    mad = float(np.median(np.abs(local_z - np.median(local_z)))) if local_z.size >= 6 else 0.0
    tol = max(0.06, 0.12 * z_mode, min(0.35 * z_mode, 2.6 * mad))

    core = pts_cv[np.abs(z - z_mode) <= tol]
    if core.shape[0] >= 12:
        return core

    z_lo, z_hi = np.percentile(z, [10, 85])
    fallback = pts_cv[(z >= z_lo) & (z <= z_hi)]
    return fallback if fallback.shape[0] >= 12 else pts_cv


def _fit_lshape_bev_yaw(xy_centered: np.ndarray) -> float:
    """Estimate BEV yaw via vectorized Minimum-Area + L-Shape boundary alignment (eliminating PCA diagonal bias)."""
    if xy_centered.shape[0] < 20:
        return 0.0
    if float(np.ptp(xy_centered[:, 0])) < 0.015 or float(np.ptp(xy_centered[:, 1])) < 0.015:
        return 0.0

    pts = (
        xy_centered[np.linspace(0, xy_centered.shape[0] - 1, 600, dtype=np.int64)]
        if xy_centered.shape[0] > 600
        else xy_centered
    )
    px = pts[:, 0][None, :]
    py = pts[:, 1][None, :]

    def _best_for_grid(thetas_deg: np.ndarray) -> float:
        rads = np.radians(thetas_deg)
        c = np.cos(rads)[:, None]
        s = np.sin(rads)[:, None]
        u = px * c + py * s
        v = -px * s + py * c

        u0 = np.percentile(u, 3, axis=1)
        u1 = np.percentile(u, 97, axis=1)
        v0 = np.percentile(v, 3, axis=1)
        v1 = np.percentile(v, 97, axis=1)
        su = np.maximum(u1 - u0, 1e-3)
        sv = np.maximum(v1 - v0, 1e-3)

        du = np.minimum(np.abs(u - u0[:, None]), np.abs(u1[:, None] - u)) / su[:, None]
        dv = np.minimum(np.abs(v - v0[:, None]), np.abs(v1[:, None] - v)) / sv[:, None]
        edge_resid = np.mean(np.minimum(du, dv), axis=1)
        losses = (su * sv) * (0.65 + 0.35 * (edge_resid / 0.25)) + 1e-6 * np.abs(thetas_deg)
        return float(thetas_deg[int(np.argmin(losses))])

    coarse_best = _best_for_grid(np.linspace(-84.0, 84.0, 43))
    fine_best = _best_for_grid(np.linspace(coarse_best - 4.0, coarse_best + 4.0, 9))
    return float(np.clip(fine_best, -85.0, 85.0))


def lift_mask_to_gemini_obb(
    points_cv: np.ndarray,
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    rgb_u8: Optional[np.ndarray] = None,
    label: Optional[str] = None,
) -> Optional[list[float]]:
    """Lift a 2D binary instance mask over metric 3D points into a robust 9-DOF Gemini OBB."""
    h, w = points_cv.shape[:2]
    if binary_mask.shape != (h, w):
        pil_m = Image.fromarray((binary_mask > 0).astype(np.uint8) * 255, mode="L").resize(
            (w, h), Image.Resampling.NEAREST
        )
        mask = np.asarray(pil_m) > 127
    else:
        mask = binary_mask > 0

    mask = mask & valid_mask & np.isfinite(points_cv).all(axis=-1)
    total_valid = int(np.sum(mask))
    if total_valid < 16:
        return None

    # Clinical specular glare suppression (Harvard Mahmood Lab): strip clipped highlights unless >70% white object
    if rgb_u8 is not None and rgb_u8.shape[:2] == (h, w):
        spec_mask, _ = compute_specular_mask(rgb_u8)
        clean_mask = mask & (~spec_mask)
        n_clean = int(np.sum(clean_mask))
        if n_clean >= 24 and n_clean >= int(0.30 * total_valid):
            mask = clean_mask

    # Erode boundary slightly to reduce background depth bleed on edges
    if cv2 is not None and int(np.sum(mask)) >= 140:
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        if int(np.sum(eroded)) >= 32:
            mask = eroded

    pts_cv = points_cv[mask]
    z_vals = pts_cv[:, 2]
    keep_range = (z_vals > 0.04) & (z_vals < 45.0)
    pts_cv = pts_cv[keep_range]
    if pts_cv.shape[0] < 16:
        return None

    # Modal foreground clustering avoids bimodal background averaging inside open masks
    core = _filter_foreground_depth_mode(pts_cv)

    # Trim lateral extreme 3% tails
    x_lo, x_hi = np.percentile(core[:, 0], [3, 97])
    y_lo, y_hi = np.percentile(core[:, 1], [3, 97])
    trimmed = core[
        (core[:, 0] >= x_lo)
        & (core[:, 0] <= x_hi)
        & (core[:, 1] >= y_lo)
        & (core[:, 1] <= y_hi)
    ]
    if trimmed.shape[0] < 10:
        trimmed = core

    pts_gem = opencv_to_gemini_points(trimmed)
    xy = pts_gem[:, :2]
    xy_mean = np.mean(xy, axis=0)
    xy_centered = xy - xy_mean

    # Vectorized BEV L-Shape / Minimum-Area yaw (avoids PCA hypotenuse bias)
    yaw_deg = _fit_lshape_bev_yaw(xy_centered)

    rad = math.radians(yaw_deg)
    c_y, s_y = math.cos(rad), math.sin(rad)
    # Rotate centered XY into local box coordinates (R_z(-yaw))
    local_x = xy_centered[:, 0] * c_y + xy_centered[:, 1] * s_y
    local_y = -xy_centered[:, 0] * s_y + xy_centered[:, 1] * c_y
    z_gem = pts_gem[:, 2]

    lx_min, lx_max = float(np.percentile(local_x, 2)), float(np.percentile(local_x, 98))
    ly_min, ly_max = float(np.percentile(local_y, 2)), float(np.percentile(local_y, 98))
    gz_min, gz_max = float(np.percentile(z_gem, 2)), float(np.percentile(z_gem, 98))

    dx = max(0.03, lx_max - lx_min)
    dy_obs = max(0.03, ly_max - ly_min)
    dz = max(0.03, gz_max - gz_min)

    # Single-view camera only observes the visible front shell: anchor front face when amodal prior applies
    norm_lbl = str(label or "").strip().lower().replace(" ", "_")
    amodal_ratio = _AMODAL_THICKNESS_RATIO.get(norm_lbl)
    if amodal_ratio is not None:
        amodal_min = min(0.65, amodal_ratio * min(dx, dz))
        if dy_obs < amodal_min:
            dy = max(0.03, amodal_min)
            cy_local = ly_min + 0.5 * dy
        else:
            dy = dy_obs
            cy_local = 0.5 * (ly_min + ly_max)
    else:
        dy = dy_obs
        cy_local = 0.5 * (ly_min + ly_max)

    cx_local = 0.5 * (lx_min + lx_max)
    cz_world = 0.5 * (gz_min + gz_max)

    cx_world = float(xy_mean[0] + cx_local * c_y - cy_local * s_y)
    cy_world = float(xy_mean[1] + cx_local * s_y + cy_local * c_y)

    # Keep positive depth strictly in front of camera
    cy_world = max(0.08, cy_world)

    return [
        round(cx_world, 3),
        round(cy_world, 3),
        round(cz_world, 3),
        round(dx, 3),
        round(dy, 3),
        round(dz, 3),
        0.0,
        0.0,
        round(yaw_deg, 1),
    ]


def _depth_layered_fallback_objects(
    points_cv: np.ndarray,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    max_objects: int = 4,
) -> list[dict[str, Any]]:
    """Cluster nearby foreground/midground depth layers over MoGe-2 points if detector finds 0 masks."""
    valid = valid_mask & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 35.0)
    if not np.any(valid):
        return []

    vals = depth_m[valid]
    q_edges = np.percentile(vals, [0, 32, 68, 96])
    layer_names = ["primer_plano_cercano", "plano_medio", "fondo_profundo"]
    results: list[dict[str, Any]] = []

    for idx in range(min(3, max_objects)):
        lo, hi = float(q_edges[idx]), float(q_edges[idx + 1])
        layer_mask = valid & (depth_m >= lo) & (depth_m <= hi)
        if int(np.sum(layer_mask)) < 40:
            continue
        box = lift_mask_to_gemini_obb(points_cv, layer_mask, valid_mask)
        if box is None:
            continue
        results.append(
            {
                "id": f"moge_layer_{idx + 1}",
                "label": layer_names[idx],
                "box_3d": box,
                "confidence": 0.72,
            }
        )

    return results


def _run_moge2_inference(
    rgb_pil: Image.Image,
    fov_hint_deg: Optional[float] = None,
) -> dict[str, Any]:
    """Run MoGe-2 ViT-S on scaled RGB image and return OpenCV points, depth, valid mask, and FOV."""
    import torch

    model, _, device, device_desc = _load_moge2_and_segmenter()
    w0, h0 = rgb_pil.size
    max_side = 768
    if max(w0, h0) > max_side:
        scale = max_side / float(max(w0, h0))
        nw = max(64, int(round(w0 * scale)))
        nh = max(64, int(round(h0 * scale)))
        rgb_proc = rgb_pil.resize((nw, nh), Image.Resampling.BILINEAR)
    else:
        rgb_proc = rgb_pil

    rgb_u8 = np.asarray(rgb_proc, dtype=np.uint8)
    rgb_f32 = rgb_u8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_f32).permute(2, 0, 1).unsqueeze(0).to(device)

    infer_kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(model.infer)
        if "fov_x" in sig.parameters and fov_hint_deg is not None:
            infer_kwargs["fov_x"] = float(fov_hint_deg)
        if "use_fp16" in sig.parameters:
            infer_kwargs["use_fp16"] = device.startswith("cuda")
    except Exception:
        pass

    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model.infer(tensor, **infer_kwargs)
        else:
            out = model.infer(tensor, **infer_kwargs)

    pts = out["points"].detach().float().cpu().numpy()
    if pts.ndim == 4:
        pts = pts[0]

    if "depth" in out and out["depth"] is not None:
        dep = out["depth"].detach().float().cpu().numpy()
        if dep.ndim == 4:
            dep = dep[0, 0]
        elif dep.ndim == 3:
            dep = dep[0]
    else:
        dep = pts[..., 2].copy()

    if "mask" in out and out["mask"] is not None:
        msk = out["mask"].detach().cpu().numpy()
        if msk.ndim == 4:
            msk = msk[0, 0]
        elif msk.ndim == 3:
            msk = msk[0]
        valid_mask = (msk > 0.5) & np.isfinite(dep) & (dep > 0.04)
    else:
        valid_mask = np.isfinite(dep) & (dep > 0.04) & (dep < 60.0)

    normals: Optional[np.ndarray] = None
    if "normal" in out and out["normal"] is not None:
        nrm = out["normal"].detach().float().cpu().numpy()
        if nrm.ndim == 4:
            nrm = nrm[0]
        if nrm.ndim == 3 and nrm.shape[-1] == 3:
            normals = nrm

    # Extract normalized horizontal FOV from MoGe-2 predicted intrinsics
    moge_fov_h = 60.0
    if "intrinsics" in out and out["intrinsics"] is not None:
        K_norm = out["intrinsics"].detach().float().cpu().numpy()
        if K_norm.ndim == 3:
            K_norm = K_norm[0]
        fx_norm = abs(float(K_norm[0, 0]))
        if fx_norm > 1e-4:
            moge_fov_h = float(np.clip(2.0 * math.degrees(math.atan(1.0 / (2.0 * fx_norm))), 25.0, 120.0))

    return {
        "points_cv": pts,
        "depth_m": dep,
        "normals_cv": normals,
        "valid_mask": valid_mask,
        "rgb_u8": rgb_u8,
        "proc_width": int(rgb_u8.shape[1]),
        "proc_height": int(rgb_u8.shape[0]),
        "moge_fov_h_deg": round(moge_fov_h, 2),
        "device": device,
        "device_desc": device_desc,
    }


def extract_depth_and_point_cloud(
    image_base64: str,
    *,
    fov_override_deg: Optional[float] = None,
    auto_fov: bool = True,
) -> dict[str, Any]:
    """Run MoGe-2 only to attach metric depth map, point cloud, and EXIF/MoGe FOV info."""
    rgb_pil, w, h, exif_fov_h_deg = extract_exif_image_info(image_base64)
    hint = exif_fov_h_deg if auto_fov else fov_override_deg
    moge_out = _run_moge2_inference(rgb_pil, fov_hint_deg=hint)

    moge_fov = moge_out["moge_fov_h_deg"]
    if not auto_fov and fov_override_deg is not None:
        effective_fov = float(fov_override_deg)
        fov_source = "manual"
    elif exif_fov_h_deg is not None:
        effective_fov = float(exif_fov_h_deg)
        fov_source = "exif_35mm"
    else:
        effective_fov = float(moge_fov)
        fov_source = "moge2_intrinsics"

    depth_b64, depth_stats = render_depth_map_png(moge_out["depth_m"], moge_out["valid_mask"])
    point_cloud = pack_point_cloud_threejs(
        moge_out["points_cv"], moge_out["rgb_u8"], moge_out["valid_mask"]
    )
    real_metrics = compute_real_photo_3d_metrics(
        points_cv=moge_out["points_cv"],
        depth_m=moge_out["depth_m"],
        normals_cv=moge_out.get("normals_cv"),
        rgb_u8=moge_out["rgb_u8"],
        valid_mask=moge_out["valid_mask"],
        effective_fov_h_deg=effective_fov,
        exif_fov_h_deg=exif_fov_h_deg,
        moge_fov_h_deg=moge_fov,
        width=w,
        height=h,
        per_object_metrology=[],
    )
    return {
        "width": w,
        "height": h,
        "effective_fov_h_deg": round(effective_fov, 2),
        "fov_source": fov_source,
        "exif_fov_h_deg": exif_fov_h_deg,
        "estimated_moge_fov_h_deg": moge_fov,
        "depth_map_base64": depth_b64,
        "depth_stats": depth_stats,
        "point_cloud": point_cloud,
        "real_photo_metrics": real_metrics,
        "device": moge_out["device"],
        "device_desc": moge_out["device_desc"],
    }


def run_local_metric_3d(
    image_base64: str,
    *,
    fov_override_deg: Optional[float] = None,
    auto_fov: bool = True,
    max_objects: int = 10,
    custom_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Full local GPU/CPU RGB -> Metric Depth + YOLOE Instance Masks -> 9-DOF 3D OBBs + Point Cloud."""
    t0 = time.perf_counter()
    rgb_pil, w, h, exif_fov_h_deg = extract_exif_image_info(image_base64)

    custom_classes: Optional[list[str]] = None
    if custom_prompt:
        parsed = [c.strip() for c in re.split(r"[,;\n]+", custom_prompt) if c.strip()]
        if parsed:
            custom_classes = parsed[:15]

    _, seg_model, device, model_desc = _load_moge2_and_segmenter(custom_classes=custom_classes)
    hint = exif_fov_h_deg if auto_fov else fov_override_deg
    moge_out = _run_moge2_inference(rgb_pil, fov_hint_deg=hint)

    moge_fov = moge_out["moge_fov_h_deg"]
    if not auto_fov and fov_override_deg is not None:
        effective_fov = float(fov_override_deg)
        fov_source = "manual"
    elif exif_fov_h_deg is not None:
        effective_fov = float(exif_fov_h_deg)
        fov_source = "exif_35mm"
    else:
        effective_fov = float(moge_fov)
        fov_source = "moge2_intrinsics"

    # Run Ultralytics YOLOE / YOLO segmentation on a PIL RGB Image (numpy arrays in Ultralytics expect BGR)
    rgb_u8 = moge_out["rgb_u8"]
    seg_results = seg_model.predict(
        source=Image.fromarray(rgb_u8, mode="RGB"),
        device=device,
        conf=0.11,
        iou=0.55,
        max_det=max(max_objects * 2, 16),
        verbose=False,
    )

    lifted_objects: list[dict[str, Any]] = []
    kept_masks_proc: list[np.ndarray] = []
    if seg_results:
        r0 = seg_results[0]
        boxes = r0.boxes
        masks = r0.masks
        names = getattr(r0, "names", {}) or {}
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.detach().cpu().numpy()
            cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
            xyxy = boxes.xyxy.detach().cpu().numpy()
            mask_stack = (
                masks.data.detach().cpu().numpy()
                if masks is not None and getattr(masks, "data", None) is not None
                else None
            )
            pw = moge_out["proc_width"]
            ph = moge_out["proc_height"]
            total_px = float(max(pw * ph, 1))

            order = np.argsort(-confs)
            for idx_num, i in enumerate(order):
                if len(lifted_objects) >= max_objects:
                    break
                if mask_stack is not None and i < mask_stack.shape[0]:
                    raw_m = mask_stack[i] > 0.5
                    if raw_m.shape != (ph, pw):
                        pil_m = Image.fromarray(raw_m.astype(np.uint8) * 255, mode="L").resize(
                            (pw, ph), Image.Resampling.NEAREST
                        )
                        bin_mask = np.asarray(pil_m) > 127
                    else:
                        bin_mask = raw_m
                else:
                    x0, y0, x1, y1 = xyxy[i]
                    ix0 = int(max(0, round(x0 + 0.15 * (x1 - x0))))
                    ix1 = int(min(pw, round(x1 - 0.15 * (x1 - x0))))
                    iy0 = int(max(0, round(y0 + 0.15 * (y1 - y0))))
                    iy1 = int(min(ph, round(y1 - 0.15 * (y1 - y0))))
                    bin_mask = np.zeros((ph, pw), dtype=bool)
                    bin_mask[iy0:iy1, ix0:ix1] = True

                # Skip composite frame-spanning bboxes when sub-objects exist
                bx0, by0, bx1, by1 = xyxy[i]
                bbox_area_frac = max(0.0, float(bx1 - bx0) * float(by1 - by0)) / total_px
                if (len(boxes) > 1 and bbox_area_frac > 0.85) or bbox_area_frac > 0.92:
                    continue

                # Skip background-spanning masks (>88% frame) & near-duplicate mask overlaps (IoU > 0.45)
                area_frac = float(np.sum(bin_mask)) / total_px
                if area_frac > 0.88 or area_frac < 0.002:
                    continue

                is_dup = False
                for prev_m in kept_masks_proc:
                    inter = float(np.sum(bin_mask & prev_m))
                    union = float(np.sum(bin_mask | prev_m))
                    if union > 0.0 and (inter / union) > 0.45:
                        is_dup = True
                        break
                if is_dup:
                    continue

                raw_name = (
                    names.get(int(cls_ids[i]), f"object_{int(cls_ids[i])}")
                    if isinstance(names, dict)
                    else f"object_{int(cls_ids[i])}"
                )
                clean_label = str(raw_name).strip().replace(" ", "_") or f"obj_{idx_num + 1}"

                box_3d = lift_mask_to_gemini_obb(
                    moge_out["points_cv"],
                    bin_mask,
                    moge_out["valid_mask"],
                    rgb_u8=rgb_u8,
                    label=clean_label,
                )
                if box_3d is None:
                    continue

                kept_masks_proc.append(bin_mask)
                lifted_objects.append(
                    {
                        "id": f"moge_{len(lifted_objects) + 1}",
                        "label": clean_label,
                        "box_3d": box_3d,
                        "confidence": round(float(confs[i]), 3),
                    }
                )

    if not lifted_objects:
        lifted_objects = _depth_layered_fallback_objects(
            moge_out["points_cv"],
            moge_out["depth_m"],
            moge_out["valid_mask"],
            max_objects=min(max_objects, 4),
        )

    spec_mask, _ = compute_specular_mask(rgb_u8)
    ransac_plane = fit_ransac_dominant_plane(
        moge_out["points_cv"],
        moge_out["valid_mask"],
        seed=42,
        specular_mask=spec_mask,
    )

    per_object_metrology: list[dict[str, Any]] = []
    for idx_m, obj_item in enumerate(lifted_objects):
        mask_m = kept_masks_proc[idx_m] if idx_m < len(kept_masks_proc) else None
        if mask_m is not None:
            per_object_metrology.append(
                compute_object_point_cloud_metrology(
                    moge_out["points_cv"],
                    moge_out.get("normals_cv"),
                    mask_m,
                    moge_out["valid_mask"],
                    obj_item["box_3d"],
                    effective_fov,
                    object_id=obj_item["id"],
                    label=obj_item["label"],
                    ransac_plane=ransac_plane,
                    specular_mask=spec_mask,
                )
            )

    depth_b64, depth_stats = render_depth_map_png(moge_out["depth_m"], moge_out["valid_mask"])
    point_cloud = pack_point_cloud_threejs(
        moge_out["points_cv"], rgb_u8, moge_out["valid_mask"]
    )
    real_metrics = compute_real_photo_3d_metrics(
        points_cv=moge_out["points_cv"],
        depth_m=moge_out["depth_m"],
        normals_cv=moge_out.get("normals_cv"),
        rgb_u8=rgb_u8,
        valid_mask=moge_out["valid_mask"],
        effective_fov_h_deg=effective_fov,
        exif_fov_h_deg=exif_fov_h_deg,
        moge_fov_h_deg=moge_fov,
        width=w,
        height=h,
        per_object_metrology=per_object_metrology,
        instance_masks=kept_masks_proc,
        objects=lifted_objects,
        ransac_plane=ransac_plane,
        specular_mask=spec_mask,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Also return canonical oriented image base64 so 2D overlay matches lifted EXIF orientation cleanly
    oriented_buf = io.BytesIO()
    rgb_pil.save(oriented_buf, format="PNG")
    oriented_b64 = base64.b64encode(oriented_buf.getvalue()).decode("ascii")

    mode_name = "moge2_yoloe_gpu" if device.startswith("cuda") else "moge2_yoloe_cpu"
    return {
        "mode": mode_name,
        "model_desc": model_desc,
        "width": w,
        "height": h,
        "image_base64": oriented_b64,
        "mime_type": "image/png",
        "effective_fov_h_deg": round(effective_fov, 2),
        "fov_source": fov_source,
        "exif_fov_h_deg": exif_fov_h_deg,
        "estimated_moge_fov_h_deg": moge_fov,
        "objects": lifted_objects,
        "scene_summary": (
            f"Reconstrucción métrica local ({model_desc}): {len(lifted_objects)} objetos en rango "
            f"{depth_stats['min_m']:.2f} m – {depth_stats['max_m']:.2f} m "
            f"(mediana {depth_stats['median_m']:.2f} m, FOV_h={effective_fov:.1f}° vía {fov_source})."
        ),
        "depth_map_base64": depth_b64,
        "depth_stats": depth_stats,
        "point_cloud": point_cloud,
        "real_photo_metrics": real_metrics,
        "latency_ms": latency_ms,
    }
