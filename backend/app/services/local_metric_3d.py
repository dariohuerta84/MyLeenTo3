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
import gc
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

from ..core.multiview_fusion import _encode_png_thumbnail, fuse_multiview_frames
from ..core.real_photo_metrics import (
    compute_object_point_cloud_metrology,
    compute_real_photo_3d_metrics,
    compute_specular_mask,
    fit_ransac_dominant_plane,
)
from ..core.scale_calibration import detect_aruco_scale_anchor

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional in CPU-only CI
    cv2 = None  # type: ignore

MOGE_MODEL_ID = os.getenv("MYLEENTO3_MOGE_MODEL", "Ruicheng/moge-2-vits-normal")
MOGE_BACKBONE_REPOS: dict[str, str] = {
    "vits": MOGE_MODEL_ID,
    "vitb": "Ruicheng/moge-2-vitb-normal",
    "vitl": "Ruicheng/moge-2-vitl-normal",
}
MOGE_BACKBONE_MAX_SIDE: dict[str, int] = {
    "vits": 768,
    "vitb": 896,
    "vitl": 1024,
}
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
_MOGE_ACTIVE_BACKBONE: Optional[str] = None
_SEG_MODEL: Any = None
_SEG_MODEL_KIND: str = "unloaded"
_PF_MODEL: Any = None
_PF_MODEL_KIND: str = "unloaded"
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

    active_repo = MOGE_BACKBONE_REPOS.get(_MOGE_ACTIVE_BACKBONE or "vits", MOGE_MODEL_ID)
    seg_name = _SEG_MODEL_KIND if _SEG_MODEL_KIND != "unloaded" else f"YOLOE-OV ({YOLOE_OV_WEIGHTS})"
    pf_name = _PF_MODEL_KIND if _PF_MODEL_KIND != "unloaded" else f"YOLOE-pf ({YOLOE_PF_WEIGHTS}, 4585 cl.)"
    models_list = [active_repo, seg_name]
    if pf_name not in models_list and YOLOE_PF_WEIGHTS:
        models_list.append(pf_name)
    return {
        "available": True,
        "device": dev_name,
        "models": models_list,
    }


def _resolve_moge_backbone(backbone: Optional[str]) -> str:
    bb = str(backbone or "vits").strip().lower()
    return bb if bb in MOGE_BACKBONE_REPOS else "vits"


def _ensure_moge_model_locked(backbone: str = "vits") -> tuple[Any, str, str, str]:
    """Load or safely hot-swap MoGe-2 backbone inside `_MODEL_LOCK` without retaining two ViTs in VRAM."""
    global _MOGE_MODEL, _MOGE_ACTIVE_BACKBONE, _LOCAL_DEVICE, _DEVICE_LABEL

    import torch
    from moge.model.v2 import MoGeModel

    bb = _resolve_moge_backbone(backbone)
    repo_id = MOGE_BACKBONE_REPOS[bb]

    if _MOGE_MODEL is None or _MOGE_ACTIVE_BACKBONE != bb:
        if _MOGE_MODEL is not None:
            try:
                _MOGE_MODEL = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                _MOGE_MODEL = None

        if torch.cuda.is_available():
            _LOCAL_DEVICE = "cuda"
            _DEVICE_LABEL = torch.cuda.get_device_name(0)
        else:
            _LOCAL_DEVICE = "cpu"
            _DEVICE_LABEL = "CPU"

        _MOGE_MODEL = MoGeModel.from_pretrained(repo_id).to(_LOCAL_DEVICE).eval()
        _MOGE_ACTIVE_BACKBONE = bb

    return _MOGE_MODEL, _LOCAL_DEVICE, _DEVICE_LABEL, bb


def _load_prompt_free_segmenter(device: Optional[str] = None) -> tuple[Any, str]:
    """Thread-safe lazy loader for YOLOE Prompt-Free (4,585 open-world categories)."""
    global _PF_MODEL, _PF_MODEL_KIND

    try:
        import torch
        from ultralytics import YOLOE  # type: ignore
    except Exception:
        return None, "unavailable"

    with _MODEL_LOCK:
        target_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if _PF_MODEL is None:
            try:
                _PF_MODEL = YOLOE(YOLOE_PF_WEIGHTS)
                if hasattr(_PF_MODEL, "to"):
                    _PF_MODEL.to(target_dev)
                _PF_MODEL_KIND = f"YOLOE-pf ({YOLOE_PF_WEIGHTS}, 4585 cl.)"
            except Exception:
                _PF_MODEL = None
                _PF_MODEL_KIND = "unavailable"
        return _PF_MODEL, _PF_MODEL_KIND


def _load_moge2_and_segmenter(
    custom_classes: Optional[list[str]] = None,
    moge_backbone: str = "vits",
) -> tuple[Any, Any, str, str]:
    """Thread-safe lazy loader for MoGe-2 (ViT-S/B/L) and Ultralytics YOLOE/YOLO-seg."""
    global _SEG_MODEL, _SEG_MODEL_KIND, _OV_MODEL, _OV_CLASSES_CACHE

    from ultralytics import YOLO

    with _MODEL_LOCK:
        moge_model, local_dev, dev_lbl, bb = _ensure_moge_model_locked(moge_backbone)
        repo_short = MOGE_BACKBONE_REPOS[bb].split("/")[-1]

        target_classes = custom_classes if custom_classes else DEFAULT_OPEN_VOCAB_CLASSES
        try:
            from ultralytics import YOLOE  # type: ignore

            if _OV_MODEL is None:
                _OV_MODEL = YOLOE(YOLOE_OV_WEIGHTS)
                if hasattr(_OV_MODEL, "to"):
                    _OV_MODEL.to(local_dev)
            target_tuple = tuple(target_classes)
            if _OV_CLASSES_CACHE != target_tuple:
                if hasattr(_OV_MODEL, "get_text_pe"):
                    pe = _OV_MODEL.get_text_pe(target_classes)
                    _OV_MODEL.set_classes(target_classes, pe)
                else:
                    _OV_MODEL.set_classes(target_classes)
                _OV_CLASSES_CACHE = target_tuple
            _SEG_MODEL_KIND = f"YOLOE-OV ({YOLOE_OV_WEIGHTS})"
            return (
                moge_model,
                _OV_MODEL,
                local_dev,
                f"{repo_short} + {_SEG_MODEL_KIND} ({dev_lbl})",
            )
        except Exception:
            pass

        if _SEG_MODEL is None:
            try:
                from ultralytics import YOLOE  # type: ignore

                _SEG_MODEL = YOLOE(YOLOE_PF_WEIGHTS)
                if hasattr(_SEG_MODEL, "to"):
                    _SEG_MODEL.to(local_dev)
                _SEG_MODEL_KIND = f"YOLOE-pf ({YOLOE_PF_WEIGHTS})"
            except Exception:
                _SEG_MODEL = YOLO(YOLO_FALLBACK_WEIGHTS)
                _SEG_MODEL_KIND = f"YOLO-seg ({YOLO_FALLBACK_WEIGHTS})"

        return moge_model, _SEG_MODEL, local_dev, f"{repo_short} + {_SEG_MODEL_KIND} ({dev_lbl})"


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


def _compute_threejs_normals(
    points_cv: np.ndarray,
    normals_cv: Optional[np.ndarray],
) -> np.ndarray:
    """Compute or normalize surface normals and convert OpenCV (X,Y,Z) -> Three.js (X,-Y,-Z)."""
    h, w = points_cv.shape[:2]
    if normals_cv is not None and normals_cv.shape[:2] == (h, w) and normals_cv.ndim == 3 and normals_cv.shape[2] == 3:
        n_cv = np.nan_to_num(normals_cv.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(n_cv, axis=-1, keepdims=True)
        valid_n = norms[..., 0] > 1e-6
        unit_cv = np.divide(n_cv, np.maximum(norms, 1e-6))
        unit_cv = np.where(valid_n[..., None], unit_cv, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    elif h >= 3 and w >= 3:
        dp_du = points_cv[1:-1, 2:] - points_cv[1:-1, :-2]
        dp_dv = points_cv[2:, 1:-1] - points_cv[:-2, 1:-1]
        raw_n = np.cross(dp_dv, dp_du).astype(np.float32)
        view_dot = np.sum(raw_n * (-points_cv[1:-1, 1:-1]), axis=-1, keepdims=True)
        raw_n = np.where(view_dot < 0.0, -raw_n, raw_n)
        padded = np.pad(raw_n, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(padded, axis=-1, keepdims=True)
        valid_n = norms[..., 0] > 1e-7
        unit_cv = np.divide(padded, np.maximum(norms, 1e-7))
        unit_cv = np.where(valid_n[..., None], unit_cv, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    else:
        unit_cv = np.zeros((h, w, 3), dtype=np.float32)
        unit_cv[..., 2] = -1.0

    n_three = np.empty_like(unit_cv, dtype=np.float32)
    n_three[..., 0] = unit_cv[..., 0]
    n_three[..., 1] = -unit_cv[..., 1]
    n_three[..., 2] = -unit_cv[..., 2]
    return n_three


def build_surface_mesh_threejs(
    points_cv: np.ndarray,
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray,
    normals_three: np.ndarray,
    *,
    target_width: int = 168,
    max_rel_depth_jump: float = 0.06,
) -> Optional[dict[str, Any]]:
    """Triangulate dense depth grid while cutting flying-pixel bridges across depth discontinuities."""
    h, w = points_cv.shape[:2]
    if h < 4 or w < 4:
        return None

    step = max(1, int(round(float(w) / float(max(target_width, 16)))))
    pts_sub = points_cv[::step, ::step]
    rgb_sub = rgb_uint8[::step, ::step]
    val_sub = valid_mask[::step, ::step]
    nrm_sub = normals_three[::step, ::step]

    gh, gw = pts_sub.shape[:2]
    if gh < 2 or gw < 2:
        return None

    z_sub = pts_sub[..., 2]
    v_ok = (
        val_sub
        & np.isfinite(pts_sub).all(axis=-1)
        & np.isfinite(z_sub)
        & (z_sub > 0.04)
        & (z_sub < 35.0)
    )
    pts_three_grid = opencv_to_threejs_points(pts_sub)

    grid_idx = np.arange(gh * gw, dtype=np.uint32).reshape(gh, gw)
    tl = grid_idx[:-1, :-1].ravel()
    tr = grid_idx[:-1, 1:].ravel()
    bl = grid_idx[1:, :-1].ravel()
    br = grid_idx[1:, 1:].ravel()

    tri_a = np.stack([tl, bl, tr], axis=1)
    tri_b = np.stack([tr, bl, br], axis=1)
    all_tris = np.concatenate([tri_a, tri_b], axis=0)

    v_flat = v_ok.ravel()
    z_flat = z_sub.ravel()
    safe_z = np.where(v_flat & (z_flat > 0.01), z_flat, 1.0)
    log_z = np.log(safe_z)

    i0, i1, i2 = all_tris[:, 0], all_tris[:, 1], all_tris[:, 2]
    ok_tri = v_flat[i0] & v_flat[i1] & v_flat[i2]
    ok_tri &= (
        (np.abs(log_z[i0] - log_z[i1]) < max_rel_depth_jump)
        & (np.abs(log_z[i1] - log_z[i2]) < max_rel_depth_jump)
        & (np.abs(log_z[i0] - log_z[i2]) < max_rel_depth_jump)
    )

    kept_tris = all_tris[ok_tri]
    if kept_tris.shape[0] == 0:
        return None

    used_verts, inverse = np.unique(kept_tris, return_inverse=True)
    compact_tris = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.uint32)

    verts_flat = pts_three_grid.reshape(-1, 3)[used_verts]
    pos_f32 = np.ascontiguousarray(np.nan_to_num(verts_flat, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float32)
    col_u8 = np.ascontiguousarray(rgb_sub.reshape(-1, 3)[used_verts], dtype=np.uint8)
    nrm_flat = np.nan_to_num(nrm_sub.reshape(-1, 3)[used_verts], nan=0.0, posinf=0.0, neginf=0.0)
    nrm_i8 = np.ascontiguousarray(np.clip(np.round(nrm_flat * 127.0), -127, 127).astype(np.int8))

    return {
        "vertex_count": int(pos_f32.shape[0]),
        "triangle_count": int(compact_tris.shape[0]),
        "positions_b64": base64.b64encode(pos_f32.tobytes()).decode("ascii"),
        "colors_b64": base64.b64encode(col_u8.tobytes()).decode("ascii"),
        "normals_b64": base64.b64encode(nrm_i8.tobytes()).decode("ascii"),
        "indices_b64": base64.b64encode(compact_tris.tobytes()).decode("ascii"),
    }


def pack_point_cloud_threejs(
    points_cv: np.ndarray,
    rgb_uint8: np.ndarray,
    valid_mask: np.ndarray,
    max_points: int = 16000,
    *,
    normals_cv: Optional[np.ndarray] = None,
    instance_masks: Optional[list[np.ndarray]] = None,
    object_ids: Optional[list[str]] = None,
    ransac_plane: Optional[dict[str, Any]] = None,
    include_surface_mesh: bool = False,
) -> dict[str, Any]:
    """Subsample valid metric points (with body-focused sampling, normals, object IDs, relief & optional mesh) into compact Three.js buffers."""
    h, w = points_cv.shape[:2]
    pts_three = opencv_to_threejs_points(points_cv)
    normals_three = _compute_threejs_normals(points_cv, normals_cv)
    z_cv = points_cv[..., 2]
    keep = (
        valid_mask
        & np.isfinite(pts_three).all(axis=-1)
        & np.isfinite(z_cv)
        & (z_cv > 0.04)
        & (z_cv < 35.0)
    )

    label_map = np.zeros((h, w), dtype=np.uint8)
    if instance_masks:
        for idx_inst, raw_m in enumerate(instance_masks[:250]):
            if raw_m.shape[:2] != (h, w):
                pil_m = Image.fromarray((raw_m > 0).astype(np.uint8) * 255, mode="L").resize(
                    (w, h), Image.Resampling.NEAREST
                )
                m_bool = np.asarray(pil_m) > 127
            else:
                m_bool = raw_m > 0
            assign_mask = m_bool & (label_map == 0)
            label_map[assign_mask] = np.uint8(idx_inst + 1)

    flat_pts = pts_three[keep]
    flat_rgb = rgb_uint8[keep]
    flat_z = z_cv[keep]
    flat_n = normals_three[keep]
    flat_lbl = label_map[keep]

    total = int(flat_pts.shape[0])
    if total == 0:
        return {
            "count": 0,
            "positions_b64": "",
            "colors_b64": "",
            "normals_b64": "",
            "object_indices_b64": "",
            "relief_cm_b64": "",
            "object_id_palette": list(object_ids or []),
            "has_normals": False,
            "body_point_count": 0,
            "max_relief_mm": 0.0,
            "median_depth_m": 0.0,
            "bounds_three": {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]},
            "surface_mesh": None,
        }

    # Compute per-pixel relief elevation (meters) above dominant RANSAC plane or p95 backdrop
    if ransac_plane is not None and ransac_plane.get("detected") and ransac_plane.get("plane_normal_gemini"):
        pts_cv_k = points_cv[keep]
        pts_gem_k = np.column_stack([pts_cv_k[:, 0], pts_cv_k[:, 2], -pts_cv_k[:, 1]]).astype(np.float64)
        n_pl = np.array(ransac_plane["plane_normal_gemini"], dtype=np.float64)
        n_pl /= max(float(np.linalg.norm(n_pl)), 1e-9)
        d_pl = float(
            ransac_plane.get(
                "plane_d_gemini",
                -float(n_pl[2]) * float(ransac_plane.get("plane_height_z_m", 0.0)),
            )
        )
        cam_sign = 1.0 if d_pl >= 0.0 else -1.0
        flat_relief = np.maximum(cam_sign * (pts_gem_k @ n_pl + d_pl), 0.0).astype(np.float32)
    else:
        p95_z = float(np.percentile(flat_z, 95))
        flat_relief = np.maximum(p95_z - flat_z, 0.0).astype(np.float32)

    if total > max_points:
        fg_pos = np.where(flat_lbl > 0)[0]
        bg_pos = np.where(flat_lbl == 0)[0]
        if fg_pos.size > 0:
            fg_target = min(int(fg_pos.size), int(round(max_points * 0.72)))
            bg_target = min(int(bg_pos.size), max_points - fg_target)
            rem = max_points - fg_target - bg_target
            if rem > 0 and fg_pos.size > fg_target:
                fg_target = min(int(fg_pos.size), fg_target + rem)
            fg_sel = fg_pos[np.linspace(0, fg_pos.size - 1, fg_target, dtype=np.int64)] if fg_target > 0 else np.empty(0, dtype=np.int64)
            bg_sel = bg_pos[np.linspace(0, bg_pos.size - 1, bg_target, dtype=np.int64)] if bg_target > 0 else np.empty(0, dtype=np.int64)
            indices = np.sort(np.concatenate([fg_sel, bg_sel]))
        else:
            indices = np.linspace(0, total - 1, max_points, dtype=np.int64)

        flat_pts = flat_pts[indices]
        flat_rgb = flat_rgb[indices]
        flat_z = flat_z[indices]
        flat_n = flat_n[indices]
        flat_lbl = flat_lbl[indices]
        flat_relief = flat_relief[indices]

    pts_f32 = np.ascontiguousarray(flat_pts, dtype=np.float32)
    cols_u8 = np.ascontiguousarray(flat_rgb, dtype=np.uint8)
    nrm_i8 = np.ascontiguousarray(np.clip(np.round(flat_n * 127.0), -127, 127).astype(np.int8))
    lbl_u8 = np.ascontiguousarray(flat_lbl, dtype=np.uint8)
    relief_mm_u16 = np.ascontiguousarray(np.clip(np.round(flat_relief * 1000.0), 0, 65535).astype(np.uint16))

    b_min = [round(float(v), 3) for v in np.min(pts_f32, axis=0)]
    b_max = [round(float(v), 3) for v in np.max(pts_f32, axis=0)]
    max_relief_mm = round(float(np.percentile(relief_mm_u16, 98)), 1) if relief_mm_u16.size > 0 else 0.0

    surface_mesh = (
        build_surface_mesh_threejs(points_cv, rgb_uint8, valid_mask, normals_three)
        if include_surface_mesh
        else None
    )

    return {
        "count": int(pts_f32.shape[0]),
        "positions_b64": base64.b64encode(pts_f32.tobytes()).decode("ascii"),
        "colors_b64": base64.b64encode(cols_u8.tobytes()).decode("ascii"),
        "normals_b64": base64.b64encode(nrm_i8.tobytes()).decode("ascii"),
        "object_indices_b64": base64.b64encode(lbl_u8.tobytes()).decode("ascii"),
        "relief_cm_b64": base64.b64encode(relief_mm_u16.tobytes()).decode("ascii"),
        "object_id_palette": list(object_ids or []),
        "has_normals": True,
        "body_point_count": int(np.sum(lbl_u8 > 0)),
        "max_relief_mm": max_relief_mm,
        "median_depth_m": round(float(np.median(flat_z)), 3),
        "bounds_three": {"min": b_min, "max": b_max},
        "surface_mesh": surface_mesh,
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


def _load_moge2_only(backbone: str = "vits") -> tuple[Any, str, str, str]:
    """Thread-safe lazy loader for MoGe-2 backbone without triggering open-vocab text encoding."""
    with _MODEL_LOCK:
        model, dev, dev_lbl, bb = _ensure_moge_model_locked(backbone)
        repo_short = MOGE_BACKBONE_REPOS[bb].split("/")[-1]
        return model, dev, f"{repo_short} ({dev_lbl})", bb


def _run_moge2_inference(
    rgb_pil: Image.Image,
    fov_hint_deg: Optional[float] = None,
    moge_backbone: str = "vits",
) -> dict[str, Any]:
    """Run MoGe-2 (ViT-S/B/L) on scaled RGB image and return OpenCV points, depth, valid mask, and FOV."""
    import torch

    model, device, device_desc, bb = _load_moge2_only(backbone=moge_backbone)
    w0, h0 = rgb_pil.size
    max_side = MOGE_BACKBONE_MAX_SIDE.get(bb, 768)
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
        "moge_backbone": bb,
        "device": device,
        "device_desc": device_desc,
    }


def _safe_detect_aruco_anchor(
    rgb_u8: np.ndarray,
    points_cv: np.ndarray,
    valid_mask: np.ndarray,
    aruco_marker_size_cm: Optional[float],
) -> Optional[dict[str, Any]]:
    try:
        marker_m = max(0.01, float(aruco_marker_size_cm or 5.0) / 100.0)
        return detect_aruco_scale_anchor(
            rgb_u8,
            points_cv,
            valid_mask,
            marker_size_m=marker_m,
        )
    except Exception:
        return None


def extract_depth_and_point_cloud(
    image_base64: str,
    *,
    fov_override_deg: Optional[float] = None,
    auto_fov: bool = True,
    moge_backbone: str = "vits",
    aruco_marker_size_cm: Optional[float] = None,
) -> dict[str, Any]:
    """Run MoGe-2 only to attach metric depth map, point cloud, ArUco anchor, and EXIF/MoGe FOV info."""
    rgb_pil, w, h, exif_fov_h_deg = extract_exif_image_info(image_base64)
    hint = exif_fov_h_deg if auto_fov else fov_override_deg
    moge_out = _run_moge2_inference(rgb_pil, fov_hint_deg=hint, moge_backbone=moge_backbone)

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
    point_cloud = pack_point_cloud_threejs(
        moge_out["points_cv"],
        moge_out["rgb_u8"],
        moge_out["valid_mask"],
        max_points=60000,
        normals_cv=moge_out.get("normals_cv"),
        ransac_plane=real_metrics.get("ransac_plane"),
        include_surface_mesh=True,
    )
    aruco_anchor = _safe_detect_aruco_anchor(
        moge_out["rgb_u8"],
        moge_out["points_cv"],
        moge_out["valid_mask"],
        aruco_marker_size_cm,
    )
    return {
        "width": w,
        "height": h,
        "effective_fov_h_deg": round(effective_fov, 2),
        "fov_source": fov_source,
        "exif_fov_h_deg": exif_fov_h_deg,
        "estimated_moge_fov_h_deg": moge_fov,
        "moge_backbone": moge_out.get("moge_backbone", "vits"),
        "aruco_anchor": aruco_anchor,
        "depth_map_base64": depth_b64,
        "depth_stats": depth_stats,
        "point_cloud": point_cloud,
        "real_photo_metrics": real_metrics,
        "device": moge_out["device"],
        "device_desc": moge_out["device_desc"],
    }


_BROAD_PF_SKIP_LABELS = {"glass_floor", "wall", "floor", "ceiling", "sky", "ground"}


def _clean_label_slug(raw_name: str, fallback: str) -> str:
    slug = re.sub(r"\s+", "_", str(raw_name).strip())
    return slug or fallback


def _bbox_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx0, by0, bx1, by1 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def lift_box2d_to_foreground_mask(
    points_cv: np.ndarray,
    valid_mask: np.ndarray,
    box_2d_norm_1000: list[float],
    proc_width: int,
    proc_height: int,
) -> Optional[np.ndarray]:
    """Isolate foreground depth cluster inside a normalized 2D `[ymin, xmin, ymax, xmax]` (0..1000) ROI."""
    if len(box_2d_norm_1000) < 4 or proc_width < 16 or proc_height < 16:
        return None
    if points_cv.shape[:2] != (proc_height, proc_width) or valid_mask.shape[:2] != (proc_height, proc_width):
        return None

    ymin, xmin, ymax, xmax = [float(np.clip(v, 0.0, 1000.0)) for v in box_2d_norm_1000[:4]]
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin

    x0 = int(max(0, math.floor((xmin / 1000.0) * proc_width)))
    x1 = int(min(proc_width, math.ceil((xmax / 1000.0) * proc_width)))
    y0 = int(max(0, math.floor((ymin / 1000.0) * proc_height)))
    y1 = int(min(proc_height, math.ceil((ymax / 1000.0) * proc_height)))

    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return None

    total_px = float(max(proc_width * proc_height, 1))
    roi_area_frac = float((x1 - x0) * (y1 - y0)) / total_px
    if roi_area_frac > 0.92 or roi_area_frac < 0.0015:
        return None

    pts_roi = points_cv[y0:y1, x0:x1]
    z_roi = pts_roi[..., 2]
    val_roi = (
        valid_mask[y0:y1, x0:x1]
        & np.isfinite(pts_roi).all(axis=-1)
        & (z_roi > 0.04)
        & (z_roi < 45.0)
    )
    if int(np.sum(val_roi)) < 16:
        return None

    fg_pts = _filter_foreground_depth_mode(pts_roi[val_roi])
    z_fg = fg_pts[:, 2]
    if z_fg.size < 8:
        return None

    z_med = float(np.median(z_fg))
    pad = max(0.04, 0.08 * z_med)
    z_lo = max(0.04, float(np.percentile(z_fg, 3)) - pad)
    z_hi = float(np.percentile(z_fg, 97)) + pad

    roi_fg_mask = val_roi & (z_roi >= z_lo) & (z_roi <= z_hi)
    if int(np.sum(roi_fg_mask)) < 16:
        return None

    full_mask = np.zeros((proc_height, proc_width), dtype=bool)
    full_mask[y0:y1, x0:x1] = roi_fg_mask
    return full_mask


def _extract_ultralytics_candidates(
    seg_results: Any,
    proc_width: int,
    proc_height: int,
    *,
    source: str,
    min_area_frac: float = 0.002,
    max_area_frac: float = 0.88,
) -> list[dict[str, Any]]:
    """Normalize Ultralytics instance segmentation predictions into sorted mask candidates."""
    if not seg_results:
        return []
    r0 = seg_results[0]
    boxes = getattr(r0, "boxes", None)
    masks = getattr(r0, "masks", None)
    names = getattr(r0, "names", {}) or {}
    if boxes is None or len(boxes) == 0:
        return []

    confs = boxes.conf.detach().cpu().numpy()
    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    xyxy = boxes.xyxy.detach().cpu().numpy()
    mask_stack = (
        masks.data.detach().cpu().numpy()
        if masks is not None and getattr(masks, "data", None) is not None
        else None
    )
    total_px = float(max(proc_width * proc_height, 1))
    order = np.argsort(-confs)

    candidates: list[dict[str, Any]] = []
    for idx_num, i in enumerate(order):
        bx0, by0, bx1, by1 = [float(v) for v in xyxy[i]]
        bbox_area_frac = max(0.0, (bx1 - bx0) * (by1 - by0)) / total_px
        if (len(boxes) > 1 and bbox_area_frac > 0.85) or bbox_area_frac > 0.92:
            continue

        if mask_stack is not None and i < mask_stack.shape[0]:
            raw_m = mask_stack[i] > 0.5
            if raw_m.shape != (proc_height, proc_width):
                pil_m = Image.fromarray(raw_m.astype(np.uint8) * 255, mode="L").resize(
                    (proc_width, proc_height), Image.Resampling.NEAREST
                )
                bin_mask = np.asarray(pil_m) > 127
            else:
                bin_mask = raw_m
        else:
            ix0 = int(max(0, round(bx0 + 0.15 * (bx1 - bx0))))
            ix1 = int(min(proc_width, round(bx1 - 0.15 * (bx1 - bx0))))
            iy0 = int(max(0, round(by0 + 0.15 * (by1 - by0))))
            iy1 = int(min(proc_height, round(by1 - 0.15 * (by1 - by0))))
            bin_mask = np.zeros((proc_height, proc_width), dtype=bool)
            bin_mask[iy0:iy1, ix0:ix1] = True

        area_frac = float(np.sum(bin_mask)) / total_px
        if area_frac > max_area_frac or area_frac < min_area_frac:
            continue

        raw_name = (
            names.get(int(cls_ids[i]), f"object_{int(cls_ids[i])}")
            if isinstance(names, dict)
            else f"object_{int(cls_ids[i])}"
        )
        clean_label = _clean_label_slug(str(raw_name), f"obj_{idx_num + 1}")
        candidates.append(
            {
                "label": clean_label,
                "conf": round(float(confs[i]), 3),
                "mask": bin_mask,
                "xyxy": [bx0, by0, bx1, by1],
                "xyxy_norm": [
                    round((by0 / max(proc_height, 1)) * 1000.0, 1),
                    round((bx0 / max(proc_width, 1)) * 1000.0, 1),
                    round((by1 / max(proc_height, 1)) * 1000.0, 1),
                    round((bx1 / max(proc_width, 1)) * 1000.0, 1),
                ],
                "area_frac": round(area_frac, 4),
                "source": source,
            }
        )
    return candidates


def fuse_discovery_and_segmentation_candidates(
    *,
    ov_candidates: list[dict[str, Any]],
    pf_candidates: list[dict[str, Any]],
    vlm_objects_2d: list[dict[str, Any]],
    points_cv: np.ndarray,
    valid_mask: np.ndarray,
    proc_width: int,
    proc_height: int,
    max_objects: int = 10,
) -> list[dict[str, Any]]:
    """Fuse VLM 2D hints, YOLOE-OV masks, and YOLOE-PF discoveries without mask duplicates."""
    fused: list[dict[str, Any]] = []

    def _overlaps(mask: np.ndarray, threshold: float = 0.45) -> bool:
        for kept in fused:
            pm = kept["mask"]
            inter = float(np.sum(mask & pm))
            union = float(np.sum(mask | pm))
            if union > 0.0 and (inter / union) > threshold:
                return True
        return False

    def _append_unique(mask: np.ndarray, label: str, conf: float, source: str) -> bool:
        if len(fused) >= max_objects or _overlaps(mask):
            return False
        fused.append(
            {
                "mask": mask,
                "label": _clean_label_slug(label, f"obj_{len(fused) + 1}"),
                "conf": round(float(conf), 3),
                "source": source,
            }
        )
        return True

    if vlm_objects_2d:
        sorted_vlm = sorted(
            vlm_objects_2d,
            key=lambda item: float(item.get("confidence", 0.85) or 0.85),
            reverse=True,
        )
        pool: list[tuple[int, str, dict[str, Any]]] = (
            [(i, "ov", c) for i, c in enumerate(ov_candidates)]
            + [(1000 + i, "pf", c) for i, c in enumerate(pf_candidates)]
        )
        used_pool: set[int] = set()

        for v_obj in sorted_vlm:
            if len(fused) >= max_objects:
                break
            box_2d = v_obj.get("box_2d") or []
            if len(box_2d) < 4:
                continue
            ymin, xmin, ymax, xmax = [float(v) for v in box_2d[:4]]
            if ymin > ymax:
                ymin, ymax = ymax, ymin
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            v_xyxy = [
                (xmin / 1000.0) * proc_width,
                (ymin / 1000.0) * proc_height,
                (xmax / 1000.0) * proc_width,
                (ymax / 1000.0) * proc_height,
            ]
            v_label = str(v_obj.get("label") or "object")
            v_conf = float(v_obj.get("confidence", 0.85) or 0.85)

            best_idx: Optional[int] = None
            best_cand: Optional[dict[str, Any]] = None
            best_score = -1.0
            for p_idx, kind, cand in pool:
                if p_idx in used_pool:
                    continue
                iou = _bbox_iou_xyxy(v_xyxy, cand["xyxy"])
                if iou >= 0.25:
                    score = iou + (0.02 if kind == "ov" else 0.0)
                    if score > best_score:
                        best_score = score
                        best_idx = p_idx
                        best_cand = cand

            if best_idx is not None and best_cand is not None:
                if _append_unique(best_cand["mask"], v_label, max(v_conf, float(best_cand["conf"])), "vlm_yoloe_fused"):
                    used_pool.add(best_idx)
            else:
                fg_mask = lift_box2d_to_foreground_mask(
                    points_cv,
                    valid_mask,
                    [ymin, xmin, ymax, xmax],
                    proc_width,
                    proc_height,
                )
                if fg_mask is not None:
                    _append_unique(fg_mask, v_label, v_conf, "vlm_depth_lifted")

        for i, cand in enumerate(ov_candidates):
            if i in used_pool:
                continue
            _append_unique(cand["mask"], cand["label"], float(cand["conf"]), "yoloe_ov")
        return fused

    # Local adaptive dual-pass mode: YOLOE-OV prioritized, high-conf YOLOE-PF fills gaps
    for cand in ov_candidates:
        _append_unique(cand["mask"], cand["label"], float(cand["conf"]), "yoloe_ov")

    for cand in pf_candidates:
        if len(fused) >= max_objects:
            break
        if float(cand.get("conf", 0.0)) < 0.20:
            continue
        if _clean_label_slug(str(cand.get("label", "")), "").lower() in _BROAD_PF_SKIP_LABELS:
            continue
        _append_unique(cand["mask"], cand["label"], float(cand["conf"]), "yoloe_pf_4585")

    return fused


def run_local_metric_3d(
    image_base64: str,
    *,
    fov_override_deg: Optional[float] = None,
    auto_fov: bool = True,
    max_objects: int = 10,
    custom_prompt: Optional[str] = None,
    vlm_discovery: Optional[dict[str, Any]] = None,
    enable_prompt_free: bool = True,
    moge_backbone: str = "vits",
    aruco_marker_size_cm: Optional[float] = None,
    enable_qubo_nms: bool = True,
) -> dict[str, Any]:
    """Full hybrid/local GPU/CPU RGB -> Metric Depth + Adaptive Discovery/YOLOE -> 9-DOF 3D OBBs + Surfels."""
    from app.services.scene_intelligence import infer_local_scene_discovery, parse_custom_prompt_classes

    t0 = time.perf_counter()
    rgb_pil, w, h, exif_fov_h_deg = extract_exif_image_info(image_base64)
    hint = exif_fov_h_deg if auto_fov else fov_override_deg
    moge_out = _run_moge2_inference(rgb_pil, fov_hint_deg=hint, moge_backbone=moge_backbone)

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
    rgb_u8 = moge_out["rgb_u8"]
    pil_proc = Image.fromarray(rgb_u8, mode="RGB")
    pw = int(moge_out["proc_width"])
    ph = int(moge_out["proc_height"])
    user_classes = parse_custom_prompt_classes(custom_prompt)

    pf_candidates: list[dict[str, Any]] = []
    pf_used_flag = False

    if vlm_discovery is not None:
        raw_classes = list(vlm_discovery.get("detected_classes") or vlm_discovery.get("active_vocabulary") or [])
        target_classes: list[str] = []
        seen_t: set[str] = set()
        for c in [*user_classes, *raw_classes]:
            cleaned_c = re.sub(r"\s+", " ", str(c).strip().replace("_", " "))
            if cleaned_c and cleaned_c.lower() not in seen_t:
                seen_t.add(cleaned_c.lower())
                target_classes.append(cleaned_c)
            if len(target_classes) >= 18:
                break
        if not target_classes:
            target_classes = list(DEFAULT_OPEN_VOCAB_CLASSES)
        discovery_meta = dict(vlm_discovery)
        discovery_meta["active_vocabulary"] = target_classes
        vlm_objects_2d = list(vlm_discovery.get("objects_2d") or [])
    else:
        if enable_prompt_free and not user_classes:
            pf_model, _ = _load_prompt_free_segmenter(device=moge_out["device"])
            if pf_model is not None:
                try:
                    pf_res = pf_model.predict(
                        source=pil_proc,
                        device=moge_out["device"],
                        conf=0.16,
                        iou=0.55,
                        max_det=16,
                        verbose=False,
                    )
                    pf_candidates = _extract_ultralytics_candidates(
                        pf_res, pw, ph, source="yoloe_pf_4585"
                    )
                    pf_used_flag = True
                except Exception:
                    pf_candidates = []

        discovery_meta = infer_local_scene_discovery(
            pf_candidates,
            depth_stats=depth_stats,
            custom_prompt=custom_prompt,
            default_classes=DEFAULT_OPEN_VOCAB_CLASSES,
        )
        target_classes = list(discovery_meta["active_vocabulary"])
        vlm_objects_2d = []

    _, seg_model, device, model_desc = _load_moge2_and_segmenter(
        custom_classes=target_classes,
        moge_backbone=moge_backbone,
    )
    if pf_used_flag:
        model_desc = f"{model_desc} + YOLOE-pf (4585 cl.)"

    seg_results = seg_model.predict(
        source=pil_proc,
        device=device,
        conf=0.11,
        iou=0.55,
        max_det=max(max_objects * 2, 16),
        verbose=False,
    )
    ov_candidates = _extract_ultralytics_candidates(seg_results, pw, ph, source="yoloe_ov")

    if vlm_discovery is not None and not ov_candidates and enable_prompt_free and not pf_candidates:
        pf_model, _ = _load_prompt_free_segmenter(device=device)
        if pf_model is not None:
            try:
                pf_res = pf_model.predict(
                    source=pil_proc,
                    device=device,
                    conf=0.16,
                    iou=0.55,
                    max_det=16,
                    verbose=False,
                )
                pf_candidates = _extract_ultralytics_candidates(
                    pf_res, pw, ph, source="yoloe_pf_4585"
                )
            except Exception:
                pf_candidates = []

    fused_candidates = fuse_discovery_and_segmentation_candidates(
        ov_candidates=ov_candidates,
        pf_candidates=pf_candidates,
        vlm_objects_2d=vlm_objects_2d,
        points_cv=moge_out["points_cv"],
        valid_mask=moge_out["valid_mask"],
        proc_width=pw,
        proc_height=ph,
        max_objects=max_objects,
    )

    lifted_objects: list[dict[str, Any]] = []
    kept_masks_proc: list[np.ndarray] = []
    sources_used: list[str] = []
    vlm_guided_count = 0

    for cand in fused_candidates:
        if len(lifted_objects) >= max_objects:
            break
        clean_label = str(cand["label"])
        bin_mask = cand["mask"]
        box_3d = lift_mask_to_gemini_obb(
            moge_out["points_cv"],
            bin_mask,
            moge_out["valid_mask"],
            rgb_u8=rgb_u8,
            label=clean_label,
        )
        if box_3d is None:
            continue

        src_tag = str(cand.get("source", "yoloe_ov"))
        if src_tag not in sources_used:
            sources_used.append(src_tag)
        if src_tag.startswith("vlm_"):
            vlm_guided_count += 1

        kept_masks_proc.append(bin_mask)
        lifted_objects.append(
            {
                "id": f"moge_{len(lifted_objects) + 1}",
                "label": clean_label,
                "box_3d": box_3d,
                "confidence": round(float(cand["conf"]), 3),
            }
        )

    if not lifted_objects:
        lifted_objects = _depth_layered_fallback_objects(
            moge_out["points_cv"],
            moge_out["depth_m"],
            moge_out["valid_mask"],
            max_objects=min(max_objects, 4),
        )
        if lifted_objects:
            sources_used.append("depth_layer_fallback")

    quantum_nms_summary: Optional[dict[str, Any]] = None
    if enable_qubo_nms and len(lifted_objects) >= 2:
        from app.core.quantum_qubo import resolve_3d_objects_mwis_qubo

        q_nms = resolve_3d_objects_mwis_qubo(
            lifted_objects,
            include_hardware_export=False,
        )
        kept_idx = list(q_nms["kept_indices"])
        if kept_idx:
            had_masks = len(kept_masks_proc) == len(lifted_objects)
            sup_labels = [str(o.get("label", o.get("id"))) for o in q_nms["suppressed_objects"]]
            lifted_objects = [lifted_objects[i] for i in kept_idx]
            if had_masks:
                kept_masks_proc = [kept_masks_proc[i] for i in kept_idx]
            qs = q_nms["qubo_stats"]
            quantum_nms_summary = {
                "enabled": True,
                "input_count": int(qs["n_vars"]),
                "kept_count": len(lifted_objects),
                "suppressed_count": len(q_nms["suppressed_indices"]),
                "conflict_edges_count": int(qs["n_conflict_edges"]),
                "energy_gain": float(qs["energy_gain"]),
                "solver_ms": float(qs["solver_ms"]),
                "exact_optimum_verified": bool(qs["exact_optimum_verified"]),
                "suppressed_labels": sup_labels,
            }

    discovery_meta["vlm_guided_count"] = vlm_guided_count
    discovery_meta["detection_sources"] = sources_used

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

    point_cloud = pack_point_cloud_threejs(
        moge_out["points_cv"],
        rgb_u8,
        moge_out["valid_mask"],
        max_points=80000,
        normals_cv=moge_out.get("normals_cv"),
        instance_masks=kept_masks_proc,
        object_ids=[str(o["id"]) for o in lifted_objects],
        ransac_plane=ransac_plane,
        include_surface_mesh=True,
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

    if vlm_discovery is not None:
        mode_name = "hybrid_vlm_moge2_gpu" if device.startswith("cuda") else "hybrid_vlm_moge2_cpu"
    else:
        mode_name = "moge2_yoloe_gpu" if device.startswith("cuda") else "moge2_yoloe_cpu"

    aruco_anchor = _safe_detect_aruco_anchor(
        rgb_u8,
        moge_out["points_cv"],
        moge_out["valid_mask"],
        aruco_marker_size_cm,
    )

    domain_es = str(discovery_meta.get("domain_label_es") or "Inspección Física General")
    return {
        "mode": mode_name,
        "model_desc": model_desc,
        "moge_backbone": moge_out.get("moge_backbone", _resolve_moge_backbone(moge_backbone)),
        "aruco_anchor": aruco_anchor,
        "width": w,
        "height": h,
        "image_base64": oriented_b64,
        "mime_type": "image/png",
        "effective_fov_h_deg": round(effective_fov, 2),
        "fov_source": fov_source,
        "exif_fov_h_deg": exif_fov_h_deg,
        "estimated_moge_fov_h_deg": moge_fov,
        "objects": lifted_objects,
        "discovery_meta": discovery_meta,
        "scene_summary": (
            f"Reconstrucción métrica ({model_desc} · {domain_es}): {len(lifted_objects)} objetos en rango "
            f"{depth_stats['min_m']:.2f} m – {depth_stats['max_m']:.2f} m "
            f"(mediana {depth_stats['median_m']:.2f} m, FOV_h={effective_fov:.1f}° vía {fov_source})."
        ),
        "depth_map_base64": depth_b64,
        "depth_stats": depth_stats,
        "point_cloud": point_cloud,
        "real_photo_metrics": real_metrics,
        "quantum_nms_summary": quantum_nms_summary,
        "latency_ms": latency_ms,
    }


def _compute_multiview_foreground_mask(depth_m: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Estimate foreground body mask for multi-view scans using central depth band."""
    valid = valid_mask & np.isfinite(depth_m) & (depth_m > 0.04) & (depth_m < 35.0)
    if not np.any(valid):
        return valid
    h, w = depth_m.shape[:2]
    y0, y1 = int(h * 0.2), max(int(h * 0.8), int(h * 0.2) + 1)
    x0, x1 = int(w * 0.2), max(int(w * 0.8), int(w * 0.2) + 1)
    center_d = depth_m[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
    if center_d.size < 32:
        center_d = depth_m[valid]
    d_med = float(np.median(center_d))
    d_lo = max(0.04, float(np.percentile(depth_m[valid], 2.0)))
    d_hi = max(d_lo + 0.08, min(float(np.percentile(depth_m[valid], 78.0)), d_med * 1.45))
    fg = valid & (depth_m >= d_lo) & (depth_m <= d_hi)
    return fg if int(np.sum(fg)) >= 64 else valid


def run_local_multiview_360(
    images: list[dict[str, Any]],
    *,
    fov_override_deg: Optional[float] = None,
    auto_fov: bool = True,
    moge_backbone: str = "vits",
    voxel_size_m: Optional[float] = None,
    allow_scale_alignment: bool = True,
    aruco_marker_size_cm: Optional[float] = None,
    use_qubo_registration: bool = True,
) -> dict[str, Any]:
    """Sequentially lift 2-6 orbital RGB images via MoGe-2 and fuse into a 360-degree 3D model."""
    t0 = time.perf_counter()
    bb = _resolve_moge_backbone(moge_backbone)

    frames: list[dict[str, Any]] = []
    first_aruco: Optional[dict[str, Any]] = None
    first_w, first_h, first_fov = 640, 480, float(fov_override_deg or 60.0)
    first_depth_stats: Optional[dict[str, float]] = None
    last_device = "cpu"
    last_device_desc = ""

    for idx, item in enumerate(images):
        img_b64 = str(item.get("image_base64") or "")
        if not img_b64:
            continue
        rgb_pil, w, h, exif_fov = extract_exif_image_info(img_b64)
        hint = exif_fov if auto_fov else fov_override_deg
        moge_out = _run_moge2_inference(rgb_pil, fov_hint_deg=hint, moge_backbone=bb)
        last_device = str(moge_out["device"])
        last_device_desc = str(moge_out["device_desc"])

        moge_fov = float(moge_out["moge_fov_h_deg"])
        if not auto_fov and fov_override_deg is not None:
            eff_fov = float(fov_override_deg)
        elif exif_fov is not None:
            eff_fov = float(exif_fov)
        else:
            eff_fov = moge_fov

        rgb_u8 = moge_out["rgb_u8"]
        pts_cv = moge_out["points_cv"]
        dep_m = moge_out["depth_m"]
        valid_mask = moge_out["valid_mask"]
        normals_cv = moge_out.get("normals_cv")

        aruco = _safe_detect_aruco_anchor(rgb_u8, pts_cv, valid_mask, aruco_marker_size_cm)
        if first_aruco is None and aruco and aruco.get("detected"):
            first_aruco = aruco

        if idx == 0:
            first_w, first_h, first_fov = int(w), int(h), round(eff_fov, 2)
            first_depth_stats = compute_depth_stats(dep_m, valid_mask)

        fg_mask = _compute_multiview_foreground_mask(dep_m, valid_mask)
        yaw_hint = item.get("yaw_hint_deg")
        frames.append(
            {
                "label": str(item.get("label") or f"Vista {idx + 1}"),
                "rgb_u8": rgb_u8,
                "points_cv": pts_cv,
                "normals_cv": normals_cv,
                "valid_mask": valid_mask,
                "foreground_mask": fg_mask,
                "fov_h_deg": eff_fov,
                "yaw_hint_deg": float(yaw_hint) if yaw_hint is not None else None,
                "aruco_anchor": aruco,
                "thumbnail_base64": _encode_png_thumbnail(rgb_u8),
            }
        )

    try:
        import torch

        if last_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if len(frames) < 2:
        raise ValueError("At least 2 valid RGB views are required for 360-degree multi-view reconstruction.")

    fused = fuse_multiview_frames(
        frames,
        voxel_size_m=voxel_size_m,
        allow_scale_alignment=allow_scale_alignment,
        use_qubo_registration=use_qubo_registration,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    mode_name = "multiview_moge2_gpu" if last_device.startswith("cuda") else "multiview_moge2_cpu"

    return {
        "mode": mode_name,
        "model_desc": last_device_desc,
        "moge_backbone": bb,
        "view_count": fused["view_count"],
        "aruco_anchor": first_aruco,
        "camera_poses": fused["camera_poses"],
        "coverage_360": fused["coverage_360"],
        "body_metrology": fused["body_metrology"],
        "point_cloud": fused["point_cloud"],
        "fused_box_3d": fused["fused_box_3d"],
        "quantum_telemetry": fused.get("quantum_telemetry"),
        "width": first_w,
        "height": first_h,
        "fov_h_deg": first_fov,
        "depth_stats": first_depth_stats,
        "latency_ms": latency_ms,
    }
