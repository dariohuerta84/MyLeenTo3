"""Universal reference-free 3D spatial diagnostics and image-adaptive hypotheses.

Evaluates any RGB image (real photos, artistic/stylized renders, World-Model
frames, or calibrated benchmarks) without requiring Ground Truth 3D labels,
quantifying latent physical/photometric variables and exposing monocular VLM
failure modes.
"""

from __future__ import annotations

import base64
import io
import math
import re
from typing import Any, Optional

import numpy as np
from PIL import Image

from .geometry_3d import (
    compute_3d_iou,
    compute_occlusion_matrix,
    compute_pairwise_geometry,
    obb_corners_gemini,
    project_obb_to_2d,
)
from .metrics import DEFAULT_FOV_SWEEP_VALUES


def _decode_rgb_array(
    image_base64: Optional[str],
    max_width: int = 320,
) -> Optional[np.ndarray]:
    """Decode base64/data-URI image to float32 RGB array in [0, 1]."""
    if not image_base64 or not image_base64.strip():
        return None
    raw = image_base64.strip()
    match = re.match(r"^data:([^;]+);base64,(.*)$", raw, flags=re.DOTALL)
    if match:
        raw = match.group(2).strip()
    try:
        data = base64.b64decode(raw, validate=False)
        with Image.open(io.BytesIO(data)) as img:
            rgb = img.convert("RGB")
            if rgb.width > max_width:
                scale = max_width / float(rgb.width)
                new_h = max(16, int(round(rgb.height * scale)))
                rgb = rgb.resize((max_width, new_h), Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
            if arr.ndim != 3 or arr.shape[2] < 3:
                return None
            return arr[:, :, :3]
    except Exception:
        return None


def _sobel_gradients(luma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Sobel (gx, gy, mag) in pure NumPy via reflect padding."""
    p = np.pad(luma, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -p[:-2, :-2]
        + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2]
        + 2.0 * p[1:-1, 2:]
        - p[2:, :-2]
        + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2]
        - 2.0 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2.0 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def _nearest_peak_distances(grad_norm: np.ndarray, threshold: float = 0.28) -> np.ndarray:
    """Compute approximate distance (in scaled px) from every pixel to nearest strong edge."""
    h, w = grad_norm.shape
    mask = grad_norm >= threshold
    if not np.any(mask):
        return np.full((h, w), float(max(h, w)) * 0.25, dtype=np.float32)

    # Fast 2-pass Chamfer L1/L2 distance transform in pure NumPy
    inf_val = float(h + w)
    dist = np.where(mask, 0.0, inf_val).astype(np.float32)

    # Forward pass
    for y in range(h):
        row = dist[y]
        if y > 0:
            prev = dist[y - 1]
            row = np.minimum(row, prev + 1.0)
            row[1:] = np.minimum(row[1:], prev[:-1] + 1.414)
            row[:-1] = np.minimum(row[:-1], prev[1:] + 1.414)
        for x in range(1, w):
            if row[x - 1] + 1.0 < row[x]:
                row[x] = row[x - 1] + 1.0
        dist[y] = row

    # Backward pass
    for y in range(h - 1, -1, -1):
        row = dist[y]
        if y + 1 < h:
            nxt = dist[y + 1]
            row = np.minimum(row, nxt + 1.0)
            row[1:] = np.minimum(row[1:], nxt[:-1] + 1.414)
            row[:-1] = np.minimum(row[:-1], nxt[1:] + 1.414)
        for x in range(w - 2, -1, -1):
            if row[x + 1] + 1.0 < row[x]:
                row[x] = row[x + 1] + 1.0
        dist[y] = row

    return dist


def _analyze_photometric_profile(
    rgb: Optional[np.ndarray],
) -> tuple[dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract photometric stress metrics and normalized Sobel edge map."""
    if rgb is None:
        profile = {
            "luminance_mean": 0.50,
            "luminance_std": 0.18,
            "dark_shadow_ratio": 0.08,
            "bloom_saturation_ratio": 0.02,
            "spectral_entropy_bits": 3.80,
            "perspective_coherence_score": 0.72,
            "domain_classification": "calibrated_studio",
        }
        return profile, None, None

    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    lum_mean = float(np.mean(luma))
    lum_std = float(np.std(luma))
    dark_ratio = float(np.mean(luma < 0.14))

    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    bloom_mask = (luma > 0.90) | ((chroma > 0.45) & (luma > 0.60))
    bloom_ratio = float(np.mean(bloom_mask))

    # Shannon entropy on 32-bin quantized luminance
    hist, _ = np.histogram(luma, bins=32, range=(0.0, 1.0))
    probs = hist.astype(np.float64)
    total = float(np.sum(probs))
    if total > 0:
        probs = probs / total
        nz = probs[probs > 0]
        entropy_bits = float(-np.sum(nz * np.log2(nz)))
    else:
        entropy_bits = 0.0

    gx, gy, mag = _sobel_gradients(luma)
    p95 = float(np.percentile(mag, 95))
    grad_norm = np.clip(mag / max(p95, 1e-4), 0.0, 1.0).astype(np.float32)

    # Perspective coherence: alignment with structured horizontal/vertical/vanishing lines
    strong_thresh = float(np.mean(mag) + 0.5 * np.std(mag))
    strong = mag > max(strong_thresh, 1e-4)
    if np.any(strong):
        theta = np.arctan2(gy[strong], gx[strong])
        # Four-fold symmetry cos(4*theta) peaks at 0, 90, 180, 270 deg (orthographic/rectilinear)
        ortho_align = float(np.mean(np.maximum(0.0, np.cos(4.0 * theta))))
        # Plus overall edge concentration
        edge_density = float(np.mean(strong))
        persp_score = float(np.clip(0.68 * ortho_align + 0.32 * min(1.0, edge_density * 4.0), 0.0, 1.0))
    else:
        persp_score = 0.25

    if dark_ratio > 0.38 and bloom_ratio > 0.035:
        domain = "high_contrast_stylized"
    elif dark_ratio > 0.42 or lum_mean < 0.25:
        domain = "low_light_high_stress"
    elif entropy_bits < 3.45 and bloom_ratio < 0.03 and persp_score >= 0.45:
        domain = "calibrated_studio"
    else:
        domain = "natural_indoor"

    dist_map = _nearest_peak_distances(grad_norm)

    profile = {
        "luminance_mean": round(lum_mean, 4),
        "luminance_std": round(lum_std, 4),
        "dark_shadow_ratio": round(dark_ratio, 4),
        "bloom_saturation_ratio": round(bloom_ratio, 4),
        "spectral_entropy_bits": round(entropy_bits, 4),
        "perspective_coherence_score": round(persp_score, 4),
        "domain_classification": domain,
    }
    return profile, grad_norm, dist_map


def _sample_edge_alignment(
    proj: dict[str, Any],
    width: int,
    height: int,
    grad_norm: Optional[np.ndarray],
    dist_map: Optional[np.ndarray],
) -> tuple[float, float]:
    """Sample Sobel gradient alignment along 12 projected edges and pixel distance residual."""
    if grad_norm is None or dist_map is None:
        return 0.62, 6.5

    gh, gw = grad_norm.shape
    sx = gw / float(max(width, 1))
    sy = gh / float(max(height, 1))
    verts = proj["vertices_2d"]
    edges = proj["edges"]

    edge_scores: list[float] = []
    px_dists: list[float] = []
    t_vals = np.linspace(0.08, 0.92, 10)

    for i0, i1 in edges:
        u0, v0 = verts[i0]
        u1, v1 = verts[i1]
        for t in t_vals:
            u = (u0 + (u1 - u0) * t) * sx
            v = (v0 + (v1 - v0) * t) * sy
            ix = int(round(u))
            iy = int(round(v))
            if 0 <= ix < gw and 0 <= iy < gh:
                x0 = max(0, ix - 2)
                x1 = min(gw, ix + 3)
                y0 = max(0, iy - 2)
                y1 = min(gh, iy + 3)
                local_peak = float(np.max(grad_norm[y0:y1, x0:x1]))
                edge_scores.append(local_peak)
                px_dists.append(float(dist_map[iy, ix]) / max(sx, 1e-4))

    if not edge_scores:
        return 0.0, float(max(width, height)) * 0.25

    mean_grad = float(np.mean(grad_norm))
    raw_align = float(np.mean(edge_scores))
    # Normalize against scene baseline gradient so dense textures don't inflate artificially
    contrast_boost = raw_align / max(raw_align + mean_grad * 0.45, 1e-4)
    score = float(np.clip(0.55 * raw_align + 0.45 * contrast_boost, 0.0, 1.0))
    return score, float(np.mean(px_dists))


def _weakness_label_for_object(
    *,
    occlusion_sev: float,
    trunc_ratio: float,
    floating_gap_m: float,
    anisotropy: float,
    edge_align: float,
    uncertainty_m: float,
) -> str:
    """Assign primary Spanish diagnostic failure tag for a single 3D box."""
    if occlusion_sev >= 0.35:
        return "Oclusión Severa"
    if trunc_ratio >= 0.25:
        return "Truncamiento de Encuadre"
    if floating_gap_m >= 0.35:
        return "Levitación sobre Plano Latente"
    if anisotropy >= 0.72:
        return "Ambigüedad Escala-Profundidad"
    if edge_align < 0.38:
        return "Baja Coherencia Fotométrica"
    if uncertainty_m >= 0.45:
        return "Incertidumbre Cuadrática de Rango"
    return "Estable"


def _compute_cascade_policy(
    *,
    objects: list[dict[str, Any]],
    image_profile: dict[str, Any],
    occlusion_matrix: list[dict[str, Any]],
    colliding_pairs: int,
    plausibility: float,
    pvi: float,
    gfi: float,
    sdas: float,
    fbr: float,
    ofi: float,
) -> dict[str, Any]:
    """Classify scene stress in C8286 regimes and emit deterministic Edge -> 3D escalation gate."""
    n = len(objects)
    occ_count = len(occlusion_matrix)
    max_occ = max((float(e.get("occlusion_ratio", 0.0)) for e in occlusion_matrix), default=0.0)
    max_abs_yaw_deg = max((abs(float(o["box_3d"][8])) for o in objects), default=0.0)

    min_clearance: Optional[float] = None
    if n >= 2:
        clearances = [
            compute_pairwise_geometry(objects[i]["box_3d"], objects[j]["box_3d"])["surface_clearance_m"]
            for i in range(n)
            for j in range(i + 1, n)
        ]
        if clearances:
            min_clearance = round(float(min(clearances)), 3)

    dark_ratio = float(image_profile.get("dark_shadow_ratio", 0.0))
    bloom_ratio = float(image_profile.get("bloom_saturation_ratio", 0.0))
    lum_mean = float(image_profile.get("luminance_mean", 0.5))

    # Separate real underexposure/localized glare from uniform bright studio backdrops
    is_illum_stress = (
        dark_ratio > 0.30
        or lum_mean < 0.28
        or (bloom_ratio > 0.035 and (dark_ratio >= 0.05 or lum_mean <= 0.70))
    )
    # Threshold >= 0.35 ray occlusion corresponds to heavy partial occlusion (>1/3 of target rays)
    is_occ_stress = (occ_count >= 1 and max_occ >= 0.35) or (fbr >= 0.25)

    if is_illum_stress and is_occ_stress:
        regime = "hybrid_stress"
    elif is_occ_stress:
        regime = "occlusion_bg"
    elif is_illum_stress:
        regime = "illumination_noise"
    else:
        regime = "clean"

    reasons: list[str] = []

    if n == 0:
        reasons.append(
            "Sin detecciones 3D activas (0 objetos); la compuerta impide inferir distancias sin respaldo."
        )
        return {
            "operational_verdict": "RECHAZADO_FISICA",
            "recommended_tier": "REJECT_RECAPTURE_OR_SENSOR",
            "recommended_model": "none",
            "c8286_degradation_regime": regime,
            "estimated_latency_savings_pct": 0.0,
            "min_surface_clearance_m": None,
            "occlusion_pairs_count": 0,
            "colliding_pairs_count": 0,
            "trigger_reasons": reasons,
            "summary": "Compuerta RECHAZADA: 0 objetos 3D detectados; requiere re-captura o verificación manual.",
        }

    # Gate 1: Reject when hard physical violations or extreme dark-range uncertainty break plausibility
    if pvi >= 0.08 or gfi >= 0.55 or plausibility < 45.0 or (is_illum_stress and ofi >= 0.55):
        if pvi >= 0.08:
            reasons.append(
                f"Interpenetración volumétrica severa SAT (PVI {pvi * 100:.1f}% >= 8.0%, {colliding_pairs} colisiones)."
            )
        if gfi >= 0.55:
            reasons.append(
                f"Levitación/hundimiento severo respecto al plano de soporte (GFI {gfi:.2f} m >= 0.55 m)."
            )
        if plausibility < 45.0:
            reasons.append(
                f"Plausibilidad física insuficiente ({plausibility:.1f}/100 < 45.0) para operar sin sensor 3D."
            )
        if is_illum_stress and ofi >= 0.55:
            reasons.append(
                f"Estrés fotométrico ({regime}) combinado con alta incertidumbre de rango (±{ofi:.2f} m >= 0.55 m)."
            )
        return {
            "operational_verdict": "RECHAZADO_FISICA",
            "recommended_tier": "REJECT_RECAPTURE_OR_SENSOR",
            "recommended_model": "gemini-4-hydrogen",
            "c8286_degradation_regime": regime,
            "estimated_latency_savings_pct": 0.0,
            "min_surface_clearance_m": min_clearance,
            "occlusion_pairs_count": occ_count,
            "colliding_pairs_count": colliding_pairs,
            "trigger_reasons": reasons,
            "summary": (
                f"Compuerta RECHAZADA por inconsistencia física severa ({reasons[0]}). "
                "Se recomienda nueva captura o validación LiDAR antes de actuar."
            ),
        }

    # Gate 2: Escalate to deep VLM reasoning when visual/geometric complexity exceeds Edge/Fast ceiling
    if (
        is_occ_stress
        or is_illum_stress
        or pvi > 0.01
        or gfi >= 0.30
        or ofi >= 0.35
        or sdas >= 0.62
        or plausibility < 68.0
        or max_abs_yaw_deg >= 45.0
    ):
        if is_occ_stress:
            reasons.append(
                f"Oclusión severa detectada ({occ_count} pares, máx. {max_occ * 100:.0f}% rayos bloqueados): "
                "supera el techo de robustez 2D de C8286 (F1=0.715) y exige razonamiento 3D profundo."
            )
        if max_abs_yaw_deg >= 45.0:
            reasons.append(
                f"Rotación 3D marcada (|yaw| máx. = {max_abs_yaw_deg:.1f}° >= 45°): requiere desambiguación completa 9-DOF."
            )
        if is_illum_stress:
            reasons.append(
                f"Degradación fotométrica activa (sombras {dark_ratio * 100:.0f}%, lum. {lum_mean:.2f}): "
                "los descriptores clásicos 2D colapsan y el tier rápido pierde anclas de escala."
            )
        if ofi >= 0.35:
            reasons.append(
                f"Incertidumbre cuadrática de rango elevada (±σ_y = {ofi:.2f} m >= 0.35 m)."
            )
        if pvi > 0.01 or gfi >= 0.30:
            reasons.append(
                f"Tensión geométrica moderada (PVI {pvi * 100:.1f}%, GFI {gfi:.2f} m): requiere verificación profunda."
            )
        if sdas >= 0.62 or plausibility < 68.0:
            reasons.append(
                f"Ambigüedad escala–profundidad (SDAS {sdas:.2f}) o plausibilidad intermedia ({plausibility:.1f}/100)."
            )
        return {
            "operational_verdict": "ESCALAR_MODELO",
            "recommended_tier": "VLM_3D_DEEP_REASONING",
            "recommended_model": "gemini-4-hydrogen",
            "c8286_degradation_regime": regime,
            "estimated_latency_savings_pct": 0.0,
            "min_surface_clearance_m": min_clearance,
            "occlusion_pairs_count": occ_count,
            "colliding_pairs_count": colliding_pairs,
            "trigger_reasons": reasons,
            "summary": (
                f"Escalar a gemini-4-hydrogen (régimen C8286 '{regime}'): {reasons[0]}"
            ),
        }

    # Gate 3: Clean geometry & lighting -> safe for cost-aware fast/balanced 3D inference
    if plausibility >= 75.0 and n <= 5 and max_occ < 0.15 and max_abs_yaw_deg < 25.0:
        rec_model = "gemini-3.5-flash-lite"
        savings = 72.0
        reasons.append(
            f"Régimen limpio (C8286 'clean'), sin oclusiones críticas ({max_occ * 100:.0f}%) ni colisiones SAT y alta plausibilidad ({plausibility:.1f}/100)."
        )
    else:
        rec_model = "gemini-4-argon"
        savings = 45.0
        reasons.append(
            f"Escena estable (plausibilidad {plausibility:.1f}/100, PVI {pvi * 100:.1f}%, ±σ_y={ofi:.2f} m) "
            "con orientación u oclusión leve: el nivel intermedio preserva precisión reduciendo latencia."
        )

    return {
        "operational_verdict": "APROBADO",
        "recommended_tier": "VLM_3D_FAST_TIER",
        "recommended_model": rec_model,
        "c8286_degradation_regime": regime,
        "estimated_latency_savings_pct": savings,
        "min_surface_clearance_m": min_clearance,
        "occlusion_pairs_count": occ_count,
        "colliding_pairs_count": colliding_pairs,
        "trigger_reasons": reasons,
        "summary": (
            f"Compuerta APROBADA en tier rápido ({rec_model}, -{savings:.0f}% latencia estimada vs. modelo profundo): "
            f"sin colisiones 3D ni estrés fotométrico crítico."
        ),
    }


def compute_intrinsic_scene_diagnostics(
    objects: list[dict[str, Any]],
    image_base64: Optional[str],
    width: int,
    height: int,
    fov_h_deg: float,
    occlusion_matrix: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compute reference-free latent 3D metrics, photometric stress, and FOV sweep."""
    rgb = _decode_rgb_array(image_base64, max_width=320)
    image_profile, grad_norm, dist_map = _analyze_photometric_profile(rgb)

    if occlusion_matrix is None:
        occlusion_matrix = compute_occlusion_matrix(objects)

    occ_by_id: dict[str, float] = {}
    for entry in occlusion_matrix:
        oid = str(entry.get("occluded_id", ""))
        ratio = float(entry.get("occlusion_ratio", 0.0))
        if oid:
            occ_by_id[oid] = max(occ_by_id.get(oid, 0.0), ratio)

    n = len(objects)
    if n == 0:
        empty_sweep = [
            {
                "fov_h_deg": int(fov),
                "reproj_iou_2d": 0.0,
                "mean_pixel_error": 0.0,
            }
            for fov in DEFAULT_FOV_SWEEP_VALUES
        ]
        empty_policy = _compute_cascade_policy(
            objects=objects,
            image_profile=image_profile,
            occlusion_matrix=occlusion_matrix,
            colliding_pairs=0,
            plausibility=0.0,
            pvi=0.0,
            gfi=0.0,
            sdas=0.0,
            fbr=0.0,
            ofi=0.0,
        )
        return {
            "image_profile": image_profile,
            "latent_metrics": {
                "physical_plausibility_score": 0.0,
                "photometric_edge_coherence": 0.0,
                "volume_interpenetration_index": 0.0,
                "gravity_floating_anomaly_m": 0.0,
                "support_plane_z_m": 0.0,
                "scale_depth_ambiguity_score": 0.0,
                "frustum_truncation_ratio": 0.0,
                "mean_depth_uncertainty_m": 0.0,
            },
            "per_object_diagnostics": [],
            "exploited_weaknesses": [
                {
                    "id": "no_objects",
                    "title": "Sin detecciones 3D activas",
                    "severity": "high",
                    "metric_value": "0 obj",
                    "explanation": "La escena no contiene objetos 3D detectados para auditar coherencia fotométrica ni física.",
                }
            ],
            "reference_free_fov_sweep": empty_sweep,
            "estimated_optimal_fov_h_deg": int(round(fov_h_deg)),
            "cascade_policy": empty_policy,
        }

    # Precompute bottom faces for latent support plane
    z_bots: list[float] = []
    volumes: list[float] = []
    depths: list[float] = []
    anisotropies: list[float] = []

    for obj in objects:
        box = obj["box_3d"]
        corners = obb_corners_gemini(box)
        z_bots.append(float(np.min(corners[:, 2])))
        dx, dy, dz = abs(float(box[3])), abs(float(box[4])), abs(float(box[5]))
        volumes.append(max(dx * dy * dz, 1e-6))
        depths.append(max(float(box[1]), 0.1))
        anisotropies.append(1.0 - min(dx, dy, dz) / max(dx, dy, dz, 1e-6))

    support_plane_z = float(np.percentile(z_bots, 25))

    # Pairwise 3D volume interpenetration (PVI)
    total_vol = float(sum(volumes))
    inter_vol = 0.0
    colliding_pairs = 0
    for i in range(n):
        bi = objects[i]["box_3d"]
        ci = np.array(bi[:3], dtype=float)
        ri = 0.5 * math.sqrt(bi[3] ** 2 + bi[4] ** 2 + bi[5] ** 2)
        for j in range(i + 1, n):
            bj = objects[j]["box_3d"]
            cj = np.array(bj[:3], dtype=float)
            rj = 0.5 * math.sqrt(bj[3] ** 2 + bj[4] ** 2 + bj[5] ** 2)
            if float(np.linalg.norm(ci - cj)) >= (ri + rj):
                continue
            iou = compute_3d_iou(bi, bj, seed=42)
            if iou > 1e-5:
                colliding_pairs += 1
                inter_vol += (iou / (1.0 + iou)) * (volumes[i] + volumes[j])

    pvi = float(np.clip(inter_vol / max(total_vol, 1e-6), 0.0, 1.0))

    # Scale-depth coupling & dispersion (SDAS)
    log_vols = np.log(np.array(volumes, dtype=float))
    log_depths = np.log(np.array(depths, dtype=float))
    std_log_vol = float(np.std(log_vols)) if n >= 2 else 0.0
    if n >= 3 and np.std(log_vols) > 1e-6 and np.std(log_depths) > 1e-6:
        abs_corr = abs(float(np.corrcoef(log_vols, log_depths)[0, 1]))
        if math.isnan(abs_corr):
            abs_corr = 0.0
    else:
        abs_corr = 0.0
    mean_anisotropy = float(np.mean(anisotropies))
    sdas = float(
        np.clip(
            0.45 * mean_anisotropy + 0.35 * abs_corr + 0.20 * min(1.0, std_log_vol / 2.5),
            0.0,
            1.0,
        )
    )

    bloom_ratio = float(image_profile["bloom_saturation_ratio"])
    per_object_diagnostics: list[dict[str, Any]] = []
    pecs_list: list[float] = []
    fbr_list: list[float] = []
    gfi_list: list[float] = []
    ofi_list: list[float] = []

    for idx, obj in enumerate(objects):
        oid = str(obj.get("id", f"obj_{idx + 1}"))
        label = str(obj.get("label", oid))
        box = obj["box_3d"]
        y_depth = depths[idx]
        roll, pitch = float(box[6]), float(box[7])
        tilt_penalty = 0.004 * (abs(roll) + abs(pitch))
        float_gap = abs(z_bots[idx] - support_plane_z) + tilt_penalty

        proj = project_obb_to_2d(box, width, height, fov_h_deg)
        out_count = 0
        for u, v in proj["vertices_2d"]:
            if u < 0.0 or u > float(width) or v < 0.0 or v > float(height):
                out_count += 1
        trunc_ratio = 1.0 if proj["behind_camera"] else (out_count / 8.0)

        edge_align, _ = _sample_edge_alignment(proj, width, height, grad_norm, dist_map)
        occ_sev = occ_by_id.get(oid, 0.0)

        # Monocular Cramér-Rao style depth uncertainty
        sigma_y = 0.025 * (y_depth ** 2) * (1.0 + 1.8 * occ_sev + 0.9 * trunc_ratio + 0.7 * bloom_ratio)

        vuln = float(
            np.clip(
                100.0
                * (
                    0.26 * (1.0 - edge_align)
                    + 0.24 * occ_sev
                    + 0.18 * trunc_ratio
                    + 0.16 * min(1.0, float_gap / 1.2)
                    + 0.16 * min(1.0, sigma_y / 1.0)
                ),
                0.0,
                100.0,
            )
        )

        dom_weakness = _weakness_label_for_object(
            occlusion_sev=occ_sev,
            trunc_ratio=trunc_ratio,
            floating_gap_m=float_gap,
            anisotropy=anisotropies[idx],
            edge_align=edge_align,
            uncertainty_m=sigma_y,
        )

        pecs_list.append(edge_align)
        fbr_list.append(trunc_ratio)
        gfi_list.append(float_gap)
        ofi_list.append(sigma_y)

        per_object_diagnostics.append(
            {
                "object_id": oid,
                "label": label,
                "depth_m": round(y_depth, 3),
                "uncertainty_depth_m": round(sigma_y, 3),
                "edge_alignment_score": round(edge_align, 4),
                "floating_gap_m": round(float_gap, 3),
                "anisotropy_ratio": round(anisotropies[idx], 4),
                "frustum_truncation_ratio": round(trunc_ratio, 4),
                "occlusion_severity": round(occ_sev, 4),
                "vulnerability_score": round(vuln, 1),
                "dominant_weakness": dom_weakness,
            }
        )

    pecs = float(np.mean(pecs_list))
    fbr = float(np.mean(fbr_list))
    gfi = float(np.mean(gfi_list))
    ofi = float(np.mean(ofi_list))

    # Composite physical plausibility index (0..100)
    plausibility = float(
        np.clip(
            100.0
            * (
                0.34 * pecs
                + 0.22 * (1.0 - min(1.0, pvi * 2.5))
                + 0.16 * (1.0 - min(1.0, gfi / 1.4))
                + 0.14 * (1.0 - fbr)
                + 0.14 * (1.0 - sdas)
            ),
            0.0,
            100.0,
        )
    )

    latent_metrics = {
        "physical_plausibility_score": round(plausibility, 1),
        "photometric_edge_coherence": round(pecs, 4),
        "volume_interpenetration_index": round(pvi, 4),
        "gravity_floating_anomaly_m": round(gfi, 3),
        "support_plane_z_m": round(support_plane_z, 3),
        "scale_depth_ambiguity_score": round(sdas, 4),
        "frustum_truncation_ratio": round(fbr, 4),
        "mean_depth_uncertainty_m": round(ofi, 3),
    }

    # Reference-free FOV sensitivity sweep using photometric edge coherence & containment
    ref_sweep: list[dict[str, Any]] = []
    for cand_fov in DEFAULT_FOV_SWEEP_VALUES:
        c_aligns: list[float] = []
        c_conts: list[float] = []
        c_px_errs: list[float] = []
        for obj in objects:
            p = project_obb_to_2d(obj["box_3d"], width, height, float(cand_fov))
            in_cnt = sum(
                1 for (u, v) in p["vertices_2d"] if 0.0 <= u <= float(width) and 0.0 <= v <= float(height)
            )
            cont = 0.0 if p["behind_camera"] else (in_cnt / 8.0)
            align, px_err = _sample_edge_alignment(p, width, height, grad_norm, dist_map)
            c_aligns.append(align)
            c_conts.append(cont)
            c_px_errs.append(px_err)
        mean_align = float(np.mean(c_aligns)) if c_aligns else 0.0
        mean_cont = float(np.mean(c_conts)) if c_conts else 0.0
        pseudo_iou = float(np.clip(0.65 * mean_align + 0.35 * mean_cont, 0.0, 1.0))
        ref_sweep.append(
            {
                "fov_h_deg": int(cand_fov),
                "reproj_iou_2d": round(pseudo_iou, 4),
                "mean_pixel_error": round(float(np.mean(c_px_errs)) if c_px_errs else 0.0, 2),
            }
        )

    best_fov_pt = max(
        ref_sweep,
        key=lambda row: (row["reproj_iou_2d"], -abs(row["fov_h_deg"] - fov_h_deg)),
    )

    # Exploited weaknesses narrative list
    weaknesses: list[dict[str, Any]] = []

    domain = image_profile["domain_classification"]
    if domain in ("high_contrast_stylized", "low_light_high_stress"):
        weaknesses.append(
            {
                "id": "photometric_stress",
                "title": "Estrés Fotométrico y Pérdida de Pistas Métricas",
                "severity": "high",
                "metric_value": f"Sombras {image_profile['dark_shadow_ratio']*100:.0f}% · Bloom {bloom_ratio*100:.1f}%",
                "explanation": (
                    "La alta proporción de sombras profundas o resplandor especular suprime gradientes de suelo y líneas de fuga, "
                    "reduciendo la referencia de escala física para modelos monoculares."
                ),
            }
        )

    if ofi >= 0.25:
        weaknesses.append(
            {
                "id": "quadratic_range_uncertainty",
                "title": "Incertidumbre Cuadrática de Profundidad (±σ_y)",
                "severity": "high" if ofi >= 0.45 else "medium",
                "metric_value": f"±{ofi:.2f} m",
                "explanation": (
                    "En visión monocular el error de rango escala con y² y se amplifica por oclusión/truncamiento, "
                    "degradando mediciones métricas lejanas sin LiDAR."
                ),
            }
        )

    if sdas >= 0.45:
        weaknesses.append(
            {
                "id": "scale_depth_coupling",
                "title": "Acoplamiento Espurio Escala–Profundidad (SDAS)",
                "severity": "high" if sdas >= 0.65 else "medium",
                "metric_value": f"SDAS {sdas:.2f}",
                "explanation": (
                    "Las dimensiones inferidas muestran alta anisotropía o correlación con la profundidad estimada, "
                    "síntoma clásico de ambigüedad monocular tamaño-distancia."
                ),
            }
        )

    if pvi > 0.01 or colliding_pairs > 0:
        weaknesses.append(
            {
                "id": "solid_interpenetration",
                "title": "Interpenetración Volumétrica entre Sólidos (PVI)",
                "severity": "high" if pvi >= 0.08 else "medium",
                "metric_value": f"{pvi*100:.1f}% ({colliding_pairs} pares)",
                "explanation": (
                    "Dos o más cuboides ocupan el mismo volumen 3D según pruebas SAT; el modelo carece de restricción "
                    "física explícita de no-colisión."
                ),
            }
        )

    if gfi >= 0.22:
        weaknesses.append(
            {
                "id": "gravity_floating",
                "title": "Anomalía de Soporte Gravitatorio (GFI)",
                "severity": "high" if gfi >= 0.50 else "medium",
                "metric_value": f"{gfi:.2f} m (plano Z={support_plane_z:.2f}m)",
                "explanation": (
                    "Varios volúmenes presentan desfase vertical o inclinación respecto al plano de soporte latente, "
                    "indicando levitación o hundimiento bajo el suelo estimado."
                ),
            }
        )

    if not weaknesses:
        weaknesses.append(
            {
                "id": "nominal_coherence",
                "title": "Coherencia Geométrica y Fotométrica Estable",
                "severity": "low",
                "metric_value": f"Plausibilidad {plausibility:.1f}/100",
                "explanation": (
                    "Las cajas proyectadas alinean con gradientes Sobel reales, respetan el plano de soporte latente "
                    "y no presentan interpenetraciones severas."
                ),
            }
        )

    cascade_policy = _compute_cascade_policy(
        objects=objects,
        image_profile=image_profile,
        occlusion_matrix=occlusion_matrix,
        colliding_pairs=colliding_pairs,
        plausibility=plausibility,
        pvi=pvi,
        gfi=gfi,
        sdas=sdas,
        fbr=fbr,
        ofi=ofi,
    )

    return {
        "image_profile": image_profile,
        "latent_metrics": latent_metrics,
        "per_object_diagnostics": per_object_diagnostics,
        "exploited_weaknesses": weaknesses[:5],
        "reference_free_fov_sweep": ref_sweep,
        "estimated_optimal_fov_h_deg": int(best_fov_pt["fov_h_deg"]),
        "cascade_policy": cascade_policy,
    }


def extract_image_adaptive_3d_hypotheses(
    image_base64: str,
    width: int,
    height: int,
    fov_h_deg: float,
    max_objects: int = 6,
) -> list[dict[str, Any]]:
    """Extract image-grounded 9-DOF 3D box hypotheses from real pixel salience."""
    rgb = _decode_rgb_array(image_base64, max_width=240)
    if rgb is None:
        return [
            {
                "id": "salient_1",
                "label": "central_focal_region",
                "box_3d": [0.0, 2.5, -0.05, 0.70, 0.55, 0.85, 0.0, 0.0, 0.0],
                "confidence": 0.72,
            }
        ]

    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    _, _, mag = _sobel_gradients(luma)
    p95 = float(np.percentile(mag, 95))
    edge_map = np.clip(mag / max(p95, 1e-4), 0.0, 1.0)

    # Border median background reference for foreground-vs-background contrast
    border = np.concatenate(
        [
            rgb[0, :, :],
            rgb[-1, :, :],
            rgb[:, 0, :],
            rgb[:, -1, :],
        ],
        axis=0,
    )
    bg_color = np.median(border, axis=0)
    color_contrast = np.linalg.norm(rgb - bg_color[None, None, :], axis=2) / 1.732
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    salience = np.clip(0.45 * edge_map + 0.35 * color_contrast + 0.20 * chroma, 0.0, 1.0)

    # Downsample salience onto 24 x 16 coarse grid
    gw, gh = 24, 16
    sh, sw = salience.shape
    grid = np.zeros((gh, gw), dtype=np.float32)
    for gy in range(gh):
        y0 = int(gy * sh / gh)
        y1 = max(y0 + 1, int((gy + 1) * sh / gh))
        for gx in range(gw):
            x0 = int(gx * sw / gw)
            x1 = max(x0 + 1, int((gx + 1) * sw / gw))
            grid[gy, gx] = float(np.mean(salience[y0:y1, x0:x1]))

    thresh = max(float(np.mean(grid) + 0.55 * np.std(grid)), float(np.percentile(grid, 66)))
    active = grid >= thresh

    visited = np.zeros_like(active, dtype=bool)
    raw_regions: list[tuple[float, int, int, int, int]] = []

    for gy in range(gh):
        for gx in range(gw):
            if not active[gy, gx] or visited[gy, gx]:
                continue
            stack = [(gy, gx)]
            visited[gy, gx] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < gh and 0 <= nx < gw and active[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            y0, y1 = min(ys), max(ys) + 1
            x0, x1 = min(xs), max(xs) + 1
            area_frac = ((y1 - y0) * (x1 - x0)) / float(gh * gw)
            if area_frac < 0.025 and len(cells) < 2:
                continue
            score = float(np.mean([grid[c[0], c[1]] for c in cells])) * (0.65 + min(area_frac, 0.35))
            raw_regions.append((score, y0, x0, y1, x1))

    raw_regions.sort(key=lambda r: r[0], reverse=True)

    # Greedy NMS on coarse grid
    kept: list[tuple[float, int, int, int, int]] = []
    for cand in raw_regions:
        _, ay0, ax0, ay1, ax1 = cand
        overlap = False
        for _, by0, bx0, by1, bx1 in kept:
            iy = max(0, min(ay1, by1) - max(ay0, by0))
            ix = max(0, min(ax1, bx1) - max(ax0, bx0))
            inter = iy * ix
            union = (ay1 - ay0) * (ax1 - ax0) + (by1 - by0) * (bx1 - bx0) - inter
            if union > 0 and (inter / float(union)) > 0.35:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
        if len(kept) >= max_objects:
            break

    if not kept:
        kept = [(0.70, 5, 8, 12, 16)]

    f = float(width) / (2.0 * math.tan(math.radians(fov_h_deg) * 0.5))
    hypotheses: list[dict[str, Any]] = []

    for idx, (score, gy0, gx0, gy1, gx1) in enumerate(kept):
        u0 = (gx0 / float(gw)) * width
        u1 = (gx1 / float(gw)) * width
        v0 = (gy0 / float(gh)) * height
        v1 = (gy1 / float(gh)) * height

        u_c = 0.5 * (u0 + u1)
        v_c = 0.5 * (v0 + v1)
        w_px = max(24.0, u1 - u0)
        h_px = max(24.0, v1 - v0)

        h_frac = h_px / float(max(height, 1))
        bottom_norm = v1 / float(max(height, 1))
        # Plausible monocular depth prior: taller & lower-grounded regions sit closer
        y_depth = float(np.clip(1.65 + 2.4 * (1.0 - h_frac) + 0.9 * (1.0 - bottom_norm), 1.5, 5.5))

        x_cam = ((u_c - width * 0.5) * y_depth) / f
        z_cam = (-(v_c - height * 0.5) * y_depth) / f
        dx = float(np.clip((w_px * y_depth) / f, 0.25, 3.2))
        dz = float(np.clip((h_px * y_depth) / f, 0.25, 3.2))
        dy = float(np.clip(0.55 * min(dx, dz) + 0.15, 0.22, 1.5))

        yaw_deg = float(np.clip(-((u_c - width * 0.5) / max(width * 0.5, 1.0)) * 18.0, -20.0, 20.0))

        # Descriptive semantic role tag
        center_dist_norm = abs(u_c / float(max(width, 1)) - 0.5)
        if idx == 0 and center_dist_norm < 0.18:
            label = "sujeto_focal_central"
        elif bottom_norm > 0.78 and y_depth < 2.8:
            label = "estructura_primer_plano"
        elif (v_c / float(max(height, 1))) < 0.35:
            label = "elemento_superior_contraste"
        elif u_c < width * 0.45:
            label = f"volumen_lateral_izq_{idx + 1}"
        else:
            label = f"volumen_lateral_der_{idx + 1}"

        conf = float(np.clip(0.58 + 0.35 * min(score, 1.0), 0.58, 0.92))
        hypotheses.append(
            {
                "id": f"salient_{idx + 1}",
                "label": label,
                "box_3d": [
                    round(x_cam, 3),
                    round(y_depth, 3),
                    round(z_cam, 3),
                    round(dx, 3),
                    round(dy, 3),
                    round(dz, 3),
                    0.0,
                    0.0,
                    round(yaw_deg, 1),
                ],
                "confidence": round(conf, 3),
            }
        )

    return hypotheses
