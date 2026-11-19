"""Multi-frame World-Model geometric consistency audit for MyLeenTo3.

Evaluates temporal stability of monocular 3D bounding box predictions across
sequential frames (inspired by auditing implicit video World Models such as
Google Genie / Veo against explicit 9-DOF metric constraints).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np

from .geometry_3d import gemini_to_threejs_pose, project_obb_to_2d
from .synthetic_benchmark import (
    DEFAULT_FOV_H_DEG,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    render_scene_png_base64,
)


def _wrap_angle_180(delta_deg: float) -> float:
    return ((float(delta_deg) + 180.0) % 360.0) - 180.0


def _box_volume(box_3d: list[float]) -> float:
    return float(abs(float(box_3d[3])) * abs(float(box_3d[4])) * abs(float(box_3d[5])))


def audit_multiframe_consistency(
    frames: list[dict[str, Any]],
    sequence_id: str = "world_model_corridor_audit",
) -> dict[str, Any]:
    """Audit temporal scale drift, volume variance, and yaw jitter across frames."""
    if not frames:
        return {
            "sequence_id": sequence_id,
            "frame_count": 0,
            "geometric_stability_score": 0.0,
            "mean_scale_drift_pct": 0.0,
            "mean_yaw_jitter_deg": 0.0,
            "object_tracks": [],
            "per_frame_summary": [],
            "drift_diagnosis": "Sin frames disponibles para auditoría temporal.",
        }

    ordered_frames = sorted(
        enumerate(frames),
        key=lambda pair: int(pair[1].get("frame_idx", pair[0])),
    )

    tracks_raw: dict[str, list[tuple[int, list[float]]]] = {}
    per_frame_summary: list[dict[str, Any]] = []

    for fallback_idx, frame in ordered_frames:
        f_idx = int(frame.get("frame_idx", fallback_idx))
        ts = float(frame.get("timestamp_s", fallback_idx * 0.5))
        objs = frame.get("objects", [])

        vols: list[float] = []
        depths: list[float] = []
        seen_labels_in_frame: set[str] = set()

        for obj in objs:
            label = str(obj.get("label", "object"))
            box = [float(v) for v in obj["box_3d"]]
            vol = _box_volume(box)
            vols.append(vol)
            depths.append(float(box[1]))

            # Avoid duplicate entries of same label in same frame
            track_key = label
            if track_key in seen_labels_in_frame:
                suffix = 2
                while f"{label}#{suffix}" in seen_labels_in_frame:
                    suffix += 1
                track_key = f"{label}#{suffix}"
            seen_labels_in_frame.add(track_key)
            tracks_raw.setdefault(track_key, []).append((f_idx, box))

        per_frame_summary.append(
            {
                "frame_idx": f_idx,
                "timestamp_s": round(ts, 3),
                "object_count": len(objs),
                "total_volume_m3": round(float(sum(vols)), 5),
                "mean_depth_m": round(float(np.mean(depths)), 4) if depths else 0.0,
            }
        )

    object_tracks: list[dict[str, Any]] = []
    drift_values: list[float] = []
    cv_values: list[float] = []
    jitter_values: list[float] = []

    for label, observations in sorted(tracks_raw.items()):
        observations.sort(key=lambda item: item[0])
        frame_indices = [obs[0] for obs in observations]
        volumes = [_box_volume(obs[1]) for obs in observations]
        yaws = [obs[1][8] for obs in observations]
        centroids = [np.array(obs[1][:3], dtype=float) for obs in observations]

        v0 = volumes[0] if volumes else 0.0
        if len(volumes) > 1 and v0 > 1e-9:
            scale_drift_pct = float(max(abs(v - v0) / v0 for v in volumes[1:]) * 100.0)
        else:
            scale_drift_pct = 0.0

        mean_v = float(np.mean(volumes)) if volumes else 0.0
        std_v = float(np.std(volumes)) if len(volumes) > 1 else 0.0
        volume_cv_pct = float((std_v / mean_v) * 100.0) if mean_v > 1e-9 else 0.0

        if len(yaws) > 1:
            yaw_deltas = [abs(_wrap_angle_180(yaws[k] - yaws[k - 1])) for k in range(1, len(yaws))]
            yaw_jitter_deg = float(np.mean(yaw_deltas))
        else:
            yaw_jitter_deg = 0.0

        if len(centroids) > 1:
            steps = [float(np.linalg.norm(centroids[k] - centroids[k - 1])) for k in range(1, len(centroids))]
            centroid_step_m = float(np.mean(steps))
        else:
            centroid_step_m = 0.0

        if len(observations) > 1:
            drift_values.append(scale_drift_pct)
            cv_values.append(volume_cv_pct)
            jitter_values.append(yaw_jitter_deg)

        object_tracks.append(
            {
                "label": label,
                "frame_indices": frame_indices,
                "volumes_m3": [round(v, 5) for v in volumes],
                "scale_drift_pct": round(scale_drift_pct, 3),
                "volume_cv_pct": round(volume_cv_pct, 3),
                "yaw_jitter_deg": round(yaw_jitter_deg, 3),
                "centroid_step_m": round(centroid_step_m, 4),
            }
        )

    mean_scale_drift_pct = float(np.mean(drift_values)) if drift_values else 0.0
    mean_cv_pct = float(np.mean(cv_values)) if cv_values else 0.0
    mean_yaw_jitter_deg = float(np.mean(jitter_values)) if jitter_values else 0.0

    raw_score = (
        100.0
        - 2.2 * mean_scale_drift_pct
        - 2.0 * mean_yaw_jitter_deg
        - 1.5 * mean_cv_pct
    )
    stability_score = float(np.clip(raw_score, 0.0, 100.0))

    worst_track = max(object_tracks, key=lambda t: t["scale_drift_pct"]) if object_tracks else None
    if stability_score >= 82.0:
        band = "Estable (Consistencia Métrica Alta)"
    elif stability_score >= 55.0:
        band = "Deriva Moderada (Típica de World-Model Implícito)"
    else:
        band = "Inestabilidad Geométrica Severa"

    if worst_track is not None:
        diagnosis = (
            f"{band}: deriva volumétrica media {mean_scale_drift_pct:.1f}% y jitter de orientación "
            f"{mean_yaw_jitter_deg:.1f}°/frame. Mayor deriva observada en '{worst_track['label']}' "
            f"({worst_track['scale_drift_pct']:.1f}% de cambio volumétrico, CV {worst_track['volume_cv_pct']:.1f}%)."
        )
    else:
        diagnosis = f"{band}: sin trayectorias multi-frame detectadas."

    return {
        "sequence_id": sequence_id,
        "frame_count": len(ordered_frames),
        "geometric_stability_score": round(stability_score, 2),
        "mean_scale_drift_pct": round(mean_scale_drift_pct, 3),
        "mean_yaw_jitter_deg": round(mean_yaw_jitter_deg, 3),
        "object_tracks": object_tracks,
        "per_frame_summary": per_frame_summary,
        "drift_diagnosis": diagnosis,
    }


@functools.lru_cache(maxsize=1)
def get_demo_multiframe_sequence() -> dict[str, Any]:
    """Return a deterministic 4-frame simulated World-Model camera pan sequence."""
    width, height, fov_h = DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FOV_H_DEG
    floor_z = -0.52

    # Physical base scene objects at t=0.0s in corridor
    base_objects = [
        ("obj_1", "equipment_rack", [-0.54, 2.75, floor_z + 0.38, 0.48, 0.36, 0.76, 0.0, 0.0, 12.0], (68, 96, 128)),
        ("obj_2", "calibration_table", [0.16, 2.30, floor_z + 0.21, 0.52, 0.36, 0.42, 0.0, 0.0, -8.0], (168, 108, 82)),
        ("obj_3", "laser_bench", [0.80, 3.45, floor_z + 0.32, 0.60, 0.40, 0.64, 0.0, 0.0, -18.0], (96, 128, 108)),
        ("obj_4", "transport_crate", [-0.18, 4.30, floor_z + 0.30, 0.64, 0.50, 0.60, 0.0, 0.0, 6.0], (184, 152, 98)),
    ]

    # Smooth camera lateral pan/dolly increments per frame (x shift, y advance)
    camera_motions = [
        (0.00, 0.00, "t=0.0s · Referencia calibrada inicial (traslación 0.00m)"),
        (0.10, 0.14, "t=0.5s · Paneo lateral +0.10m / avance +0.14m"),
        (0.20, 0.28, "t=1.0s · Paneo lateral +0.20m / avance +0.28m"),
        (0.30, 0.42, "t=1.5s · Paneo lateral +0.30m / avance +0.42m"),
    ]

    # Realistic World-Model scale drift factors (per object across 4 frames) & yaw jitter (deg)
    scale_drifts = {
        "equipment_rack": [1.00, 1.025, 1.055, 1.085],
        "calibration_table": [1.00, 0.985, 1.020, 1.045],
        "laser_bench": [1.00, 1.040, 1.095, 1.135],
        "transport_crate": [1.00, 1.050, 1.115, 1.170],
    }
    yaw_jitters = {
        "equipment_rack": [0.0, 2.4, -1.8, 3.6],
        "calibration_table": [0.0, -1.6, 2.1, -2.5],
        "laser_bench": [0.0, 4.2, -3.5, 5.1],
        "transport_crate": [0.0, -4.8, 5.6, -6.2],
    }

    frames: list[dict[str, Any]] = []
    for f_idx, (cam_dx, cam_dy, note) in enumerate(camera_motions):
        true_render_objs: list[dict[str, Any]] = []
        detected_objs: list[dict[str, Any]] = []

        for oid, label, base_box, color_rgb in base_objects:
            bx, by, bz, dx, dy, dz, r, p, yaw = base_box
            # Camera moves by (+cam_dx, +cam_dy), so objects shift by (-cam_dx, -cam_dy) in camera frame
            true_box = [bx - cam_dx, by - cam_dy, bz, dx, dy, dz, r, p, yaw]
            true_render_objs.append({"id": oid, "label": label, "box_3d": true_box, "color_rgb": list(color_rgb)})

            # Apply isotropic/anisotropic scale drift (cube root on dims so volume shifts by factor) + yaw jitter
            vol_mult = scale_drifts[label][f_idx]
            dim_mult = vol_mult ** (1.0 / 3.0)
            yj = yaw_jitters[label][f_idx]
            pred_box = [
                round(bx - cam_dx + 0.012 * f_idx * (1 if f_idx % 2 == 0 else -1), 4),
                round(by - cam_dy + 0.025 * f_idx, 4),
                round(bz, 4),
                round(dx * dim_mult, 4),
                round(dy * dim_mult, 4),
                round(dz * dim_mult, 4),
                round(r, 4),
                round(p, 4),
                round(yaw + yj, 4),
            ]
            proj = project_obb_to_2d(pred_box, width, height, fov_h)
            pose = gemini_to_threejs_pose(pred_box)
            detected_objs.append(
                {
                    "id": f"f{f_idx}_{oid}",
                    "label": label,
                    "box_3d": pred_box,
                    "projected_2d": proj,
                    "threejs_pose": pose,
                    "volume_m3": round(_box_volume(pred_box), 5),
                    "distance_m": round(math.sqrt(pred_box[0] ** 2 + pred_box[1] ** 2 + pred_box[2] ** 2), 4),
                }
            )

        img_b64 = render_scene_png_base64(
            true_render_objs,
            width=width,
            height=height,
            fov_h_deg=fov_h,
            floor_z=floor_z,
            max_grid_y=6.5,
            scene_seed=10 + f_idx,
        )
        frames.append(
            {
                "frame_idx": f_idx,
                "timestamp_s": round(f_idx * 0.5, 2),
                "camera_note": note,
                "image_base64": img_b64,
                "objects": detected_objs,
            }
        )

    audit = audit_multiframe_consistency(frames, sequence_id="world_model_corridor_audit")
    return {
        "sequence_id": "world_model_corridor_audit",
        "title": "Auditoría Multi-Frame: World-Model vs. Geometría Explícita",
        "description": (
            "Secuencia de 4 frames simulando un paneo de cámara en corredor para auditar "
            "deriva volumétrica y jitter angular (inspirado en evaluación de World Models tipo Google Genie/Veo)."
        ),
        "width": width,
        "height": height,
        "fov_h_deg": fov_h,
        "frames": frames,
        "audit": audit,
    }
