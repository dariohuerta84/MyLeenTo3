"""Deterministic 3D Spatial Copilot powered by Google GenAI Interactions API Function Calling."""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from typing import Any, Optional

from google import genai

from app.core.geometry_3d import (
    compute_occlusion_matrix,
    compute_pairwise_geometry,
    gemini_to_threejs_pose,
    project_obb_to_2d,
    compute_box_2d_iou,
)
from app.core.intrinsic_diagnostics import compute_intrinsic_scene_diagnostics
from app.core.metrics import fov_sensitivity_sweep
from app.services.gemini_vision import (
    extract_output_text,
    extract_usage,
    resolve_thinking_level,
)

COPILOT_SYSTEM_INSTRUCTION = (
    "Eres el Copiloto Espacial 3D de MyLeenTo3, un asistente científico de visión computacional 3D. "
    "NUNCA calcules distancias, volúmenes, colisiones, oclusiones ni métricas mentalmente: "
    "SIEMPRE invoca las herramientas deterministas provistas (`focus_object`, `measure_distance`, "
    "`set_camera_view`, `analyze_occlusions_and_collisions`, `filter_objects`, "
    "`run_fov_sensitivity_sweep`, `evaluate_escalation_policy`) para obtener resultados exactos y controlar el visor 3D. "
    "Responde en español técnico claro, conciso y cuantitativo usando exclusivamente los valores "
    "retornados por las herramientas (metros, grados, porcentajes y veredictos)."
)

COPILOT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "focus_object",
        "description": "Focuses the 3D camera viewport on a specific detected object and returns its metric 9-DOF pose.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "Object ID (e.g. 'pred_1') or exact label to focus in the 3D scene.",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "type": "function",
        "name": "measure_distance",
        "description": "Computes exact Euclidean center distance, OBB surface clearance, SAT collision status, 3D IoU, and Three.js ruler endpoints between two objects.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_a_id": {
                    "type": "string",
                    "description": "ID or label of the first object.",
                },
                "object_b_id": {
                    "type": "string",
                    "description": "ID or label of the second object.",
                },
            },
            "required": ["object_a_id", "object_b_id"],
        },
    },
    {
        "type": "function",
        "name": "set_camera_view",
        "description": "Switches the interactive 3D viewport camera preset.",
        "parameters": {
            "type": "object",
            "properties": {
                "view_mode": {
                    "type": "string",
                    "enum": ["camera_aligned", "orbit", "bev", "isometric"],
                    "description": "Target camera preset: camera_aligned, orbit, bev (Bird's-Eye View), or isometric.",
                }
            },
            "required": ["view_mode"],
        },
    },
    {
        "type": "function",
        "name": "analyze_occlusions_and_collisions",
        "description": "Runs exact ray-OBB slab occlusion analysis from the camera origin and pairwise SAT collision checks across all scene objects, enabling 3D occlusion rays.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "filter_objects",
        "description": "Filters scene objects by forward depth range (meters) and/or label substring, highlighting the matching subset in the 3D viewport.",
        "parameters": {
            "type": "object",
            "properties": {
                "min_depth_m": {
                    "type": "number",
                    "description": "Minimum forward depth Y in meters (inclusive).",
                },
                "max_depth_m": {
                    "type": "number",
                    "description": "Maximum forward depth Y in meters (inclusive).",
                },
                "label_contains": {
                    "type": "string",
                    "description": "Optional case-insensitive substring to match against object labels.",
                },
            },
        },
    },
    {
        "type": "function",
        "name": "run_fov_sensitivity_sweep",
        "description": "Runs a horizontal Field-of-View sensitivity sweep (40°..110°) to evaluate 2D pinhole reprojection IoU and pixel error, opening the metrics panel.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "type": "function",
        "name": "evaluate_escalation_policy",
        "description": "Evaluates the reference-free Edge -> VLM 3D Physical Reliability Gate (C8286 degradation regime, operational verdict APROBADO/ESCALAR_MODELO/RECHAZADO_FISICA, recommended model tier, estimated latency savings, and minimum pairwise clearance).",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


def _normalize_text(text: str) -> str:
    """Lowercase and strip accents for robust Spanish/English matching."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


class SpatialToolbox:
    """Executes deterministic 3D geometry tools against the current scene state."""

    def __init__(
        self,
        *,
        objects: list[dict[str, Any]],
        gt_objects: Optional[list[dict[str, Any]]],
        image_base64: Optional[str] = None,
        width: int,
        height: int,
        fov_h_deg: float,
    ) -> None:
        self.objects = [{k: v for k, v in obj.items() if v is not None} for obj in objects]
        self.gt_objects = (
            [{k: v for k, v in obj.items() if v is not None} for obj in gt_objects]
            if gt_objects is not None
            else None
        )
        self.image_base64 = image_base64
        self.width = width
        self.height = height
        self.fov_h_deg = fov_h_deg
        self.tool_trace: list[dict[str, Any]] = []
        self.scene_actions: list[dict[str, Any]] = []

    def _find_object(self, query: str) -> Optional[dict[str, Any]]:
        if not self.objects:
            return None
        q_norm = _normalize_text(str(query).strip())
        for obj in self.objects:
            if _normalize_text(obj["id"]) == q_norm:
                return obj
        for obj in self.objects:
            if _normalize_text(obj["label"]) == q_norm:
                return obj
        for obj in self.objects:
            if q_norm and (q_norm in _normalize_text(obj["label"]) or q_norm in _normalize_text(obj["id"])):
                return obj
        return None

    def execute(self, name: str, arguments: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        args = dict(arguments or {})
        handler_map = {
            "focus_object": self.focus_object,
            "measure_distance": self.measure_distance,
            "set_camera_view": self.set_camera_view,
            "analyze_occlusions_and_collisions": self.analyze_occlusions_and_collisions,
            "filter_objects": self.filter_objects,
            "run_fov_sensitivity_sweep": self.run_fov_sensitivity_sweep,
            "evaluate_escalation_policy": self.evaluate_escalation_policy,
        }
        handler = handler_map.get(name)
        if handler is None:
            result = {"error": f"Unknown tool '{name}'"}
        else:
            try:
                result = handler(**args)
            except TypeError as exc:
                result = {"error": f"Invalid arguments for '{name}': {exc}"}
            except Exception as exc:
                result = {"error": f"Tool execution error in '{name}': {exc}"}

        self.tool_trace.append({"name": name, "arguments": args, "result": result})
        return result

    def focus_object(self, object_id: str) -> dict[str, Any]:
        obj = self._find_object(object_id)
        if obj is None:
            return {
                "found": False,
                "requested_id": object_id,
                "available_ids": [o["id"] for o in self.objects],
            }
        b = obj["box_3d"]
        pose = gemini_to_threejs_pose(b)
        distance_m = round(math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2), 4)
        volume_m3 = round(b[3] * b[4] * b[5], 4)
        self.scene_actions.append({"type": "focus_object", "object_id": obj["id"]})
        return {
            "found": True,
            "object_id": obj["id"],
            "label": obj["label"],
            "depth_y_m": round(float(b[1]), 4),
            "distance_m": distance_m,
            "volume_m3": volume_m3,
            "dimensions_m": [round(float(b[3]), 3), round(float(b[4]), 3), round(float(b[5]), 3)],
            "yaw_deg": round(float(b[8]), 2),
            "position_three": [round(float(v), 4) for v in pose["position"]],
        }

    def measure_distance(self, object_a_id: str, object_b_id: str) -> dict[str, Any]:
        obj_a = self._find_object(object_a_id)
        obj_b = self._find_object(object_b_id)
        if obj_a is None or obj_b is None:
            return {
                "found": False,
                "error": "One or both objects were not found in the active scene.",
                "available_objects": [{"id": o["id"], "label": o["label"]} for o in self.objects],
            }
        pair = compute_pairwise_geometry(obj_a["box_3d"], obj_b["box_3d"])
        pose_a = gemini_to_threejs_pose(obj_a["box_3d"])
        pose_b = gemini_to_threejs_pose(obj_b["box_3d"])
        data = {
            "object_a_id": obj_a["id"],
            "object_a_label": obj_a["label"],
            "object_b_id": obj_b["id"],
            "object_b_label": obj_b["label"],
            **pair,
            "ruler_endpoints_three": {
                "center_a": pose_a["position"],
                "center_b": pose_b["position"],
                "surface_a": pair["closest_point_a_three"],
                "surface_b": pair["closest_point_b_three"],
            },
        }
        self.scene_actions.append(
            {
                "type": "show_measurement",
                "object_a_id": obj_a["id"],
                "object_b_id": obj_b["id"],
                "data": data,
            }
        )
        return {"found": True, **data}

    def set_camera_view(self, view_mode: str) -> dict[str, Any]:
        aliases = {
            "camera_aligned": "camera_aligned",
            "camera": "camera_aligned",
            "camara": "camera_aligned",
            "orbit": "orbit",
            "orbita": "orbit",
            "bev": "bev",
            "top": "bev",
            "top_down": "bev",
            "planta": "bev",
            "cenital": "bev",
            "isometric": "isometric",
            "isometrica": "isometric",
            "iso": "isometric",
        }
        normalized = aliases.get(_normalize_text(view_mode), "orbit")
        self.scene_actions.append({"type": "set_camera_view", "view_mode": normalized})
        return {"view_mode": normalized, "applied": True}

    def analyze_occlusions_and_collisions(self) -> dict[str, Any]:
        occlusions = compute_occlusion_matrix(self.objects)
        collisions: list[dict[str, Any]] = []
        near_contacts: list[dict[str, Any]] = []
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                a = self.objects[i]
                b = self.objects[j]
                pair = compute_pairwise_geometry(a["box_3d"], b["box_3d"])
                if pair["is_colliding"]:
                    collisions.append(
                        {
                            "object_a_id": a["id"],
                            "object_a_label": a["label"],
                            "object_b_id": b["id"],
                            "object_b_label": b["label"],
                            "iou_3d": pair["iou_3d"],
                            "center_distance_m": pair["center_distance_m"],
                        }
                    )
                elif pair["surface_clearance_m"] <= 0.15:
                    near_contacts.append(
                        {
                            "object_a_id": a["id"],
                            "object_a_label": a["label"],
                            "object_b_id": b["id"],
                            "object_b_label": b["label"],
                            "surface_clearance_m": pair["surface_clearance_m"],
                        }
                    )

        self.scene_actions.append({"type": "toggle_occlusion_rays", "enabled": True})
        compact_occlusions = [
            {
                "occluder_id": item["occluder_id"],
                "occluder_label": item["occluder_label"],
                "occluded_id": item["occluded_id"],
                "occluded_label": item["occluded_label"],
                "occlusion_ratio": item["occlusion_ratio"],
                "depth_gap_m": item["depth_gap_m"],
            }
            for item in occlusions
        ]
        return {
            "occlusion_count": len(compact_occlusions),
            "occlusions": compact_occlusions,
            "collision_count": len(collisions),
            "collisions": collisions,
            "near_contact_pairs": near_contacts,
        }

    def filter_objects(
        self,
        min_depth_m: Optional[float] = None,
        max_depth_m: Optional[float] = None,
        label_contains: Optional[str] = None,
    ) -> dict[str, Any]:
        needle = _normalize_text(label_contains) if label_contains else None
        matched: list[dict[str, Any]] = []
        for obj in self.objects:
            depth = float(obj["box_3d"][1])
            if min_depth_m is not None and depth < float(min_depth_m):
                continue
            if max_depth_m is not None and depth > float(max_depth_m):
                continue
            if needle and needle not in _normalize_text(obj["label"]):
                continue
            matched.append(
                {
                    "id": obj["id"],
                    "label": obj["label"],
                    "depth_y_m": round(depth, 3),
                }
            )
        object_ids = [m["id"] for m in matched]
        self.scene_actions.append({"type": "highlight_subset", "object_ids": object_ids})
        return {
            "min_depth_m": min_depth_m,
            "max_depth_m": max_depth_m,
            "label_contains": label_contains,
            "matched_count": len(matched),
            "object_ids": object_ids,
            "objects": matched,
        }

    def run_fov_sensitivity_sweep(self) -> dict[str, Any]:
        has_gt = bool(self.gt_objects)
        if has_gt and self.gt_objects is not None:
            sweep = fov_sensitivity_sweep(
                self.objects,
                self.gt_objects,
                self.width,
                self.height,
            )
        else:
            # Self-consistency sweep relative to active fov_h_deg when GT is absent
            fov_values = [40, 50, 60, 70, 80, 90, 100, 110]
            ref_projs = [
                project_obb_to_2d(obj["box_3d"], self.width, self.height, self.fov_h_deg)
                for obj in self.objects
            ]
            sweep = []
            for fov in fov_values:
                ious: list[float] = []
                px_errs: list[float] = []
                for obj, ref in zip(self.objects, ref_projs):
                    cur = project_obb_to_2d(obj["box_3d"], self.width, self.height, float(fov))
                    ious.append(compute_box_2d_iou(cur["box_2d_reproj"], ref["box_2d_reproj"]))
                    du = cur["center_2d"][0] - ref["center_2d"][0]
                    dv = cur["center_2d"][1] - ref["center_2d"][1]
                    px_errs.append(math.hypot(du, dv))
                sweep.append(
                    {
                        "fov_h_deg": fov,
                        "reproj_iou_2d": round(sum(ious) / len(ious), 4) if ious else 0.0,
                        "mean_pixel_error": round(sum(px_errs) / len(px_errs), 2) if px_errs else 0.0,
                    }
                )

        best = max(sweep, key=lambda item: item["reproj_iou_2d"]) if sweep else None
        self.scene_actions.append({"type": "open_tab", "tab": "metrics"})
        return {
            "has_ground_truth": has_gt,
            "current_fov_h_deg": self.fov_h_deg,
            "optimal_fov_h_deg": best["fov_h_deg"] if best else self.fov_h_deg,
            "best_reproj_iou_2d": best["reproj_iou_2d"] if best else 0.0,
            "sweep": sweep,
        }

    def evaluate_escalation_policy(self) -> dict[str, Any]:
        diag = compute_intrinsic_scene_diagnostics(
            self.objects,
            self.image_base64,
            self.width,
            self.height,
            self.fov_h_deg,
        )
        policy = dict(diag.get("cascade_policy") or {})
        latent = dict(diag.get("latent_metrics") or {})
        if policy.get("occlusion_pairs_count", 0) > 0:
            self.scene_actions.append({"type": "toggle_occlusion_rays", "enabled": True})
        return {
            **policy,
            "physical_plausibility_score": latent.get("physical_plausibility_score", 0.0),
            "volume_interpenetration_index": latent.get("volume_interpenetration_index", 0.0),
            "gravity_floating_anomaly_m": latent.get("gravity_floating_anomaly_m", 0.0),
            "mean_depth_uncertainty_m": latent.get("mean_depth_uncertainty_m", 0.0),
        }


def _scene_context_block(
    objects: list[dict[str, Any]],
    gt_objects: Optional[list[dict[str, Any]]],
    width: int,
    height: int,
    fov_h_deg: float,
) -> str:
    compact = [
        {
            "id": o["id"],
            "label": o["label"],
            "box_3d": [round(float(v), 3) for v in o["box_3d"]],
        }
        for o in objects
    ]
    return (
        f"Contexto de escena 3D activa ({width}x{height}px, FOV_h={fov_h_deg:.1f}°, "
        f"ground_truth_disponible={bool(gt_objects)}):\n"
        f"{json.dumps(compact, ensure_ascii=False)}"
    )


def _extract_function_calls(interaction: Any) -> list[dict[str, Any]]:
    """Extract function_call steps from an Interaction object."""
    calls: list[dict[str, Any]] = []
    steps = getattr(interaction, "steps", None) or []
    for step in steps:
        step_type = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
        if step_type == "function_call":
            call_id = getattr(step, "id", None) or (step.get("id") if isinstance(step, dict) else "")
            name = getattr(step, "name", None) or (step.get("name") if isinstance(step, dict) else "")
            arguments = getattr(step, "arguments", None)
            if arguments is None and isinstance(step, dict):
                arguments = step.get("arguments", {})
            calls.append(
                {
                    "id": str(call_id),
                    "name": str(name),
                    "arguments": dict(arguments or {}),
                }
            )
    return calls


def _merge_usage(acc: dict[str, int], new: dict[str, Any]) -> dict[str, int]:
    return {
        "input_tokens": acc["input_tokens"] + int(new.get("input_tokens", 0)),
        "output_tokens": acc["output_tokens"] + int(new.get("output_tokens", 0)),
        "thought_tokens": acc["thought_tokens"] + int(new.get("thought_tokens", 0)),
        "total_tokens": acc["total_tokens"] + int(new.get("total_tokens", 0)),
    }


def _run_offline_router(message: str, toolbox: SpatialToolbox) -> str:
    """Deterministic Spanish/English command router invoking the exact same 7 Python tools."""
    norm = _normalize_text(message)
    objects = toolbox.objects
    parts: list[str] = []

    if not objects:
        return (
            "No hay objetos 3D detectados en la escena activa todavía. "
            "Analiza un benchmark calibrado o sube una imagen RGB primero."
        )

    # Sort objects by forward depth Y
    sorted_by_depth = sorted(objects, key=lambda o: float(o["box_3d"][1]))

    # 1. Camera view preset
    if any(k in norm for k in ["vista", "camara", "camera", "view", "bev", "planta", "cenital", "isometric", "orbita", "orbit"]):
        target_view = None
        if any(k in norm for k in ["bev", "planta", "cenital", "top", "arriba"]):
            target_view = "bev"
        elif any(k in norm for k in ["isometric", "iso"]):
            target_view = "isometric"
        elif any(k in norm for k in ["camara", "camera", "alinead", "aligned", "pinhole"]):
            target_view = "camera_aligned"
        elif any(k in norm for k in ["orbit", "orbita", "libre"]):
            target_view = "orbit"
        if target_view:
            res = toolbox.execute("set_camera_view", {"view_mode": target_view})
            parts.append(f"Cámara cambiada a preset **`{res['view_mode']}`**.")

    # 2. Occlusions & collisions
    if any(k in norm for k in ["oclusi", "colisi", "choque", "intersect", "rayo", "occlusion", "collision"]):
        res = toolbox.execute("analyze_occlusions_and_collisions", {})
        occ_n = res["occlusion_count"]
        col_n = res["collision_count"]
        summary = (
            f"Auditoría geométrica completada: **{occ_n} par(es) con oclusión** "
            f"y **{col_n} colisión(es) 3D (SAT)** (rayos de oclusión activados en el visor)."
        )
        if res["occlusions"]:
            top = res["occlusions"][0]
            summary += (
                f" Mayor oclusión: `{top['occluder_label']}` bloquea "
                f"**{top['occlusion_ratio'] * 100:.1f}%** de `{top['occluded_label']}` "
                f"(brecha de profundidad: {top['depth_gap_m']:.2f} m)."
            )
        parts.append(summary)

    # 3. Distance / clearance measurement
    if any(k in norm for k in ["mide", "medir", "distancia", "separacion", "clearance", "measure", "distance", "cercanos"]):
        if len(objects) >= 2:
            # Check if user named two specific objects
            mentioned = [
                obj
                for obj in objects
                if _normalize_text(obj["id"]) in norm or _normalize_text(obj["label"]) in norm
            ]
            if len(mentioned) >= 2:
                obj_a, obj_b = mentioned[0], mentioned[1]
            elif any(k in norm for k in ["lejanos", "farthest"]):
                obj_a, obj_b = sorted_by_depth[-2], sorted_by_depth[-1]
            elif any(k in norm for k in ["entre si", "mas proximos entre", "closest pair"]):
                best_pair = min(
                    ((objects[i], objects[j]) for i in range(len(objects)) for j in range(i + 1, len(objects))),
                    key=lambda p: compute_pairwise_geometry(p[0]["box_3d"], p[1]["box_3d"])["center_distance_m"],
                )
                obj_a, obj_b = best_pair
            else:
                obj_a, obj_b = sorted_by_depth[0], sorted_by_depth[1]

            res = toolbox.execute(
                "measure_distance",
                {"object_a_id": obj_a["id"], "object_b_id": obj_b["id"]},
            )
            if res.get("found"):
                status = "COLISIÓN DETECTADA" if res["is_colliding"] else "sin colisión"
                parts.append(
                    f"Medición exacta entre **`{res['object_a_label']}`** (`{res['object_a_id']}`) y "
                    f"**`{res['object_b_label']}`** (`{res['object_b_id']}`): "
                    f"distancia entre centros = **{res['center_distance_m']:.3f} m**, "
                    f"holgura entre superficies = **{res['surface_clearance_m']:.3f} m**, "
                    f"IoU 3D = **{res['iou_3d']:.3f}** ({status})."
                )
        else:
            parts.append("Se requieren al menos 2 objetos en la escena para medir distancias entre pares.")

    # 4. Focus object
    if any(k in norm for k in ["enfoca", "focus", "selecciona", "inspecciona", "acerca", "mas lejano", "mas cercano"]):
        target_obj = None
        for obj in objects:
            if _normalize_text(obj["id"]) in norm or _normalize_text(obj["label"]) in norm:
                target_obj = obj
                break
        if target_obj is None:
            if any(k in norm for k in ["lejano", "farthest", "fondo", "ultimo"]):
                target_obj = sorted_by_depth[-1]
            elif any(k in norm for k in ["cercano", "nearest", "proximo", "frente"]):
                target_obj = sorted_by_depth[0]
            elif any(k in norm for k in ["grande", "mayor", "largest"]):
                target_obj = max(objects, key=lambda o: o["box_3d"][3] * o["box_3d"][4] * o["box_3d"][5])
            else:
                target_obj = sorted_by_depth[0]

        res = toolbox.execute("focus_object", {"object_id": target_obj["id"]})
        if res.get("found"):
            parts.append(
                f"Objeto enfocado: **`{res['label']}`** (`{res['object_id']}`) a profundidad "
                f"Y = **{res['depth_y_m']:.2f} m** (distancia radial {res['distance_m']:.2f} m, "
                f"volumen {res['volume_m3']:.3f} m³, yaw {res['yaw_deg']:.1f}°)."
            )

    # 5. FOV sensitivity sweep
    if any(k in norm for k in ["fov", "barrido", "sensibilidad", "calibracion", "sweep"]):
        res = toolbox.execute("run_fov_sensitivity_sweep", {})
        parts.append(
            f"Barrido de sensibilidad FOV ejecutado (40°–110°): óptimo en "
            f"**FOV_h = {res['optimal_fov_h_deg']}°** con IoU de reproyección 2D = "
            f"**{res['best_reproj_iou_2d']:.3f}** (pestaña de Métricas abierta)."
        )

    # 6. Filter objects by depth / label
    if any(k in norm for k in ["filtra", "filter", "menos de", "mas de", "menor a", "mayor a", "entre ", "solo "]):
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", norm)]
        min_d: Optional[float] = None
        max_d: Optional[float] = None
        label_q: Optional[str] = None
        if "entre" in norm and len(nums) >= 2:
            min_d, max_d = min(nums[0], nums[1]), max(nums[0], nums[1])
        elif any(k in norm for k in ["menos de", "menor a", "menor que", "under", "below", "hasta"]) and nums:
            max_d = nums[0]
        elif any(k in norm for k in ["mas de", "mayor a", "mayor que", "over", "above", "desde"]) and nums:
            min_d = nums[0]
        for obj in objects:
            lbl = _normalize_text(obj["label"])
            if lbl in norm:
                label_q = obj["label"]
                break
        res = toolbox.execute(
            "filter_objects",
            {"min_depth_m": min_d, "max_depth_m": max_d, "label_contains": label_q},
        )
        matched_labels = ", ".join(f"`{o['label']}` ({o['depth_y_m']} m)" for o in res["objects"]) or "ninguno"
        parts.append(
            f"Filtro aplicado: **{res['matched_count']} objeto(s)** resaltados ({matched_labels})."
        )

    # 6b. Edge -> VLM 3D Escalation Policy & Physical Reliability Gate (herencia C8286)
    if any(
        k in norm
        for k in [
            "escalar",
            "cascada",
            "confiable",
            "veredicto",
            "politica",
            "costo",
            "tier",
            "c8286",
            "conviene",
            "aprobado",
            "alucinacion",
            "riesgo",
            "compuerta",
        ]
    ):
        res = toolbox.execute("evaluate_escalation_policy", {})
        clr_txt = (
            f"holgura mínima = **{res['min_surface_clearance_m']:.3f} m**"
            if res.get("min_surface_clearance_m") is not None
            else "sin pares suficientes para holgura"
        )
        reasons_txt = "; ".join(res.get("trigger_reasons") or [])
        savings = float(res.get("estimated_latency_savings_pct", 0.0))
        savings_txt = (
            f"ahorro estimado de latencia = **-{savings:.0f}%**"
            if savings > 0
            else "requiere inferencia profunda (`0%` recorte)"
        )
        parts.append(
            f"Veredicto de compuerta Edge → 3D: **`{res.get('operational_verdict')}`** "
            f"(régimen C8286: `{res.get('c8286_degradation_regime')}`, "
            f"modelo recomendado: **`{res.get('recommended_model')}`**, "
            f"{savings_txt}, plausibilidad física = **{res.get('physical_plausibility_score', 0.0):.1f}/100**, "
            f"{clr_txt}). Motivo: {reasons_txt}"
        )

    # Fallback overview if no specific command matched
    if not parts:
        nearest = sorted_by_depth[0]
        farthest = sorted_by_depth[-1]
        res = toolbox.execute("focus_object", {"object_id": nearest["id"]})
        parts.append(
            f"Escena activa con **{len(objects)} objetos** en rango de profundidad "
            f"**{nearest['box_3d'][1]:.2f} m – {farthest['box_3d'][1]:.2f} m**. "
            f"Enfoqué el más cercano (`{res['label']}`). "
            "Puedes pedirme: medir distancias, evaluar la compuerta de escalamiento Edge → 3D, "
            "analizar oclusiones y colisiones, cambiar la vista (`bev`, `orbit`, `isometric`, `camera_aligned`), "
            "filtrar por profundidad o ejecutar el barrido de sensibilidad FOV."
        )

    return " ".join(parts)


def run_spatial_copilot(
    *,
    message: str,
    objects: list[dict[str, Any]],
    gt_objects: Optional[list[dict[str, Any]]] = None,
    image_base64: Optional[str] = None,
    width: int = 960,
    height: int = 640,
    fov_h_deg: float = 60.0,
    model: str = "gemini-4-hydrogen",
    previous_interaction_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict[str, Any]:
    """Run Spatial Copilot via live Gemini Function Calling or deterministic offline router."""
    t0 = time.perf_counter()
    toolbox = SpatialToolbox(
        objects=objects,
        gt_objects=gt_objects,
        image_base64=image_base64,
        width=width,
        height=height,
        fov_h_deg=fov_h_deg,
    )

    if not api_key:
        reply = _run_offline_router(message, toolbox)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "reply": reply,
            "mode": "offline_demo",
            "notice": (
                "Ejecutado con el enrutador determinista local (sin API key de Google AI Studio). "
                "Se invocaron exactamente las mismas 7 herramientas geométricas Python."
            ),
            "model": model,
            "interaction_id": None,
            "tool_trace": toolbox.tool_trace,
            "scene_actions": toolbox.scene_actions,
            "latency_ms": latency_ms,
            "usage": {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0},
        }

    thinking_level = resolve_thinking_level(model, None)
    gen_config = {"temperature": 0.2, "thinking_level": thinking_level}
    scene_ctx = _scene_context_block(objects, gt_objects, width, height, fov_h_deg)
    user_input: Any = f"{scene_ctx}\n\nConsulta del usuario: {message}"

    total_usage = {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0}
    last_interaction_id = previous_interaction_id

    try:
        client = genai.Client(api_key=api_key)
        kwargs: dict[str, Any] = {
            "model": model,
            "system_instruction": COPILOT_SYSTEM_INSTRUCTION,
            "input": user_input,
            "tools": COPILOT_TOOLS,
            "generation_config": gen_config,
        }
        if previous_interaction_id:
            kwargs["previous_interaction_id"] = previous_interaction_id

        interaction = client.interactions.create(**kwargs)
        last_interaction_id = getattr(interaction, "id", last_interaction_id)
        total_usage = _merge_usage(total_usage, extract_usage(interaction))

        max_rounds = 5
        for _ in range(max_rounds):
            calls = _extract_function_calls(interaction)
            if not calls:
                break

            tool_results_input: list[dict[str, Any]] = []
            for call in calls:
                tool_res = toolbox.execute(call["name"], call["arguments"])
                tool_results_input.append(
                    {
                        "type": "function_result",
                        "name": call["name"],
                        "call_id": call["id"],
                        "result": [
                            {
                                "type": "text",
                                "text": json.dumps(tool_res, ensure_ascii=False),
                            }
                        ],
                    }
                )

            # Re-specify tools, system_instruction, generation_config, and model every turn
            interaction = client.interactions.create(
                model=model,
                system_instruction=COPILOT_SYSTEM_INSTRUCTION,
                input=tool_results_input,
                tools=COPILOT_TOOLS,
                generation_config=gen_config,
                previous_interaction_id=last_interaction_id,
            )
            last_interaction_id = getattr(interaction, "id", last_interaction_id)
            total_usage = _merge_usage(total_usage, extract_usage(interaction))

        reply_text = extract_output_text(interaction)
        if not reply_text:
            reply_text = "Herramientas geométricas ejecutadas correctamente en el visor 3D."

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "reply": reply_text,
            "mode": "live_gemini",
            "notice": None,
            "model": model,
            "interaction_id": last_interaction_id,
            "tool_trace": toolbox.tool_trace,
            "scene_actions": toolbox.scene_actions,
            "latency_ms": latency_ms,
            "usage": total_usage,
        }

    except Exception as exc:
        # Clean fallback to offline deterministic router without leaking API key
        fallback_toolbox = SpatialToolbox(
            objects=objects,
            gt_objects=gt_objects,
            image_base64=image_base64,
            width=width,
            height=height,
            fov_h_deg=fov_h_deg,
        )
        reply = _run_offline_router(message, fallback_toolbox)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "reply": reply,
            "mode": "deterministic_fallback",
            "notice": f"Llamada live a Google AI Studio no disponible ({type(exc).__name__}); respuesta generada por el enrutador determinista.",
            "model": model,
            "interaction_id": None,
            "tool_trace": fallback_toolbox.tool_trace,
            "scene_actions": fallback_toolbox.scene_actions,
            "latency_ms": latency_ms,
            "usage": total_usage,
        }
