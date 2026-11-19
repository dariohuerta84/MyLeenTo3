"""Google AI Studio Interactions API vision service for 9-DOF 3D bounding box estimation."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Optional

from google import genai
from PIL import Image

from app.schemas import DEFAULT_THINKING_LEVELS, RawDetectionList, SUPPORTED_MODELS

SPATIAL_SYSTEM_INSTRUCTION = (
    "You are a calibrated monocular 3D spatial perception engine. "
    "Estimate accurate 9-DOF metric 3D oriented bounding boxes in the camera coordinate frame: "
    "origin (0, 0, 0) at camera optical center, +X right, +Y forward (depth > 0), +Z up. "
    "Each box_3d MUST be [x, y, z, dx, dy, dz, roll, pitch, yaw] where x, y, z are center "
    "coordinates in meters; dx, dy, dz are full side lengths in meters (> 0); and roll, pitch, yaw "
    "are intrinsic ZYX Euler orientation angles in degrees. "
    "Return strictly valid JSON matching the requested schema."
)


def resolve_api_key(header_api_key: Optional[str] = None) -> Optional[str]:
    """Resolve Gemini API key from HTTP header first, then GEMINI_API_KEY env var."""
    if header_api_key and header_api_key.strip():
        return header_api_key.strip()
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    return env_key if env_key else None


def resolve_thinking_level(model: str, thinking_level: Optional[str] = None) -> str:
    """Resolve thinking level using model-specific default when omitted."""
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model}'. Allowed: {SUPPORTED_MODELS}")
    if thinking_level:
        return thinking_level
    return DEFAULT_THINKING_LEVELS[model]


def strip_data_uri(image_base64: str, default_mime: str = "image/png") -> tuple[str, str]:
    """Strip optional data URL prefix and extract MIME type if present."""
    raw = image_base64.strip()
    match = re.match(r"^data:([^;]+);base64,(.*)$", raw, flags=re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1).strip()
    return raw, default_mime


def infer_image_dimensions(image_base64: str, fallback_width: int = 960, fallback_height: int = 640) -> tuple[int, int]:
    """Decode base64 image header via PIL to obtain exact (width, height)."""
    clean_b64, _ = strip_data_uri(image_base64)
    try:
        raw_bytes = base64.b64decode(clean_b64, validate=False)
        with Image.open(io.BytesIO(raw_bytes)) as img:
            return int(img.width), int(img.height)
    except Exception:
        return fallback_width, fallback_height


def extract_usage(interaction: Any) -> dict[str, Any]:
    """Safely extract token usage dictionary from a Google GenAI Interaction object."""
    usage = getattr(interaction, "usage", None)
    if usage is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        }
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("total_input_tokens", usage.get("input_tokens", 0)) or 0),
            "output_tokens": int(usage.get("total_output_tokens", usage.get("output_tokens", 0)) or 0),
            "thought_tokens": int(usage.get("total_thought_tokens", usage.get("thought_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    return {
        "input_tokens": int(getattr(usage, "total_input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "total_output_tokens", 0) or 0),
        "thought_tokens": int(getattr(usage, "total_thought_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def extract_output_text(interaction: Any) -> str:
    """Extract final textual content from an Interaction response (output_text or steps)."""
    direct_text = getattr(interaction, "output_text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    chunks: list[str] = []
    steps = getattr(interaction, "steps", None) or []
    for step in steps:
        step_type = getattr(step, "type", None) or (step.get("type") if isinstance(step, dict) else None)
        if step_type == "model_output":
            content = getattr(step, "content", None) or (step.get("content") if isinstance(step, dict) else [])
            for item in content or []:
                item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
                if item_type == "text":
                    txt = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
                    if txt:
                        chunks.append(str(txt))
        elif step_type == "text":
            txt = getattr(step, "text", None) or (step.get("text") if isinstance(step, dict) else "")
            if txt:
                chunks.append(str(txt))

    return "\n".join(chunks).strip()


def _parse_raw_detections(raw_text: str) -> RawDetectionList:
    """Parse JSON structured output into RawDetectionList, stripping optional code fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return RawDetectionList.model_validate_json(cleaned)
    except Exception:
        data = json.loads(cleaned)
        if isinstance(data, list):
            data = {"objects": data, "scene_summary": ""}
        return RawDetectionList.model_validate(data)


def build_spatial_prompt(
    max_objects: int = 10,
    fov_h_deg: float = 60.0,
    custom_prompt: Optional[str] = None,
) -> str:
    """Construct the structured 3D detection instruction prompt."""
    base = (
        f"Detect up to {max_objects} prominent rigid physical objects in this monocular RGB image "
        f"(assumed horizontal FOV = {fov_h_deg:.1f} deg). "
        "For every object, output its label, confidence (0..1), and metric 9-DOF box_3d "
        "[x, y, z, dx, dy, dz, roll, pitch, yaw] in camera coordinates "
        "(+X right, +Y forward depth > 0.3m, +Z up; dx, dy, dz > 0 in meters; angles in degrees). "
        "Also include a concise scene_summary describing spatial depth ordering."
    )
    if custom_prompt and custom_prompt.strip():
        return f"{base}\nAdditional user focus: {custom_prompt.strip()}"
    return base


def detect_3d_objects_live(
    *,
    api_key: str,
    image_base64: str,
    mime_type: str = "image/png",
    model: str = "gemini-4-hydrogen",
    thinking_level: Optional[str] = None,
    fov_h_deg: float = 60.0,
    max_objects: int = 10,
    custom_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Execute live 3D object detection via Google GenAI Interactions API (v1)."""
    resolved_level = resolve_thinking_level(model, thinking_level)
    clean_b64, resolved_mime = strip_data_uri(image_base64, default_mime=mime_type)
    prompt = build_spatial_prompt(
        max_objects=max_objects,
        fov_h_deg=fov_h_deg,
        custom_prompt=custom_prompt,
    )

    client = genai.Client(api_key=api_key)
    t0 = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        system_instruction=SPATIAL_SYSTEM_INSTRUCTION,
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "data": clean_b64, "mime_type": resolved_mime},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": RawDetectionList.model_json_schema(),
        },
        generation_config={
            "temperature": 0.2,
            "thinking_level": resolved_level,
        },
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    raw_text = extract_output_text(interaction)
    if not raw_text:
        raise ValueError("Gemini Interactions API returned an empty structured output payload.")

    parsed = _parse_raw_detections(raw_text)
    objects: list[dict[str, Any]] = []
    for idx, det in enumerate(parsed.objects[:max_objects]):
        x, y, z, dx, dy, dz, roll, pitch, yaw = det.box_3d
        # Ensure positive depth in front of camera
        safe_y = max(y, 0.25)
        objects.append(
            {
                "id": f"pred_{idx + 1}",
                "label": det.label.strip() or f"object_{idx + 1}",
                "box_3d": [x, safe_y, z, dx, dy, dz, roll, pitch, yaw],
                "confidence": round(float(det.confidence), 4),
            }
        )

    return {
        "objects": objects,
        "scene_summary": parsed.scene_summary,
        "latency_ms": latency_ms,
        "usage": extract_usage(interaction),
        "interaction_id": getattr(interaction, "id", None),
        "model": model,
        "thinking_level": resolved_level,
    }


def build_deterministic_fallback_detections(
    *,
    width: int,
    height: int,
    fov_h_deg: float,
    model: str,
    thinking_level: str,
    max_objects: int = 10,
) -> dict[str, Any]:
    """Generate deterministic fallback 3D hypotheses when analyzing custom images offline."""
    candidates = [
        {
            "id": "fallback_1",
            "label": "primary_subject",
            "box_3d": [0.0, 2.35, -0.10, 0.62, 0.48, 0.52, 0.0, 0.0, 12.0],
            "confidence": 0.84,
        },
        {
            "id": "fallback_2",
            "label": "left_reference_object",
            "box_3d": [-0.72, 2.95, -0.18, 0.44, 0.38, 0.42, 0.0, 0.0, -18.0],
            "confidence": 0.78,
        },
        {
            "id": "fallback_3",
            "label": "right_background_volume",
            "box_3d": [0.78, 3.40, 0.05, 0.55, 0.50, 0.68, 0.0, 0.0, 25.0],
            "confidence": 0.74,
        },
    ]
    return {
        "objects": candidates[:max_objects],
        "scene_summary": (
            f"Modo determinista offline ({width}x{height}, FOV_h={fov_h_deg:.1f}°): "
            "hipótesis 3D de referencia generadas localmente sin clave API de Google AI Studio."
        ),
        "latency_ms": 4.2,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "thought_tokens": 0,
            "total_tokens": 0,
        },
        "interaction_id": None,
        "model": model,
        "thinking_level": thinking_level,
    }
