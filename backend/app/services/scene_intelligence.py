"""Hybrid Scene Intelligence: VLM / Prompt-Free open-world discovery & 3D-grounded diagnosis.

Connects semantic image understanding (either live Gemini Interactions API VLM or
local autonomous YOLOE-PF 4,585-class discovery) with open-vocabulary YOLOE-OV
segmentation, physical scale-prior checks, and metric 3D diagnosis grounded strictly
in MoGe-2 / Kay-Kajiya / RANSAC / SAT computed telemetry.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Literal, Optional

import numpy as np
from google import genai
from pydantic import BaseModel, Field

from app.services.gemini_vision import (
    extract_output_text,
    extract_usage,
    resolve_thinking_level,
    strip_data_uri,
)

# Canonical inspection domain catalog tailored for adaptive detector vocabularies
# and downstream 3D-grounded diagnostics.
DOMAIN_PROFILES: dict[str, dict[str, Any]] = {
    "electrical_industrial": {
        "label_es": "Inspección Eléctrica e Industrial",
        "keywords": {
            "wire", "cable", "cord", "switch", "electrical switch", "wall outlet",
            "power outlet", "socket", "power plug", "plug", "circuit", "breaker",
            "junction box", "electrical box", "relay", "oscilloscope", "multimeter",
            "tool", "screwdriver", "pliers", "tape", "electronic device", "pcb",
            "cableado", "interruptor", "enchufe", "tomacorriente", "tablero", "alambre",
        },
        "detector_classes": [
            "human hand",
            "electrical switch",
            "wall outlet",
            "power plug",
            "electrical wire",
            "cable",
            "cord",
            "junction box",
            "electronic device",
            "plastic cover",
            "leather bag",
            "tool",
            "screwdriver",
            "oscilloscope",
            "connector",
        ],
        "focus_es": [
            "Verificar holguras libres entre manos/herramientas y puntos energizados o cables expuestos.",
            "Medir longitud libre, relieve Z y curvatura de conductores sobre la pared o tablero.",
            "Detectar reflejos especulares en cubiertas plásticas y oclusiones de empalmes.",
        ],
    },
    "clinical_anatomical": {
        "label_es": "Inspección Clínica y Anatómica",
        "keywords": {
            "hand", "human hand", "arm", "finger", "wrist", "leg", "foot", "face",
            "skin", "lesion", "wound", "bandage", "gauze", "medical", "syringe",
            "stethoscope", "patient", "glove", "surgical", "clinic", "hospital",
            "anatomical", "mano", "brazo", "herida", "piel", "clinica", "medico",
        },
        "detector_classes": [
            "human hand",
            "finger",
            "arm",
            "face",
            "person",
            "medical glove",
            "bandage",
            "medical device",
            "bottle",
            "table",
            "surgical instrument",
            "syringe",
            "cloth",
            "scale ruler",
        ],
        "focus_es": [
            "Cuantificar elevación de relieve en milímetros (relieve 3D sobre plano de soporte).",
            "Verificar pérdida métrica causada por brillos especulares en piel o tejido húmedo.",
            "Validar proporción anatómica frente a priors de tamaño físico humano.",
        ],
    },
    "desk_electronics": {
        "label_es": "Estación de Trabajo y Electrónica",
        "keywords": {
            "laptop", "computer", "monitor", "screen", "keyboard", "mouse",
            "computer mouse", "phone", "smartphone", "desk", "notebook", "book",
            "tablet", "headphones", "webcam", "speaker", "microphone", "printer",
            "oscilloscope", "spectrometer", "coffee mug", "mug", "cup",
            "escritorio", "computadora", "teclado", "pantalla", "telefono", "oficina",
        },
        "detector_classes": [
            "laptop",
            "monitor",
            "keyboard",
            "computer mouse",
            "smartphone",
            "desk",
            "mug",
            "cup",
            "bottle",
            "book",
            "headphones",
            "electronic device",
            "cable",
            "chair",
            "human hand",
        ],
        "focus_es": [
            "Evaluar alineación Manhattan-World y coplanaridad respecto al plano del escritorio (RANSAC).",
            "Auditar distancia entre dispositivos y colisiones o bloqueos visuales frente a cámara.",
            "Contrastación métrica de ancho/alto con dimensiones estándar de hardware (laptop/monitor).",
        ],
    },
    "logistics_packaging": {
        "label_es": "Logística y Empaque",
        "keywords": {
            "box", "cardboard box", "crate", "storage crate", "pallet", "package",
            "parcel", "carton", "container", "warehouse", "forklift", "shelf",
            "bag", "backpack", "suitcase", "luggage", "bin", "basket",
            "caja", "paquete", "almacen", "palet", "contenedor", "embalaje",
        },
        "detector_classes": [
            "cardboard box",
            "storage crate",
            "package",
            "pallet",
            "container",
            "backpack",
            "bag",
            "bottle",
            "shelf",
            "hand truck",
            "plastic bin",
            "label",
            "table",
            "person",
        ],
        "focus_es": [
            "Calcular volumen ocupado en m³ y huella en planta (BEV) de cada bulto o caja.",
            "Detectar apilamiento, interpenetración SAT o brechas de contacto con el plano de soporte.",
            "Verificar proporción cúbica real frente a priors estándar de embalaje.",
        ],
    },
    "interior_architecture": {
        "label_es": "Arquitectura e Interiores",
        "keywords": {
            "chair", "sofa", "couch", "table", "door", "window", "cabinet",
            "wardrobe", "bed", "shelf", "bookshelf", "lamp", "plant", "potted plant",
            "rug", "carpet", "wall", "floor", "mirror", "sink", "toilet", "fridge",
            "mueble", "silla", "mesa", "puerta", "ventana", "habitacion", "interior",
        },
        "detector_classes": [
            "chair",
            "table",
            "sofa",
            "desk",
            "cabinet",
            "shelf",
            "door",
            "window",
            "lamp",
            "potted plant",
            "bed",
            "mirror",
            "backpack",
            "person",
        ],
        "focus_es": [
            "Medir ortogonalidad arquitectónica (Manhattan score) e inclinación del plano de suelo/pared.",
            "Auditar corredores libres, holguras entre mobiliario y truncamiento en bordes de encuadre.",
            "Estimar consistencia de FOV efectivo entre EXIF 35mm y las intrínsecas de MoGe-2.",
        ],
    },
    "automotive_mechanical": {
        "label_es": "Automotriz y Mecánica",
        "keywords": {
            "car", "vehicle", "truck", "motorcycle", "bicycle", "wheel", "tire",
            "engine", "gear", "wrench", "hammer", "drill", "bolt", "nut", "pipe",
            "metal", "machine", "motor", "brake", "chassis", "helmet",
            "vehiculo", "auto", "motor", "llanta", "herramienta", "mecanica", "taller",
        },
        "detector_classes": [
            "car",
            "wheel",
            "tire",
            "engine part",
            "metal pipe",
            "wrench",
            "power drill",
            "tool",
            "toolbox",
            "machine",
            "motorcycle",
            "helmet",
            "human hand",
            "person",
        ],
        "focus_es": [
            "Controlar brillos metálicos especulares que deforman la estimación monocular de profundidad.",
            "Inspeccionar separaciones críticas y relieve entre componentes mecánicos o herramientas.",
            "Auditar consistencia normal-profundidad (NDC) en piezas curvas.",
        ],
    },
    "outdoor_urban": {
        "label_es": "Entorno Urbano y Exterior",
        "keywords": {
            "street", "road", "sidewalk", "building", "tree", "traffic light",
            "stop sign", "sign", "bench", "crosswalk", "bus", "bicycle", "pole",
            "fence", "sky", "pedestrian", "calle", "edificio", "exterior", "urbano",
        },
        "detector_classes": [
            "person",
            "car",
            "bicycle",
            "motorcycle",
            "bus",
            "traffic light",
            "street sign",
            "bench",
            "tree",
            "building",
            "pole",
            "backpack",
            "trash can",
        ],
        "focus_es": [
            "Analizar gradiente de profundidad cercana vs. lejana y separación de corredores de paso.",
            "Detectar oclusiones entre peatones, mobiliario urbano y vehículos.",
            "Evaluar truncamiento lateral en bordes del frustum de cámara.",
        ],
    },
    "general_physical": {
        "label_es": "Inspección Física General",
        "keywords": set(),
        "detector_classes": [
            "person",
            "human hand",
            "table",
            "chair",
            "bottle",
            "cup",
            "phone",
            "book",
            "bag",
            "cardboard box",
            "electronic device",
            "cable",
            "tool",
            "door",
        ],
        "focus_es": [
            "Validar rango de profundidad en metros, relieves relativos y soporte RANSAC.",
            "Auditar pares cercanos por colisión SAT y oclusión parcial Kay-Kajiya.",
            "Verificar consistencia de escala física respecto a objetos cotidianos de referencia.",
        ],
    },
}

# Physical largest-extent priors (min_m, typical_m, max_m) for zero-shot scale sanity checks.
PHYSICAL_SCALE_PRIORS_M: dict[str, tuple[float, float, float]] = {
    "hand": (0.14, 0.19, 0.24),
    "human_hand": (0.14, 0.19, 0.24),
    "finger": (0.05, 0.08, 0.12),
    "person": (1.20, 1.70, 2.05),
    "phone": (0.12, 0.15, 0.19),
    "smartphone": (0.13, 0.16, 0.19),
    "laptop": (0.28, 0.35, 0.44),
    "monitor": (0.45, 0.62, 0.90),
    "keyboard": (0.30, 0.43, 0.50),
    "computer_mouse": (0.08, 0.11, 0.14),
    "mouse": (0.08, 0.11, 0.14),
    "cup": (0.07, 0.10, 0.15),
    "mug": (0.08, 0.11, 0.15),
    "bottle": (0.14, 0.23, 0.34),
    "book": (0.18, 0.25, 0.36),
    "wall_outlet": (0.07, 0.12, 0.18),
    "power_outlet": (0.07, 0.12, 0.18),
    "electrical_switch": (0.07, 0.12, 0.20),
    "socket": (0.07, 0.12, 0.18),
    "power_plug": (0.04, 0.07, 0.12),
    "oscilloscope": (0.24, 0.36, 0.48),
    "cardboard_box": (0.18, 0.38, 0.85),
    "storage_crate": (0.30, 0.52, 0.90),
    "backpack": (0.32, 0.46, 0.62),
    "bag": (0.22, 0.40, 0.68),
    "leather_bag": (0.22, 0.38, 0.62),
    "chair": (0.55, 0.85, 1.15),
    "desk": (0.95, 1.40, 2.20),
    "table": (0.80, 1.35, 2.40),
    "door": (1.75, 2.05, 2.40),
    "wheel": (0.42, 0.64, 0.85),
    "tire": (0.45, 0.66, 0.90),
    "car": (3.20, 4.45, 5.40),
    "screwdriver": (0.14, 0.22, 0.35),
    "wrench": (0.15, 0.25, 0.45),
}

_NOISE_PF_TERMS = {"tattoo artist", "glass floor", "handcuff", "corset", "passbook"}


class Detected2DHint(BaseModel):
    """Optional 2D normalized bounding hint `[ymin, xmin, ymax, xmax]` in `0..1000`."""

    label: str
    box_2d: list[float] = Field(..., min_length=4, max_length=4)
    typical_size_m: Optional[float] = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class ScalePriorItem(BaseModel):
    """Physical real-world scale anchor proposed by the VLM for metrology verification."""

    label: str
    typical_extent_m: float = Field(..., gt=0.01, le=25.0)
    rationale: str = ""


class SceneVocabularyProposal(BaseModel):
    """Structured VLM scene understanding & open-vocabulary proposal before 3D segmentation."""

    domain: str
    domain_label_es: str
    scene_context_es: str
    detected_classes: list[str] = Field(default_factory=list)
    objects_2d: list[Detected2DHint] = Field(default_factory=list)
    scale_priors: list[ScalePriorItem] = Field(default_factory=list)
    inspection_focus_es: list[str] = Field(default_factory=list)


class DiagnosisFinding(BaseModel):
    """Single quantitative finding strictly supported by computed 3D geometry/metrics."""

    title: str
    metric_evidence: str
    severity: Literal["info", "low", "medium", "high"]
    interpretation: str


class SceneDiagnosisPayload(BaseModel):
    """Structured 3D-grounded adaptive diagnostic report in Spanish."""

    domain: str
    domain_label_es: str
    diagnosis_title: str
    executive_summary: str
    model_predictions_es: str
    confidence_level: Literal["alta", "media", "baja"]
    scale_consistency_note: str
    key_findings: list[DiagnosisFinding] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


def _norm_token(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("_", " ").replace("-", " "))


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _strip_json_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_custom_prompt_classes(custom_prompt: Optional[str]) -> list[str]:
    """Parse user comma/semicolon/newline custom classes into clean deduplicated noun phrases."""
    if not custom_prompt or not custom_prompt.strip():
        return []
    raw_items = re.split(r"[,;\n]+", custom_prompt)
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        token = item.strip(" \t\r\n\"'[]")
        if not token:
            continue
        low = token.lower()
        if low in seen:
            continue
        seen.add(low)
        cleaned.append(token)
        if len(cleaned) >= 14:
            break
    return cleaned


def infer_local_scene_discovery(
    pf_detections: list[dict[str, Any]],
    *,
    depth_stats: Optional[dict[str, float]] = None,
    custom_prompt: Optional[str] = None,
    default_classes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Infer scene domain and dynamic open-vocab class list from local Prompt-Free discoveries."""
    user_classes = parse_custom_prompt_classes(custom_prompt)
    user_tokens = {_norm_token(c) for c in user_classes}
    if custom_prompt and custom_prompt.strip():
        for w in re.findall(r"[a-zA-ZáéíóúñÁÉÍÓÚÑ0-9_]+", custom_prompt.lower()):
            user_tokens.add(_norm_token(w))

    scores: dict[str, float] = {}
    for dom_key, prof in DOMAIN_PROFILES.items():
        if dom_key == "general_physical":
            continue
        kw_set = {_norm_token(k) for k in prof["keywords"]}
        cls_set = {_norm_token(k) for k in prof["detector_classes"]}
        all_terms = kw_set | cls_set
        score = 0.0

        for det in pf_detections:
            lbl = _norm_token(str(det.get("label", "")))
            if not lbl or lbl in _NOISE_PF_TERMS:
                continue
            conf = float(det.get("conf", det.get("confidence", 0.25)) or 0.25)
            # Exclude generic "person" alone from biasing toward clinical vs electrical heavily
            weight = 0.25 if lbl == "person" else 1.0
            if lbl in all_terms:
                score += (0.8 + conf) * weight
            else:
                lbl_words = set(lbl.split())
                if lbl_words & all_terms:
                    score += (0.45 + 0.5 * conf) * weight

        for u_tok in user_tokens:
            if not u_tok:
                continue
            if u_tok in all_terms:
                score += 1.6
            elif any(u_tok in term or term in u_tok for term in all_terms if len(u_tok) >= 3):
                score += 0.9

        scores[dom_key] = score

    median_m = float((depth_stats or {}).get("median_m", 0.0) or 0.0)
    if 0.05 < median_m < 0.9:
        for close_dom in ("electrical_industrial", "clinical_anatomical"):
            if scores.get(close_dom, 0.0) > 0.0:
                scores[close_dom] += 0.15

    best_dom = max(scores, key=lambda k: scores[k]) if scores else "general_physical"
    best_score = scores.get(best_dom, 0.0)
    if best_score <= 0.15:
        best_dom = "general_physical"

    profile = DOMAIN_PROFILES[best_dom]

    # Build prioritized deduplicated open-vocab class list (max 28)
    vocab: list[str] = []
    seen_vocab: set[str] = set()

    def _add_term(term: str) -> None:
        t = re.sub(r"\s+", " ", term.strip().replace("_", " "))
        if not t:
            return
        k = t.lower()
        if k in seen_vocab or len(vocab) >= 28:
            return
        seen_vocab.add(k)
        vocab.append(t)

    for uc in user_classes:
        _add_term(uc)
    for dc in profile["detector_classes"]:
        _add_term(dc)

    clean_pf: list[dict[str, Any]] = []
    for det in pf_detections:
        raw_lbl = str(det.get("label", "")).strip()
        conf = float(det.get("conf", det.get("confidence", 0.0)) or 0.0)
        norm_l = _norm_token(raw_lbl)
        if not norm_l:
            continue
        clean_pf.append(
            {
                "label": raw_lbl,
                "confidence": round(conf, 3),
                "box_2d": det.get("xyxy_norm") or det.get("box_2d"),
            }
        )
        if (
            conf >= 0.22
            and norm_l not in _NOISE_PF_TERMS
            and len(norm_l.split()) <= 3
        ):
            _add_term(raw_lbl)

    if default_classes:
        for fb_c in default_classes:
            if len(vocab) >= 28:
                break
            _add_term(fb_c)

    for anchor in ("person", "hand"):
        _add_term(anchor)

    top_pf_txt = ", ".join(p["label"] for p in clean_pf[:4]) if clean_pf else "ninguna previa"
    range_txt = (
        f"rango mediano {median_m:.2f} m"
        if median_m > 0.02
        else "profundidad por estimar"
    )
    scene_context_es = (
        f"Descubrimiento autónomo local ({profile['label_es']} · {range_txt}). "
        f"Pistas Prompt-Free (4,585 cl.): {top_pf_txt}."
    )

    return {
        "discovery_mode": "local_yoloe_pf_adaptive",
        "domain": best_dom,
        "domain_label_es": profile["label_es"],
        "scene_context_es": scene_context_es,
        "detected_classes": list(vocab),
        "active_vocabulary": vocab,
        "prompt_free_discoveries": clean_pf[:10],
        "objects_2d": [],
        "scale_priors": [],
        "inspection_focus_es": list(profile["focus_es"]),
        "vlm_guided_count": 0,
    }


def discover_scene_vocabulary_vlm(
    *,
    api_key: str,
    image_base64: str,
    mime_type: str = "image/png",
    model: str = "gemini-4-hydrogen",
    custom_prompt: Optional[str] = None,
) -> dict[str, Any]:
    """Use Gemini Interactions API VLM to identify domain, dynamic classes, and 2D anchors."""
    clean_b64, resolved_mime = strip_data_uri(image_base64, default_mime=mime_type)
    user_classes = parse_custom_prompt_classes(custom_prompt)
    allowed_domains = ", ".join(DOMAIN_PROFILES.keys())

    prompt = (
        "Inspect this RGB image for calibrated 3D spatial perception and open-vocabulary segmentation.\n"
        f"1. Classify `domain` strictly as ONE of: {allowed_domains}.\n"
        "2. Provide `domain_label_es` and `scene_context_es` in concise Spanish.\n"
        "3. Propose 6 to 14 crisp English noun-phrase `detected_classes` for an open-vocab detector "
        "(e.g., 'human hand', 'electrical switch', 'braided wire', 'leather bag', 'oscilloscope').\n"
        "4. Detect up to 10 visible rigid or anatomical physical items in `objects_2d` with "
        "normalized `box_2d` = [ymin, xmin, ymax, xmax] in 0..1000 coordinates, clear `label`, "
        "`typical_size_m` (meters), and `confidence` (0..1).\n"
        "5. Provide physical `scale_priors` (real-world typical largest dimension in meters) and 2-3 "
        "`inspection_focus_es` bullets relevant to this scene."
    )
    if custom_prompt and custom_prompt.strip():
        prompt += f"\n6. User explicitly requested focus/classes: {custom_prompt.strip()}"

    client = genai.Client(api_key=api_key)
    t0 = time.perf_counter()
    interaction = client.interactions.create(
        model=model,
        system_instruction=(
            "You are a multimodal scene intelligence front-end that feeds open-vocabulary 2D instance "
            "segmentation before metric 3D reconstruction. Return strictly valid JSON matching the schema."
        ),
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "data": clean_b64, "mime_type": resolved_mime},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": SceneVocabularyProposal.model_json_schema(),
        },
        generation_config={
            "temperature": 0.15,
            "thinking_level": "low",
        },
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    raw_text = extract_output_text(interaction)
    if not raw_text:
        raise ValueError("Empty structured response from VLM scene discovery.")

    cleaned = _strip_json_fences(raw_text)
    try:
        proposal = SceneVocabularyProposal.model_validate_json(cleaned)
    except Exception:
        proposal = SceneVocabularyProposal.model_validate(json.loads(cleaned))

    domain_key = proposal.domain.strip().lower()
    if domain_key not in DOMAIN_PROFILES:
        domain_key = "general_physical"
    profile = DOMAIN_PROFILES[domain_key]

    vocab: list[str] = []
    seen: set[str] = set()
    for term in [*user_classes, *proposal.detected_classes, *[o.label for o in proposal.objects_2d]]:
        t = re.sub(r"\s+", " ", str(term).strip().replace("_", " "))
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        vocab.append(t)
        if len(vocab) >= 16:
            break

    if not vocab:
        vocab = list(profile["detector_classes"][:12])

    objects_2d_clean: list[dict[str, Any]] = []
    for item in proposal.objects_2d[:10]:
        vals = [float(np.clip(v, 0.0, 1000.0)) if isinstance(v, (int, float)) else 0.0 for v in item.box_2d]
        ymin, xmin, ymax, xmax = min(vals[0], vals[2]), min(vals[1], vals[3]), max(vals[0], vals[2]), max(vals[1], vals[3])
        if (xmax - xmin) < 4.0 or (ymax - ymin) < 4.0:
            continue
        objects_2d_clean.append(
            {
                "label": re.sub(r"\s+", " ", item.label.strip()),
                "box_2d": [round(ymin, 1), round(xmin, 1), round(ymax, 1), round(xmax, 1)],
                "typical_size_m": round(float(item.typical_size_m), 3) if item.typical_size_m else None,
                "confidence": round(float(item.confidence), 3),
            }
        )

    scale_priors = [
        {
            "label": sp.label.strip(),
            "typical_extent_m": round(float(sp.typical_extent_m), 3),
            "rationale": sp.rationale.strip(),
        }
        for sp in proposal.scale_priors[:8]
    ]

    return {
        "discovery_mode": "vlm_gemini",
        "domain": domain_key,
        "domain_label_es": proposal.domain_label_es.strip() or profile["label_es"],
        "scene_context_es": proposal.scene_context_es.strip() or f"Escena analizada en dominio {profile['label_es']}.",
        "detected_classes": vocab,
        "active_vocabulary": vocab,
        "prompt_free_discoveries": [],
        "objects_2d": objects_2d_clean,
        "scale_priors": scale_priors,
        "inspection_focus_es": [f.strip() for f in proposal.inspection_focus_es if f.strip()][:4] or list(profile["focus_es"]),
        "usage": extract_usage(interaction),
        "latency_ms": latency_ms,
    }


def _lookup_physical_prior(
    label: str,
    extra_prior_map: dict[str, tuple[float, float, float, str]],
) -> Optional[tuple[float, float, float, str]]:
    k_lbl = _norm_key(label)
    if k_lbl in extra_prior_map:
        return extra_prior_map[k_lbl]
    if k_lbl in PHYSICAL_SCALE_PRIORS_M:
        lo, typ, hi = PHYSICAL_SCALE_PRIORS_M[k_lbl]
        return lo, typ, hi, "prior físico estándar"

    words = [w for w in k_lbl.split("_") if len(w) >= 3]
    for candidate_key, val in extra_prior_map.items():
        if candidate_key in k_lbl or k_lbl in candidate_key:
            return val
    for candidate_key, bounds in PHYSICAL_SCALE_PRIORS_M.items():
        if candidate_key in k_lbl or any(w == candidate_key for w in words):
            lo, typ, hi = bounds
            return lo, typ, hi, "prior físico estándar"
    return None


def evaluate_scale_priors(
    objects: list[dict[str, Any]],
    extra_priors: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], str]:
    """Compare measured 3D OBB max dimensions against physical scale priors."""
    extra_map: dict[str, tuple[float, float, float, str]] = {}
    for p in extra_priors or []:
        lbl = _norm_key(str(p.get("label", "")))
        typ = p.get("typical_extent_m")
        if lbl and typ and float(typ) > 0.01:
            t_val = float(typ)
            extra_map[lbl] = (0.55 * t_val, t_val, 1.75 * t_val, str(p.get("rationale", "prior VLM")))

    checks: list[dict[str, Any]] = []
    ratios: list[float] = []

    for obj in objects:
        box = obj.get("box_3d") or []
        if len(box) < 6:
            continue
        raw_lbl = str(obj.get("label", ""))
        if _norm_key(raw_lbl) == "person" and float(box[1]) < 1.8 and "person" not in extra_map:
            # Partial close-up human crops (<1.8m depth) must not be penalized against full standing stature (1.70m).
            continue
        measured_max_m = max(float(box[3]), float(box[4]), float(box[5]))
        prior = _lookup_physical_prior(raw_lbl, extra_map)
        if prior is None:
            continue
        min_m, typical_m, max_m, source_txt = prior
        ratio = measured_max_m / max(typical_m, 1e-4)
        ratios.append(ratio)

        if 0.55 <= ratio <= 1.75 or (0.70 * min_m <= measured_max_m <= 1.30 * max_m):
            status = "consistente"
        elif ratio < 0.55:
            status = "subestimada"
        else:
            status = "sobreestimada"

        checks.append(
            {
                "object_id": str(obj.get("id", "")),
                "label": str(obj.get("label", "")),
                "measured_max_m": round(measured_max_m, 3),
                "expected_typical_m": round(typical_m, 3),
                "ratio": round(ratio, 2),
                "status": status,
                "source": source_txt,
            }
        )

    if not checks:
        return (
            [],
            "Sin anclas con prior dimensional rígido en la escena; se mantiene escala métrica intrínseca (MoGe-2 / FOV).",
        )

    ratios_sorted = sorted(ratios)
    mid = len(ratios_sorted) // 2
    median_ratio = (
        ratios_sorted[mid]
        if len(ratios_sorted) % 2 == 1
        else 0.5 * (ratios_sorted[mid - 1] + ratios_sorted[mid])
    )
    consistent_n = sum(1 for c in checks if c["status"] == "consistente")

    if median_ratio < 0.65 or median_ratio > 1.35:
        corr = 1.0 / max(median_ratio, 1e-4)
        direction = "sobreestimada" if median_ratio > 1.35 else "subestimada"
        note = (
            f"Escala global {direction} (ratio mediano {median_ratio:.2f}× sobre {len(checks)} ancla(s)); "
            f"multiplicador sugerido = {corr:.2f}× o calibrar con referencia física conocida."
        )
    else:
        note = (
            f"Escala física consistente en {consistent_n}/{len(checks)} ancla(s) verificadas "
            f"(ratio mediano {median_ratio:.2f}× respecto a tamaños reales estándar)."
        )

    return checks, note


def _compact_3d_telemetry_for_diagnosis(
    *,
    discovery: dict[str, Any],
    objects: list[dict[str, Any]],
    occlusion_matrix: list[dict[str, Any]],
    intrinsic_diagnostics: Optional[dict[str, Any]],
    real_photo_metrics: Optional[dict[str, Any]],
    depth_stats: Optional[dict[str, float]],
    point_cloud: Optional[dict[str, Any]],
    scale_checks: list[dict[str, Any]],
    scale_note: str,
) -> dict[str, Any]:
    obj_rows = []
    for o in objects[:10]:
        b = o.get("box_3d") or [0.0] * 9
        obj_rows.append(
            {
                "id": o.get("id"),
                "label": o.get("label"),
                "depth_m": round(float(b[1]), 3),
                "extents_cm": [
                    round(float(b[3]) * 100.0, 1),
                    round(float(b[4]) * 100.0, 1),
                    round(float(b[5]) * 100.0, 1),
                ],
                "yaw_deg": round(float(b[8]), 1),
                "volume_m3": round(float(b[3]) * float(b[4]) * float(b[5]), 5),
                "confidence": o.get("confidence", 0.0),
            }
        )

    ndc = (real_photo_metrics or {}).get("normal_depth_consistency") or {}
    plane = (real_photo_metrics or {}).get("ransac_plane") or {}
    bounds = (real_photo_metrics or {}).get("depth_boundaries") or {}
    opt = (real_photo_metrics or {}).get("optical_telemetry") or {}
    gate = (intrinsic_diagnostics or {}).get("cascade_policy") or {}
    latent = (intrinsic_diagnostics or {}).get("latent_metrics") or {}

    return {
        "domain": discovery.get("domain", "general_physical"),
        "domain_label_es": discovery.get("domain_label_es", "Inspección Física General"),
        "object_count": len(objects),
        "objects": obj_rows,
        "depth_stats_m": depth_stats or {},
        "point_cloud": {
            "count": (point_cloud or {}).get("count", 0),
            "body_point_count": (point_cloud or {}).get("body_point_count", 0),
            "max_relief_mm": (point_cloud or {}).get("max_relief_mm", 0.0),
            "triangle_count": ((point_cloud or {}).get("surface_mesh") or {}).get("triangle_count", 0),
        },
        "real_photo_metrics": {
            "ndc_score_0_100": ndc.get("ndc_score_0_100"),
            "mean_normal_err_deg": ndc.get("mean_angular_error_deg"),
            "manhattan_alignment_score": (real_photo_metrics or {}).get("manhattan_alignment_score"),
            "ransac_plane_detected": plane.get("detected"),
            "ransac_plane_orientation": plane.get("plane_orientation"),
            "ransac_plane_tilt_deg": plane.get("plane_tilt_deg"),
            "ransac_inlier_ratio": plane.get("inlier_ratio"),
            "depth_rgb_boundary_f1": bounds.get("depth_rgb_boundary_f1"),
            "flying_pixel_ratio": bounds.get("flying_pixel_ratio"),
            "specular_reflection_ratio": opt.get("specular_reflection_ratio"),
            "focus_quality": opt.get("focus_quality"),
            "overall_3d_geometry_score": (real_photo_metrics or {}).get("overall_3d_geometry_score_0_100"),
        },
        "cascade_gate": {
            "verdict": gate.get("operational_verdict"),
            "regime": gate.get("c8286_degradation_regime"),
            "min_surface_clearance_m": gate.get("min_surface_clearance_m"),
            "colliding_pairs": gate.get("colliding_pairs_count", 0),
            "occlusion_pairs": len(occlusion_matrix),
            "physical_plausibility": latent.get("physical_plausibility_score"),
        },
        "top_occlusions": [
            {
                "occluder": oc.get("occluder_label"),
                "occluded": oc.get("occluded_label"),
                "ratio": oc.get("occlusion_ratio"),
                "depth_gap_m": oc.get("depth_gap_m"),
            }
            for oc in occlusion_matrix[:3]
        ],
        "scale_prior_checks": scale_checks,
        "scale_consistency_note": scale_note,
    }


def build_deterministic_scene_diagnosis(
    *,
    discovery: dict[str, Any],
    objects: list[dict[str, Any]],
    occlusion_matrix: list[dict[str, Any]],
    intrinsic_diagnostics: Optional[dict[str, Any]] = None,
    real_photo_metrics: Optional[dict[str, Any]] = None,
    depth_stats: Optional[dict[str, float]] = None,
    point_cloud: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Generate a structured 3D-grounded Spanish diagnosis strictly from computed geometry & metrology."""
    domain = str(discovery.get("domain") or "general_physical")
    if domain not in DOMAIN_PROFILES:
        domain = "general_physical"
    profile = DOMAIN_PROFILES[domain]
    domain_label_es = str(discovery.get("domain_label_es") or profile["label_es"])

    scale_checks, scale_note = evaluate_scale_priors(
        objects, extra_priors=discovery.get("scale_priors")
    )

    findings: list[dict[str, str]] = []
    if objects:
        sorted_objs = sorted(objects, key=lambda o: float(o["box_3d"][1]))
        near_o = sorted_objs[0]
        far_o = sorted_objs[-1]
        near_b = near_o["box_3d"]
        far_b = far_o["box_3d"]
        largest_o = max(objects, key=lambda o: float(o["box_3d"][3]) * float(o["box_3d"][4]) * float(o["box_3d"][5]))
        lb = largest_o["box_3d"]
        findings.append(
            {
                "title": "Distribución Espacial y Envolventes 9-DOF",
                "metric_evidence": (
                    f"{len(objects)} obj · {float(near_b[1]):.2f} m → {float(far_b[1]):.2f} m"
                ),
                "severity": "info",
                "interpretation": (
                    f"Elemento frontal más próximo: '{near_o['label']}' a {float(near_b[1]):.2f} m "
                    f"({float(near_b[3])*100:.1f}×{float(near_b[4])*100:.1f}×{float(near_b[5])*100:.1f} cm). "
                    f"Mayor envolvente: '{largest_o['label']}' "
                    f"({float(lb[3])*100:.1f}×{float(lb[4])*100:.1f}×{float(lb[5])*100:.1f} cm)."
                ),
            }
        )
    else:
        findings.append(
            {
                "title": "Sin Instancias 3D Segmentadas",
                "metric_evidence": "0 objetos",
                "severity": "high",
                "interpretation": "No se aislaron instancias válidas; intente especificar clases objetivo en Vocabulario / Enfoque.",
            }
        )

    if real_photo_metrics:
        ndc = real_photo_metrics.get("normal_depth_consistency") or {}
        plane = real_photo_metrics.get("ransac_plane") or {}
        ndc_score = float(ndc.get("ndc_score_0_100", 0.0) or 0.0)
        relief_mm = float((point_cloud or {}).get("max_relief_mm", 0.0) or 0.0)
        pc_cnt = int((point_cloud or {}).get("count", 0) or 0)
        tri_cnt = int(((point_cloud or {}).get("surface_mesh") or {}).get("triangle_count", 0) or 0)
        surf_sev = "info" if ndc_score >= 70.0 else ("medium" if ndc_score >= 45.0 else "high")
        plane_desc = (
            f"plano {plane.get('plane_orientation', 'detectado')} "
            f"(inliers {float(plane.get('inlier_ratio', 0.0))*100:.0f}%, tilt {float(plane.get('plane_tilt_deg', 0.0)):.1f}°)"
            if plane.get("detected")
            else "sin plano dominante RANSAC claro"
        )
        findings.append(
            {
                "title": "Coherencia Superficial MoGe-2 y Relieve 2DGS",
                "metric_evidence": f"NDC {ndc_score:.1f}/100 · Relieve {relief_mm:.0f} mm · {pc_cnt:,} surfels",
                "severity": surf_sev,
                "interpretation": (
                    f"Superficie reconstruida con {tri_cnt:,} triángulos 2.5D sobre {plane_desc}. "
                    f"Error angular normal-profundidad medio: {float(ndc.get('mean_angular_error_deg', 0.0)):.1f}°."
                ),
            }
        )

    if scale_checks:
        biased = [c for c in scale_checks if c["status"] != "consistente"]
        scale_sev = "medium" if biased else "info"
        ex = biased[0] if biased else scale_checks[0]
        findings.append(
            {
                "title": "Verificación contra Priors de Tamaño Físico",
                "metric_evidence": f"{ex['label']}: {ex['measured_max_m']:.2f} m vs ~{ex['expected_typical_m']:.2f} m ({ex['ratio']:.2f}×)",
                "severity": scale_sev,
                "interpretation": scale_note,
            }
        )

    gate = (intrinsic_diagnostics or {}).get("cascade_policy") or {}
    col_n = int(gate.get("colliding_pairs_count", 0) or 0)
    occ_n = len(occlusion_matrix)
    min_clr = gate.get("min_surface_clearance_m")
    if occ_n > 0 or col_n > 0 or min_clr is not None:
        clr_str = f"Holgura mín {float(min_clr):.3f} m" if min_clr is not None else "Sin pares"
        occ_sev = "high" if col_n > 0 else ("medium" if occ_n > 0 else "info")
        top_occ_note = ""
        if occlusion_matrix:
            t0 = occlusion_matrix[0]
            top_occ_note = (
                f" Bloqueo principal: '{t0['occluder_label']}' oculta "
                f"{float(t0['occlusion_ratio'])*100:.0f}% de '{t0['occluded_label']}' "
                f"(brecha {float(t0['depth_gap_m']):.2f} m)."
            )
        findings.append(
            {
                "title": "Oclusiones Kay-Kajiya y Holguras SAT",
                "metric_evidence": f"{clr_str} · {occ_n} oclus. · {col_n} colisiones SAT",
                "severity": occ_sev,
                "interpretation": (
                    f"Compuerta Edge→3D en veredicto '{gate.get('operational_verdict', 'N/A')}' "
                    f"(régimen {gate.get('c8286_degradation_regime', 'clean')}).{top_occ_note}"
                ),
            }
        )

    opt = (real_photo_metrics or {}).get("optical_telemetry") or (intrinsic_diagnostics or {}).get("image_profile") or {}
    spec_ratio = float(opt.get("specular_reflection_ratio", 0.0) or 0.0)
    focus_q = str(opt.get("focus_quality", "sharp"))
    if spec_ratio > 0.04 or focus_q == "blurry":
        findings.append(
            {
                "title": "Degradación Óptica y Brillo Especular",
                "metric_evidence": f"Especularidad {spec_ratio*100:.1f}% · Enfoque {focus_q}",
                "severity": "medium" if spec_ratio > 0.08 or focus_q == "blurry" else "low",
                "interpretation": (
                    "Los reflejos saturados o desenfoque reducen la precisión de profundidad monocular; "
                    "se aplicó enmascaramiento especular en el levantamiento OBB."
                ),
            }
        )

    verdict = str(gate.get("operational_verdict") or "")
    if not objects or verdict == "RECHAZADO_FISICA":
        conf_lvl: Literal["alta", "media", "baja"] = "baja"
    elif verdict == "ESCALAR_MODELO" or any(c["status"] != "consistente" for c in scale_checks):
        conf_lvl = "media"
    else:
        conf_lvl = "alta"

    depth_span_txt = (
        f"rango {float(depth_stats['min_m']):.2f}–{float(depth_stats['max_m']):.2f} m "
        f"(mediana {float(depth_stats['median_m']):.2f} m)"
        if depth_stats and float(depth_stats.get("max_m", 0.0)) > 0.01
        else "sin mapa denso de profundidad"
    )
    labels_txt = ", ".join(str(o["label"]) for o in objects[:5]) or "sin instancias"

    domain_actions_map: dict[str, list[str]] = {
        "electrical_industrial": [
            "Confirmar holgura dieléctrica / física respecto a conductores o cajas energizadas usando Corte X/Y/Z en el visor 3D.",
            "Si los cables delgados pierden continuidad, añadir términos específicos en Vocabulario/Enfoque (ej. 'insulated copper wire, terminal block').",
        ],
        "clinical_anatomical": [
            "Usar el modo Relieve (mm) y el Cortador Transversal para medir elevación topográfica sobre el plano base.",
            "Incluir un marcador físico de tamaño conocido en la escena si se requiere tolerancia subcentimétrica.",
        ],
        "desk_electronics": [
            "Alternar a vista BEV (tecla 3) para auditar holguras entre equipos y alineación respecto al plano del escritorio.",
        ],
        "logistics_packaging": [
            "Revisar volúmenes en m³ y posibles colisiones SAT antes de planificación de cubicaje o agarre robótico.",
        ],
    }
    actions = list(domain_actions_map.get(domain, []))
    actions.extend(list(profile["focus_es"][:2]))
    if any(c["status"] != "consistente" for c in scale_checks):
        actions.insert(0, "Aplicar calibración con objeto de referencia conocida debido a desviación respecto a priors físicos.")

    exec_summary = (
        f"Escena clasificada como {domain_label_es} con {len(objects)} instancia(s) medidas "
        f"({labels_txt}) en {depth_span_txt}."
    )
    model_pred = (
        f"Motor 3D proyecta envolventes métricas fundadas en superficie local y {scale_note} "
        f"Se recomienda priorizar inspección sobre los {min(len(objects), 3)} cuerpos principales "
        f"en modo Surfels / Híbrido."
    )

    return {
        "domain": domain,
        "domain_label_es": domain_label_es,
        "diagnosis_title": f"Diagnóstico Métrico 3D — {domain_label_es}",
        "executive_summary": exec_summary,
        "model_predictions_es": model_pred,
        "confidence_level": conf_lvl,
        "scale_consistency_note": scale_note,
        "key_findings": findings[:5],
        "recommended_actions": actions[:4],
        "scale_prior_checks": scale_checks,
        "diagnosis_source": "deterministic_3d_grounded",
    }


def generate_scene_diagnosis_vlm(
    *,
    api_key: str,
    image_base64: str,
    mime_type: str,
    model: str,
    discovery: dict[str, Any],
    objects: list[dict[str, Any]],
    occlusion_matrix: list[dict[str, Any]],
    intrinsic_diagnostics: Optional[dict[str, Any]] = None,
    real_photo_metrics: Optional[dict[str, Any]] = None,
    depth_stats: Optional[dict[str, float]] = None,
    point_cloud: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use Gemini VLM grounded strictly in verified 3D telemetry to predict & diagnose the scene."""
    scale_checks, scale_note = evaluate_scale_priors(
        objects, extra_priors=discovery.get("scale_priors")
    )
    telemetry = _compact_3d_telemetry_for_diagnosis(
        discovery=discovery,
        objects=objects,
        occlusion_matrix=occlusion_matrix,
        intrinsic_diagnostics=intrinsic_diagnostics,
        real_photo_metrics=real_photo_metrics,
        depth_stats=depth_stats,
        point_cloud=point_cloud,
        scale_checks=scale_checks,
        scale_note=scale_note,
    )
    clean_b64, resolved_mime = strip_data_uri(image_base64, default_mime=mime_type)
    thinking_level = resolve_thinking_level(model, None)

    prompt = (
        "Analyze the attached RGB image together with the EXACT measured 3D geometry telemetry JSON below.\n"
        "Respond in technical Spanish tailored to the detected domain. EVERY distance, dimension, relief, "
        "angle, or ratio you cite MUST come directly from the JSON telemetry (NEVER invent unseen numbers).\n"
        "In `model_predictions_es`, synthesize what is physically happening in the scene, whether the 3D "
        "metric reconstruction scale is trustworthy or biased, and what critical physical interaction or "
        "inspection conclusion follows from combining image semantics + 3D measurements.\n\n"
        f"Verified 3D Telemetry JSON:\n{json.dumps(telemetry, ensure_ascii=False)}"
    )

    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        model=model,
        system_instruction=(
            "You are MyLeenTo3's 3D-Grounded Domain Diagnostic LLM. Combine visual semantics with measured "
            "metric 3D telemetry (MoGe-2, 2DGS surfels, RANSAC plane, SAT clearance, Kay-Kajiya occlusions). "
            "Never fabricate coordinates or distances outside the supplied JSON telemetry. Return strictly JSON."
        ),
        input=[
            {"type": "text", "text": prompt},
            {"type": "image", "data": clean_b64, "mime_type": resolved_mime},
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": SceneDiagnosisPayload.model_json_schema(),
        },
        generation_config={
            "temperature": 0.2,
            "thinking_level": thinking_level,
        },
    )

    raw_text = extract_output_text(interaction)
    if not raw_text:
        raise ValueError("Empty structured response from VLM 3D-grounded diagnosis.")

    cleaned = _strip_json_fences(raw_text)
    try:
        parsed = SceneDiagnosisPayload.model_validate_json(cleaned)
    except Exception:
        parsed = SceneDiagnosisPayload.model_validate(json.loads(cleaned))

    domain_key = parsed.domain.strip().lower()
    if domain_key not in DOMAIN_PROFILES:
        domain_key = str(discovery.get("domain") or "general_physical")
        if domain_key not in DOMAIN_PROFILES:
            domain_key = "general_physical"

    diag_dict = parsed.model_dump()
    diag_dict["domain"] = domain_key
    diag_dict["domain_label_es"] = (
        parsed.domain_label_es.strip() or DOMAIN_PROFILES[domain_key]["label_es"]
    )
    diag_dict["scale_prior_checks"] = scale_checks
    if not diag_dict.get("scale_consistency_note"):
        diag_dict["scale_consistency_note"] = scale_note
    diag_dict["diagnosis_source"] = "vlm_3d_grounded"

    return diag_dict, extract_usage(interaction)


def build_scene_intelligence_payload(
    *,
    discovery: Optional[dict[str, Any]],
    objects: list[dict[str, Any]],
    occlusion_matrix: list[dict[str, Any]],
    intrinsic_diagnostics: Optional[dict[str, Any]] = None,
    real_photo_metrics: Optional[dict[str, Any]] = None,
    depth_stats: Optional[dict[str, float]] = None,
    point_cloud: Optional[dict[str, Any]] = None,
    custom_prompt: Optional[str] = None,
    image_base64: Optional[str] = None,
    mime_type: str = "image/png",
    model: str = "gemini-4-hydrogen",
    api_key: Optional[str] = None,
    allow_live_vlm: bool = False,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build complete `scene_intelligence` response section and optional token usage."""
    if discovery:
        disc = dict(discovery)
    else:
        pseudo_dets = [
            {"label": str(o.get("label", "")), "conf": float(o.get("confidence", 0.7) or 0.7)}
            for o in objects
        ]
        disc = infer_local_scene_discovery(
            pseudo_dets,
            depth_stats=depth_stats,
            custom_prompt=custom_prompt,
        )

    extra_usage = {"input_tokens": 0, "output_tokens": 0, "thought_tokens": 0, "total_tokens": 0}
    vlm_error_type: Optional[str] = None

    if allow_live_vlm and api_key and image_base64 and objects:
        try:
            diagnosis, raw_u = generate_scene_diagnosis_vlm(
                api_key=api_key,
                image_base64=image_base64,
                mime_type=mime_type,
                model=model,
                discovery=disc,
                objects=objects,
                occlusion_matrix=occlusion_matrix,
                intrinsic_diagnostics=intrinsic_diagnostics,
                real_photo_metrics=real_photo_metrics,
                depth_stats=depth_stats,
                point_cloud=point_cloud,
            )
            extra_usage = {
                "input_tokens": int(raw_u.get("input_tokens", 0)),
                "output_tokens": int(raw_u.get("output_tokens", 0)),
                "thought_tokens": int(raw_u.get("thought_tokens", 0)),
                "total_tokens": int(raw_u.get("total_tokens", 0)),
            }
        except Exception as exc:
            vlm_error_type = type(exc).__name__
            diagnosis = build_deterministic_scene_diagnosis(
                discovery=disc,
                objects=objects,
                occlusion_matrix=occlusion_matrix,
                intrinsic_diagnostics=intrinsic_diagnostics,
                real_photo_metrics=real_photo_metrics,
                depth_stats=depth_stats,
                point_cloud=point_cloud,
            )
    else:
        diagnosis = build_deterministic_scene_diagnosis(
            discovery=disc,
            objects=objects,
            occlusion_matrix=occlusion_matrix,
            intrinsic_diagnostics=intrinsic_diagnostics,
            real_photo_metrics=real_photo_metrics,
            depth_stats=depth_stats,
            point_cloud=point_cloud,
        )

    payload: dict[str, Any] = {
        "discovery_mode": str(disc.get("discovery_mode", "local_yoloe_pf_adaptive")),
        "domain": str(disc.get("domain", diagnosis["domain"])),
        "domain_label_es": str(disc.get("domain_label_es", diagnosis["domain_label_es"])),
        "scene_context_es": str(disc.get("scene_context_es", "")),
        "active_vocabulary": list(disc.get("active_vocabulary") or disc.get("detected_classes") or [])[:28],
        "prompt_free_discoveries": list(disc.get("prompt_free_discoveries") or [])[:10],
        "vlm_guided_count": int(disc.get("vlm_guided_count", 0) or 0),
        "detection_sources": list(disc.get("detection_sources") or []),
        "inspection_focus_es": list(disc.get("inspection_focus_es") or [])[:4],
        "diagnosis": diagnosis,
    }
    if vlm_error_type:
        payload["vlm_fallback_reason"] = vlm_error_type

    return payload, extra_usage
