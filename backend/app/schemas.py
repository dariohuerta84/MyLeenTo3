"""Pydantic schemas and shared API constants for MyLeenTo3."""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator

AllowedModel = Literal[
    "gemini-4-hydrogen",
    "gemini-4-argon",
    "gemini-3.5-flash-lite",
]

ThinkingLevel = Literal["low", "medium", "high"]
MoGeBackbone = Literal["vits", "vitb", "vitl"]

SUPPORTED_MODELS: list[str] = [
    "gemini-4-hydrogen",
    "gemini-4-argon",
    "gemini-3.5-flash-lite",
]

DEFAULT_THINKING_LEVELS: dict[str, str] = {
    "gemini-4-hydrogen": "high",
    "gemini-4-argon": "medium",
    "gemini-3.5-flash-lite": "low",
}


class RawDetection(BaseModel):
    """Single 9-DOF 3D object detection returned by Gemini Structured Output."""

    label: str = Field(..., description="Semantic class name in concise lowercase/English or Spanish.")
    box_3d: list[float] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="[x, y, z, dx, dy, dz, roll, pitch, yaw] in meters and degrees (Gemini X-Right, Y-Forward, Z-Up).",
    )
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("box_3d")
    @classmethod
    def validate_box_3d(cls, v: list[float]) -> list[float]:
        if len(v) != 9:
            raise ValueError("box_3d must have exactly 9 numeric values.")
        x, y, z, dx, dy, dz, roll, pitch, yaw = [float(val) for val in v]
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError("box_3d dimensions (dx, dy, dz) must be strictly positive.")
        return [x, y, z, dx, dy, dz, roll, pitch, yaw]


class RawDetectionList(BaseModel):
    """Structured Output schema passed to Gemini Interactions API."""

    objects: list[RawDetection] = Field(default_factory=list)
    scene_summary: str = Field(
        default="",
        description="Brief metric summary of the 3D scene layout and dominant depth planes.",
    )


class ObjectInput(BaseModel):
    """Input 3D object pose accepted by reproject, measure, and copilot endpoints."""

    id: str
    label: str
    box_3d: list[float] = Field(..., min_length=9, max_length=9)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    box_2d: Optional[list[float]] = None

    @field_validator("box_3d")
    @classmethod
    def validate_box_3d(cls, v: list[float]) -> list[float]:
        if len(v) != 9:
            raise ValueError("box_3d must contain 9 floats: [x, y, z, dx, dy, dz, roll, pitch, yaw].")
        vals = [float(val) for val in v]
        if vals[3] <= 0 or vals[4] <= 0 or vals[5] <= 0:
            raise ValueError("Dimensions dx, dy, dz must be strictly positive.")
        return vals


class EnrichedObject(BaseModel):
    """Fully projected 3D object with Three.js pose, pinhole projection, volume, and depth."""

    id: str
    label: str
    box_3d: list[float]
    confidence: float = 0.9
    volume_m3: float
    distance_m: float
    projected_2d: dict[str, Any]
    threejs_pose: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    gemini_env_configured: bool
    supported_models: list[str]
    sdk_version: str
    local_metric_3d_available: bool = False
    local_3d_device: str = "disabled"
    local_3d_models: list[str] = Field(default_factory=list)


class BenchmarkSummary(BaseModel):
    id: str
    title: str
    subtitle: str
    description: str
    width: int
    height: int
    fov_h_deg: float
    image_base64: str
    gt_objects: list[dict[str, Any]]


class BenchmarksResponse(BaseModel):
    benchmarks: list[BenchmarkSummary]


class AnalyzeRequest(BaseModel):
    benchmark_id: Optional[str] = None
    image_base64: Optional[str] = None
    mime_type: str = "image/png"
    width: Optional[int] = Field(default=None, gt=0)
    height: Optional[int] = Field(default=None, gt=0)
    model: AllowedModel = "gemini-4-hydrogen"
    thinking_level: Optional[ThinkingLevel] = None
    fov_h_deg: float = Field(default=60.0, ge=20.0, le=140.0)
    max_objects: int = Field(default=10, ge=1, le=30)
    custom_prompt: Optional[str] = None
    force_demo: bool = False
    engine: Literal["auto", "hybrid_vlm_3d", "moge2_local", "gemini_live", "salience_fallback"] = "auto"
    auto_fov: bool = True
    enable_diagnosis: bool = True
    moge_backbone: MoGeBackbone = "vits"
    aruco_marker_size_cm: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    enable_qubo_nms: bool = True


class AnalyzeResponse(BaseModel):
    benchmark_id: Optional[str] = None
    model: str
    thinking_level: str
    mode: str
    notice: Optional[str] = None
    width: int
    height: int
    fov_h_deg: float
    fov_v_deg: float
    fov_source: Optional[str] = None
    exif_fov_h_deg: Optional[float] = None
    estimated_moge_fov_h_deg: Optional[float] = None
    moge_backbone: Optional[str] = None
    scale_factor: float = 1.0
    aruco_anchor: Optional[dict[str, Any]] = None
    image_base64: str
    mime_type: str
    scene_summary: str = ""
    objects: list[EnrichedObject]
    gt_objects: Optional[list[EnrichedObject]] = None
    occlusion_matrix: list[dict[str, Any]]
    metrics: Optional[dict[str, Any]] = None
    fov_sweep: Optional[list[dict[str, Any]]] = None
    intrinsic_diagnostics: Optional[dict[str, Any]] = None
    depth_map_base64: Optional[str] = None
    depth_stats: Optional[dict[str, float]] = None
    point_cloud: Optional[dict[str, Any]] = None
    real_photo_metrics: Optional[dict[str, Any]] = None
    scene_intelligence: Optional[dict[str, Any]] = None
    quantum_nms_summary: Optional[dict[str, Any]] = None
    latency_ms: float
    usage: dict[str, Any]


class ReprojectRequest(BaseModel):
    objects: list[ObjectInput]
    gt_objects: Optional[list[ObjectInput]] = None
    image_base64: Optional[str] = None
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fov_h_deg: float = Field(..., ge=20.0, le=140.0)
    reference_fov_h_deg: Optional[float] = Field(default=None, ge=20.0, le=140.0)
    rescale_depth_with_fov: bool = False


class ReprojectResponse(BaseModel):
    width: int
    height: int
    fov_h_deg: float
    fov_v_deg: float
    objects: list[EnrichedObject]
    gt_objects: Optional[list[EnrichedObject]] = None
    occlusion_matrix: list[dict[str, Any]]
    metrics: Optional[dict[str, Any]] = None
    fov_sweep: Optional[list[dict[str, Any]]] = None
    intrinsic_diagnostics: Optional[dict[str, Any]] = None
    latency_ms: float


class CalibrateScaleRequest(BaseModel):
    objects: list[ObjectInput]
    gt_objects: Optional[list[ObjectInput]] = None
    image_base64: Optional[str] = None
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    fov_h_deg: float = Field(..., ge=20.0, le=140.0)
    scale_factor: Optional[float] = Field(default=None, gt=0.0, le=20.0)
    measured_distance_m: Optional[float] = Field(default=None, gt=0.0)
    reference_distance_m: Optional[float] = Field(default=None, gt=0.0)
    reference_label: Optional[str] = None
    depth_stats: Optional[dict[str, float]] = None
    real_photo_metrics: Optional[dict[str, Any]] = None
    scene_intelligence: Optional[dict[str, Any]] = None


class CalibrateScaleResponse(BaseModel):
    scale_factor: float
    reference_label: Optional[str] = None
    width: int
    height: int
    fov_h_deg: float
    fov_v_deg: float
    objects: list[EnrichedObject]
    gt_objects: Optional[list[EnrichedObject]] = None
    occlusion_matrix: list[dict[str, Any]]
    metrics: Optional[dict[str, Any]] = None
    intrinsic_diagnostics: Optional[dict[str, Any]] = None
    depth_stats: Optional[dict[str, float]] = None
    real_photo_metrics: Optional[dict[str, Any]] = None
    scene_intelligence: Optional[dict[str, Any]] = None
    latency_ms: float


class MeasureRequest(BaseModel):
    object_a: ObjectInput
    object_b: ObjectInput


class MeasureResponse(BaseModel):
    object_a_id: str
    object_a_label: str
    object_b_id: str
    object_b_label: str
    center_distance_m: float
    surface_clearance_m: float
    is_colliding: bool
    iou_3d: float
    iou_bev: float
    closest_point_a_three: list[float]
    closest_point_b_three: list[float]
    centroid_delta_m: list[float]
    ruler_endpoints_three: dict[str, list[float]]


class CompareModelsRequest(BaseModel):
    benchmark_id: str = "lab_desk_calibrated"
    fov_h_deg: float = Field(default=60.0, ge=20.0, le=140.0)
    use_live: bool = False


class ModelComparisonEntry(BaseModel):
    model: str
    thinking_level: str
    mode: str
    notice: Optional[str] = None
    latency_ms: float
    usage: dict[str, Any]
    object_count: int
    metrics: dict[str, Any]
    fov_sweep: list[dict[str, Any]]


class CompareModelsResponse(BaseModel):
    benchmark_id: str
    fov_h_deg: float
    mode: str
    notice: Optional[str] = None
    results: list[ModelComparisonEntry]
    fov_sweep: list[dict[str, Any]]


class MultiFrameAuditRequest(BaseModel):
    sequence_id: Optional[str] = None
    frames: list[dict[str, Any]] = Field(..., min_length=1)
    use_live: bool = False
    model: AllowedModel = "gemini-4-hydrogen"
    fov_h_deg: float = Field(default=60.0, ge=20.0, le=140.0)


class ToolTraceEntry(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class CopilotRequest(BaseModel):
    message: str = Field(..., min_length=1)
    objects: list[ObjectInput] = Field(default_factory=list)
    gt_objects: Optional[list[ObjectInput]] = None
    image_base64: Optional[str] = None
    width: int = Field(default=960, gt=0)
    height: int = Field(default=640, gt=0)
    fov_h_deg: float = Field(default=60.0, ge=20.0, le=140.0)
    model: AllowedModel = "gemini-4-hydrogen"
    previous_interaction_id: Optional[str] = None
    real_photo_metrics: Optional[dict[str, Any]] = None
    scene_intelligence: Optional[dict[str, Any]] = None


class CopilotResponse(BaseModel):
    reply: str
    mode: str
    notice: Optional[str] = None
    model: str
    interaction_id: Optional[str] = None
    tool_trace: list[ToolTraceEntry] = Field(default_factory=list)
    scene_actions: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float
    usage: dict[str, Any] = Field(default_factory=dict)


class MultiViewInputImage(BaseModel):
    image_base64: str = Field(..., min_length=16)
    mime_type: str = "image/png"
    label: Optional[str] = None
    yaw_hint_deg: Optional[float] = Field(default=None, ge=-360.0, le=360.0)


class MultiViewReconstructRequest(BaseModel):
    images: list[MultiViewInputImage] = Field(default_factory=list)
    use_demo: bool = False
    moge_backbone: MoGeBackbone = "vits"
    fov_h_deg: float = Field(default=60.0, ge=20.0, le=140.0)
    auto_fov: bool = True
    allow_scale_alignment: bool = True
    voxel_size_mm: Optional[float] = Field(default=None, ge=2.0, le=50.0)
    aruco_marker_size_cm: Optional[float] = Field(default=None, ge=1.0, le=100.0)
    use_qubo_optimizer: bool = True


class MultiViewCameraPose(BaseModel):
    view_index: int
    label: str
    center_three: list[float]
    quaternion_three: list[float]
    forward_three: list[float]
    up_three: list[float]
    azimuth_deg: float
    fov_h_deg: float
    registration_method: str
    inliers: int
    alignment_rmse_m: float
    overlap_ratio: float
    estimated_scale: float
    color: str
    thumbnail_base64: Optional[str] = None


class MultiViewCoverage360(BaseModel):
    sector_scores: list[float]
    angular_coverage_pct: float
    largest_blind_gap_deg: float
    recommended_next_azimuth_deg: float
    is_watertight_360: bool
    next_best_view_advice_es: str
    qubo_nbv_plan: Optional[dict[str, Any]] = None


class MultiViewBodyMetrology(BaseModel):
    convex_hull_volume_m3: float
    convex_hull_surface_area_m2: float
    voxel_solid_volume_m3: float
    voxel_shell_volume_m3: float
    voxel_resolution_mm: float
    occupied_voxels: int
    bounding_dimensions_m: list[float]
    fused_center_gemini: list[float]
    fused_box_3d: list[float]
    mean_icp_rmse_mm: float
    single_view_obb_volume_m3: Optional[float] = None
    gt_volume_m3: Optional[float] = None
    gt_volume_error_pct: Optional[float] = None


class MultiViewReconstructResponse(BaseModel):
    mode: str
    notice: Optional[str] = None
    moge_backbone: str = "vits"
    view_count: int
    width: int
    height: int
    fov_h_deg: float
    fov_v_deg: float
    scale_factor: float = 1.0
    aruco_anchor: Optional[dict[str, Any]] = None
    camera_poses: list[MultiViewCameraPose]
    coverage_360: MultiViewCoverage360
    body_metrology: MultiViewBodyMetrology
    objects: list[EnrichedObject] = Field(default_factory=list)
    point_cloud: dict[str, Any]
    depth_stats: Optional[dict[str, float]] = None
    quantum_telemetry: Optional[dict[str, Any]] = None
    latency_ms: float


QuantumTaskType = Literal["registration_stress", "nms_3d", "nbv_plan"]


class QuantumOptimizeRequest(BaseModel):
    task: QuantumTaskType = "registration_stress"
    n_correspondences: int = Field(default=72, ge=12, le=140)
    outlier_ratio: float = Field(default=0.75, ge=0.0, le=0.90)
    noise_mm: float = Field(default=4.0, ge=0.0, le=30.0)
    allow_scale: bool = True
    objects: Optional[list[ObjectInput]] = None
    conflict_iou_threshold: float = Field(default=0.28, ge=0.05, le=0.80)
    existing_azimuths_deg: Optional[list[float]] = None
    sector_scores: Optional[list[float]] = None
    budget_k: Optional[int] = Field(default=None, ge=1, le=6)
    seed: int = Field(default=42, ge=0, le=100000)


class QuantumOptimizeResponse(BaseModel):
    task: str
    title_es: str
    summary_es: str
    qubo_stats: dict[str, Any]
    baseline_comparison: dict[str, Any]
    energy_history: list[dict[str, Any]]
    graph_preview: dict[str, Any]
    details: dict[str, Any]
    hardware_export: dict[str, Any]
    latency_ms: float


class QuantumDemoResponse(BaseModel):
    seed: int
    engine_version: str
    references: list[str]
    scenarios: dict[str, QuantumOptimizeResponse]
    latency_ms: float
