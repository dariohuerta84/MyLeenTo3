"""Core 3D computer vision, metrics, benchmarks, and multi-frame audit package."""

from .geometry_3d import (
    BASIS_GEMINI_TO_THREE,
    OBB_EDGES,
    R_VIEW_GEMINI_TO_OPENCV,
    compute_3d_iou,
    compute_bev_iou,
    compute_box_2d_iou,
    compute_fov_v_deg,
    compute_occlusion_matrix,
    compute_pairwise_geometry,
    euler_deg_to_quaternion,
    gemini_to_threejs_pose,
    obb_corners_gemini,
    obb_intersects_sat,
    project_obb_to_2d,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from .intrinsic_diagnostics import (
    compute_intrinsic_scene_diagnostics,
    extract_image_adaptive_3d_hypotheses,
)
from .metrics import (
    DEFAULT_FOV_SWEEP_VALUES,
    evaluate_scene_predictions,
    fov_sensitivity_sweep,
)
from .multiframe_audit import (
    audit_multiframe_consistency,
    get_demo_multiframe_sequence,
)
from .synthetic_benchmark import (
    DEFAULT_FOV_H_DEG,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    SEED,
    get_benchmark,
    list_benchmarks,
    render_scene_png_base64,
)

__all__ = [
    "BASIS_GEMINI_TO_THREE",
    "DEFAULT_FOV_H_DEG",
    "DEFAULT_FOV_SWEEP_VALUES",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "OBB_EDGES",
    "R_VIEW_GEMINI_TO_OPENCV",
    "SEED",
    "audit_multiframe_consistency",
    "compute_3d_iou",
    "compute_bev_iou",
    "compute_box_2d_iou",
    "compute_fov_v_deg",
    "compute_intrinsic_scene_diagnostics",
    "compute_occlusion_matrix",
    "compute_pairwise_geometry",
    "euler_deg_to_quaternion",
    "evaluate_scene_predictions",
    "extract_image_adaptive_3d_hypotheses",
    "fov_sensitivity_sweep",
    "gemini_to_threejs_pose",
    "get_benchmark",
    "get_demo_multiframe_sequence",
    "list_benchmarks",
    "obb_corners_gemini",
    "obb_intersects_sat",
    "project_obb_to_2d",
    "quaternion_to_rotation_matrix",
    "render_scene_png_base64",
    "rotation_matrix_to_quaternion",
]
