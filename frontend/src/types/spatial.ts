export type GeminiModelId =
  | 'gemini-4-hydrogen'
  | 'gemini-4-argon'
  | 'gemini-3.5-flash-lite';

export type ThinkingLevel = 'low' | 'medium' | 'high';

export type CameraViewMode = 'camera_aligned' | 'orbit' | 'bev' | 'isometric';

export type AppSection = 'landing' | 'lab' | 'multiframe' | 'metrics';

export interface Projected2D {
  vertices_2d: [number, number][];
  vertices_norm: [number, number][];
  edges: [number, number][];
  center_2d: [number, number];
  label_anchor_2d: [number, number];
  front_arrow_2d: [[number, number], [number, number]];
  box_2d_reproj: [number, number, number, number];
  behind_camera: boolean;
  min_depth_m: number;
  max_depth_m: number;
}

export interface ThreeJSPose {
  position: [number, number, number];
  quaternion: [number, number, number, number];
  size: [number, number, number];
  corners_three: [number, number, number][];
  forward_dir_three: [number, number, number];
}

export interface SpatialObject {
  id: string;
  label: string;
  box_3d: [number, number, number, number, number, number, number, number, number];
  confidence?: number;
  box_2d?: [number, number, number, number];
  projected_2d?: Projected2D;
  threejs_pose?: ThreeJSPose;
  volume_m3?: number;
  distance_m?: number;
}

export interface OcclusionRayThree {
  origin: [number, number, number];
  hit: [number, number, number];
  target: [number, number, number];
}

export interface OcclusionEntry {
  occluder_id: string;
  occluder_label: string;
  occluded_id: string;
  occluded_label: string;
  occlusion_ratio: number;
  depth_gap_m: number;
  occluded_rays_three: OcclusionRayThree[];
}

export interface PerObjectMatch {
  pred_id: string;
  gt_id: string;
  label: string;
  iou_3d: number;
  iou_bev: number;
  ate_m: number;
  aoe_deg: number;
  reproj_iou_2d: number;
  centroid_error_vec_three: [number, number, number];
}

export interface BootstrapCI95 {
  miou_3d_low: number;
  miou_3d_high: number;
  ate_low: number;
  ate_high: number;
  n_boot: number;
}

export interface SceneMetrics {
  mIoU_3D: number;
  mIoU_BEV: number;
  ap_25: number;
  ap_50: number;
  ate_m: number;
  ase: number;
  aoe_deg: number;
  reproj_iou_2d: number;
  matched_count: number;
  gt_count: number;
  pred_count: number;
  bootstrap_ci_95: BootstrapCI95;
  per_object_matches: PerObjectMatch[];
}

export interface FovSweepPoint {
  fov_h_deg: number;
  reproj_iou_2d: number;
  mean_pixel_error: number;
}

export interface ImagePhotometricProfile {
  luminance_mean: number;
  luminance_std: number;
  dark_shadow_ratio: number;
  bloom_saturation_ratio: number;
  spectral_entropy_bits: number;
  perspective_coherence_score: number;
  domain_classification:
    | 'calibrated_studio'
    | 'natural_indoor'
    | 'high_contrast_stylized'
    | 'low_light_high_stress'
    | string;
}

export interface LatentIntrinsicMetrics {
  physical_plausibility_score: number;
  photometric_edge_coherence: number;
  volume_interpenetration_index: number;
  gravity_floating_anomaly_m: number;
  support_plane_z_m: number;
  scale_depth_ambiguity_score: number;
  frustum_truncation_ratio: number;
  mean_depth_uncertainty_m: number;
}

export interface PerObjectIntrinsicDiagnostic {
  object_id: string;
  label: string;
  depth_m: number;
  uncertainty_depth_m: number;
  edge_alignment_score: number;
  floating_gap_m: number;
  anisotropy_ratio: number;
  frustum_truncation_ratio: number;
  occlusion_severity: number;
  vulnerability_score: number;
  dominant_weakness: string;
}

export interface ExploitedWeaknessEntry {
  id: string;
  title: string;
  severity: 'low' | 'medium' | 'high' | string;
  metric_value: string;
  explanation: string;
}

export interface CascadePolicy {
  operational_verdict: 'APROBADO' | 'ESCALAR_MODELO' | 'RECHAZADO_FISICA' | string;
  recommended_tier: string;
  recommended_model: string;
  c8286_degradation_regime: 'clean' | 'occlusion_bg' | 'illumination_noise' | 'hybrid_stress' | string;
  estimated_latency_savings_pct: number;
  min_surface_clearance_m: number | null;
  occlusion_pairs_count: number;
  colliding_pairs_count: number;
  trigger_reasons: string[];
  summary: string;
}

export interface IntrinsicDiagnostics {
  image_profile: ImagePhotometricProfile;
  latent_metrics: LatentIntrinsicMetrics;
  per_object_diagnostics: PerObjectIntrinsicDiagnostic[];
  exploited_weaknesses: ExploitedWeaknessEntry[];
  reference_free_fov_sweep: FovSweepPoint[];
  estimated_optimal_fov_h_deg: number;
  cascade_policy?: CascadePolicy;
}

export interface BenchmarkScene {
  id: string;
  title: string;
  subtitle: string;
  description: string;
  width: number;
  height: number;
  fov_h_deg: number;
  image_base64: string;
  gt_objects: SpatialObject[];
}

export interface HealthResponse {
  status: string;
  gemini_env_configured: boolean;
  supported_models: GeminiModelId[];
  sdk_version?: string;
}

export interface AnalyzeResponse {
  benchmark_id?: string | null;
  model: GeminiModelId;
  thinking_level?: ThinkingLevel | string;
  width: number;
  height: number;
  fov_h_deg: number;
  fov_v_deg: number;
  objects: SpatialObject[];
  gt_objects?: SpatialObject[] | null;
  occlusion_matrix: OcclusionEntry[];
  metrics?: SceneMetrics | null;
  fov_sweep?: FovSweepPoint[] | null;
  intrinsic_diagnostics?: IntrinsicDiagnostics | null;
  latency_ms: number;
  usage?: Record<string, unknown> | null;
  mode: string;
  notice?: string | null;
}

export interface ReprojectResponse {
  width: number;
  height: number;
  fov_h_deg: number;
  fov_v_deg: number;
  objects: SpatialObject[];
  gt_objects?: SpatialObject[] | null;
  occlusion_matrix: OcclusionEntry[];
  metrics?: SceneMetrics | null;
  fov_sweep?: FovSweepPoint[] | null;
  intrinsic_diagnostics?: IntrinsicDiagnostics | null;
}

export interface PairwiseGeometry {
  center_distance_m: number;
  surface_clearance_m: number;
  is_colliding: boolean;
  iou_3d: number;
  iou_bev: number;
  closest_point_a_three: [number, number, number];
  closest_point_b_three: [number, number, number];
  centroid_delta_m: [number, number, number];
}

export interface MeasurementState {
  object_a_id: string;
  object_a_label: string;
  object_b_id: string;
  object_b_label: string;
  geometry: PairwiseGeometry;
}

export interface ModelComparisonRow {
  model: GeminiModelId | string;
  thinking_level?: string;
  mode?: string;
  latency_ms: number;
  metrics: SceneMetrics;
  object_count?: number;
}

export interface CompareModelsResponse {
  benchmark_id: string;
  fov_h_deg: number;
  results: ModelComparisonRow[];
  fov_sweep?: FovSweepPoint[];
}

export interface MultiFrameItem {
  frame_idx: number;
  timestamp_s: number;
  camera_note: string;
  image_base64?: string;
  width?: number;
  height?: number;
  fov_h_deg?: number;
  objects: SpatialObject[];
}

export interface ObjectTrackAudit {
  label: string;
  frame_indices: number[];
  volumes_m3: number[];
  scale_drift_pct: number;
  volume_cv_pct: number;
  yaw_jitter_deg: number;
  centroid_step_m: number;
}

export interface MultiFrameAuditResult {
  sequence_id: string;
  frame_count: number;
  geometric_stability_score: number;
  mean_scale_drift_pct: number;
  mean_yaw_jitter_deg: number;
  object_tracks: ObjectTrackAudit[];
  per_frame_summary: Array<{
    frame_idx: number;
    timestamp_s?: number;
    object_count: number;
    mean_depth_m?: number;
    total_volume_m3?: number;
  }>;
  drift_diagnosis: string;
}

export interface MultiFrameDemoResponse {
  sequence_id: string;
  title?: string;
  description?: string;
  width: number;
  height: number;
  fov_h_deg: number;
  frames: MultiFrameItem[];
  audit: MultiFrameAuditResult;
}

export interface CopilotToolTrace {
  name: string;
  tool_name?: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
}

export type SceneAction =
  | { type: 'focus_object'; object_id: string }
  | {
      type: 'show_measurement';
      object_a_id: string;
      object_b_id: string;
      data?: PairwiseGeometry;
    }
  | { type: 'set_camera_view'; view_mode: CameraViewMode }
  | { type: 'toggle_occlusion_rays'; enabled: boolean }
  | { type: 'highlight_subset'; object_ids: string[] }
  | { type: 'open_tab'; tab: 'lab' | 'metrics' | 'multiframe' };

export interface CopilotResponse {
  reply: string;
  mode: string;
  model?: string;
  interaction_id?: string | null;
  tool_trace: CopilotToolTrace[];
  scene_actions: SceneAction[];
  notice?: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  mode?: string;
  tool_trace?: CopilotToolTrace[];
  scene_actions?: SceneAction[];
  timestamp: string;
}
