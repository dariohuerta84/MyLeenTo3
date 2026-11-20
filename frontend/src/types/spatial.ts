export type GeminiModelId =
  | 'gemini-4-hydrogen'
  | 'gemini-4-argon'
  | 'gemini-3.5-flash-lite';

export type ThinkingLevel = 'low' | 'medium' | 'high';

export type CameraViewMode = 'camera_aligned' | 'orbit' | 'bev' | 'isometric';

export type AppSection = 'landing' | 'lab' | 'multiview' | 'quantum' | 'multiframe' | 'metrics';

export type AnalyzeEngine =
  | 'auto'
  | 'hybrid_vlm_3d'
  | 'moge2_local'
  | 'gemini_live'
  | 'salience_fallback';

export type MoGeBackbone = 'vits' | 'vitb' | 'vitl';

export interface ArucoAnchor {
  detected: boolean;
  dictionary: string;
  marker_id: number;
  corners_px: [number, number][];
  measured_side_m: number;
  reference_side_m: number;
  suggested_scale_factor: number;
}

export interface ScalePriorCheck {
  object_id: string;
  label: string;
  measured_max_m: number;
  expected_typical_m: number;
  ratio: number;
  status: 'consistente' | 'subestimada' | 'sobreestimada' | string;
  source?: string;
}

export interface SceneDiagnosisFinding {
  title: string;
  metric_evidence: string;
  severity: 'info' | 'low' | 'medium' | 'high' | string;
  interpretation: string;
}

export interface SceneDiagnosis {
  domain: string;
  domain_label_es: string;
  diagnosis_title: string;
  executive_summary: string;
  model_predictions_es: string;
  confidence_level: 'alta' | 'media' | 'baja' | string;
  scale_consistency_note: string;
  key_findings: SceneDiagnosisFinding[];
  recommended_actions: string[];
  scale_prior_checks?: ScalePriorCheck[];
  diagnosis_source?: string;
}

export interface SceneIntelligence {
  discovery_mode: string;
  domain: string;
  domain_label_es: string;
  scene_context_es: string;
  active_vocabulary: string[];
  prompt_free_discoveries?: Array<{
    label: string;
    confidence: number;
    box_2d?: number[] | null;
  }>;
  vlm_guided_count?: number;
  detection_sources?: string[];
  inspection_focus_es?: string[];
  vlm_fallback_reason?: string;
  diagnosis: SceneDiagnosis;
}

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

export interface DepthBinMetric {
  bin: string;
  label: string;
  gt_count: number;
  matched_count: number;
  mIoU_3D: number;
  ate_m: number;
}

export interface SceneMetrics {
  mIoU_3D: number;
  mIoU_BEV: number;
  ap_25: number;
  ap_50: number;
  nds_static?: number;
  ate_m: number;
  ase: number;
  aoe_deg: number;
  reproj_iou_2d: number;
  matched_count: number;
  gt_count: number;
  pred_count: number;
  depth_bins?: DepthBinMetric[];
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
  laplacian_variance?: number;
  focus_quality?: 'sharp' | 'moderate' | 'blurry' | string;
  noise_sigma_est?: number;
  dynamic_contrast_ratio?: number;
  equivalent_focal_35mm_mm?: number;
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
  support_plane_mode?: string;
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

export interface SceneGraphRelation {
  from_id: string;
  from_label: string;
  to_id: string;
  to_label: string;
  relation_horizontal: string;
  relation_depth: string;
  center_distance_m: number;
  surface_clearance_m: number;
  is_colliding: boolean;
  occlusion_ratio: number;
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
  viewpoint_ambiguity_score?: number;
  next_best_view_advice?: string;
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
  scene_graph_3d?: SceneGraphRelation[];
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

export interface DepthStats {
  min_m: number;
  p10_m: number;
  median_m: number;
  p90_m: number;
  max_m: number;
}

export interface SurfaceMeshData {
  vertex_count: number;
  triangle_count: number;
  positions_b64: string;
  colors_b64: string;
  normals_b64: string;
  indices_b64: string;
}

export interface PointCloudData {
  count: number;
  positions_b64: string;
  colors_b64: string;
  normals_b64?: string;
  object_indices_b64?: string;
  relief_cm_b64?: string;
  object_id_palette?: string[];
  has_normals?: boolean;
  body_point_count?: number;
  max_relief_mm?: number;
  median_depth_m: number;
  bounds_three?: {
    min: [number, number, number];
    max: [number, number, number];
  };
  surface_mesh?: SurfaceMeshData | null;
}

export interface PerObjectPointCloudMetrology {
  object_id: string;
  label: string;
  point_count: number;
  surface_fit_rmse_m: number;
  fill_solidity_ratio: number;
  visible_surface_area_m2: number;
  depth_iqr_m: number;
  angular_span_deg: [number, number];
  normal_consistency_deg: number;
  mean_relief_height_m?: number;
  max_relief_height_m?: number;
  plane_contact_gap_m?: number;
  integrated_volume_above_plane_m3?: number;
  specular_ratio_in_mask?: number;
}

export interface RealPhotoMetrics {
  normal_depth_consistency: {
    has_model_normals: boolean;
    mean_angular_error_deg: number;
    median_angular_error_deg: number;
    consistent_ratio_15deg: number;
    ndc_score_0_100: number;
    manhattan_alignment_score: number;
  };
  manhattan_alignment_score: number;
  ransac_plane: {
    detected: boolean;
    inlier_ratio: number;
    rmse_residual_m: number;
    plane_tilt_deg: number;
    plane_height_z_m: number;
    plane_d_gemini?: number;
    plane_normal_gemini: [number, number, number];
    plane_orientation: string;
  };
  depth_boundaries: {
    depth_rgb_boundary_f1: number;
    depth_pro_multiscale_f1?: number;
    silhouette_boundary_recall?: number | null;
    flying_pixel_ratio: number;
    depth_smoothness_index: number;
  };
  depth_profile: {
    depth_dynamic_range_ratio: number;
    foreground_ratio: number;
    metric_scale_regime: string;
    scene_span_xyz_m: [number, number, number];
  };
  optical_telemetry: {
    laplacian_variance: number;
    focus_quality: string;
    noise_sigma_est: number;
    dynamic_contrast_ratio: number;
    equivalent_focal_35mm_mm: number;
    specular_reflection_ratio?: number;
  };
  free_space_consistency?: {
    free_space_violation_ratio: number;
    mean_depth_residual_m: number;
    consistent_objects: number;
    total_objects: number;
  };
  fov_exif_vs_moge_diff_deg?: number | null;
  per_object_metrology: PerObjectPointCloudMetrology[];
  overall_3d_geometry_score_0_100: number;
}

export interface HealthResponse {
  status: string;
  gemini_env_configured: boolean;
  supported_models: GeminiModelId[];
  sdk_version?: string;
  local_metric_3d_available?: boolean;
  local_3d_device?: string;
  local_3d_models?: string[];
}

export interface AnalyzeResponse {
  benchmark_id?: string | null;
  model: GeminiModelId;
  thinking_level?: ThinkingLevel | string;
  width: number;
  height: number;
  fov_h_deg: number;
  fov_v_deg: number;
  fov_source?: string | null;
  exif_fov_h_deg?: number | null;
  estimated_moge_fov_h_deg?: number | null;
  moge_backbone?: MoGeBackbone | string | null;
  scale_factor?: number;
  aruco_anchor?: ArucoAnchor | null;
  image_base64?: string;
  mime_type?: string;
  objects: SpatialObject[];
  gt_objects?: SpatialObject[] | null;
  occlusion_matrix: OcclusionEntry[];
  metrics?: SceneMetrics | null;
  fov_sweep?: FovSweepPoint[] | null;
  intrinsic_diagnostics?: IntrinsicDiagnostics | null;
  depth_map_base64?: string | null;
  depth_stats?: DepthStats | null;
  point_cloud?: PointCloudData | null;
  real_photo_metrics?: RealPhotoMetrics | null;
  scene_intelligence?: SceneIntelligence | null;
  quantum_nms_summary?: QuantumNmsSummary | null;
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

export interface CalibrateScaleResponse {
  scale_factor: number;
  reference_label?: string | null;
  width: number;
  height: number;
  fov_h_deg: number;
  fov_v_deg: number;
  objects: SpatialObject[];
  gt_objects?: SpatialObject[] | null;
  occlusion_matrix: OcclusionEntry[];
  metrics?: SceneMetrics | null;
  intrinsic_diagnostics?: IntrinsicDiagnostics | null;
  depth_stats?: DepthStats | null;
  real_photo_metrics?: RealPhotoMetrics | null;
  scene_intelligence?: SceneIntelligence | null;
  quantum_nms_summary?: QuantumNmsSummary | null;
  latency_ms: number;
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

export interface MultiViewInputImage {
  image_base64: string;
  mime_type?: string;
  label?: string;
  yaw_hint_deg?: number | null;
}

export interface MultiViewCameraPose {
  view_index: number;
  label: string;
  center_three: [number, number, number];
  quaternion_three: [number, number, number, number];
  forward_three: [number, number, number];
  up_three: [number, number, number];
  azimuth_deg: number;
  fov_h_deg: number;
  registration_method: string;
  inliers: number;
  alignment_rmse_m: number;
  overlap_ratio: number;
  estimated_scale: number;
  color: string;
  thumbnail_base64?: string | null;
}

export interface QuantumNmsSummary {
  enabled: boolean;
  input_count: number;
  kept_count: number;
  suppressed_count: number;
  conflict_edges_count: number;
  energy_gain: number;
  solver_ms: number;
  exact_optimum_verified: boolean;
  suppressed_labels: string[];
}

export interface QuboNbvPlan {
  budget_k: number;
  optimal_shot_plan_deg: number[];
  optimal_sector_indices: number[];
  current_coverage_pct: number;
  predicted_coverage_after_plan_pct: number;
  predicted_sector_scores: number[];
  shots_saved_vs_uniform_6: number;
  shadow_uncertainty_by_sector: number[];
  qubo_energy: number;
  greedy_energy: number;
  energy_gain: number;
  greedy_plan_deg: number[];
  greedy_predicted_coverage_pct: number;
  solver_ms: number;
  energy_history: Array<{ step: number; best_energy: number; mean_energy: number }>;
  advice_es: string;
}

export interface QuEraAtomNode {
  id: number;
  label: string;
  x_um: number;
  y_um: number;
  selected: boolean;
}

export interface QuantumHardwareExport {
  bqm_braket: {
    format: string;
    vartype: string;
    problem_name: string;
    num_variables: number;
    num_quadratic_couplings: number;
    offset: number;
    linear: Record<string, number>;
    quadratic: Record<string, number>;
    compatible_backends: string[];
  };
  quera_aquila_register: {
    device: string;
    reference: string;
    rydberg_blockade_radius_um: number;
    min_atom_spacing_um: number;
    observed_min_spacing_um: number;
    field_of_view_um: [number, number];
    max_physical_qubits: number;
    embedded_nodes: number;
    total_problem_nodes: number;
    unit_disk_fidelity: number;
    non_unit_disk_violations: number;
    nguyen_2023_gadget_qubits_est: number;
    total_estimated_qubits: number;
    hardware_ready: boolean;
    atoms_um: QuEraAtomNode[];
  };
}

export interface QuantumTelemetrySummary {
  enabled: boolean;
  solver_name: string;
  views_optimized_by_qubo: number;
  total_qubo_vars: number;
  total_conflict_edges: number;
  total_qubo_inliers: number;
  total_ransac_inliers: number;
  mean_energy_gain: number;
  total_solver_ms: number;
  per_view_pairs: Array<{
    view_index: number;
    anchor_index: number;
    method_won: string;
    n_vars: number;
    n_conflict_edges: number;
    qubo_inliers: number;
    ransac_inliers: number;
    energy_gain: number;
    solver_ms: number;
  }>;
  hardware_export?: QuantumHardwareExport | null;
}

export type QuantumTaskType = 'registration_stress' | 'nms_3d' | 'nbv_plan';

export interface QuantumOptimizeResponse {
  task: QuantumTaskType | string;
  title_es: string;
  summary_es: string;
  qubo_stats: Record<string, unknown>;
  baseline_comparison: Record<string, unknown>;
  energy_history: Array<{ step: number; best_energy: number; mean_energy: number }>;
  graph_preview: {
    nodes: Array<{
      id: number;
      label: string;
      weight: number;
      selected: boolean;
      x: number;
      y: number;
    }>;
    edges: Array<{ source: number; target: number; kind: string }>;
    shown_nodes: number;
    total_nodes: number;
  };
  details: Record<string, unknown>;
  hardware_export: QuantumHardwareExport;
  latency_ms: number;
}

export interface QuantumDemoResponse {
  seed: number;
  engine_version: string;
  references: string[];
  scenarios: Record<string, QuantumOptimizeResponse>;
  latency_ms: number;
}

export interface MultiViewCoverage360 {
  sector_scores: number[];
  angular_coverage_pct: number;
  largest_blind_gap_deg: number;
  recommended_next_azimuth_deg: number;
  is_watertight_360: boolean;
  next_best_view_advice_es: string;
  qubo_nbv_plan?: QuboNbvPlan | null;
}

export interface MultiViewBodyMetrology {
  convex_hull_volume_m3: number;
  convex_hull_surface_area_m2: number;
  voxel_solid_volume_m3: number;
  voxel_shell_volume_m3: number;
  voxel_resolution_mm: number;
  occupied_voxels: number;
  bounding_dimensions_m: [number, number, number];
  fused_center_gemini: [number, number, number];
  fused_box_3d: [number, number, number, number, number, number, number, number, number];
  mean_icp_rmse_mm: number;
  single_view_obb_volume_m3?: number | null;
  gt_volume_m3?: number | null;
  gt_volume_error_pct?: number | null;
}

export interface MultiViewReconstructResponse {
  mode: string;
  notice?: string | null;
  moge_backbone: MoGeBackbone | string;
  view_count: number;
  width: number;
  height: number;
  fov_h_deg: number;
  fov_v_deg: number;
  scale_factor: number;
  aruco_anchor?: ArucoAnchor | null;
  camera_poses: MultiViewCameraPose[];
  coverage_360: MultiViewCoverage360;
  body_metrology: MultiViewBodyMetrology;
  objects: SpatialObject[];
  point_cloud: PointCloudData;
  depth_stats?: DepthStats | null;
  quantum_telemetry?: QuantumTelemetrySummary | null;
  latency_ms: number;
}
