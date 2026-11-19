import React, { useEffect, useRef } from 'react';
import {
  AlertCircle,
  Box,
  Compass,
  Eye,
  Layers,
  Loader2,
  Play,
  Ruler,
  ScanLine,
  ShieldCheck,
  Upload,
  X,
} from 'lucide-react';
import type {
  BenchmarkScene,
  CameraViewMode,
  DepthStats,
  GeminiModelId,
  IntrinsicDiagnostics,
  MeasurementState,
  OcclusionEntry,
  PointCloudData,
  SceneAction,
  SceneMetrics,
  SpatialObject,
  ThinkingLevel,
} from '../types/spatial';
import { PinholeOverlay2D } from './PinholeOverlay2D';
import { Scene3DViewport } from './Scene3DViewport';
import { SpatialCopilotPanel } from './SpatialCopilotPanel';
import { formatDegrees, formatMeters, formatPercent } from '../utils/geometry';

interface LabWorkspaceProps {
  benchmarks: BenchmarkScene[];
  activeBenchmarkId: string;
  customImageTitle: string | null;
  imageBase64: string;
  imageMime: string;
  width: number;
  height: number;
  fovHDeg: number;
  thinkingLevel: ThinkingLevel;
  selectedModel: GeminiModelId;
  unfoldProgress: number;
  showGroundTruth: boolean;
  showOcclusionRays: boolean;
  showPointCloud?: boolean;
  showDepthMap?: boolean;
  measureMode: boolean;
  measureFirstId: string | null;
  cameraMode: CameraViewMode;
  objects: SpatialObject[];
  gtObjects?: SpatialObject[] | null;
  occlusionMatrix: OcclusionEntry[];
  metrics?: SceneMetrics | null;
  intrinsicDiagnostics?: IntrinsicDiagnostics | null;
  pointCloud?: PointCloudData | null;
  depthMapBase64?: string | null;
  depthStats?: DepthStats | null;
  fovSource?: string | null;
  local3dAvailable?: boolean;
  local3dDevice?: string;
  selectedObjectId: string | null;
  hoveredObjectId: string | null;
  highlightedIds: string[] | null;
  measurement: MeasurementState | null;
  analyzing: boolean;
  modeBadge: string;
  notice?: string | null;
  error?: string | null;
  apiKey: string;
  onSelectBenchmark: (id: string) => void;
  onUploadFile: (file: File) => void;
  onChangeFovH: (fov: number) => void;
  onChangeThinkingLevel: (level: ThinkingLevel) => void;
  onChangeUnfoldProgress: (val: number) => void;
  onToggleGroundTruth: () => void;
  onToggleOcclusionRays: () => void;
  onTogglePointCloud?: () => void;
  onToggleDepthMap?: () => void;
  onToggleMeasureMode: () => void;
  onChangeCameraMode: (mode: CameraViewMode) => void;
  onAnalyzeScene: () => void;
  onSelectObject: (id: string, shiftKey: boolean) => void;
  onHoverObject: (id: string | null) => void;
  onClearSelection: () => void;
  onExecuteSceneActions: (actions: SceneAction[]) => void;
}

export const LabWorkspace: React.FC<LabWorkspaceProps> = ({
  benchmarks,
  activeBenchmarkId,
  customImageTitle,
  imageBase64,
  imageMime,
  width,
  height,
  fovHDeg,
  thinkingLevel,
  selectedModel,
  unfoldProgress,
  showGroundTruth,
  showOcclusionRays,
  showPointCloud = true,
  showDepthMap = false,
  measureMode,
  measureFirstId,
  cameraMode,
  objects,
  gtObjects,
  occlusionMatrix,
  metrics,
  intrinsicDiagnostics,
  pointCloud,
  depthMapBase64,
  depthStats,
  fovSource,
  local3dAvailable = false,
  local3dDevice,
  selectedObjectId,
  hoveredObjectId,
  highlightedIds,
  measurement,
  analyzing,
  modeBadge,
  notice,
  error,
  apiKey,
  onSelectBenchmark,
  onUploadFile,
  onChangeFovH,
  onChangeThinkingLevel,
  onChangeUnfoldProgress,
  onToggleGroundTruth,
  onToggleOcclusionRays,
  onTogglePointCloud,
  onToggleDepthMap,
  onToggleMeasureMode,
  onChangeCameraMode,
  onAnalyzeScene,
  onSelectObject,
  onHoverObject,
  onClearSelection,
  onExecuteSceneActions,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Global workstation keyboard shortcuts (1-4, G, M, Esc) when not typing in an input
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }

      if (e.key === '1') onChangeCameraMode('camera_aligned');
      else if (e.key === '2') onChangeCameraMode('orbit');
      else if (e.key === '3') onChangeCameraMode('bev');
      else if (e.key === '4') onChangeCameraMode('isometric');
      else if (e.key === 'g' || e.key === 'G') onToggleGroundTruth();
      else if (e.key === 'm' || e.key === 'M') onToggleMeasureMode();
      else if (e.key === 'Escape') onClearSelection();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [
    onChangeCameraMode,
    onToggleGroundTruth,
    onToggleMeasureMode,
    onClearSelection,
  ]);

  const selectedObj = objects.find((o) => o.id === selectedObjectId) ?? null;

  return (
    <div className="mx-auto max-w-[1600px] space-y-4 px-4 py-4 sm:px-6">
      {/* Precision Scene Toolbar */}
      <section
        aria-label="Barra de calibración de escena"
        className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-3.5"
      >
        <div className="flex flex-wrap items-center gap-3">
          {/* Scene picker + upload */}
          <div className="flex flex-wrap items-center gap-2">
            <label htmlFor="benchmark-select" className="text-xs font-medium text-[#4B5563]">
              Escena
            </label>
            <select
              id="benchmark-select"
              value={customImageTitle ? '__custom__' : activeBenchmarkId}
              onChange={(e) => {
                if (e.target.value !== '__custom__') {
                  onSelectBenchmark(e.target.value);
                }
              }}
              className="font-mono-tech rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] px-2.5 py-1.5 text-xs font-medium text-[#111827] focus:border-[#1D4ED8] focus:outline-none"
            >
              {benchmarks.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.id}
                </option>
              ))}
              {customImageTitle && (
                <option value="__custom__">Foto: {customImageTitle}</option>
              )}
            </select>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onUploadFile(file);
                e.target.value = '';
              }}
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-2.5 py-1.5 text-xs font-medium text-[#111827] transition-colors hover:bg-[#F7F7F5]"
            >
              <Upload className="h-3.5 w-3.5 text-[#4B5563]" />
              <span>Subir RGB</span>
            </button>
          </div>

          <div className="hidden h-5 w-px bg-[#E2E5EA] lg:block" />

          {/* FOV_h live slider */}
          <div className="flex items-center gap-2 min-w-[205px]">
            <label
              htmlFor="fov-slider"
              className="font-mono-tech text-xs font-medium text-[#111827] whitespace-nowrap"
            >
              FOV_h {fovHDeg.toFixed(1)}°
            </label>
            <input
              id="fov-slider"
              type="range"
              min={25}
              max={120}
              step={0.5}
              value={fovHDeg}
              onChange={(e) => onChangeFovH(Number(e.target.value))}
              className="optical-slider w-24 sm:w-28"
            />
            {fovSource && (
              <span
                className="font-mono-tech rounded border border-[#E2E5EA] bg-[#F7F7F5] px-1.5 py-0.5 text-[10px] text-[#4B5563]"
                title="Fuente de distancia focal / FOV horizontal"
              >
                {fovSource === 'moge2_intrinsics'
                  ? 'MoGe-2'
                  : fovSource === 'exif_35mm'
                  ? 'EXIF'
                  : fovSource === 'manual'
                  ? 'Manual'
                  : fovSource}
              </span>
            )}
          </div>

          {/* 2D -> 3D Unfold Scrubber */}
          <div className="flex items-center gap-2 min-w-[195px]">
            <label
              htmlFor="unfold-slider"
              className="font-mono-tech text-xs font-medium text-[#1D4ED8] whitespace-nowrap"
            >
              2D→3D {Math.round(unfoldProgress * 100)}%
            </label>
            <input
              id="unfold-slider"
              type="range"
              min={0}
              max={100}
              step={1}
              value={Math.round(unfoldProgress * 100)}
              onChange={(e) => onChangeUnfoldProgress(Number(e.target.value) / 100)}
              className="optical-slider w-24 sm:w-28"
            />
          </div>

          <div className="hidden h-5 w-px bg-[#E2E5EA] xl:block" />

          {/* Thinking level */}
          <div className="flex items-center gap-1 rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] p-0.5">
            <span className="px-1.5 text-[10px] font-medium text-[#4B5563]">
              Razonamiento
            </span>
            {(['low', 'medium', 'high'] as ThinkingLevel[]).map((lvl) => (
              <button
                key={lvl}
                type="button"
                onClick={() => onChangeThinkingLevel(lvl)}
                className={`font-mono-tech rounded-md px-2 py-1 text-[11px] transition-colors ${
                  thinkingLevel === lvl
                    ? 'bg-[#FFFFFF] font-semibold text-[#111827] border border-[#E2E5EA]'
                    : 'text-[#4B5563] hover:text-[#111827]'
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>

          {/* Toggles */}
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              type="button"
              onClick={onToggleGroundTruth}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                showGroundTruth
                  ? 'border-[#0F766E] bg-[#0F766E]/10 text-[#0F766E]'
                  : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#4B5563] hover:text-[#111827]'
              }`}
              title="Mostrar/ocultar Ground Truth (G)"
            >
              <Layers className="h-3.5 w-3.5" />
              <span>GT</span>
              <kbd className="font-mono-tech text-[10px] opacity-70">G</kbd>
            </button>

            <button
              type="button"
              onClick={onToggleOcclusionRays}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                showOcclusionRays
                  ? 'border-[#B45309] bg-[#B45309]/10 text-[#B45309]'
                  : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#4B5563] hover:text-[#111827]'
              }`}
              title="Rayos de oclusión Kay-Kajiya"
            >
              <ScanLine className="h-3.5 w-3.5" />
              <span>Oclusión ({occlusionMatrix.length})</span>
            </button>

            <button
              type="button"
              onClick={onToggleMeasureMode}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                measureMode
                  ? 'border-[#BE123C] bg-[#BE123C]/10 text-[#BE123C]'
                  : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#4B5563] hover:text-[#111827]'
              }`}
              title="Medir distancia entre 2 objetos (M)"
            >
              <Ruler className="h-3.5 w-3.5" />
              <span>Medir</span>
              <kbd className="font-mono-tech text-[10px] opacity-70">M</kbd>
            </button>

            <button
              type="button"
              onClick={() => onTogglePointCloud?.()}
              disabled={!pointCloud || pointCloud.count === 0}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors disabled:opacity-45 ${
                showPointCloud && pointCloud && pointCloud.count > 0
                  ? 'border-[#1D4ED8] bg-[#1D4ED8]/10 text-[#1D4ED8]'
                  : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#4B5563] hover:text-[#111827]'
              }`}
              title="Mostrar/ocultar nube de puntos 3D métrica (MoGe-2)"
            >
              <Box className="h-3.5 w-3.5" />
              <span>Nube 3D</span>
              {pointCloud && pointCloud.count > 0 && (
                <span className="font-mono-tech text-[10px] opacity-80">
                  {pointCloud.count >= 1000
                    ? `${Math.round(pointCloud.count / 1000)}k`
                    : pointCloud.count}
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => onToggleDepthMap?.()}
              disabled={!depthMapBase64}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors disabled:opacity-45 ${
                showDepthMap && depthMapBase64
                  ? 'border-[#0F766E] bg-[#0F766E]/10 text-[#0F766E]'
                  : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#4B5563] hover:text-[#111827]'
              }`}
              title="Superponer mapa de profundidad métrica sobre la foto 2D"
            >
              <Compass className="h-3.5 w-3.5" />
              <span>Profundidad</span>
            </button>
          </div>

          {/* Primary analyze action */}
          <div className="ml-auto flex items-center gap-2">
            {local3dAvailable && (
              <span
                className="font-mono-tech hidden items-center rounded-md border border-[#E2E5EA] bg-[#F7F7F5] px-2 py-1 text-[10px] text-[#0F766E] 2xl:inline-flex"
                title={local3dDevice || 'GPU Métrica Local'}
              >
                MoGe-2 GPU
              </span>
            )}
            <button
              type="button"
              onClick={onAnalyzeScene}
              disabled={analyzing}
              className="inline-flex items-center gap-1.5 rounded-lg bg-[#1D4ED8] px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-[#1E40AF] disabled:opacity-60"
            >
              {analyzing ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Play className="h-3.5 w-3.5" />
              )}
              <span>Analizar Escena</span>
            </button>
          </div>
        </div>

        {/* Optional notice / error / active subset banner */}
        {(notice || error || highlightedIds) && (
          <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 border-t border-[#E2E5EA] pt-2.5 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              {error && (
                <span className="inline-flex items-center gap-1.5 text-[#BE123C]">
                  <AlertCircle className="h-3.5 w-3.5" />
                  {error}
                </span>
              )}
              {!error && notice && (
                <span className="inline-flex items-center gap-1.5 text-[#4B5563]">
                  <Eye className="h-3.5 w-3.5 text-[#0F766E]" />
                  {notice}
                </span>
              )}
              {highlightedIds && (
                <span className="font-mono-tech inline-flex items-center gap-1.5 rounded bg-[#1D4ED8]/10 px-2 py-0.5 text-[11px] text-[#1D4ED8]">
                  Subconjunto filtrado: {highlightedIds.length} objetos
                  <button
                    type="button"
                    onClick={onClearSelection}
                    className="ml-1 hover:text-[#111827]"
                    aria-label="Limpiar filtro"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              )}
            </div>

            {metrics && (
              <div className="font-mono-tech flex items-center gap-3 text-[11px] text-[#4B5563]">
                <span>
                  mIoU_3D <strong className="text-[#1D4ED8]">{formatPercent(metrics.mIoU_3D)}</strong>
                </span>
                <span>
                  ATE <strong className="text-[#111827]">{formatMeters(metrics.ate_m, 3)}</strong>
                </span>
                <span>
                  AOE <strong className="text-[#B45309]">{formatDegrees(metrics.aoe_deg, 1)}</strong>
                </span>
              </div>
            )}
          </div>
        )}
      </section>

      {/* Compuerta Edge -> 3D (Herencia C8286) Strip */}
      {intrinsicDiagnostics?.cascade_policy && (
        <section
          aria-label="Compuerta de confiabilidad Edge a VLM 3D"
          className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] px-4 py-2.5 text-xs"
        >
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="inline-flex items-center gap-1.5 font-semibold text-[#111827]">
              <ShieldCheck className="h-3.5 w-3.5 text-[#1D4ED8]" />
              <span>Compuerta Edge → 3D (Herencia C8286):</span>
            </span>

            {(() => {
              const cp = intrinsicDiagnostics.cascade_policy;
              const verdictStyle =
                cp.operational_verdict === 'APROBADO'
                  ? 'border-[#0F766E]/35 bg-[#0F766E]/10 text-[#0F766E]'
                  : cp.operational_verdict === 'ESCALAR_MODELO'
                  ? 'border-[#B45309]/35 bg-[#B45309]/10 text-[#B45309]'
                  : 'border-[#BE123C]/35 bg-[#BE123C]/10 text-[#BE123C]';
              const verdictLabel =
                cp.operational_verdict === 'APROBADO'
                  ? 'APROBADO'
                  : cp.operational_verdict === 'ESCALAR_MODELO'
                  ? 'ESCALAR MODELO'
                  : 'RECHAZADO FÍSICA';
              return (
                <>
                  <span
                    className={`font-mono-tech rounded border px-2 py-0.5 text-[11px] font-semibold ${verdictStyle}`}
                  >
                    {verdictLabel}
                  </span>
                  <span className="font-mono-tech rounded border border-[#E2E5EA] bg-[#F7F7F5] px-2 py-0.5 text-[11px] text-[#111827]">
                    Régimen: {cp.c8286_degradation_regime}
                  </span>
                  <span className="font-mono-tech text-[11px] text-[#4B5563]">
                    Tier: <strong className="text-[#111827]">{cp.recommended_model}</strong>
                    {cp.estimated_latency_savings_pct > 0
                      ? ` (−${cp.estimated_latency_savings_pct.toFixed(0)}% latencia)`
                      : cp.operational_verdict === 'ESCALAR_MODELO'
                      ? ' (Razonamiento Profundo)'
                      : ' (Re-captura / Sensor)'}
                  </span>
                  {cp.min_surface_clearance_m !== null && (
                    <span className="font-mono-tech text-[11px] text-[#4B5563]">
                      Holgura mín: <strong className="text-[#111827]">{cp.min_surface_clearance_m.toFixed(2)} m</strong>
                    </span>
                  )}
                </>
              );
            })()}
          </div>

          {intrinsicDiagnostics.cascade_policy.trigger_reasons[0] && (
            <p className="max-w-2xl text-[11px] text-[#4B5563]">
              {intrinsicDiagnostics.cascade_policy.trigger_reasons[0]}
            </p>
          )}
        </section>
      )}

      {/* Main Synchronized 2D <-> 3D Split */}
      <div className="grid gap-4 lg:grid-cols-12">
        <div className="lg:col-span-5">
          <PinholeOverlay2D
            imageBase64={imageBase64}
            imageMime={imageMime}
            width={width}
            height={height}
            fovHDeg={fovHDeg}
            objects={objects}
            gtObjects={gtObjects}
            showGroundTruth={showGroundTruth}
            depthMapBase64={depthMapBase64}
            depthStats={depthStats}
            showDepthMap={showDepthMap}
            selectedObjectId={selectedObjectId}
            hoveredObjectId={hoveredObjectId}
            highlightedIds={highlightedIds}
            measureMode={measureMode}
            measureFirstId={measureFirstId}
            measurement={measurement}
            onSelectObject={onSelectObject}
            onHoverObject={onHoverObject}
            onUploadFile={onUploadFile}
          />
        </div>

        <div className="lg:col-span-7">
          <Scene3DViewport
            imageBase64={imageBase64}
            imageMime={imageMime}
            width={width}
            height={height}
            fovHDeg={fovHDeg}
            objects={objects}
            gtObjects={gtObjects}
            perObjectMatches={metrics?.per_object_matches}
            occlusionMatrix={occlusionMatrix}
            showGroundTruth={showGroundTruth}
            showOcclusionRays={showOcclusionRays}
            showPointCloud={showPointCloud}
            pointCloud={pointCloud}
            depthStats={depthStats}
            unfoldProgress={unfoldProgress}
            cameraMode={cameraMode}
            onChangeCameraMode={onChangeCameraMode}
            selectedObjectId={selectedObjectId}
            hoveredObjectId={hoveredObjectId}
            highlightedIds={highlightedIds}
            measurement={measurement}
            modeBadge={modeBadge}
            onSelectObject={onSelectObject}
            onHoverObject={onHoverObject}
          />
        </div>
      </div>

      {/* Bottom Split: Selected Pose & Object Telemetry + Spatial Copilot */}
      <div className="grid gap-4 lg:grid-cols-12">
        {/* Compact Object Inspector & List */}
        <section
          aria-label="Inspector de objetos 9-DOF"
          className="flex flex-col overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] lg:col-span-5"
        >
          <div className="flex items-center justify-between border-b border-[#E2E5EA] bg-[#F7F7F5] px-4 py-2.5">
            <h2 className="text-xs font-semibold text-[#111827]">
              Objetos Detectados ({objects.length})
            </h2>
            {(selectedObjectId || measurement) && (
              <button
                type="button"
                onClick={onClearSelection}
                className="text-[11px] font-medium text-[#4B5563] hover:text-[#111827]"
              >
                Limpiar selección (Esc)
              </button>
            )}
          </div>

          {measurement && (
            <div className="border-b border-[#BE123C]/25 bg-[#BE123C]/[0.04] px-4 py-2.5 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-[#BE123C]">
                  Regla 3D: {measurement.object_a_label} ↔ {measurement.object_b_label}
                </span>
                <span className="font-mono-tech text-[11px] text-[#111827]">
                  {measurement.geometry.is_colliding ? 'COLISIÓN SAT' : 'Sin colisión'}
                </span>
              </div>
              <div className="font-mono-tech mt-1 flex flex-wrap gap-3 text-[11px] text-[#4B5563]">
                <span>Centro: {formatMeters(measurement.geometry.center_distance_m, 3)}</span>
                <span>Holgura: {formatMeters(measurement.geometry.surface_clearance_m, 3)}</span>
                <span>IoU 3D: {formatPercent(measurement.geometry.iou_3d)}</span>
              </div>
            </div>
          )}

          {selectedObj && (
            <div className="border-b border-[#E2E5EA] bg-[#F7F7F5]/60 px-4 py-2.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[#111827]">
                  {selectedObj.label}{' '}
                  <span className="font-mono-tech text-[11px] text-[#4B5563]">
                    ({selectedObj.id})
                  </span>
                </span>
                <span className="font-mono-tech text-[11px] text-[#1D4ED8]">
                  d = {formatMeters(selectedObj.distance_m ?? selectedObj.box_3d[1], 2)}
                </span>
              </div>
              <div className="font-mono-tech mt-1.5 grid grid-cols-3 gap-2 text-[10px] text-[#4B5563]">
                <div>
                  Pos: [{selectedObj.box_3d[0].toFixed(2)}, {selectedObj.box_3d[1].toFixed(2)},{' '}
                  {selectedObj.box_3d[2].toFixed(2)}]
                </div>
                <div>
                  Dim: [{selectedObj.box_3d[3].toFixed(2)}, {selectedObj.box_3d[4].toFixed(2)},{' '}
                  {selectedObj.box_3d[5].toFixed(2)}]
                </div>
                <div>
                  Yaw: {selectedObj.box_3d[8].toFixed(1)}°
                </div>
              </div>
            </div>
          )}

          <div className="scroll-affordance-y max-h-[260px] divide-y divide-[#E2E5EA]">
            {objects.map((obj) => {
              const isSelected = selectedObjectId === obj.id;
              const isHovered = hoveredObjectId === obj.id;
              return (
                <button
                  key={obj.id}
                  type="button"
                  onMouseEnter={() => onHoverObject(obj.id)}
                  onMouseLeave={() => onHoverObject(null)}
                  onClick={(e) => onSelectObject(obj.id, e.shiftKey)}
                  className={`flex w-full items-center justify-between px-4 py-2.5 text-left text-xs transition-colors ${
                    isSelected
                      ? 'bg-[#BE123C]/[0.06]'
                      : isHovered
                        ? 'bg-[#F7F7F5]'
                        : 'bg-[#FFFFFF]'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-2 w-2 rounded-full ${
                        isSelected ? 'bg-[#BE123C]' : 'bg-[#1D4ED8]'
                      }`}
                    />
                    <span className="font-medium text-[#111827]">{obj.label}</span>
                    <span className="font-mono-tech text-[10px] text-[#9CA3AF]">
                      {obj.id}
                    </span>
                  </div>
                  <div className="font-mono-tech flex items-center gap-3 text-[11px] text-[#4B5563]">
                    <span>{formatMeters(obj.distance_m ?? obj.box_3d[1], 2)}</span>
                    <span>{(obj.volume_m3 ?? obj.box_3d[3] * obj.box_3d[4] * obj.box_3d[5]).toFixed(2)} m³</span>
                  </div>
                </button>
              );
            })}
            {objects.length === 0 && (
              <div className="p-6 text-center text-xs text-[#4B5563]">
                Sin objetos cargados todavía. Haz clic en <strong>Analizar Escena</strong>.
              </div>
            )}
          </div>
        </section>

        {/* Spatial Copilot */}
        <div className="lg:col-span-7">
          <SpatialCopilotPanel
            objects={objects}
            gtObjects={gtObjects}
            imageBase64={imageBase64}
            width={width}
            height={height}
            fovHDeg={fovHDeg}
            selectedModel={selectedModel}
            apiKey={apiKey}
            onExecuteSceneActions={onExecuteSceneActions}
          />
        </div>
      </div>
    </div>
  );
};
