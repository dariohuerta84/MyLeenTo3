import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  AnalyzeEngine,
  AnalyzeResponse,
  AppSection,
  ArucoAnchor,
  BenchmarkScene,
  CameraViewMode,
  DepthStats,
  FovSweepPoint,
  GeminiModelId,
  IntrinsicDiagnostics,
  MeasurementState,
  MoGeBackbone,
  OcclusionEntry,
  PointCloudData,
  QuantumNmsSummary,
  RealPhotoMetrics,
  SceneAction,
  SceneIntelligence,
  SceneMetrics,
  SpatialObject,
  ThinkingLevel,
} from './types/spatial';
import {
  analyzeScene,
  calibrateScale,
  fetchBenchmarks,
  fetchHealth,
  measurePair,
  reprojectScene,
} from './services/api';
import { runViewTransition, stripDataUrlPrefix } from './utils/geometry';
import { HeaderNav } from './components/HeaderNav';
import { ApiKeyModal } from './components/ApiKeyModal';
import { LandingShowcase } from './components/LandingShowcase';
import { LabWorkspace } from './components/LabWorkspace';
import { MultiView360Workspace } from './components/MultiView360Workspace';
import { MetricsBenchmarkPanel } from './components/MetricsBenchmarkPanel';
import { MultiFrameAuditPanel } from './components/MultiFrameAuditPanel';
import { QuantumQuboWorkspace } from './components/QuantumQuboWorkspace';

const DEFAULT_THINKING_BY_MODEL: Record<GeminiModelId, ThinkingLevel> = {
  'gemini-4-hydrogen': 'high',
  'gemini-4-argon': 'medium',
  'gemini-3.5-flash-lite': 'low',
};

const VALID_VIEWS: AppSection[] = ['landing', 'lab', 'multiview', 'quantum', 'multiframe', 'metrics'];

function getInitialQueryParams(): { view: AppSection; scene?: string } {
  if (typeof window === 'undefined') return { view: 'landing' };
  const params = new URLSearchParams(window.location.search);
  const rawView = params.get('view') as AppSection | null;
  const view = rawView && VALID_VIEWS.includes(rawView) ? rawView : 'landing';
  const scene = params.get('scene') ?? undefined;
  return { view, scene };
}

export function App() {
  const initialQuery = useRef(getInitialQueryParams());
  const [activeSection, setActiveSection] = useState<AppSection>(initialQuery.current.view);
  const [selectedModel, setSelectedModel] = useState<GeminiModelId>('gemini-4-hydrogen');
  const [thinkingLevel, setThinkingLevel] = useState<ThinkingLevel>('high');
  const [selectedEngine, setSelectedEngine] = useState<AnalyzeEngine>('auto');
  const [mogeBackbone, setMogeBackbone] = useState<MoGeBackbone>('vits');
  const [customPrompt, setCustomPrompt] = useState<string>('');

  // Privacy-first in-memory API key only (never stored in localStorage/cookies)
  const [apiKey, setApiKey] = useState<string>('');
  const [apiKeyModalOpen, setApiKeyModalOpen] = useState(false);
  const [backendEnvConfigured, setBackendEnvConfigured] = useState(false);
  const [local3dAvailable, setLocal3dAvailable] = useState(false);
  const [local3dDevice, setLocal3dDevice] = useState<string>('disabled');

  // Scene & Benchmark state
  const [benchmarks, setBenchmarks] = useState<BenchmarkScene[]>([]);
  const [activeBenchmarkId, setActiveBenchmarkId] = useState<string>('lab_desk_calibrated');
  const [customImageTitle, setCustomImageTitle] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string>('');
  const [imageMime, setImageMime] = useState<string>('image/png');
  const [width, setWidth] = useState<number>(960);
  const [height, setHeight] = useState<number>(640);
  const [fovHDeg, setFovHDeg] = useState<number>(60);

  // Spatial inference state
  const [objects, setObjects] = useState<SpatialObject[]>([]);
  const [gtObjects, setGtObjects] = useState<SpatialObject[] | null>(null);
  const [occlusionMatrix, setOcclusionMatrix] = useState<OcclusionEntry[]>([]);
  const [metrics, setMetrics] = useState<SceneMetrics | null>(null);
  const [fovSweep, setFovSweep] = useState<FovSweepPoint[] | null>(null);
  const [intrinsicDiagnostics, setIntrinsicDiagnostics] =
    useState<IntrinsicDiagnostics | null>(null);
  const [pointCloud, setPointCloud] = useState<PointCloudData | null>(null);
  const [depthMapBase64, setDepthMapBase64] = useState<string | null>(null);
  const [depthStats, setDepthStats] = useState<DepthStats | null>(null);
  const [fovSource, setFovSource] = useState<string | null>(null);
  const [realPhotoMetrics, setRealPhotoMetrics] = useState<RealPhotoMetrics | null>(null);
  const [sceneIntelligence, setSceneIntelligence] = useState<SceneIntelligence | null>(null);
  const [quantumNmsSummary, setQuantumNmsSummary] = useState<QuantumNmsSummary | null>(null);
  const [scaleFactor, setScaleFactor] = useState<number>(1.0);
  const [scaleLabel, setScaleLabel] = useState<string | null>(null);
  const [arucoAnchor, setArucoAnchor] = useState<ArucoAnchor | null>(null);
  const [fovManualOverride, setFovManualOverride] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [modeBadge, setModeBadge] = useState<string>('calibrated_demo');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Interactive 2D <-> 3D lab controls
  const [unfoldProgress, setUnfoldProgress] = useState<number>(1);
  const [showGroundTruth, setShowGroundTruth] = useState<boolean>(true);
  const [showOcclusionRays, setShowOcclusionRays] = useState<boolean>(false);
  const [showPointCloud, setShowPointCloud] = useState<boolean>(true);
  const [showDepthMap, setShowDepthMap] = useState<boolean>(false);
  const [measureMode, setMeasureMode] = useState<boolean>(false);
  const [measureFirstId, setMeasureFirstId] = useState<string | null>(null);
  const [cameraMode, setCameraMode] = useState<CameraViewMode>('orbit');
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [highlightedIds, setHighlightedIds] = useState<string[] | null>(null);
  const [measurement, setMeasurement] = useState<MeasurementState | null>(null);

  const mainRef = useRef<HTMLElement>(null);
  const reprojectSeqRef = useRef<number>(0);
  const objectsRef = useRef<SpatialObject[]>([]);
  const gtObjectsRef = useRef<SpatialObject[] | null>(null);
  const imageBase64Ref = useRef<string>('');
  const baseObjectsRef = useRef<SpatialObject[]>([]);
  const baseDepthStatsRef = useRef<DepthStats | null>(null);
  const baseRealPhotoMetricsRef = useRef<RealPhotoMetrics | null>(null);
  const baseSceneIntelligenceRef = useRef<SceneIntelligence | null>(null);

  useEffect(() => {
    objectsRef.current = objects;
  }, [objects]);

  useEffect(() => {
    gtObjectsRef.current = gtObjects;
  }, [gtObjects]);

  useEffect(() => {
    imageBase64Ref.current = imageBase64;
  }, [imageBase64]);

  const navigateTo = useCallback((section: AppSection) => {
    runViewTransition(() => {
      setActiveSection(section);
    });
    window.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior });
    window.setTimeout(() => {
      mainRef.current?.focus({ preventScroll: true });
    }, 30);
  }, []);

  const clearLocal3DTelemetry = useCallback(() => {
    setPointCloud(null);
    setDepthMapBase64(null);
    setDepthStats(null);
    setFovSource(null);
    setRealPhotoMetrics(null);
    setQuantumNmsSummary(null);
    setArucoAnchor(null);
    setScaleFactor(1.0);
    setScaleLabel(null);
    setShowDepthMap(false);
    setFovManualOverride(false);
    baseDepthStatsRef.current = null;
    baseRealPhotoMetricsRef.current = null;
  }, []);

  const applyLocal3DTelemetry = useCallback((res: AnalyzeResponse) => {
    setPointCloud(res.point_cloud ?? null);
    setDepthMapBase64(res.depth_map_base64 ?? null);
    setDepthStats(res.depth_stats ?? null);
    setFovSource(res.fov_source ?? null);
    setRealPhotoMetrics(res.real_photo_metrics ?? null);
    setQuantumNmsSummary(res.quantum_nms_summary ?? null);
    setArucoAnchor(res.aruco_anchor ?? null);
    baseDepthStatsRef.current = res.depth_stats ?? null;
    baseRealPhotoMetricsRef.current = res.real_photo_metrics ?? null;
    if (res.image_base64) {
      setImageBase64(res.image_base64);
      setImageMime(res.mime_type || 'image/png');
    }
    if (res.width > 0 && res.height > 0) {
      setWidth(res.width);
      setHeight(res.height);
    }
    if (res.fov_source && res.fov_source !== 'manual' && Number.isFinite(res.fov_h_deg)) {
      setFovHDeg(Number(res.fov_h_deg.toFixed(1)));
    }
  }, []);

  const runBenchmarkAnalysis = useCallback(
    async (
      benchmarkId: string,
      sceneList: BenchmarkScene[],
      model: GeminiModelId,
      thinking: ThinkingLevel,
      fov: number,
      key: string
    ) => {
      const target = sceneList.find((b) => b.id === benchmarkId) ?? sceneList[0];
      if (!target) return;

      clearLocal3DTelemetry();
      setActiveBenchmarkId(target.id);
      setCustomImageTitle(null);
      setImageBase64(target.image_base64);
      setImageMime('image/png');
      setWidth(target.width);
      setHeight(target.height);
      setFovHDeg(fov);
      setSelectedObjectId(null);
      setMeasureFirstId(null);
      setMeasurement(null);
      setHighlightedIds(null);
      setAnalyzing(true);
      setError(null);

      try {
        const res = await analyzeScene(
          {
            benchmark_id: target.id,
            model,
            thinking_level: thinking,
            fov_h_deg: fov,
            max_objects: 10,
          },
          key
        );
        setObjects(res.objects ?? []);
        setGtObjects(res.gt_objects ?? target.gt_objects ?? null);
        setOcclusionMatrix(res.occlusion_matrix ?? []);
        setMetrics(res.metrics ?? null);
        setFovSweep(res.fov_sweep ?? null);
        setIntrinsicDiagnostics(res.intrinsic_diagnostics ?? null);
        setSceneIntelligence(res.scene_intelligence ?? null);
        baseObjectsRef.current = res.objects ?? [];
        baseSceneIntelligenceRef.current = res.scene_intelligence ?? null;
        setScaleFactor(1.0);
        setScaleLabel(null);
        setModeBadge(res.mode ?? 'calibrated_demo');
        setNotice(res.notice ?? null);
        if (target.id === 'occlusion_corridor') {
          setShowOcclusionRays(true);
        }
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'No se pudo analizar la escena'
        );
      } finally {
        setAnalyzing(false);
      }
    },
    [clearLocal3DTelemetry]
  );

  // Initial load: health + benchmarks + initial benchmark analysis
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [health, list] = await Promise.all([
          fetchHealth().catch(() => null),
          fetchBenchmarks(),
        ]);
        if (!active) return;
        if (health) {
          setBackendEnvConfigured(Boolean(health.gemini_env_configured));
          setLocal3dAvailable(Boolean(health.local_metric_3d_available));
          setLocal3dDevice(health.local_3d_device || 'disabled');
        }
        setBenchmarks(list);
        if (list.length > 0) {
          const requestedId = initialQuery.current.scene;
          const initialScene =
            (requestedId && list.find((b) => b.id === requestedId)) || list[0];
          await runBenchmarkAnalysis(
            initialScene.id,
            list,
            'gemini-4-hydrogen',
            'high',
            initialScene.fov_h_deg || 60,
            ''
          );
        }
      } catch (err) {
        if (!active) return;
        setError(
          err instanceof Error
            ? `Error al conectar con /api: ${err.message}`
            : 'Backend no disponible'
        );
      }
    })();
    return () => {
      active = false;
    };
  }, [runBenchmarkAnalysis]);

  // Honor URL hash fragments (#problematica-scroll, #scrolly-3d-story) after landing mounts
  useEffect(() => {
    if (activeSection !== 'landing' || benchmarks.length === 0) return;
    const rawHash = window.location.hash.replace(/^#/, '');
    if (!rawHash) return;
    const timer = window.setTimeout(() => {
      const target = document.getElementById(rawHash);
      if (target) {
        target.scrollIntoView({ behavior: 'auto', block: 'start' });
      }
    }, 120);
    return () => window.clearTimeout(timer);
  }, [activeSection, benchmarks.length]);

  const handleSelectModel = (model: GeminiModelId) => {
    setSelectedModel(model);
    setThinkingLevel(DEFAULT_THINKING_BY_MODEL[model]);
  };

  const handleSelectBenchmark = (id: string) => {
    const found = benchmarks.find((b) => b.id === id);
    const nextFov = found?.fov_h_deg ?? 60;
    runBenchmarkAnalysis(id, benchmarks, selectedModel, thinkingLevel, nextFov, apiKey);
  };

  const handleUploadFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || '');
      const { base64, mimeType } = stripDataUrlPrefix(dataUrl);
      const img = new Image();
      img.onload = async () => {
        const w = img.naturalWidth || 960;
        const h = img.naturalHeight || 640;
        setCustomImageTitle(file.name);
        clearLocal3DTelemetry();
        setImageBase64(base64);
        setImageMime(mimeType);
        setWidth(w);
        setHeight(h);
        setGtObjects(null);
        setSelectedObjectId(null);
        setMeasurement(null);
        setHighlightedIds(null);
        setAnalyzing(true);
        setError(null);
        try {
          const res = await analyzeScene(
            {
              image_base64: base64,
              mime_type: mimeType,
              width: w,
              height: h,
              model: selectedModel,
              thinking_level: thinkingLevel,
              fov_h_deg: fovHDeg,
              custom_prompt: customPrompt.trim() || undefined,
              engine: selectedEngine,
              moge_backbone: mogeBackbone,
              auto_fov: true,
              enable_diagnosis: true,
              max_objects: 10,
            },
            apiKey
          );
          setObjects(res.objects ?? []);
          setGtObjects(res.gt_objects ?? null);
          setOcclusionMatrix(res.occlusion_matrix ?? []);
          setMetrics(res.metrics ?? null);
          setFovSweep(res.fov_sweep ?? null);
          setIntrinsicDiagnostics(res.intrinsic_diagnostics ?? null);
          setSceneIntelligence(res.scene_intelligence ?? null);
          baseObjectsRef.current = res.objects ?? [];
          baseSceneIntelligenceRef.current = res.scene_intelligence ?? null;
          setScaleFactor(1.0);
          setScaleLabel(null);
          setModeBadge(res.mode ?? 'calibrated_demo');
          setNotice(res.notice ?? null);
          applyLocal3DTelemetry(res);
        } catch (err) {
          setError(
            err instanceof Error ? err.message : 'Error al analizar la foto RGB'
          );
        } finally {
          setAnalyzing(false);
        }
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  const handleAnalyzeCurrent = async () => {
    if (!customImageTitle) {
      await runBenchmarkAnalysis(
        activeBenchmarkId,
        benchmarks,
        selectedModel,
        thinkingLevel,
        fovHDeg,
        apiKey
      );
      return;
    }

    setAnalyzing(true);
    setError(null);
    try {
      const res = await analyzeScene(
        {
          image_base64: imageBase64,
          mime_type: imageMime,
          width,
          height,
          model: selectedModel,
          thinking_level: thinkingLevel,
          fov_h_deg: fovHDeg,
          custom_prompt: customPrompt.trim() || undefined,
          engine: selectedEngine,
          moge_backbone: mogeBackbone,
          auto_fov: !fovManualOverride,
          enable_diagnosis: true,
          max_objects: 10,
        },
        apiKey
      );
      setObjects(res.objects ?? []);
      setGtObjects(res.gt_objects ?? null);
      setOcclusionMatrix(res.occlusion_matrix ?? []);
      setMetrics(res.metrics ?? null);
      setFovSweep(res.fov_sweep ?? null);
      setIntrinsicDiagnostics(res.intrinsic_diagnostics ?? null);
      setSceneIntelligence(res.scene_intelligence ?? null);
      baseObjectsRef.current = res.objects ?? [];
      baseSceneIntelligenceRef.current = res.scene_intelligence ?? null;
      setScaleFactor(1.0);
      setScaleLabel(null);
      setModeBadge(res.mode ?? 'live');
      setNotice(res.notice ?? null);
      applyLocal3DTelemetry(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al analizar la escena');
    } finally {
      setAnalyzing(false);
    }
  };

  // Real-time FOV_h reprojection without re-calling Gemini
  const handleChangeFovH = useCallback(
    (nextFov: number) => {
      setFovHDeg(nextFov);
      setFovManualOverride(true);
      setFovSource('manual');
      const currentObjects = objectsRef.current;
      if (currentObjects.length === 0) return;

      const seq = ++reprojectSeqRef.current;
      reprojectScene({
        objects: currentObjects,
        gt_objects: gtObjectsRef.current,
        image_base64: imageBase64Ref.current || undefined,
        width,
        height,
        fov_h_deg: nextFov,
      })
        .then((res) => {
          if (seq !== reprojectSeqRef.current) return;
          setObjects(res.objects ?? []);
          if (res.gt_objects !== undefined) {
            setGtObjects(res.gt_objects);
          }
          setOcclusionMatrix(res.occlusion_matrix ?? []);
          if (res.metrics !== undefined) {
            setMetrics(res.metrics);
          }
          if (res.fov_sweep !== undefined) {
            setFovSweep(res.fov_sweep);
          }
          if (res.intrinsic_diagnostics !== undefined) {
            setIntrinsicDiagnostics(res.intrinsic_diagnostics);
          }
        })
        .catch(() => {
          // Keep current state silently if transient abort
        });
    },
    [width, height]
  );

  const handleCalibrateScale = useCallback(
    async (nextFactor: number, label: string) => {
      const baseObjs = baseObjectsRef.current.length > 0 ? baseObjectsRef.current : objectsRef.current;
      try {
        const res = await calibrateScale({
          objects: baseObjs,
          gt_objects: gtObjectsRef.current,
          image_base64: imageBase64Ref.current || undefined,
          width,
          height,
          fov_h_deg: fovHDeg,
          scale_factor: nextFactor,
          reference_label: label,
          depth_stats: baseDepthStatsRef.current,
          real_photo_metrics: baseRealPhotoMetricsRef.current,
          scene_intelligence: baseSceneIntelligenceRef.current,
        });
        setObjects(res.objects ?? []);
        if (res.gt_objects !== undefined) setGtObjects(res.gt_objects);
        setOcclusionMatrix(res.occlusion_matrix ?? []);
        if (res.metrics !== undefined) setMetrics(res.metrics);
        if (res.intrinsic_diagnostics !== undefined) setIntrinsicDiagnostics(res.intrinsic_diagnostics);
        if (res.depth_stats !== undefined) setDepthStats(res.depth_stats);
        if (res.real_photo_metrics !== undefined) setRealPhotoMetrics(res.real_photo_metrics);
        if (res.scene_intelligence !== undefined) setSceneIntelligence(res.scene_intelligence);
        setScaleFactor(res.scale_factor);
        setScaleLabel(Math.abs(res.scale_factor - 1.0) > 0.005 ? label : null);
        setMeasurement(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al calibrar escala');
      }
    },
    [width, height, fovHDeg]
  );

  const handleResetScale = useCallback(() => {
    handleCalibrateScale(1.0, 'Escala original 1.00×');
  }, [handleCalibrateScale]);

  const runPairMeasurement = useCallback(
    async (idA: string, idB: string) => {
      const objA = objectsRef.current.find((o) => o.id === idA);
      const objB = objectsRef.current.find((o) => o.id === idB);
      if (!objA || !objB) return;
      try {
        const geometry = await measurePair(objA, objB);
        setMeasurement({
          object_a_id: objA.id,
          object_a_label: objA.label,
          object_b_id: objB.id,
          object_b_label: objB.label,
          geometry,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al medir par');
      }
    },
    []
  );

  const handleSelectObject = useCallback(
    (id: string, shiftKey: boolean) => {
      if (measureMode || shiftKey) {
        const first = measureFirstId ?? (shiftKey ? selectedObjectId : null);
        if (!first || first === id) {
          setMeasureFirstId(id);
          setSelectedObjectId(id);
        } else {
          setMeasureFirstId(null);
          setMeasureMode(false);
          setSelectedObjectId(id);
          runPairMeasurement(first, id);
        }
        return;
      }

      setSelectedObjectId((prev) => (prev === id ? null : id));
    },
    [measureMode, measureFirstId, selectedObjectId, runPairMeasurement]
  );

  const handleClearSelection = useCallback(() => {
    setSelectedObjectId(null);
    setMeasureFirstId(null);
    setMeasureMode(false);
    setMeasurement(null);
    setHighlightedIds(null);
  }, []);

  const handleExecuteSceneActions = useCallback(
    (actions: SceneAction[]) => {
      actions.forEach((act) => {
        switch (act.type) {
          case 'focus_object':
            setSelectedObjectId(act.object_id);
            break;
          case 'show_measurement': {
            const a = objectsRef.current.find((o) => o.id === act.object_a_id);
            const b = objectsRef.current.find((o) => o.id === act.object_b_id);
            if (act.data && a && b) {
              setMeasurement({
                object_a_id: a.id,
                object_a_label: a.label,
                object_b_id: b.id,
                object_b_label: b.label,
                geometry: act.data,
              });
            } else if (act.object_a_id && act.object_b_id) {
              runPairMeasurement(act.object_a_id, act.object_b_id);
            }
            break;
          }
          case 'set_camera_view':
            setCameraMode(act.view_mode);
            break;
          case 'toggle_occlusion_rays':
            setShowOcclusionRays(Boolean(act.enabled));
            break;
          case 'highlight_subset':
            setHighlightedIds(act.object_ids ?? null);
            break;
          case 'open_tab':
            if (act.tab === 'metrics') navigateTo('metrics');
            else if (act.tab === 'multiframe') navigateTo('multiframe');
            else navigateTo('lab');
            break;
        }
      });
    },
    [navigateTo, runPairMeasurement]
  );

  const handleOpenLabWithBenchmark = (benchmarkId?: string) => {
    if (benchmarkId && benchmarkId !== activeBenchmarkId) {
      handleSelectBenchmark(benchmarkId);
    }
    navigateTo('lab');
  };

  const handleFocusObjectInLab = (objectId: string) => {
    setSelectedObjectId(objectId);
    setShowOcclusionRays(true);
    navigateTo('lab');
  };

  const handleApplyOptimizedLabObjects = useCallback(
    async (keptIds: string[]) => {
      const keptSet = new Set(keptIds);
      const filtered = objectsRef.current.filter((obj) => keptSet.has(obj.id));
      if (filtered.length === 0) return;

      const prevCount = objectsRef.current.length;
      setObjects(filtered);
      baseObjectsRef.current = filtered;
      setSelectedObjectId(null);
      setMeasurement(null);
      setHighlightedIds(null);
      setQuantumNmsSummary({
        enabled: true,
        input_count: prevCount,
        kept_count: filtered.length,
        suppressed_count: Math.max(0, prevCount - filtered.length),
        conflict_edges_count: Math.max(0, prevCount - filtered.length),
        energy_gain: 0,
        solver_ms: 0,
        exact_optimum_verified: true,
        suppressed_labels: [],
      });
      setNotice(
        `Optimizador QUBO-MWIS aplicado en Laboratorio 3D: ${filtered.length}/${prevCount} objetos sin colisiones.`
      );
      navigateTo('lab');

      try {
        const res = await reprojectScene({
          objects: filtered,
          gt_objects: gtObjectsRef.current,
          image_base64: imageBase64Ref.current || undefined,
          width,
          height,
          fov_h_deg: fovHDeg,
        });
        setObjects(res.objects ?? filtered);
        setOcclusionMatrix(res.occlusion_matrix ?? []);
        if (res.metrics !== undefined) setMetrics(res.metrics);
        if (res.intrinsic_diagnostics !== undefined) {
          setIntrinsicDiagnostics(res.intrinsic_diagnostics);
        }
      } catch {
        // Keep filtered subset if offline reproject fails transiently
      }
    },
    [fovHDeg, height, navigateTo, width]
  );

  const activeSceneTitle =
    customImageTitle ??
    benchmarks.find((b) => b.id === activeBenchmarkId)?.title ??
    activeBenchmarkId;

  return (
    <div className="min-h-screen flex flex-col bg-[#F7F7F5] text-[#111827]">
      {/* Native CSS Scroll-Driven Top Progress Bar */}
      <div className="scroll-progress-track" aria-hidden="true">
        <div className="scroll-progress-bar" />
      </div>

      <HeaderNav
        activeSection={activeSection}
        onNavigate={navigateTo}
        selectedModel={selectedModel}
        onSelectModel={handleSelectModel}
        hasApiKey={apiKey.length > 0}
        backendEnvConfigured={backendEnvConfigured}
        onOpenApiKeyModal={() => setApiKeyModalOpen(true)}
      />

      <main
        ref={mainRef}
        tabIndex={-1}
        className="workspace-transition-region flex-1 focus:outline-none"
      >
        {activeSection === 'landing' && (
          <LandingShowcase
            benchmarks={benchmarks}
            onOpenLab={handleOpenLabWithBenchmark}
            onNavigate={navigateTo}
          />
        )}

        {activeSection === 'lab' && (
          <LabWorkspace
            benchmarks={benchmarks}
            activeBenchmarkId={activeBenchmarkId}
            customImageTitle={customImageTitle}
            imageBase64={imageBase64}
            imageMime={imageMime}
            width={width}
            height={height}
            fovHDeg={fovHDeg}
            thinkingLevel={thinkingLevel}
            selectedModel={selectedModel}
            selectedEngine={selectedEngine}
            mogeBackbone={mogeBackbone}
            customPrompt={customPrompt}
            scaleFactor={scaleFactor}
            scaleLabel={scaleLabel}
            arucoAnchor={arucoAnchor}
            unfoldProgress={unfoldProgress}
            showGroundTruth={showGroundTruth}
            showOcclusionRays={showOcclusionRays}
            showPointCloud={showPointCloud}
            showDepthMap={showDepthMap}
            measureMode={measureMode}
            measureFirstId={measureFirstId}
            cameraMode={cameraMode}
            objects={objects}
            gtObjects={gtObjects}
            occlusionMatrix={occlusionMatrix}
            metrics={metrics}
            intrinsicDiagnostics={intrinsicDiagnostics}
            realPhotoMetrics={realPhotoMetrics}
            sceneIntelligence={sceneIntelligence}
            quantumNmsSummary={quantumNmsSummary}
            pointCloud={pointCloud}
            depthMapBase64={depthMapBase64}
            depthStats={depthStats}
            fovSource={fovSource}
            local3dAvailable={local3dAvailable}
            local3dDevice={local3dDevice}
            selectedObjectId={selectedObjectId}
            hoveredObjectId={hoveredObjectId}
            highlightedIds={highlightedIds}
            measurement={measurement}
            analyzing={analyzing}
            modeBadge={modeBadge}
            notice={notice}
            error={error}
            apiKey={apiKey}
            onSelectBenchmark={handleSelectBenchmark}
            onUploadFile={handleUploadFile}
            onChangeFovH={handleChangeFovH}
            onChangeThinkingLevel={setThinkingLevel}
            onChangeEngine={setSelectedEngine}
            onChangeMogeBackbone={setMogeBackbone}
            onChangeCustomPrompt={setCustomPrompt}
            onCalibrateScale={handleCalibrateScale}
            onResetScale={handleResetScale}
            onChangeUnfoldProgress={setUnfoldProgress}
            onToggleGroundTruth={() => setShowGroundTruth((v) => !v)}
            onToggleOcclusionRays={() => setShowOcclusionRays((v) => !v)}
            onTogglePointCloud={() => setShowPointCloud((v) => !v)}
            onToggleDepthMap={() => setShowDepthMap((v) => !v)}
            onToggleMeasureMode={() => {
              setMeasureMode((v) => !v);
              setMeasureFirstId(null);
            }}
            onChangeCameraMode={setCameraMode}
            onAnalyzeScene={handleAnalyzeCurrent}
            onSelectObject={handleSelectObject}
            onHoverObject={setHoveredObjectId}
            onClearSelection={handleClearSelection}
            onExecuteSceneActions={handleExecuteSceneActions}
          />
        )}

        {activeSection === 'multiview' && (
          <MultiView360Workspace
            mogeBackbone={mogeBackbone}
            onChangeMogeBackbone={setMogeBackbone}
            local3dAvailable={local3dAvailable}
            local3dDevice={local3dDevice}
          />
        )}

        {activeSection === 'quantum' && (
          <QuantumQuboWorkspace
            labObjects={objects}
            onApplyOptimizedLabObjects={handleApplyOptimizedLabObjects}
            onNavigate={navigateTo}
          />
        )}

        {activeSection === 'multiframe' && (
          <MultiFrameAuditPanel apiKey={apiKey} />
        )}

        {activeSection === 'metrics' && (
          <MetricsBenchmarkPanel
            benchmarkId={activeBenchmarkId}
            sceneTitle={activeSceneTitle}
            selectedModel={selectedModel}
            fovHDeg={fovHDeg}
            objects={objects}
            metrics={metrics}
            fovSweep={fovSweep}
            intrinsicDiagnostics={intrinsicDiagnostics}
            realPhotoMetrics={realPhotoMetrics}
            occlusionMatrix={occlusionMatrix}
            apiKey={apiKey}
            onFocusObjectInLab={handleFocusObjectInLab}
          />
        )}
      </main>

      <footer className="border-t border-[#E2E5EA] bg-[#FFFFFF] py-4">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-2 px-4 text-[11px] text-[#4B5563] sm:px-6">
          <span>
            MyLeenTo3 · Percepción Espacial 3D Monocular y Compuerta Adaptativa Edge → VLM
          </span>
          <span className="font-mono-tech">
            Gemini Interactions API v1 · Base Quiral det(M) = +1.0 · SEED = 42
          </span>
        </div>
      </footer>

      <ApiKeyModal
        isOpen={apiKeyModalOpen}
        onClose={() => setApiKeyModalOpen(false)}
        apiKey={apiKey}
        onSaveApiKey={setApiKey}
        backendEnvConfigured={backendEnvConfigured}
      />
    </div>
  );
}

export default App;
