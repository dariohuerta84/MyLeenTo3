import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  Compass,
  Cpu,
  Download,
  GitBranch,
  Layers,
  Loader2,
  Play,
  RotateCcw,
  ShieldCheck,
} from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type {
  AppSection,
  QuantumDemoResponse,
  QuantumOptimizeResponse,
  QuantumTaskType,
  SpatialObject,
} from '../types/spatial';
import { fetchQuantumDemo, runQuantumOptimization } from '../services/api';

interface QuantumQuboWorkspaceProps {
  labObjects: SpatialObject[];
  onApplyOptimizedLabObjects?: (keptIds: string[]) => void;
  onNavigate: (section: AppSection) => void;
}

const TAB_ITEMS: Array<{
  id: QuantumTaskType;
  code: string;
  label: string;
  ref: string;
}> = [
  {
    id: 'registration_stress',
    code: 'MÓDULO 01',
    label: 'Registro 3D Anti-Simetría',
    ref: 'MAC / QuAnt + Nguyen PRX Quantum 2023',
  },
  {
    id: 'nms_3d',
    code: 'MÓDULO 02',
    label: 'NMS Global 3D (Rydberg)',
    ref: 'Ebadi et al. Science 2022 (MWIS Bloqueo)',
  },
  {
    id: 'nbv_plan',
    code: 'MÓDULO 03',
    label: 'Planificador Multi-Disparo NBV',
    ref: 'Classical Shadows (Huang, Preskill, Lukin 2022)',
  },
];

function polarToCartesian(cx: number, cy: number, r: number, azDeg: number): [number, number] {
  const rad = ((azDeg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

function buildWedgePath(
  cx: number,
  cy: number,
  rIn: number,
  rOut: number,
  startAzDeg: number,
  endAzDeg: number
): string {
  const [x1, y1] = polarToCartesian(cx, cy, rOut, startAzDeg);
  const [x2, y2] = polarToCartesian(cx, cy, rOut, endAzDeg);
  const [x3, y3] = polarToCartesian(cx, cy, rIn, endAzDeg);
  const [x4, y4] = polarToCartesian(cx, cy, rIn, startAzDeg);
  return [
    `M ${x1.toFixed(2)} ${y1.toFixed(2)}`,
    `A ${rOut} ${rOut} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`,
    `L ${x3.toFixed(2)} ${y3.toFixed(2)}`,
    `A ${rIn} ${rIn} 0 0 0 ${x4.toFixed(2)} ${y4.toFixed(2)}`,
    'Z',
  ].join(' ');
}

function downloadJsonFile(filename: string, payload: unknown): void {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export const QuantumQuboWorkspace: React.FC<QuantumQuboWorkspaceProps> = ({
  labObjects,
  onApplyOptimizedLabObjects,
  onNavigate,
}) => {
  const [activeTask, setActiveTask] = useState<QuantumTaskType>('registration_stress');
  const [demoSuite, setDemoSuite] = useState<QuantumDemoResponse | null>(null);
  const [scenarios, setScenarios] = useState<Record<string, QuantumOptimizeResponse>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [runningTask, setRunningTask] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [appliedFlash, setAppliedFlash] = useState<string | null>(null);

  // Module 1 controls (registration_stress)
  const [regCount, setRegCount] = useState<number>(72);
  const [regOutlierPct, setRegOutlierPct] = useState<number>(75);
  const [regAllowScale, setRegAllowScale] = useState<boolean>(true);

  // Module 2 controls (nms_3d)
  const [nmsIouThresh, setNmsIouThresh] = useState<number>(0.28);
  const [nmsUseLabScene, setNmsUseLabScene] = useState<boolean>(false);

  // Module 3 controls (nbv_plan)
  const [nbvSectorsActive, setNbvSectorsActive] = useState<number[]>([0, 1]);
  const [nbvBudgetK, setNbvBudgetK] = useState<number>(2);

  const loadSuite = useCallback(async () => {
    setLoading(true);
    setError(null);
    setAppliedFlash(null);
    try {
      const suite = await fetchQuantumDemo();
      setDemoSuite(suite);
      setScenarios(suite.scenarios ?? {});
      setRegCount(72);
      setRegOutlierPct(75);
      setRegAllowScale(true);
      setNmsIouThresh(0.28);
      setNmsUseLabScene(false);
      setNbvSectorsActive([0, 1]);
      setNbvBudgetK(2);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'No se pudo inicializar el motor QUBO-MWIS (Fase 4)'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSuite();
  }, [loadSuite]);

  const currentResult: QuantumOptimizeResponse | null =
    scenarios[activeTask] ?? null;

  const executeLiveOptimization = async (
    taskOverride?: QuantumTaskType,
    useLabOverride?: boolean,
    customAzimuths?: number[],
    customBudget?: number
  ) => {
    const task = taskOverride ?? activeTask;
    setRunningTask(true);
    setError(null);
    setAppliedFlash(null);
    try {
      let res: QuantumOptimizeResponse;
      if (task === 'registration_stress') {
        res = await runQuantumOptimization({
          task,
          n_correspondences: regCount,
          outlier_ratio: regOutlierPct / 100.0,
          allow_scale: regAllowScale,
        });
      } else if (task === 'nms_3d') {
        const useLab = useLabOverride ?? nmsUseLabScene;
        res = await runQuantumOptimization({
          task,
          conflict_iou_threshold: nmsIouThresh,
          objects: useLab && labObjects.length >= 2 ? labObjects : undefined,
        });
      } else {
        const sectors = customAzimuths ?? nbvSectorsActive;
        const azDegrees = sectors.map((secIdx) => secIdx * 30.0 + 15.0);
        res = await runQuantumOptimization({
          task: 'nbv_plan',
          existing_azimuths_deg: azDegrees,
          budget_k: customBudget ?? nbvBudgetK,
        });
      }
      setScenarios((prev) => ({ ...prev, [task]: res }));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Error ejecutando optimización QUBO-MWIS'
      );
    } finally {
      setRunningTask(false);
    }
  };

  const handleToggleNbvSector = (secIdx: number) => {
    const exists = nbvSectorsActive.includes(secIdx);
    const next = exists
      ? nbvSectorsActive.filter((s) => s !== secIdx)
      : [...nbvSectorsActive, secIdx].sort((a, b) => a - b);
    const safeNext = next.length > 0 ? next : [secIdx];
    setNbvSectorsActive(safeNext);
    executeLiveOptimization('nbv_plan', undefined, safeNext, nbvBudgetK);
  };

  const qStats = (currentResult?.qubo_stats ?? {}) as Record<string, number | boolean | string>;
  const baseCmp = (currentResult?.baseline_comparison ?? {}) as Record<string, unknown>;
  const details = (currentResult?.details ?? {}) as Record<string, unknown>;
  const hwExport = currentResult?.hardware_export;
  const aquila = hwExport?.quera_aquila_register;
  const brk = hwExport?.bqm_braket;

  const nodeCoordsMap = useMemo(() => {
    const map = new Map<number, { x: number; y: number }>();
    for (const n of currentResult?.graph_preview?.nodes ?? []) {
      map.set(n.id, { x: n.x, y: n.y });
    }
    return map;
  }, [currentResult?.graph_preview?.nodes]);

  const greedyRefEnergy = Number(qStats.greedy_energy ?? 0);

  const nbvPredScores = (details.predicted_sector_scores as number[] | undefined) ?? Array(12).fill(0);
  const nbvShadowUnc = (details.shadow_uncertainty_by_sector as number[] | undefined) ?? Array(12).fill(1);
  const nbvOptimalAngles = (details.optimal_shot_plan_deg as number[] | undefined) ?? [];

  return (
    <div className="mx-auto max-w-[1600px] space-y-4 px-4 py-5 sm:px-6">
      {/* Editorial Header */}
      <header className="border-b border-[#15284B]/18 pb-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#9F1239]">
              FASE 4 · MOTOR COMBINATORIO QUBO-MWIS (HARVARD QUANTUM INITIATIVE / RYDBERG BLOQUEO)
            </p>
            <h1 className="font-serif-editorial mt-1 text-2xl font-medium tracking-tight text-[#111827] sm:text-3xl">
              Optimizador Combinatorio Cuántico-Inspirado y Puente QPU
            </h1>
            <p className="mt-1 max-w-4xl text-xs leading-relaxed text-[#4B5563]">
              Formula correspondencias 3D isométricas, supresión de colisiones SAT 15-ejes y planificación multi-disparo ActivePose NBV como Hamiltoniano Ising/QUBO de átomos neutros resuelto por Bifurcación Simulada Balística (bSB) + Branch-and-Bound, con exportación directa BQM / QuEra Aquila.
            </p>
          </div>

          <div className="font-mono-tech flex items-center gap-3 text-[11px] text-[#4B5563]">
            <span>
              Motor: <strong className="text-[#15284B]">{demoSuite?.engine_version ?? 'bSB + MWIS'}</strong>
            </span>
            {currentResult && (
              <>
                <span className="text-[#CBD5E1]">|</span>
                <span>
                  Solver: <strong className="text-[#0F766E]">{Number(currentResult.latency_ms).toFixed(1)} ms</strong>
                </span>
              </>
            )}
            <button
              type="button"
              onClick={loadSuite}
              disabled={loading || runningTask}
              className="inline-flex items-center gap-1.5 border border-[#15284B]/22 bg-[#FFFFFF] px-2.5 py-1 text-[11px] font-medium text-[#15284B] transition-colors hover:border-[#15284B] disabled:opacity-50"
            >
              <RotateCcw className="h-3 w-3" />
              <span>Restaurar SEED=42</span>
            </button>
          </div>
        </div>

        {/* Clean Bottom-Border Tabs (Zero Pill Badges) */}
        <nav
          aria-label="Módulos combinatorios QUBO"
          className="mt-4 flex flex-wrap gap-6 border-t border-[#E2E5EA] pt-3"
        >
          {TAB_ITEMS.map((tab) => {
            const active = activeTask === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => {
                  setActiveTask(tab.id);
                  setAppliedFlash(null);
                }}
                className={`border-b-2 pb-2 text-left transition-colors ${
                  active
                    ? 'border-[#15284B] text-[#111827]'
                    : 'border-transparent text-[#4B5563] hover:text-[#111827]'
                }`}
              >
                <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#0F766E]">
                  {tab.code}
                </div>
                <div className="text-xs font-semibold">{tab.label}</div>
                <div className="font-mono-tech text-[10px] text-[#4B5563]">{tab.ref}</div>
              </button>
            );
          })}
        </nav>

        {(error || appliedFlash) && (
          <div className="mt-2.5 flex flex-wrap items-center gap-3 text-xs">
            {error && (
              <span className="inline-flex items-center gap-1.5 text-[#BE123C]">
                <AlertCircle className="h-3.5 w-3.5" />
                {error}
              </span>
            )}
            {appliedFlash && (
              <span className="inline-flex items-center gap-1.5 text-[#0F766E]">
                <ShieldCheck className="h-3.5 w-3.5" />
                {appliedFlash}
              </span>
            )}
          </div>
        )}
      </header>

      {/* Parameter Control Strip for Active Task */}
      <section className="flex flex-wrap items-center gap-4 border border-[#15284B]/16 bg-[#FFFFFF] px-4 py-3 text-xs">
        {activeTask === 'registration_stress' && (
          <>
            <label className="flex items-center gap-2">
              <span className="text-[#4B5563]">Correspondencias</span>
              <input
                type="range"
                min={24}
                max={96}
                step={6}
                value={regCount}
                onChange={(e) => setRegCount(Number(e.target.value))}
                className="optical-slider w-24"
              />
              <span className="font-mono-tech font-semibold text-[#111827]">{regCount}</span>
            </label>

            <label className="flex items-center gap-2">
              <span className="text-[#4B5563]">Contaminación outliers</span>
              <input
                type="range"
                min={20}
                max={85}
                step={5}
                value={regOutlierPct}
                onChange={(e) => setRegOutlierPct(Number(e.target.value))}
                className="optical-slider w-24"
              />
              <span className="font-mono-tech font-semibold text-[#BE123C]">{regOutlierPct}%</span>
            </label>

            <label className="inline-flex cursor-pointer items-center gap-1.5 border border-[#E2E5EA] px-2.5 py-1">
              <input
                type="checkbox"
                checked={regAllowScale}
                onChange={(e) => setRegAllowScale(e.target.checked)}
                className="accent-[#15284B]"
              />
              <span>Estimación Sim(3) con escala</span>
            </label>
          </>
        )}

        {activeTask === 'nms_3d' && (
          <>
            <label className="flex items-center gap-2">
              <span className="text-[#4B5563]">Umbral conflicto IoU 3D</span>
              <input
                type="range"
                min={0.1}
                max={0.6}
                step={0.02}
                value={nmsIouThresh}
                onChange={(e) => setNmsIouThresh(Number(e.target.value))}
                className="optical-slider w-28"
              />
              <span className="font-mono-tech font-semibold text-[#111827]">
                {nmsIouThresh.toFixed(2)}
              </span>
            </label>

            <button
              type="button"
              onClick={() => {
                const next = !nmsUseLabScene;
                setNmsUseLabScene(next);
                executeLiveOptimization('nms_3d', next);
              }}
              disabled={labObjects.length < 2 || runningTask}
              className={`inline-flex items-center gap-1.5 border px-3 py-1.5 font-medium transition-colors disabled:opacity-45 ${
                nmsUseLabScene
                  ? 'border-[#0F766E] bg-[#0F766E] text-white'
                  : 'border-[#15284B]/22 bg-[#FAF8F4] text-[#15284B] hover:border-[#15284B]'
              }`}
            >
              <Layers className="h-3.5 w-3.5" />
              <span>
                {nmsUseLabScene
                  ? `Escena Laboratorio Activa (${labObjects.length} OBBs)`
                  : `Usar Escena Activa del Lab (${labObjects.length} OBBs)`}
              </span>
            </button>

            {nmsUseLabScene &&
              onApplyOptimizedLabObjects &&
              Array.isArray(details.kept_ids) &&
              (details.suppressed_labels as string[] | undefined)?.length ? (
              <button
                type="button"
                onClick={() => {
                  const keptIds = (details.kept_ids as string[]) ?? [];
                  onApplyOptimizedLabObjects(keptIds);
                  setAppliedFlash(
                    `Subconjunto óptimo (${keptIds.length} objetos) aplicado al Laboratorio 3D.`
                  );
                }}
                className="inline-flex items-center gap-1.5 border border-[#0F766E] bg-[#F4FBF9] px-3 py-1.5 font-medium text-[#0F766E] hover:bg-[#0F766E] hover:text-white"
              >
                <span>Aplicar Selección Sin Colisiones al Lab</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            ) : null}
          </>
        )}

        {activeTask === 'nbv_plan' && (
          <>
            <div className="flex items-center gap-2">
              <span className="text-[#4B5563]">Presupuesto disparos K</span>
              {[1, 2, 3, 4].map((kVal) => (
                <button
                  key={kVal}
                  type="button"
                  onClick={() => {
                    setNbvBudgetK(kVal);
                    executeLiveOptimization('nbv_plan', undefined, nbvSectorsActive, kVal);
                  }}
                  className={`font-mono-tech border px-2.5 py-1 text-xs font-semibold ${
                    nbvBudgetK === kVal
                      ? 'border-[#15284B] bg-[#15284B] text-white'
                      : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#4B5563] hover:text-[#111827]'
                  }`}
                >
                  K={kVal}
                </button>
              ))}
            </div>
            <span className="font-mono-tech text-[11px] text-[#4B5563]">
              Haz clic en los sectores del anillo polar para simular cámaras ya capturadas.
            </span>
          </>
        )}

        <button
          type="button"
          onClick={() => executeLiveOptimization()}
          disabled={loading || runningTask}
          className="ml-auto inline-flex items-center gap-1.5 bg-[#15284B] px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-[#1D4ED8] disabled:opacity-50"
        >
          {runningTask || loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5" />
          )}
          <span>Resolver Hamiltoniano QUBO</span>
        </button>
      </section>

      {loading && !currentResult ? (
        <div className="flex h-72 items-center justify-center border border-[#E2E5EA] bg-[#FFFFFF] text-xs text-[#4B5563]">
          <Loader2 className="mr-2 h-4 w-4 animate-spin text-[#15284B]" />
          Construyendo instancia QUBO-MWIS SEED=42…
        </div>
      ) : currentResult ? (
        <>
          {/* Summary Narrative + Architectural Metrics Ledger (Divide-Y Hairlines) */}
          <section className="grid gap-4 lg:grid-cols-12">
            <div className="border border-[#15284B]/18 border-l-2 border-l-[#15284B] bg-[#FFFFFF] p-4 lg:col-span-5">
              <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#15284B]">
                Resumen Cuantitativo · Veredicto Combinatorio
              </div>
              <h2 className="font-serif-editorial mt-1 text-xl text-[#111827]">
                {currentResult.title_es}
              </h2>
              <p className="mt-2 text-xs leading-relaxed text-[#4B5563]">
                {currentResult.summary_es}
              </p>

              <dl className="mt-4 divide-y divide-[#E2E5EA] border-t border-[#E2E5EA] text-xs">
                <div className="flex items-center justify-between py-2">
                  <dt className="text-[#4B5563]">Variables binarias QUBO</dt>
                  <dd className="font-mono-tech font-semibold text-[#111827]">
                    {String(qStats.n_vars ?? 0)} qubits lógicos ·{' '}
                    {String(qStats.n_conflict_edges ?? 0)} acoplamientos
                  </dd>
                </div>
                <div className="flex items-center justify-between py-2">
                  <dt className="text-[#4B5563]">Energía QUBO vs. Greedy</dt>
                  <dd className="font-mono-tech font-semibold text-[#0F766E]">
                    {Number(qStats.qubo_energy ?? 0).toFixed(2)} vs.{' '}
                    {Number(qStats.greedy_energy ?? 0).toFixed(2)} (Δ{' '}
                    {Number(qStats.energy_gain ?? 0).toFixed(2)})
                  </dd>
                </div>
                {activeTask === 'registration_stress' && (
                  <>
                    <div className="flex items-center justify-between py-2">
                      <dt className="text-[#4B5563]">Inliers (QUBO-MWIS vs. RANSAC)</dt>
                      <dd className="font-mono-tech font-semibold text-[#1D4ED8]">
                        {String(baseCmp.qubo_inliers ?? 0)} vs. {String(baseCmp.ransac_inliers ?? 0)}{' '}
                        ({String(baseCmp.qubo_rmse_mm ?? 0)} mm RMSE)
                      </dd>
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <dt className="text-[#4B5563]">Error rotacional / traslación / escala</dt>
                      <dd className="font-mono-tech font-semibold text-[#111827]">
                        {String(baseCmp.rotation_error_deg ?? 0)}° ·{' '}
                        {String(baseCmp.translation_error_mm ?? 0)} mm ·{' '}
                        {String(baseCmp.scale_error_pct ?? 0)}%
                      </dd>
                    </div>
                  </>
                )}
                {activeTask === 'nms_3d' && (
                  <>
                    <div className="flex items-center justify-between py-2">
                      <dt className="text-[#4B5563]">Objetos retenidos (QUBO vs. Greedy)</dt>
                      <dd className="font-mono-tech font-semibold text-[#1D4ED8]">
                        {String(baseCmp.qubo_kept_count ?? 0)} vs. {String(baseCmp.greedy_kept_count ?? 0)}{' '}
                        (peso {Number(baseCmp.qubo_weight_sum ?? 0).toFixed(2)} vs.{' '}
                        {Number(baseCmp.greedy_weight_sum ?? 0).toFixed(2)})
                      </dd>
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <dt className="text-[#4B5563]">Cajas suprimidas por bloqueo</dt>
                      <dd className="font-mono-tech max-w-[220px] truncate text-right text-[#BE123C]">
                        {((details.suppressed_labels as string[] | undefined) ?? []).join(', ') || 'ninguna'}
                      </dd>
                    </div>
                  </>
                )}
                {activeTask === 'nbv_plan' && (
                  <>
                    <div className="flex items-center justify-between py-2">
                      <dt className="text-[#4B5563]">Cobertura 360° (Actual → Plan QUBO)</dt>
                      <dd className="font-mono-tech font-semibold text-[#0F766E]">
                        {String(baseCmp.current_coverage_pct ?? 0)}% →{' '}
                        {String(baseCmp.qubo_predicted_coverage_pct ?? 0)}%
                      </dd>
                    </div>
                    <div className="flex items-center justify-between py-2">
                      <dt className="text-[#4B5563]">Tomas óptimas (ahorro vs. 6 fijas)</dt>
                      <dd className="font-mono-tech font-semibold text-[#1D4ED8]">
                        [{nbvOptimalAngles.map((a) => `${a.toFixed(0)}°`).join(', ')}] (−
                        {String(baseCmp.shots_saved_vs_uniform_6 ?? 0)} fotos)
                      </dd>
                    </div>
                  </>
                )}
              </dl>
            </div>

            {/* Energy Convergence LineChart (Simulated Bifurcation vs Classical Greedy) */}
            <div className="border border-[#15284B]/18 bg-[#FFFFFF] p-4 lg:col-span-7">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E2E5EA] pb-2">
                <h3 className="flex items-center gap-1.5 text-xs font-semibold text-[#111827]">
                  <GitBranch className="h-3.5 w-3.5 text-[#1D4ED8]" />
                  <span>Trayectoria Simpléctica bSB vs. Baseline Greedy Clásico</span>
                </h3>
                <span className="font-mono-tech text-[10px] text-[#4B5563]">
                  H(x) = xᵀ Q x · Minimizando energía de bloqueo Rydberg
                </span>
              </div>
              <div className="mt-3 h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={currentResult.energy_history}
                    margin={{ top: 8, right: 16, left: 4, bottom: 4 }}
                  >
                    <CartesianGrid stroke="#E2E5EA" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="step"
                      tick={{ fontSize: 10, fill: '#4B5563', fontFamily: 'JetBrains Mono' }}
                      label={{
                        value: 'Iteración adiabática',
                        position: 'insideBottomRight',
                        offset: -2,
                        fontSize: 10,
                        fill: '#4B5563',
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: '#4B5563', fontFamily: 'JetBrains Mono' }}
                      width={52}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#FFFFFF',
                        borderColor: '#15284B',
                        fontSize: '11px',
                        fontFamily: 'JetBrains Mono',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', fontFamily: 'JetBrains Mono' }} />
                    <ReferenceLine
                      y={greedyRefEnergy}
                      stroke="#B45309"
                      strokeDasharray="5 4"
                      label={{
                        value: `Greedy (${greedyRefEnergy.toFixed(2)})`,
                        position: 'insideTopRight',
                        fill: '#B45309',
                        fontSize: 10,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="best_energy"
                      name="QUBO bSB Mejor"
                      stroke="#1D4ED8"
                      strokeWidth={2.2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="mean_energy"
                      name="Media Enjambre"
                      stroke="#0F766E"
                      strokeWidth={1.5}
                      strokeDasharray="4 2"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          {/* Bottom Split: Left Graph / Polar Set-Cover Diagram + Right QuEra Aquila Hardware Register */}
          <section className="grid gap-4 lg:grid-cols-12">
            {/* Compatibility / Blockade Graph OR Interactive Polar NBV Ring */}
            <div className="border border-[#15284B]/18 bg-[#FFFFFF] p-4 lg:col-span-6">
              <div className="flex items-center justify-between border-b border-[#E2E5EA] pb-2">
                <h3 className="flex items-center gap-1.5 text-xs font-semibold text-[#111827]">
                  <Compass className="h-3.5 w-3.5 text-[#0F766E]" />
                  <span>
                    {activeTask === 'nbv_plan'
                      ? 'Anillo Polar Set-Cover (Classical Shadows + Clic Sectores)'
                      : 'Grafo de Conflicto / Bloqueo Rydberg (MWIS)'}
                  </span>
                </h3>
                <span className="font-mono-tech text-[10px] text-[#4B5563]">
                  {activeTask === 'nbv_plan'
                    ? `${nbvSectorsActive.length} cámaras base`
                    : `${currentResult.graph_preview.shown_nodes}/${currentResult.graph_preview.total_nodes} nodos`}
                </span>
              </div>

              {activeTask === 'nbv_plan' ? (
                <div className="mt-3 flex flex-col items-center gap-4 sm:flex-row">
                  <svg
                    viewBox="0 0 220 220"
                    className="h-60 w-60 shrink-0"
                    aria-label="Anillo interactivo NBV Set-Cover QUBO"
                  >
                    <circle cx="110" cy="110" r="92" fill="#FAF8F4" stroke="#E2E5EA" strokeWidth="1" />
                    <circle cx="110" cy="110" r="52" fill="#FFFFFF" stroke="#E2E5EA" strokeWidth="1" />

                    {Array.from({ length: 12 }).map((_, idx) => {
                      const sAz = idx * 30;
                      const eAz = sAz + 30;
                      const isExisting = nbvSectorsActive.includes(idx);
                      const isPlanned = (
                        (details.optimal_sector_indices as number[] | undefined) ?? []
                      ).includes(idx);
                      const pred = nbvPredScores[idx] ?? 0;
                      const shadowVal = nbvShadowUnc[idx] ?? 1.0;

                      const fillColor = isExisting
                        ? '#1D4ED8'
                        : isPlanned
                        ? '#B45309'
                        : pred >= 0.25
                        ? '#0F766E'
                        : '#FFFFFF';
                      const fillOp = isExisting || isPlanned ? 0.85 : pred >= 0.25 ? 0.35 : 1.0;

                      return (
                        <path
                          key={`nbv-sec-${idx}`}
                          d={buildWedgePath(110, 110, 52, 90, sAz, eAz)}
                          fill={fillColor}
                          fillOpacity={fillOp}
                          stroke="#CBD5E1"
                          strokeWidth="1"
                          className="cursor-pointer transition-opacity hover:opacity-80"
                          onClick={() => handleToggleNbvSector(idx)}
                        >
                          <title>
                            Sector {sAz}–{eAz}° · Incertidumbre Shadow {shadowVal.toFixed(2)} (clic para alternar cámara)
                          </title>
                        </path>
                      );
                    })}

                    {/* Optimal QUBO Shot Vectors */}
                    {nbvOptimalAngles.map((ang, sIdx) => {
                      const [tx, ty] = polarToCartesian(110, 110, 82, ang);
                      return (
                        <g key={`shot-vec-${sIdx}`}>
                          <line
                            x1="110"
                            y1="110"
                            x2={tx}
                            y2={ty}
                            stroke="#9F1239"
                            strokeWidth="2"
                            strokeDasharray="4 2"
                          />
                          <circle cx={tx} cy={ty} r="5.5" fill="#9F1239" stroke="#FFFFFF" strokeWidth="1.5" />
                        </g>
                      );
                    })}

                    <text
                      x="110"
                      y="107"
                      textAnchor="middle"
                      fontSize="16"
                      fontWeight="700"
                      fontFamily="JetBrains Mono"
                      fill="#111827"
                    >
                      {String(baseCmp.qubo_predicted_coverage_pct ?? 0)}%
                    </text>
                    <text
                      x="110"
                      y="122"
                      textAnchor="middle"
                      fontSize="8.5"
                      fontFamily="JetBrains Mono"
                      fill="#4B5563"
                    >
                      PLAN K={nbvBudgetK} TOMAS
                    </text>
                  </svg>

                  <div className="space-y-2 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-2.5 w-2.5 bg-[#1D4ED8]" />
                      <span>Cámara base activa (clic para toggle)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-2.5 w-2.5 bg-[#B45309]" />
                      <span>Disparo óptimo recomendado QUBO</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-2.5 w-2.5 bg-[#0F766E]/60" />
                      <span>Sector cubierto por cono 52°</span>
                    </div>
                    <p className="border-t border-[#E2E5EA] pt-2 text-[11px] leading-relaxed text-[#4B5563]">
                      La ponderación de sombras aleatorias (<em>Classical Shadows</em>) prioriza brechas con mayor déficit informacional para minimizar el número de capturas necesarias.
                    </p>
                  </div>
                </div>
              ) : (
                <div className="mt-3 grid items-center gap-4 sm:grid-cols-12">
                  <svg
                    viewBox="0 0 320 260"
                    className="h-60 w-full border border-[#E2E5EA] bg-[#FAF8F4] sm:col-span-7"
                    aria-label="Grafo de conflicto y conjunto independiente máximo MWIS"
                  >
                    {currentResult.graph_preview.edges.map((edge, idx) => {
                      const p1 = nodeCoordsMap.get(edge.source);
                      const p2 = nodeCoordsMap.get(edge.target);
                      if (!p1 || !p2) return null;
                      return (
                        <line
                          key={`ge-${idx}`}
                          x1={p1.x * 320}
                          y1={p1.y * 260}
                          x2={p2.x * 320}
                          y2={p2.y * 260}
                          stroke="#BE123C"
                          strokeOpacity="0.24"
                          strokeWidth="1"
                        />
                      );
                    })}
                    {currentResult.graph_preview.nodes.map((node) => (
                      <g key={`gn-${node.id}`}>
                        <circle
                          cx={node.x * 320}
                          cy={node.y * 260}
                          r={node.selected ? 6.2 : 4.2}
                          fill={node.selected ? '#0F766E' : '#CBD5E1'}
                          stroke={node.selected ? '#15284B' : '#64748B'}
                          strokeWidth={node.selected ? 1.5 : 0.8}
                        >
                          <title>
                            {node.label} · peso={node.weight} · {node.selected ? 'SELECCIONADO' : 'suprimido'}
                          </title>
                        </circle>
                      </g>
                    ))}
                  </svg>

                  <div className="space-y-2.5 text-xs sm:col-span-5">
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-3 w-3 rounded-full border border-[#15284B] bg-[#0F766E]" />
                      <span className="font-medium text-[#111827]">Nodo retenido (Conjunto Independiente)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-2.5 w-2.5 rounded-full border border-[#64748B] bg-[#CBD5E1]" />
                      <span className="text-[#4B5563]">Nodo bloqueado / outlier</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-block h-0.5 w-4 bg-[#BE123C]/55" />
                      <span className="text-[#4B5563]">Arista de conflicto físico (U_ij &gt; λ)</span>
                    </div>
                    <div className="border-t border-[#E2E5EA] pt-2 font-mono-tech text-[11px] text-[#4B5563]">
                      Densidad conflicto: {(Number(qStats.conflict_density ?? 0) * 100).toFixed(1)}%
                      <br />
                      Violaciones MWIS: <strong className="text-[#0F766E]">0</strong>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* QuEra Aquila 256-qubit Physical Register & AWS Braket Export */}
            {aquila && brk && (
              <div className="border border-[#15284B]/18 bg-[#101C31] p-4 text-[#F8FAFC] lg:col-span-6">
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/15 pb-2">
                  <h3 className="flex items-center gap-1.5 text-xs font-semibold text-[#F8FAFC]">
                    <Cpu className="h-3.5 w-3.5 text-[#F59E0B]" />
                    <span>Puente Hardware: QuEra Aquila (Átomos Neutros) & AWS Braket BQM</span>
                  </h3>
                  <span className="font-mono-tech text-[10px] text-[#14B8A6]">
                    R_b = {aquila.rydberg_blockade_radius_um} µm · d_min ≥ {aquila.min_atom_spacing_um} µm
                  </span>
                </div>

                <div className="mt-3 grid items-center gap-4 sm:grid-cols-12">
                  {/* 75 x 76 um Optical Tweezer Atom Array SVG */}
                  <svg
                    viewBox="0 0 150 152"
                    className="h-56 w-full border border-white/15 bg-[#0A1324] sm:col-span-6"
                    aria-label="Registro físico de pinzas ópticas 75x76 micrómetros para QuEra Aquila"
                  >
                    {/* FOV calibration grid */}
                    <rect x="4" y="4" width="142" height="144" fill="none" stroke="#334155" strokeDasharray="3 3" strokeWidth="0.8" />
                    {aquila.atoms_um
                      .filter((a) => a.selected)
                      .map((a) => (
                        <circle
                          key={`blk-${a.id}`}
                          cx={a.x_um * 2}
                          cy={a.y_um * 2}
                          r={aquila.rydberg_blockade_radius_um * 2}
                          fill="#14B8A6"
                          fillOpacity="0.09"
                          stroke="#14B8A6"
                          strokeOpacity="0.45"
                          strokeDasharray="2.5 2.5"
                          strokeWidth="0.7"
                        />
                      ))}
                    {aquila.atoms_um.map((atom) => (
                      <circle
                        key={`atom-${atom.id}`}
                        cx={atom.x_um * 2}
                        cy={atom.y_um * 2}
                        r={atom.selected ? 3.2 : 2.3}
                        fill={atom.selected ? '#F59E0B' : '#60A5FA'}
                        stroke="#FFFFFF"
                        strokeWidth={atom.selected ? 0.9 : 0.4}
                      >
                        <title>
                          {atom.label} ({atom.x_um}, {atom.y_um} µm)
                        </title>
                      </circle>
                    ))}
                  </svg>

                  <div className="space-y-3 sm:col-span-6">
                    <dl className="divide-y divide-white/12 text-[11px]">
                      <div className="flex items-center justify-between py-1.5">
                        <dt className="text-[#CBD5E1]">Campo pinzas ópticas</dt>
                        <dd className="font-mono-tech text-[#F8FAFC]">
                          {aquila.field_of_view_um[0]} × {aquila.field_of_view_um[1]} µm
                        </dd>
                      </div>
                      <div className="flex items-center justify-between py-1.5">
                        <dt className="text-[#CBD5E1]">Separación mínima medida</dt>
                        <dd className="font-mono-tech text-[#14B8A6]">
                          {aquila.observed_min_spacing_um.toFixed(2)} µm
                        </dd>
                      </div>
                      <div className="flex items-center justify-between py-1.5">
                        <dt className="text-[#CBD5E1]">Fidelidad Unit-Disk 2D</dt>
                        <dd className="font-mono-tech text-[#F59E0B]">
                          {(aquila.unit_disk_fidelity * 100).toFixed(1)}%
                        </dd>
                      </div>
                      <div className="flex items-center justify-between py-1.5">
                        <dt className="text-[#CBD5E1]">Qubits estimados (+gadgets PRX23)</dt>
                        <dd className="font-mono-tech text-[#F8FAFC]">
                          {aquila.total_estimated_qubits} / {aquila.max_physical_qubits}
                        </dd>
                      </div>
                      <div className="flex items-center justify-between py-1.5">
                        <dt className="text-[#CBD5E1]">Despacho Aquila directo</dt>
                        <dd
                          className={`font-mono-tech font-semibold ${
                            aquila.hardware_ready ? 'text-[#14B8A6]' : 'text-[#F59E0B]'
                          }`}
                        >
                          {aquila.hardware_ready ? 'LISTO (≤48 nodos emb.)' : 'Subgrafo híbrido clásico'}
                        </dd>
                      </div>
                    </dl>

                    <div className="flex flex-col gap-2 pt-1">
                      <button
                        type="button"
                        onClick={() =>
                          downloadJsonFile(`myleento3_bqm_${activeTask}.json`, brk)
                        }
                        className="inline-flex items-center justify-center gap-1.5 border border-[#60A5FA]/50 bg-[#162540] px-3 py-1.5 text-[11px] font-medium text-[#F8FAFC] transition-colors hover:border-[#60A5FA]"
                      >
                        <Download className="h-3.5 w-3.5 text-[#60A5FA]" />
                        <span>Descargar BQM / AWS Braket JSON ({brk.num_variables} vars)</span>
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          downloadJsonFile(`myleento3_quera_aquila_${activeTask}.json`, aquila)
                        }
                        className="inline-flex items-center justify-center gap-1.5 border border-[#F59E0B]/50 bg-[#162540] px-3 py-1.5 text-[11px] font-medium text-[#F8FAFC] transition-colors hover:border-[#F59E0B]"
                      >
                        <Download className="h-3.5 w-3.5 text-[#F59E0B]" />
                        <span>Descargar Registro QuEra Aquila (µm)</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* Footer Quick Cross-Navigation */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[#E2E5EA] pt-3 text-[11px] text-[#4B5563]">
            <span>
              Integrado nativamente en <strong className="text-[#111827]">/api/analyze</strong> (NMS 3D),{' '}
              <strong className="text-[#111827]">/api/multiview-reconstruct</strong> (Sim(3)+NBV) y{' '}
              <strong className="text-[#111827]">Copiloto Espacial</strong> (función #9).
            </span>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => onNavigate('multiview')}
                className="font-medium text-[#1D4ED8] hover:underline"
              >
                Abrir Reconstrucción 360° →
              </button>
              <button
                type="button"
                onClick={() => onNavigate('lab')}
                className="font-medium text-[#0F766E] hover:underline"
              >
                Volver al Laboratorio 3D →
              </button>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};
