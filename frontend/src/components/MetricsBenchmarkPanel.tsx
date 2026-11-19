import React, { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  ArrowUpRight,
  Download,
  GitCompare,
  Layers,
  Loader2,
  ShieldAlert,
} from 'lucide-react';
import type {
  CompareModelsResponse,
  FovSweepPoint,
  OcclusionEntry,
  SceneMetrics,
  SpatialObject,
} from '../types/spatial';
import { compareModels } from '../services/api';
import {
  exportPosesToCsv,
  exportPosesToJson,
  formatDegrees,
  formatMeters,
  formatPercent,
} from '../utils/geometry';

interface MetricsBenchmarkPanelProps {
  benchmarkId: string;
  sceneTitle: string;
  selectedModel: string;
  fovHDeg: number;
  objects: SpatialObject[];
  metrics?: SceneMetrics | null;
  fovSweep?: FovSweepPoint[] | null;
  occlusionMatrix: OcclusionEntry[];
  apiKey: string;
  onFocusObjectInLab: (objectId: string) => void;
}

export const MetricsBenchmarkPanel: React.FC<MetricsBenchmarkPanelProps> = ({
  benchmarkId,
  sceneTitle,
  selectedModel,
  fovHDeg,
  objects,
  metrics,
  fovSweep,
  occlusionMatrix,
  apiKey,
  onFocusObjectInLab,
}) => {
  const [comparison, setComparison] = useState<CompareModelsResponse | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  const handleCompareModels = async () => {
    setComparing(true);
    setCompareError(null);
    try {
      const res = await compareModels(benchmarkId || 'lab_desk_calibrated', fovHDeg, false, apiKey);
      setComparison(res);
    } catch (err) {
      setCompareError(
        err instanceof Error ? err.message : 'Error al comparar modelos'
      );
    } finally {
      setComparing(false);
    }
  };

  const activeFovSweep = comparison?.fov_sweep ?? fovSweep ?? [];

  const comparisonChartData =
    comparison?.results.map((r) => ({
      model: String(r.model).replace('gemini-', ''),
      mIoU_3D: Number(((r.metrics.mIoU_3D ?? 0) * 100).toFixed(1)),
      mIoU_BEV: Number(((r.metrics.mIoU_BEV ?? 0) * 100).toFixed(1)),
      AP25: Number(((r.metrics.ap_25 ?? 0) * 100).toFixed(1)),
    })) ?? [];

  return (
    <div className="mx-auto max-w-[1440px] space-y-6 px-4 py-6 sm:px-6">
      {/* Header strip */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#E2E5EA] pb-4">
        <div>
          <h1 className="font-serif-editorial text-3xl text-[#111827]">
            Evaluación Métrica & Comparativa Multi-Modelo
          </h1>
          <p className="mt-1 text-xs text-[#4B5563]">
            Emparejamiento Húngaro, IoU 3D/BEV exacto, intervalos Bootstrap 95% (n=1000) y curva de sensibilidad intrínseca FOV.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={handleCompareModels}
            disabled={comparing}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#1D4ED8] px-3.5 py-2 text-xs font-medium text-white transition-colors hover:bg-[#1E40AF] disabled:opacity-60"
          >
            {comparing ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <GitCompare className="h-3.5 w-3.5" />
            )}
            <span>Comparar 3 Modelos Gemini</span>
          </button>

          <button
            type="button"
            onClick={() =>
              exportPosesToJson(objects, {
                scene: benchmarkId || 'custom',
                model: selectedModel,
                fov_h_deg: fovHDeg,
              })
            }
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-3 py-2 text-xs font-medium text-[#111827] hover:bg-[#F7F7F5]"
          >
            <Download className="h-3.5 w-3.5 text-[#4B5563]" />
            <span>JSON</span>
          </button>

          <button
            type="button"
            onClick={() => exportPosesToCsv(objects, benchmarkId || 'custom')}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-3 py-2 text-xs font-medium text-[#111827] hover:bg-[#F7F7F5]"
          >
            <Download className="h-3.5 w-3.5 text-[#4B5563]" />
            <span>CSV</span>
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      {metrics ? (
        <section aria-label="Indicadores de desempeño 3D" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">mIoU 3D</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#1D4ED8]">
              {formatPercent(metrics.mIoU_3D)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              IC95% [{formatPercent(metrics.bootstrap_ci_95?.miou_3d_low)},{' '}
              {formatPercent(metrics.bootstrap_ci_95?.miou_3d_high)}]
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">mIoU BEV</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#0F766E]">
              {formatPercent(metrics.mIoU_BEV)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Sutherland-Hodgman XY
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">AP@0.25</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#111827]">
              {formatPercent(metrics.ap_25)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              AP@0.50: {formatPercent(metrics.ap_50)}
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">ATE (Traslación)</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#111827]">
              {formatMeters(metrics.ate_m, 3)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              IC95% [{formatMeters(metrics.bootstrap_ci_95?.ate_low, 3)},{' '}
              {formatMeters(metrics.bootstrap_ci_95?.ate_high, 3)}]
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">AOE (Orientación)</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#B45309]">
              {formatDegrees(metrics.aoe_deg, 1)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              ASE escala: {(metrics.ase ?? 0).toFixed(3)}
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">Reproj IoU 2D</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#111827]">
              {formatPercent(metrics.reproj_iou_2d)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Emparejados: {metrics.matched_count}/{metrics.gt_count}
            </div>
          </div>
        </section>
      ) : (
        <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5 text-xs text-[#4B5563]">
          Esta escena personalizada no incluye anotaciones Ground Truth 3D. Selecciona un benchmark calibrado en el Laboratorio 3D o ejecuta <span className="font-medium text-[#111827]">Comparar 3 Modelos Gemini</span> para ver la matriz completa de métricas.
        </div>
      )}

      {/* Charts Row: FOV Sensitivity + Multi-Model Comparison */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* FOV Sensitivity Chart */}
        <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5 lg:col-span-6">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-[#111827]">
                Curva de Sensibilidad FOV_h (Proyección Pinhole)
              </h2>
              <p className="text-xs text-[#4B5563]">
                Impacto de un descalibrado de ángulo visual sobre IoU 2D reproyectado y error en píxeles.
              </p>
            </div>
            <span className="font-mono-tech rounded border border-[#E2E5EA] bg-[#F7F7F5] px-2 py-0.5 text-[11px] text-[#111827]">
              Actual: {fovHDeg.toFixed(0)}°
            </span>
          </div>

          <div className="mt-4 h-[260px] w-full">
            {activeFovSweep.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activeFovSweep} margin={{ top: 10, right: 16, left: -6, bottom: 4 }}>
                  <CartesianGrid stroke="#E2E5EA" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="fov_h_deg"
                    tick={{ fontSize: 11, fill: '#4B5563' }}
                    tickFormatter={(v) => `${v}°`}
                  />
                  <YAxis
                    yAxisId="iou"
                    domain={[0, 1]}
                    tick={{ fontSize: 11, fill: '#1D4ED8' }}
                    tickFormatter={(v) => `${Math.round(v * 100)}%`}
                  />
                  <YAxis
                    yAxisId="px"
                    orientation="right"
                    tick={{ fontSize: 11, fill: '#B45309' }}
                    tickFormatter={(v) => `${v}px`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E2E5EA',
                      fontSize: '11px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line
                    yAxisId="iou"
                    type="monotone"
                    dataKey="reproj_iou_2d"
                    name="Reproj IoU 2D"
                    stroke="#1D4ED8"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    yAxisId="px"
                    type="monotone"
                    dataKey="mean_pixel_error"
                    name="Error medio (px)"
                    stroke="#B45309"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-[#CBD5E1] text-xs text-[#4B5563]">
                Sin datos de barrido FOV disponibles para esta escena.
              </div>
            )}
          </div>
        </section>

        {/* Multi-Model Comparison Card */}
        <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5 lg:col-span-6">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-[#111827]">
                Comparativa Multi-Modelo ({sceneTitle || benchmarkId})
              </h2>
              <p className="text-xs text-[#4B5563]">
                gemini-4-hydrogen vs. gemini-4-argon vs. gemini-3.5-flash-lite
              </p>
            </div>
          </div>

          {compareError && (
            <div className="mt-3 rounded-lg border border-[#BE123C]/30 bg-[#BE123C]/5 px-3 py-2 text-xs text-[#BE123C]">
              {compareError}
            </div>
          )}

          {comparison ? (
            <div className="mt-4 space-y-4">
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonChartData} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
                    <CartesianGrid stroke="#E2E5EA" strokeDasharray="3 3" />
                    <XAxis dataKey="model" tick={{ fontSize: 11, fill: '#111827' }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#4B5563' }} unit="%" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#FFFFFF',
                        borderColor: '#E2E5EA',
                        fontSize: '11px',
                        fontFamily: 'JetBrains Mono, monospace',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Bar dataKey="mIoU_3D" name="mIoU 3D (%)" fill="#1D4ED8" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="mIoU_BEV" name="mIoU BEV (%)" fill="#0F766E" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="AP25" name="AP@0.25 (%)" fill="#64748B" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="scroll-affordance-x rounded-lg border border-[#E2E5EA]">
                <table className="font-mono-tech w-full min-w-[480px] text-left text-[11px]">
                  <thead className="border-b border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563]">
                    <tr>
                      <th className="px-3 py-2">Modelo</th>
                      <th className="px-3 py-2">mIoU 3D</th>
                      <th className="px-3 py-2">ATE</th>
                      <th className="px-3 py-2">AOE</th>
                      <th className="px-3 py-2">Latencia</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E2E5EA]">
                    {comparison.results.map((r) => (
                      <tr key={r.model}>
                        <td className="px-3 py-2 font-semibold text-[#111827]">{r.model}</td>
                        <td className="px-3 py-2 text-[#1D4ED8]">
                          {formatPercent(r.metrics.mIoU_3D)}
                        </td>
                        <td className="px-3 py-2">{formatMeters(r.metrics.ate_m, 3)}</td>
                        <td className="px-3 py-2">{formatDegrees(r.metrics.aoe_deg, 1)}</td>
                        <td className="px-3 py-2 text-[#4B5563]">
                          {Math.round(r.latency_ms)} ms
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="mt-4 flex h-[260px] flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-[#CBD5E1] bg-[#F7F7F5] p-6 text-center">
              <GitCompare className="h-5 w-5 text-[#1D4ED8]" />
              <p className="max-w-xs text-xs text-[#4B5563]">
                Evalúa simultáneamente los 3 perfiles de Gemini sobre el benchmark actual para contrastar precisión 3D vs. latencia.
              </p>
              <button
                type="button"
                onClick={handleCompareModels}
                disabled={comparing}
                className="rounded-lg bg-[#111827] px-3.5 py-1.5 text-xs font-medium text-white hover:bg-[#1D4ED8]"
              >
                Ejecutar Comparativa
              </button>
            </div>
          )}
        </section>
      </div>

      {/* Occlusion Matrix Table */}
      <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-[#B45309]" />
            <h2 className="text-sm font-semibold text-[#111827]">
              Matriz de Oclusión por Rayos Kay-Kajiya ({occlusionMatrix.length})
            </h2>
          </div>
          <span className="text-xs text-[#4B5563]">
            Haz clic en una fila para enfocar el objeto ocluido en el Laboratorio 3D
          </span>
        </div>

        {occlusionMatrix.length > 0 ? (
          <div className="scroll-affordance-x mt-4 rounded-lg border border-[#E2E5EA]">
            <table className="w-full min-w-[600px] text-left text-xs">
              <thead className="border-b border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563]">
                <tr>
                  <th className="px-3.5 py-2.5 font-medium">Oclusor (Frente)</th>
                  <th className="px-3.5 py-2.5 font-medium">Ocluido (Fondo)</th>
                  <th className="px-3.5 py-2.5 font-medium">Ratio de Oclusión</th>
                  <th className="px-3.5 py-2.5 font-medium">Brecha de Profundidad</th>
                  <th className="px-3.5 py-2.5 font-medium text-right">Inspección</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#E2E5EA]">
                {occlusionMatrix.map((entry, idx) => (
                  <tr
                    key={`${entry.occluder_id}-${entry.occluded_id}-${idx}`}
                    className="hover:bg-[#F7F7F5]"
                  >
                    <td className="px-3.5 py-2.5 font-medium text-[#111827]">
                      {entry.occluder_label}{' '}
                      <span className="font-mono-tech text-[10px] text-[#4B5563]">
                        ({entry.occluder_id})
                      </span>
                    </td>
                    <td className="px-3.5 py-2.5 font-medium text-[#B45309]">
                      {entry.occluded_label}{' '}
                      <span className="font-mono-tech text-[10px] text-[#4B5563]">
                        ({entry.occluded_id})
                      </span>
                    </td>
                    <td className="font-mono-tech px-3.5 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="h-1.5 w-20 overflow-hidden rounded-full bg-[#E2E5EA]">
                          <div
                            className="h-full bg-[#B45309]"
                            style={{ width: `${Math.round(entry.occlusion_ratio * 100)}%` }}
                          />
                        </div>
                        <span>{formatPercent(entry.occlusion_ratio)}</span>
                      </div>
                    </td>
                    <td className="font-mono-tech px-3.5 py-2.5 text-[#111827]">
                      {formatMeters(entry.depth_gap_m, 2)}
                    </td>
                    <td className="px-3.5 py-2.5 text-right">
                      <button
                        type="button"
                        onClick={() => onFocusObjectInLab(entry.occluded_id)}
                        className="inline-flex items-center gap-1 text-xs font-medium text-[#1D4ED8] hover:underline"
                      >
                        <span>Ver en 3D</span>
                        <ArrowUpRight className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="mt-3 text-xs text-[#4B5563]">
            Sin oclusiones directas detectadas desde el origen óptico en esta escena. Prueba el benchmark <span className="font-mono-tech text-[#111827]">occlusion_corridor</span>.
          </p>
        )}
      </section>

      {/* 9-DOF Poses Table */}
      <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-[#1D4ED8]" />
            <h2 className="text-sm font-semibold text-[#111827]">
              Tabla de Poses 9-DOF (Marco Gemini: X-Der, Y-Profundidad, Z-Arriba)
            </h2>
          </div>
          <span className="font-mono-tech text-xs text-[#4B5563]">
            {objects.length} objetos
          </span>
        </div>

        <div className="scroll-affordance-x mt-4 rounded-lg border border-[#E2E5EA]">
          <table className="font-mono-tech w-full min-w-[780px] text-left text-[11px]">
            <thead className="border-b border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563]">
              <tr>
                <th className="px-3 py-2.5">ID / Etiqueta</th>
                <th className="px-3 py-2.5">Centro (x, y, z) m</th>
                <th className="px-3 py-2.5">Dimensiones (dx, dy, dz) m</th>
                <th className="px-3 py-2.5">Euler ZYX (r, p, y) °</th>
                <th className="px-3 py-2.5">Volumen</th>
                <th className="px-3 py-2.5">Distancia</th>
                <th className="px-3 py-2.5 text-right">3D</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E2E5EA]">
              {objects.map((o) => {
                const [x, y, z, dx, dy, dz, roll, pitch, yaw] = o.box_3d;
                const vol = o.volume_m3 ?? dx * dy * dz;
                const dist = o.distance_m ?? Math.hypot(x, y, z);
                return (
                  <tr key={o.id} className="hover:bg-[#F7F7F5]">
                    <td className="px-3 py-2.5">
                      <span className="font-sans font-semibold text-[#111827]">{o.label}</span>{' '}
                      <span className="text-[#4B5563]">({o.id})</span>
                    </td>
                    <td className="px-3 py-2.5">
                      [{x.toFixed(2)}, {y.toFixed(2)}, {z.toFixed(2)}]
                    </td>
                    <td className="px-3 py-2.5">
                      [{dx.toFixed(2)}, {dy.toFixed(2)}, {dz.toFixed(2)}]
                    </td>
                    <td className="px-3 py-2.5">
                      [{roll.toFixed(1)}, {pitch.toFixed(1)}, {yaw.toFixed(1)}]
                    </td>
                    <td className="px-3 py-2.5">{vol.toFixed(3)} m³</td>
                    <td className="px-3 py-2.5 text-[#1D4ED8]">{dist.toFixed(2)} m</td>
                    <td className="px-3 py-2.5 text-right">
                      <button
                        type="button"
                        onClick={() => onFocusObjectInLab(o.id)}
                        className="font-sans text-xs font-medium text-[#1D4ED8] hover:underline"
                      >
                        Enfocar
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
