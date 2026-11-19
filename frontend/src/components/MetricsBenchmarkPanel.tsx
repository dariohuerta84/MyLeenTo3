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
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Download,
  Eye,
  GitCompare,
  Layers,
  Loader2,
  ShieldAlert,
  ShieldCheck,
} from 'lucide-react';
import type {
  CompareModelsResponse,
  FovSweepPoint,
  IntrinsicDiagnostics,
  OcclusionEntry,
  RealPhotoMetrics,
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
  intrinsicDiagnostics?: IntrinsicDiagnostics | null;
  realPhotoMetrics?: RealPhotoMetrics | null;
  occlusionMatrix: OcclusionEntry[];
  apiKey: string;
  onFocusObjectInLab: (objectId: string) => void;
}

const DOMAIN_LABELS: Record<string, string> = {
  calibrated_studio: 'Estudio Calibrado',
  natural_indoor: 'Escena Natural / Interior',
  high_contrast_stylized: 'Alto Contraste / Estilizado',
  low_light_high_stress: 'Baja Luminancia / Estrés Alto',
};

export const MetricsBenchmarkPanel: React.FC<MetricsBenchmarkPanelProps> = ({
  benchmarkId,
  sceneTitle,
  selectedModel,
  fovHDeg,
  objects,
  metrics,
  fovSweep,
  intrinsicDiagnostics,
  realPhotoMetrics,
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

  const activeFovSweep =
    comparison?.fov_sweep ??
    fovSweep ??
    intrinsicDiagnostics?.reference_free_fov_sweep ??
    [];
  const latent = intrinsicDiagnostics?.latent_metrics;
  const imgProf = intrinsicDiagnostics?.image_profile;
  const cascadePolicy = intrinsicDiagnostics?.cascade_policy;
  const exploitedWeaknesses = intrinsicDiagnostics?.exploited_weaknesses ?? [];
  const perObjectDiag = intrinsicDiagnostics?.per_object_diagnostics ?? [];

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

      {/* KPI Cards: Supervised GT if present, otherwise Reference-Free Universal 3D KPIs */}
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
            <div className="text-[11px] font-medium text-[#4B5563]">AP@0.25 · NDS</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#111827]">
              {formatPercent(metrics.ap_25)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              AP@0.50: {formatPercent(metrics.ap_50)} · NDS:{' '}
              {((metrics.nds_static ?? 0) * 100).toFixed(1)}%
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

          {metrics.depth_bins && metrics.depth_bins.length > 0 && (
            <div className="sm:col-span-2 lg:col-span-6 grid gap-3 border border-[#E2E5EA] bg-[#FFFFFF] p-3.5 sm:grid-cols-3">
              {metrics.depth_bins.map((b) => (
                <div key={b.bin} className="border-l-2 border-[#1D4ED8]/35 pl-3">
                  <div className="font-mono-tech text-[10px] uppercase tracking-wider text-[#4B5563]">
                    Rango Omni3D · {b.label}
                  </div>
                  <div className="font-mono-tech mt-1 flex items-baseline justify-between text-xs text-[#111827]">
                    <span>
                      mIoU 3D:{' '}
                      <strong className="text-[#1D4ED8]">
                        {b.gt_count > 0 ? formatPercent(b.mIoU_3D) : '—'}
                      </strong>
                    </span>
                    <span className="text-[#4B5563]">
                      ATE {b.gt_count > 0 ? formatMeters(b.ate_m, 3) : '—'} ({b.matched_count}/{b.gt_count})
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      ) : latent ? (
        <section aria-label="Diagnóstico intrínseco sin Ground Truth" className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">Plausibilidad Física 3D</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#1D4ED8]">
              {latent.physical_plausibility_score.toFixed(1)}
              <span className="text-xs font-normal text-[#4B5563]">/100</span>
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Sin Ground Truth (Universal)
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">Coherencia Sobel (PECS)</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#0F766E]">
              {formatPercent(latent.photometric_edge_coherence)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Alineación bordes 2D↔3D
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">Interpenetración (PVI)</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#BE123C]">
              {formatPercent(latent.volume_interpenetration_index)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Solapamiento sólido SAT
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">Anomalía Gravedad (GFI)</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#B45309]">
              {formatMeters(latent.gravity_floating_anomaly_m, 2)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Plano Z={latent.support_plane_z_m.toFixed(2)} m
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">Ambigüedad Escala (SDAS)</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#111827]">
              {latent.scale_depth_ambiguity_score.toFixed(2)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Acoplamiento ln(V)–ln(y)
            </div>
          </div>

          <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="text-[11px] font-medium text-[#4B5563]">Incertidumbre (±σ_y)</div>
            <div className="font-mono-tech mt-1 text-2xl font-semibold text-[#111827]">
              ±{formatMeters(latent.mean_depth_uncertainty_m, 2)}
            </div>
            <div className="font-mono-tech mt-1 text-[10px] text-[#4B5563]">
              Truncamiento: {formatPercent(latent.frustum_truncation_ratio)}
            </div>
          </div>
        </section>
      ) : (
        <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5 text-xs text-[#4B5563]">
          Analiza una escena en el Laboratorio 3D para calcular métricas intrínsecas y de coherencia geométrica.
        </div>
      )}

      {/* Compuerta de Escalamiento Adaptativo Edge -> 3D (Evolución de C8286) */}
      {cascadePolicy && (
        <section className="space-y-3 rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#E2E5EA] pb-3">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-[#1D4ED8]" />
              <div>
                <h2 className="text-sm font-semibold text-[#111827]">
                  Compuerta de Escalamiento Adaptativo Edge → 3D (Evolución de C8286)
                </h2>
                <p className="text-xs text-[#4B5563]">
                  Extiende las cascadas costo–precisión 2D (HSV/LBP/HOG vs. MobileNetV2 INT8) para decidir cuándo basta un tier rápido y cuándo escalar a razonamiento 3D profundo.
                </p>
              </div>
            </div>

            {(() => {
              const badgeCls =
                cascadePolicy.operational_verdict === 'APROBADO'
                  ? 'border-[#0F766E]/35 bg-[#0F766E]/10 text-[#0F766E]'
                  : cascadePolicy.operational_verdict === 'ESCALAR_MODELO'
                  ? 'border-[#B45309]/35 bg-[#B45309]/10 text-[#B45309]'
                  : 'border-[#BE123C]/35 bg-[#BE123C]/10 text-[#BE123C]';
              return (
                <span
                  className={`font-mono-tech rounded-md border px-3 py-1 text-xs font-semibold ${badgeCls}`}
                >
                  {cascadePolicy.operational_verdict}
                </span>
              );
            })()}
          </div>

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] p-3">
              <div className="text-[11px] text-[#4B5563]">Régimen Visual C8286</div>
              <div className="font-mono-tech mt-1 text-sm font-semibold text-[#111827]">
                {cascadePolicy.c8286_degradation_regime}
              </div>
              <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                {cascadePolicy.occlusion_pairs_count} oclusiones · {cascadePolicy.colliding_pairs_count} colisiones
              </div>
            </div>

            <div className="rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] p-3">
              <div className="text-[11px] text-[#4B5563]">Tier &amp; Modelo Recomendado</div>
              <div className="font-mono-tech mt-1 text-sm font-semibold text-[#1D4ED8]">
                {cascadePolicy.recommended_model}
              </div>
              <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                {cascadePolicy.recommended_tier}
              </div>
            </div>

            <div className="rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] p-3">
              <div className="text-[11px] text-[#4B5563]">Ahorro Estimado de Latencia</div>
              <div className="font-mono-tech mt-1 text-sm font-semibold text-[#0F766E]">
                {cascadePolicy.estimated_latency_savings_pct > 0
                  ? `−${cascadePolicy.estimated_latency_savings_pct.toFixed(0)}% vs. Deep Tier`
                  : '0% (Escalado completo)'}
              </div>
              <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                Prioridad: confiabilidad física
              </div>
            </div>

            <div className="rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] p-3">
              <div className="text-[11px] text-[#4B5563]">Holgura Mínima de Paso</div>
              <div className="font-mono-tech mt-1 text-sm font-semibold text-[#111827]">
                {cascadePolicy.min_surface_clearance_m !== null
                  ? formatMeters(cascadePolicy.min_surface_clearance_m, 3)
                  : 'N/A (< 2 obj)'}
              </div>
              <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                Distancia superficie-a-superficie
              </div>
            </div>
          </div>

          <div className="rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] p-3 text-xs">
            <div className="font-semibold text-[#111827]">{cascadePolicy.summary}</div>
            {cascadePolicy.next_best_view_advice && (
              <div className="font-mono-tech mt-2 border-l-2 border-[#1D4ED8] pl-2.5 text-[11px] text-[#111827]">
                Next-Best-View (ActivePose
                {typeof cascadePolicy.viewpoint_ambiguity_score === 'number'
                  ? ` · ambigüedad ${(cascadePolicy.viewpoint_ambiguity_score * 100).toFixed(0)}%`
                  : ''}
                ): {cascadePolicy.next_best_view_advice}
              </div>
            )}
            {cascadePolicy.trigger_reasons.length > 0 && (
              <ul className="mt-1.5 list-disc space-y-1 pl-4 text-[11px] text-[#4B5563]">
                {cascadePolicy.trigger_reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            )}
          </div>
        </section>
      )}

      {/* Universal Reference-Free Latent Diagnostics & Photometric Vulnerability Section */}
      {latent && imgProf && (
        <section className="space-y-4 rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E2E5EA] pb-3">
            <div className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#1D4ED8]" />
              <div>
                <h2 className="text-sm font-semibold text-[#111827]">
                  Diagnóstico Intrínseco Universal &amp; Variables Latentes Poco Medibles
                </h2>
                <p className="text-xs text-[#4B5563]">
                  Auditoría física, fotométrica y estadística sin necesidad de Ground Truth para exponer debilidades monoculares.
                </p>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-md border border-[#E2E5EA] bg-[#F7F7F5] px-2.5 py-1 text-[11px] font-medium text-[#111827]">
                Dominio: {DOMAIN_LABELS[imgProf.domain_classification] ?? imgProf.domain_classification}
              </span>
              {metrics && (
                <span className="font-mono-tech rounded-md border border-[#1D4ED8]/25 bg-[#1D4ED8]/5 px-2.5 py-1 text-[11px] font-semibold text-[#1D4ED8]">
                  Plausibilidad {latent.physical_plausibility_score.toFixed(1)}/100
                </span>
              )}
            </div>
          </div>

          <div className="grid gap-5 lg:grid-cols-12">
            {/* Photometric Stress Profile */}
            <div className="space-y-3 rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] p-4 lg:col-span-5">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-[#111827]">
                <Eye className="h-3.5 w-3.5 text-[#1D4ED8]" />
                <span>Perfil de Estrés Fotométrico de la Imagen</span>
              </div>

              {[
                {
                  label: 'Sombras profundas (<14% lum)',
                  value: imgProf.dark_shadow_ratio,
                  display: formatPercent(imgProf.dark_shadow_ratio),
                  color: imgProf.dark_shadow_ratio > 0.35 ? '#BE123C' : '#4B5563',
                },
                {
                  label: 'Bloom / Saturación especular',
                  value: Math.min(1, imgProf.bloom_saturation_ratio * 3),
                  display: formatPercent(imgProf.bloom_saturation_ratio),
                  color: imgProf.bloom_saturation_ratio > 0.04 ? '#B45309' : '#4B5563',
                },
                {
                  label: 'Estructura de perspectiva (líneas)',
                  value: imgProf.perspective_coherence_score,
                  display: formatPercent(imgProf.perspective_coherence_score),
                  color: '#0F766E',
                },
                {
                  label: 'Entropía espectral (complejidad)',
                  value: Math.min(1, imgProf.spectral_entropy_bits / 5.0),
                  display: `${imgProf.spectral_entropy_bits.toFixed(2)} bits`,
                  color: '#1D4ED8',
                },
                {
                  label: 'Truncamiento de encuadre (FBR)',
                  value: latent.frustum_truncation_ratio,
                  display: formatPercent(latent.frustum_truncation_ratio),
                  color: latent.frustum_truncation_ratio > 0.2 ? '#BE123C' : '#4B5563',
                },
              ].map((bar) => (
                <div key={bar.label} className="space-y-1">
                  <div className="flex items-center justify-between text-[11px]">
                    <span className="text-[#4B5563]">{bar.label}</span>
                    <span className="font-mono-tech font-medium text-[#111827]">
                      {bar.display}
                    </span>
                  </div>
                  <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#E2E5EA]">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${Math.round(Math.max(0.03, Math.min(1, bar.value)) * 100)}%`,
                        backgroundColor: bar.color,
                      }}
                    />
                  </div>
                </div>
              ))}

              <div className="font-mono-tech grid grid-cols-3 gap-2 border-t border-[#E2E5EA] pt-2.5 text-[10px] text-[#4B5563]">
                <div>
                  Nitidez: <strong className="text-[#111827]">{imgProf.focus_quality ?? 'sharp'}</strong>{' '}
                  ({(imgProf.laplacian_variance ?? 0).toFixed(0)})
                </div>
                <div>
                  Ruido σ: <strong className="text-[#111827]">{(imgProf.noise_sigma_est ?? 0).toFixed(4)}</strong>
                </div>
                <div>
                  Eq. 35mm: <strong className="text-[#111827]">{(imgProf.equivalent_focal_35mm_mm ?? 31.2).toFixed(1)} mm</strong>
                </div>
              </div>

              {metrics && (
                <div className="font-mono-tech grid grid-cols-3 gap-2 border-t border-[#E2E5EA] pt-2 text-[10px] text-[#4B5563]">
                  <div>PECS: {formatPercent(latent.photometric_edge_coherence)}</div>
                  <div>GFI: {formatMeters(latent.gravity_floating_anomaly_m, 2)}</div>
                  <div>±σ_y: {formatMeters(latent.mean_depth_uncertainty_m, 2)}</div>
                </div>
              )}
            </div>

            {/* Exploited Weaknesses Cards */}
            <div className="space-y-2.5 lg:col-span-7">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-[#111827]">
                <AlertTriangle className="h-3.5 w-3.5 text-[#B45309]" />
                <span>Debilidades Monoculares Explotadas en Esta Escena</span>
              </div>

              <div className="grid gap-2.5 sm:grid-cols-2">
                {exploitedWeaknesses.map((w) => {
                  const sevStyle =
                    w.severity === 'high'
                      ? 'border-[#BE123C]/30 bg-[#BE123C]/5 text-[#BE123C]'
                      : w.severity === 'medium'
                      ? 'border-[#B45309]/30 bg-[#B45309]/5 text-[#B45309]'
                      : 'border-[#0F766E]/30 bg-[#0F766E]/5 text-[#0F766E]';
                  return (
                    <div
                      key={w.id}
                      className="rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] p-3"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <span className="text-xs font-semibold text-[#111827]">
                          {w.title}
                        </span>
                        <span
                          className={`font-mono-tech shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-semibold ${sevStyle}`}
                        >
                          {w.metric_value}
                        </span>
                      </div>
                      <p className="mt-1.5 text-[11px] leading-relaxed text-[#4B5563]">
                        {w.explanation}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Per-object fragility table */}
          {perObjectDiag.length > 0 && (
            <div className="scroll-affordance-x rounded-lg border border-[#E2E5EA]">
              <table className="font-mono-tech w-full min-w-[760px] text-left text-[11px]">
                <thead className="border-b border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563]">
                  <tr>
                    <th className="px-3 py-2 font-sans font-medium">Objeto</th>
                    <th className="px-3 py-2 font-sans font-medium">Profundidad ± σ_y</th>
                    <th className="px-3 py-2 font-sans font-medium">Sobel PECS</th>
                    <th className="px-3 py-2 font-sans font-medium">Levitación</th>
                    <th className="px-3 py-2 font-sans font-medium">Anisotropía</th>
                    <th className="px-3 py-2 font-sans font-medium">Fragilidad</th>
                    <th className="px-3 py-2 font-sans font-medium">Modo de Fallo Dominante</th>
                    <th className="px-3 py-2 font-sans font-medium text-right">3D</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E5EA]">
                  {perObjectDiag.map((d) => {
                    const vulnColor =
                      d.vulnerability_score >= 55
                        ? 'text-[#BE123C]'
                        : d.vulnerability_score >= 30
                        ? 'text-[#B45309]'
                        : 'text-[#0F766E]';
                    return (
                      <tr key={d.object_id} className="hover:bg-[#F7F7F5]">
                        <td className="px-3 py-2 font-sans font-semibold text-[#111827]">
                          {d.label}{' '}
                          <span className="font-mono-tech font-normal text-[#4B5563]">
                            ({d.object_id})
                          </span>
                        </td>
                        <td className="px-3 py-2">
                          {d.depth_m.toFixed(2)} m{' '}
                          <span className="text-[#B45309]">
                            ±{d.uncertainty_depth_m.toFixed(2)} m
                          </span>
                        </td>
                        <td className="px-3 py-2">{formatPercent(d.edge_alignment_score)}</td>
                        <td className="px-3 py-2">{d.floating_gap_m.toFixed(2)} m</td>
                        <td className="px-3 py-2">{d.anisotropy_ratio.toFixed(2)}</td>
                        <td className={`px-3 py-2 font-semibold ${vulnColor}`}>
                          {d.vulnerability_score.toFixed(1)}/100
                        </td>
                        <td className="px-3 py-2 font-sans text-[#111827]">
                          {d.dominant_weakness}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <button
                            type="button"
                            onClick={() => onFocusObjectInLab(d.object_id)}
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
          )}
        </section>
      )}

      {/* Charts Row: FOV Sensitivity + Multi-Model Comparison */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* FOV Sensitivity Chart */}
        <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5 lg:col-span-6">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-[#111827]">
                {metrics
                  ? 'Curva de Sensibilidad FOV_h (Proyección Pinhole)'
                  : 'Curva FOV_h Sin Referencia (Alineación Sobel PECS)'}
              </h2>
              <p className="text-xs text-[#4B5563]">
                {metrics
                  ? 'Impacto de un descalibrado de ángulo visual sobre IoU 2D reproyectado y error en píxeles.'
                  : `Coherencia fotométrica vs. distancia al pico Sobel (FOV sugerido: ${
                      intrinsicDiagnostics?.estimated_optimal_fov_h_deg ?? fovHDeg.toFixed(0)
                    }°).`}
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

      {/* Real-Photo SOTA 3D Metrology (MoGe-2 Normals + RANSAC + Depth Pro DBE) */}
      {realPhotoMetrics && (
        <section className="space-y-4 rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E2E5EA] pb-3">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-[#1D4ED8]" />
              <div>
                <h2 className="text-sm font-semibold text-[#111827]">
                  Metrología 3D de Foto Real (MoGe-2 Normales + RANSAC + Depth Pro DBE)
                </h2>
                <p className="text-xs text-[#4B5563]">
                  Consistencia normal–profundidad, plano dominante RANSAC, nitidez de bordes de profundidad y ajuste superficial por objeto en metros reales.
                </p>
              </div>
            </div>
            <span className="font-mono-tech rounded border border-[#1D4ED8]/25 bg-[#F7F7F5] px-2.5 py-1 text-xs font-semibold text-[#1D4ED8]">
              Score Geométrico {realPhotoMetrics.overall_3d_geometry_score_0_100.toFixed(1)}/100
            </span>
          </div>

          <div className="grid gap-4 border-b border-[#E2E5EA] pb-4 sm:grid-cols-2 lg:grid-cols-3">
            <div className="border-l-2 border-[#1D4ED8] pl-3">
              <div className="text-[11px] font-medium text-[#4B5563]">Normal-Depth Consistency (NDC)</div>
              <div className="font-mono-tech mt-1 text-base font-semibold text-[#111827]">
                {realPhotoMetrics.normal_depth_consistency.median_angular_error_deg.toFixed(1)}° med ·{' '}
                {realPhotoMetrics.normal_depth_consistency.ndc_score_0_100.toFixed(0)}/100
              </div>
              <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                {(realPhotoMetrics.normal_depth_consistency.consistent_ratio_15deg * 100).toFixed(1)}% píxeles &lt; 15° · Manhattan{' '}
                {(realPhotoMetrics.manhattan_alignment_score * 100).toFixed(0)}%
              </div>
            </div>

            <div className="border-l-2 border-[#0F766E] pl-3">
              <div className="text-[11px] font-medium text-[#4B5563]">Plano Dominante RANSAC</div>
              <div className="font-mono-tech mt-1 text-base font-semibold text-[#111827]">
                {realPhotoMetrics.ransac_plane.detected
                  ? `${(realPhotoMetrics.ransac_plane.rmse_residual_m * 100).toFixed(2)} cm RMSE · ${realPhotoMetrics.ransac_plane.plane_tilt_deg.toFixed(1)}°`
                  : 'No detectado'}
              </div>
              <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                Orientación: {realPhotoMetrics.ransac_plane.plane_orientation} · Inliers{' '}
                {(realPhotoMetrics.ransac_plane.inlier_ratio * 100).toFixed(0)}%
              </div>
            </div>

            <div className="border-l-2 border-[#B45309] pl-3">
              <div className="text-[11px] font-medium text-[#4B5563]">Bordes Profundidad-RGB (DBE) &amp; Espacio Libre</div>
              <div className="font-mono-tech mt-1 text-base font-semibold text-[#111827]">
                F1 {(realPhotoMetrics.depth_boundaries.depth_rgb_boundary_f1 * 100).toFixed(1)}%
                {typeof realPhotoMetrics.depth_boundaries.silhouette_boundary_recall === 'number'
                  ? ` · Silueta ${(realPhotoMetrics.depth_boundaries.silhouette_boundary_recall * 100).toFixed(1)}%`
                  : ''}{' '}
                · Fly {(realPhotoMetrics.depth_boundaries.flying_pixel_ratio * 100).toFixed(2)}%
              </div>
              <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                Envergadura {realPhotoMetrics.depth_profile.scene_span_xyz_m.map((v) => `${v.toFixed(2)}m`).join(' × ')}
                {realPhotoMetrics.free_space_consistency && realPhotoMetrics.free_space_consistency.total_objects > 0
                  ? ` · Free-Space ${realPhotoMetrics.free_space_consistency.consistent_objects}/${realPhotoMetrics.free_space_consistency.total_objects}`
                  : ''}
                {typeof realPhotoMetrics.optical_telemetry.specular_reflection_ratio === 'number'
                  ? ` · Especular ${(realPhotoMetrics.optical_telemetry.specular_reflection_ratio * 100).toFixed(1)}%`
                  : ''}
              </div>
            </div>
          </div>

          {realPhotoMetrics.per_object_metrology.length > 0 && (
            <div className="scroll-affordance-x rounded-lg border border-[#E2E5EA]">
              <table className="font-mono-tech w-full min-w-[900px] text-left text-[11px]">
                <thead className="border-b border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563]">
                  <tr>
                    <th className="px-3 py-2 font-sans font-medium">Objeto</th>
                    <th className="px-3 py-2 font-sans font-medium">Puntos 3D</th>
                    <th className="px-3 py-2 font-sans font-medium">Área Visible (m²)</th>
                    <th className="px-3 py-2 font-sans font-medium">Relieve RANSAC (cm)</th>
                    <th className="px-3 py-2 font-sans font-medium">Vol. s/Plano (cm³)</th>
                    <th className="px-3 py-2 font-sans font-medium">Ajuste Caja RMSE (cm)</th>
                    <th className="px-3 py-2 font-sans font-medium">Solidez</th>
                    <th className="px-3 py-2 font-sans font-medium">Espesor IQR (cm)</th>
                    <th className="px-3 py-2 font-sans font-medium">Error Normales (°)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E2E5EA]">
                  {realPhotoMetrics.per_object_metrology.map((m) => (
                    <tr key={m.object_id} className="hover:bg-[#F7F7F5]">
                      <td className="px-3 py-2 font-sans font-semibold text-[#111827]">
                        {m.label} <span className="font-mono-tech font-normal text-[#4B5563]">({m.object_id})</span>
                      </td>
                      <td className="px-3 py-2">{m.point_count.toLocaleString()}</td>
                      <td className="px-3 py-2 text-[#1D4ED8]">{m.visible_surface_area_m2.toFixed(4)}</td>
                      <td className="px-3 py-2 text-[#0F766E]">
                        {typeof m.max_relief_height_m === 'number'
                          ? `${(m.max_relief_height_m * 100).toFixed(1)} cm`
                          : '—'}
                      </td>
                      <td className="px-3 py-2">
                        {typeof m.integrated_volume_above_plane_m3 === 'number'
                          ? (m.integrated_volume_above_plane_m3 * 1e6).toFixed(0)
                          : '—'}
                      </td>
                      <td className="px-3 py-2">{(m.surface_fit_rmse_m * 100).toFixed(2)} cm</td>
                      <td className="px-3 py-2">{(m.fill_solidity_ratio * 100).toFixed(1)}%</td>
                      <td className="px-3 py-2">{(m.depth_iqr_m * 100).toFixed(2)} cm</td>
                      <td className="px-3 py-2">{m.normal_consistency_deg.toFixed(1)}°</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

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
