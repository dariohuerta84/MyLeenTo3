import React, { useEffect, useMemo, useState } from 'react';
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
import { Activity, Film, Loader2, Play, Pause, RefreshCw } from 'lucide-react';
import type { MultiFrameDemoResponse } from '../types/spatial';
import { fetchMultiFrameDemo, runMultiFrameAudit } from '../services/api';
import { normalizeBase64Image } from '../utils/geometry';

interface MultiFrameAuditPanelProps {
  apiKey: string;
}

const TRACK_COLORS = ['#1D4ED8', '#0F766E', '#B45309', '#BE123C', '#475569'];

export const MultiFrameAuditPanel: React.FC<MultiFrameAuditPanelProps> = ({
  apiKey,
}) => {
  const [data, setData] = useState<MultiFrameDemoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [auditing, setAuditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  const loadDemo = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchMultiFrameDemo();
      setData(res);
      setFrameIdx(0);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : 'Error al cargar auditoría multi-frame'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDemo();
  }, []);

  useEffect(() => {
    if (!playing || !data || data.frames.length === 0) return;
    const id = window.setInterval(() => {
      setFrameIdx((prev) => (prev + 1) % data.frames.length);
    }, 1100);
    return () => window.clearInterval(id);
  }, [playing, data]);

  const handleReAudit = async () => {
    if (!data) return;
    setAuditing(true);
    setError(null);
    try {
      const freshAudit = await runMultiFrameAudit(data.frames, apiKey);
      setData((prev) => (prev ? { ...prev, audit: freshAudit } : prev));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al re-auditar frames');
    } finally {
      setAuditing(false);
    }
  };

  // Transform object_tracks into Recharts series by frame_idx
  const volumeChartData = useMemo(() => {
    if (!data?.audit?.object_tracks) return [];
    const frames = data.frames.map((f) => f.frame_idx);
    return frames.map((idx) => {
      const row: Record<string, number | string> = { frame: `Frame ${idx}` };
      data.audit.object_tracks.forEach((track) => {
        const pos = track.frame_indices.indexOf(idx);
        if (pos !== -1) {
          row[track.label] = Number((track.volumes_m3[pos] ?? 0).toFixed(4));
        }
      });
      return row;
    });
  }, [data]);

  const jitterChartData = useMemo(() => {
    if (!data?.audit?.object_tracks) return [];
    return data.audit.object_tracks.map((t) => ({
      label: t.label,
      scale_drift_pct: Number((t.scale_drift_pct ?? 0).toFixed(2)),
      yaw_jitter_deg: Number((t.yaw_jitter_deg ?? 0).toFixed(2)),
    }));
  }, [data]);

  if (loading) {
    return (
      <div className="mx-auto flex max-w-[1440px] items-center justify-center gap-2 px-4 py-20 text-xs text-[#4B5563]">
        <Loader2 className="h-4 w-4 animate-spin text-[#1D4ED8]" />
        <span>Cargando secuencia de auditoría World-Model…</span>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="mx-auto max-w-[1440px] px-4 py-12 sm:px-6">
        <div className="rounded-xl border border-[#BE123C]/30 bg-[#FFFFFF] p-6 text-xs text-[#BE123C]">
          <p>{error}</p>
          <button
            type="button"
            onClick={loadDemo}
            className="mt-3 rounded-lg bg-[#111827] px-3 py-1.5 text-white"
          >
            Reintentar
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const currentFrame = data.frames[frameIdx] ?? data.frames[0];
  const audit = data.audit;
  const stabilityColor =
    audit.geometric_stability_score >= 80
      ? 'text-[#0F766E]'
      : audit.geometric_stability_score >= 60
        ? 'text-[#B45309]'
        : 'text-[#BE123C]';

  return (
    <div className="mx-auto max-w-[1440px] space-y-6 px-4 py-6 sm:px-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#E2E5EA] pb-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-[#4B5563]">
            <Film className="h-4 w-4 text-[#1D4ED8]" />
            <span className="font-mono-tech">{audit.sequence_id}</span>
          </div>
          <h1 className="font-serif-editorial mt-1 text-3xl text-[#111827]">
            Auditoría de Consistencia Geométrica Multi-Frame
          </h1>
          <p className="mt-1 max-w-3xl text-xs text-[#4B5563]">
            Cuantifica deriva de escala (volume drift) y fluctuación angular (yaw jitter) a lo largo de una secuencia de paneo de cámara — contrastando percepción 3D explícita frente a World Models implícitos tipo Google Genie / Veo.
          </p>
        </div>

        <button
          type="button"
          onClick={handleReAudit}
          disabled={auditing}
          className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-3.5 py-2 text-xs font-medium text-[#111827] hover:bg-[#F7F7F5] disabled:opacity-60"
        >
          {auditing ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5 text-[#1D4ED8]" />
          )}
          <span>Re-ejecutar Auditoría</span>
        </button>
      </div>

      {/* Top KPI summary */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
          <div className="text-[11px] font-medium text-[#4B5563]">
            Geometric Stability Score
          </div>
          <div className={`font-mono-tech mt-1 text-3xl font-semibold ${stabilityColor}`}>
            {audit.geometric_stability_score.toFixed(1)}
            <span className="text-sm text-[#4B5563]"> / 100</span>
          </div>
          <div className="mt-1 text-[11px] text-[#4B5563]">
            {audit.frame_count} frames auditados
          </div>
        </div>

        <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
          <div className="text-[11px] font-medium text-[#4B5563]">
            Deriva Media de Escala
          </div>
          <div className="font-mono-tech mt-1 text-3xl font-semibold text-[#B45309]">
            {audit.mean_scale_drift_pct.toFixed(2)}%
          </div>
          <div className="mt-1 text-[11px] text-[#4B5563]">
            Variación relativa de volumen 3D
          </div>
        </div>

        <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
          <div className="text-[11px] font-medium text-[#4B5563]">
            Jitter Medio de Yaw
          </div>
          <div className="font-mono-tech mt-1 text-3xl font-semibold text-[#1D4ED8]">
            {audit.mean_yaw_jitter_deg.toFixed(2)}°
          </div>
          <div className="mt-1 text-[11px] text-[#4B5563]">
            Inestabilidad angular cuadro a cuadro
          </div>
        </div>

        <div className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4">
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-[#4B5563]">
            <Activity className="h-3.5 w-3.5 text-[#0F766E]" />
            <span>Diagnóstico Automático</span>
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-[#111827]">
            {audit.drift_diagnosis}
          </p>
        </div>
      </div>

      {/* Main Split: 4-Frame Scrubber + Charts */}
      <div className="grid gap-6 lg:grid-cols-12">
        {/* Frame scrubber card */}
        <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5 lg:col-span-6">
          <div className="flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-[#111827]">
                Visor Temporal (Frame {currentFrame?.frame_idx} · t ={' '}
                {currentFrame?.timestamp_s?.toFixed(2)}s)
              </h2>
              <p className="text-xs text-[#4B5563]">{currentFrame?.camera_note}</p>
            </div>
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] px-3 py-1.5 text-xs font-medium text-[#111827] hover:border-[#1D4ED8]"
            >
              {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
              <span>{playing ? 'Pausar' : 'Reproducir'}</span>
            </button>
          </div>

          <div className="mt-4 overflow-hidden rounded-lg border border-[#E2E5EA] bg-[#F0F2F5]">
            {currentFrame?.image_base64 ? (
              <img
                src={normalizeBase64Image(currentFrame.image_base64)}
                alt={`Frame ${currentFrame.frame_idx}`}
                className="block w-full h-auto"
              />
            ) : (
              <div className="flex aspect-[3/2] items-center justify-center text-xs text-[#4B5563]">
                Frame sin textura embebida ({currentFrame?.objects?.length ?? 0} objetos)
              </div>
            )}
          </div>

          {/* Scrubber slider + direct frame pills */}
          <div className="mt-4 space-y-3">
            <div className="flex items-center gap-3">
              <label htmlFor="multiframe-scrubber" className="font-mono-tech text-xs text-[#4B5563]">
                F{frameIdx}
              </label>
              <input
                id="multiframe-scrubber"
                type="range"
                min={0}
                max={Math.max(0, data.frames.length - 1)}
                step={1}
                value={frameIdx}
                onChange={(e) => {
                  setPlaying(false);
                  setFrameIdx(Number(e.target.value));
                }}
                className="optical-slider flex-1"
              />
              <span className="font-mono-tech text-xs text-[#4B5563]">
                {frameIdx + 1}/{data.frames.length}
              </span>
            </div>

            <div className="grid grid-cols-4 gap-2">
              {data.frames.map((fr, idx) => (
                <button
                  key={fr.frame_idx}
                  type="button"
                  onClick={() => {
                    setPlaying(false);
                    setFrameIdx(idx);
                  }}
                  className={`rounded-lg border px-2.5 py-2 text-left transition-colors ${
                    idx === frameIdx
                      ? 'border-[#1D4ED8] bg-[#1D4ED8]/[0.06] text-[#111827]'
                      : 'border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563] hover:text-[#111827]'
                  }`}
                >
                  <div className="font-mono-tech text-[11px] font-semibold">
                    Frame {fr.frame_idx}
                  </div>
                  <div className="font-mono-tech text-[10px]">
                    t={fr.timestamp_s.toFixed(1)}s · {fr.objects.length} obj
                  </div>
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Right charts */}
        <div className="space-y-6 lg:col-span-6">
          <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
            <h2 className="text-sm font-semibold text-[#111827]">
              Evolución de Volumen 3D por Objeto (m³)
            </h2>
            <p className="text-xs text-[#4B5563]">
              Un modelo geométricamente estable debería mantener líneas horizontales planas entre cuadros.
            </p>
            <div className="mt-3 h-[210px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={volumeChartData} margin={{ top: 8, right: 14, left: -10, bottom: 0 }}>
                  <CartesianGrid stroke="#E2E5EA" strokeDasharray="3 3" />
                  <XAxis dataKey="frame" tick={{ fontSize: 11, fill: '#4B5563' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#4B5563' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E2E5EA',
                      fontSize: '11px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  {audit.object_tracks.map((track, idx) => (
                    <Line
                      key={track.label}
                      type="monotone"
                      dataKey={track.label}
                      stroke={TRACK_COLORS[idx % TRACK_COLORS.length]}
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
            <h2 className="text-sm font-semibold text-[#111827]">
              Deriva de Escala (%) vs. Jitter de Yaw (°) por Trayectoria
            </h2>
            <div className="mt-3 h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={jitterChartData} margin={{ top: 8, right: 14, left: -12, bottom: 0 }}>
                  <CartesianGrid stroke="#E2E5EA" strokeDasharray="3 3" />
                  <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#111827' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#4B5563' }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFFFFF',
                      borderColor: '#E2E5EA',
                      fontSize: '11px',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="scale_drift_pct" name="Drift escala (%)" fill="#B45309" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="yaw_jitter_deg" name="Yaw jitter (°)" fill="#1D4ED8" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>
      </div>

      {/* Object tracks audit table */}
      <section className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-5">
        <h2 className="text-sm font-semibold text-[#111827]">
          Trayectorias Auditadas por Objeto
        </h2>
        <div className="scroll-affordance-x mt-3 rounded-lg border border-[#E2E5EA]">
          <table className="font-mono-tech w-full min-w-[640px] text-left text-[11px]">
            <thead className="border-b border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563]">
              <tr>
                <th className="px-3.5 py-2.5">Objeto</th>
                <th className="px-3.5 py-2.5">Frames</th>
                <th className="px-3.5 py-2.5">Drift Escala (%)</th>
                <th className="px-3.5 py-2.5">CV Volumen (%)</th>
                <th className="px-3.5 py-2.5">Yaw Jitter (°)</th>
                <th className="px-3.5 py-2.5">Paso Centroide (m)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E2E5EA]">
              {audit.object_tracks.map((t) => (
                <tr key={t.label} className="hover:bg-[#F7F7F5]">
                  <td className="px-3.5 py-2.5 font-sans font-semibold text-[#111827]">
                    {t.label}
                  </td>
                  <td className="px-3.5 py-2.5 text-[#4B5563]">
                    {t.frame_indices.join(', ')}
                  </td>
                  <td className="px-3.5 py-2.5 text-[#B45309]">
                    {t.scale_drift_pct.toFixed(2)}%
                  </td>
                  <td className="px-3.5 py-2.5">{t.volume_cv_pct.toFixed(2)}%</td>
                  <td className="px-3.5 py-2.5 text-[#1D4ED8]">
                    {t.yaw_jitter_deg.toFixed(2)}°
                  </td>
                  <td className="px-3.5 py-2.5">{t.centroid_step_m.toFixed(3)} m</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};
