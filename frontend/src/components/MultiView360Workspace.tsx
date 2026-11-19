import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Camera,
  Compass,
  Layers,
  Loader2,
  Play,
  RotateCcw,
  Trash2,
  Upload,
} from 'lucide-react';
import type {
  CameraViewMode,
  MoGeBackbone,
  MultiViewInputImage,
  MultiViewReconstructResponse,
  SpatialObject,
} from '../types/spatial';
import { fetchMultiViewDemo, reconstructMultiView360 } from '../services/api';
import { normalizeBase64Image, stripDataUrlPrefix } from '../utils/geometry';
import { Scene3DViewport } from './Scene3DViewport';

interface UploadedViewDraft {
  id: string;
  label: string;
  image_base64: string;
  mime_type: string;
  yawHint: string;
}

interface MultiView360WorkspaceProps {
  mogeBackbone: MoGeBackbone;
  onChangeMogeBackbone: (bb: MoGeBackbone) => void;
  local3dAvailable: boolean;
  local3dDevice: string;
}

function fileToBase64(file: File): Promise<{ base64: string; mime: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result !== 'string') {
        reject(new Error(`No se pudo leer ${file.name}`));
        return;
      }
      const parsed = stripDataUrlPrefix(reader.result);
      resolve({
        base64: parsed.base64,
        mime: file.type || parsed.mimeType || 'image/png',
      });
    };
    reader.onerror = () => reject(new Error(`Error leyendo ${file.name}`));
    reader.readAsDataURL(file);
  });
}

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

export const MultiView360Workspace: React.FC<MultiView360WorkspaceProps> = ({
  mogeBackbone,
  onChangeMogeBackbone,
  local3dAvailable,
  local3dDevice,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [drafts, setDrafts] = useState<UploadedViewDraft[]>([]);
  const [allowScaleAlignment, setAllowScaleAlignment] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MultiViewReconstructResponse | null>(null);

  const [cameraMode, setCameraMode] = useState<CameraViewMode>('orbit');
  const [selectedObjectId, setSelectedObjectId] = useState<string | null>(null);
  const [hoveredObjectId, setHoveredObjectId] = useState<string | null>(null);
  const [scaleFactor, setScaleFactor] = useState<number>(1.0);
  const [scaleLabel, setScaleLabel] = useState<string | null>(null);

  const loadCalibratedDemo = useCallback(async () => {
    setLoading(true);
    setError(null);
    setScaleFactor(1.0);
    setScaleLabel(null);
    setSelectedObjectId(null);
    try {
      const res = await fetchMultiViewDemo();
      setResult(res);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'No se pudo ejecutar la reconstrucción demo 360°'
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCalibratedDemo();
  }, [loadCalibratedDemo]);

  const handleUploadFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    setError(null);
    try {
      const remaining = Math.max(0, 6 - drafts.length);
      const selected = Array.from(fileList).slice(0, remaining > 0 ? remaining : 6);
      const decoded = await Promise.all(
        selected.map(async (f, idx) => {
          const { base64, mime } = await fileToBase64(f);
          return {
            id: `${Date.now()}-${idx}-${f.name}`,
            label: f.name.replace(/\.[^/.]+$/, '').slice(0, 26) || `Vista ${drafts.length + idx + 1}`,
            image_base64: base64,
            mime_type: mime,
            yawHint: '',
          };
        })
      );
      setDrafts((prev) => {
        const merged = remaining > 0 ? [...prev, ...decoded] : decoded;
        return merged.slice(0, 6);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error al leer imágenes');
    }
  };

  const handleRunFusion = async () => {
    if (drafts.length === 0) {
      await loadCalibratedDemo();
      return;
    }
    if (drafts.length < 2) {
      setError('Sube entre 2 y 6 fotografías orbitales alrededor del cuerpo para fusionar en 360°.');
      return;
    }

    setLoading(true);
    setError(null);
    setScaleFactor(1.0);
    setScaleLabel(null);
    setSelectedObjectId(null);

    const payloadImages: MultiViewInputImage[] = drafts.map((d, idx) => {
      const parsedYaw = d.yawHint.trim() !== '' ? Number(d.yawHint) : NaN;
      return {
        image_base64: d.image_base64,
        mime_type: d.mime_type,
        label: d.label || `Vista ${idx + 1}`,
        yaw_hint_deg: Number.isFinite(parsedYaw) ? parsedYaw : undefined,
      };
    });

    try {
      const res = await reconstructMultiView360({
        images: payloadImages,
        use_demo: false,
        moge_backbone: mogeBackbone,
        auto_fov: true,
        allow_scale_alignment: allowScaleAlignment,
      });
      setResult(res);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'No se pudo completar la fusión multi-vista 360°'
      );
    } finally {
      setLoading(false);
    }
  };

  const scaledObjects = useMemo<SpatialObject[]>(() => {
    if (!result?.objects) return [];
    if (Math.abs(scaleFactor - 1.0) <= 1e-4) return result.objects;
    return result.objects.map((o) => {
      const b = o.box_3d;
      const scaledBox: SpatialObject['box_3d'] = [
        b[0] * scaleFactor,
        b[1] * scaleFactor,
        b[2] * scaleFactor,
        b[3] * scaleFactor,
        b[4] * scaleFactor,
        b[5] * scaleFactor,
        b[6],
        b[7],
        b[8],
      ];
      return {
        ...o,
        box_3d: scaledBox,
        distance_m: (o.distance_m ?? b[1]) * scaleFactor,
        volume_m3: (o.volume_m3 ?? b[3] * b[4] * b[5]) * scaleFactor ** 3,
        threejs_pose: o.threejs_pose
          ? {
              ...o.threejs_pose,
              position: [
                o.threejs_pose.position[0] * scaleFactor,
                o.threejs_pose.position[1] * scaleFactor,
                o.threejs_pose.position[2] * scaleFactor,
              ],
              size: [
                o.threejs_pose.size[0] * scaleFactor,
                o.threejs_pose.size[1] * scaleFactor,
                o.threejs_pose.size[2] * scaleFactor,
              ],
            }
          : undefined,
      };
    });
  }, [result?.objects, scaleFactor]);

  const met = result?.body_metrology;
  const cov = result?.coverage_360;
  const sLin = scaleFactor > 0.05 && Number.isFinite(scaleFactor) ? scaleFactor : 1.0;
  const sArea = sLin * sLin;
  const sVol = sLin * sLin * sLin;

  return (
    <div className="mx-auto max-w-[1600px] space-y-4 px-4 py-5 sm:px-6">
      {/* Editorial Title Header & Control Strip */}
      <header className="border-b border-[#E2E5EA] pb-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="font-mono-tech text-[11px] text-[#4B5563]">
              FASE 3 · REGISTRO SIFT + SIM(3) UMEYAMA + ICP PUNTO-A-PLANO + FUSIÓN VOXEL 360°
            </p>
            <h1 className="font-serif-editorial mt-1 text-2xl sm:text-3xl font-semibold tracking-tight text-[#111827]">
              Reconstrucción Multi-Vista 360° y Volumetría Cerrada
            </h1>
            <p className="mt-1 max-w-3xl text-xs text-[#4B5563]">
              Registra 2 a 6 vistas orbitales en un marco quiral único (`det(M) = +1`), alinea diferencias de escala entre tomas mediante `Sim(3)`, integra la envolvente cerrada 360° y guía la siguiente captura ciega con ActivePose NBV.
            </p>
          </div>

          <div className="font-mono-tech flex items-center gap-3 text-[11px] text-[#4B5563]">
            <span>
              Motor:{' '}
              <strong className="text-[#111827]">
                {local3dAvailable ? `GPU (${local3dDevice})` : 'Offline fallback'}
              </strong>
            </span>
            {result && (
              <>
                <span className="text-[#CBD5E1]">|</span>
                <span>
                  Latencia: <strong className="text-[#0F766E]">{result.latency_ms.toFixed(0)} ms</strong>
                </span>
              </>
            )}
          </div>
        </div>

        {/* Control Bar */}
        <div className="mt-4 flex flex-wrap items-center gap-2.5 border-t border-[#E2E5EA] pt-3">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => {
              handleUploadFiles(e.target.files);
              e.target.value = '';
            }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-3 py-1.5 text-xs font-medium text-[#111827] transition-colors hover:bg-[#F7F7F5] disabled:opacity-50"
          >
            <Upload className="h-3.5 w-3.5 text-[#1D4ED8]" />
            <span>Subir 2–6 Fotos ({drafts.length}/6)</span>
          </button>

          <button
            type="button"
            onClick={() => {
              setDrafts([]);
              loadCalibratedDemo();
            }}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] px-3 py-1.5 text-xs font-medium text-[#111827] transition-colors hover:bg-[#FFFFFF] disabled:opacity-50"
          >
            <RotateCcw className="h-3.5 w-3.5 text-[#0F766E]" />
            <span>Demo Calibrada 360° (4 Vistas)</span>
          </button>

          <div className="hidden h-5 w-px bg-[#E2E5EA] sm:block" />

          <div className="flex items-center gap-2">
            <label htmlFor="mv-backbone-select" className="text-xs font-medium text-[#4B5563]">
              Precisión 3D
            </label>
            <select
              id="mv-backbone-select"
              value={mogeBackbone}
              onChange={(e) => onChangeMogeBackbone(e.target.value as MoGeBackbone)}
              className="font-mono-tech rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-2.5 py-1.5 text-xs font-medium text-[#111827] focus:border-[#1D4ED8] focus:outline-none"
            >
              <option value="vits">MoGe-2 ViT-S (Rápido)</option>
              <option value="vitb">MoGe-2 ViT-B (Preciso)</option>
              <option value="vitl">MoGe-2 ViT-L (Máxima)</option>
            </select>
          </div>

          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-2.5 py-1.5 text-xs text-[#111827]">
            <input
              type="checkbox"
              checked={allowScaleAlignment}
              onChange={(e) => setAllowScaleAlignment(e.target.checked)}
              className="accent-[#1D4ED8]"
            />
            <span>Alinear escala Sim(3)</span>
          </label>

          <button
            type="button"
            onClick={handleRunFusion}
            disabled={loading || drafts.length === 1}
            className="ml-auto inline-flex items-center gap-1.5 rounded-lg bg-[#1D4ED8] px-4 py-1.5 text-xs font-medium text-white transition-colors hover:bg-[#1E40AF] disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Play className="h-3.5 w-3.5" />
            )}
            <span>{drafts.length >= 2 ? `Fusionar ${drafts.length} Vistas 360°` : 'Re-fusionar 360°'}</span>
          </button>
        </div>

        {(error || result?.notice) && (
          <div className="mt-2.5 flex items-center gap-2 text-xs">
            {error ? (
              <span className="inline-flex items-center gap-1.5 text-[#BE123C]">
                <AlertCircle className="h-3.5 w-3.5" />
                {error}
              </span>
            ) : result?.notice ? (
              <span className="font-mono-tech text-[11px] text-[#4B5563]">{result.notice}</span>
            ) : null}
          </div>
        )}
      </header>

      {/* Capture / Uploaded View Filmstrip */}
      {drafts.length > 0 ? (
        <section
          aria-label="Vistas subidas para fusión 360"
          className="border border-[#E2E5EA] bg-[#FFFFFF] p-3"
        >
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-semibold text-[#111827]">
              Cola de Captura Orbital ({drafts.length}/6)
            </span>
            <button
              type="button"
              onClick={() => setDrafts([])}
              className="text-[11px] text-[#4B5563] hover:text-[#BE123C]"
            >
              Vaciar cola
            </button>
          </div>
          <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-6">
            {drafts.map((d, idx) => (
              <div key={d.id} className="border border-[#E2E5EA] bg-[#F7F7F5] p-2">
                <div className="relative aspect-[4/3] overflow-hidden bg-[#FFFFFF]">
                  <img
                    src={normalizeBase64Image(d.image_base64, d.mime_type)}
                    alt={d.label}
                    className="h-full w-full object-cover"
                  />
                  <button
                    type="button"
                    onClick={() => setDrafts((prev) => prev.filter((item) => item.id !== d.id))}
                    aria-label={`Quitar ${d.label}`}
                    className="absolute right-1 top-1 border border-[#E2E5EA] bg-[#FFFFFF]/90 p-1 text-[#4B5563] hover:text-[#BE123C]"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
                <div className="mt-1.5 flex items-center justify-between gap-1 text-[11px]">
                  <span className="font-mono-tech truncate font-semibold text-[#111827]">
                    V{idx + 1} · {d.label}
                  </span>
                </div>
                <div className="mt-1 flex items-center gap-1.5">
                  <label htmlFor={`yaw-hint-${d.id}`} className="text-[10px] text-[#4B5563]">
                    Azimut (°)
                  </label>
                  <input
                    id={`yaw-hint-${d.id}`}
                    type="number"
                    min={-360}
                    max={360}
                    step={5}
                    placeholder="Auto"
                    value={d.yawHint}
                    onChange={(e) => {
                      const val = e.target.value;
                      setDrafts((prev) =>
                        prev.map((item) =>
                          item.id === d.id ? { ...item, yawHint: val } : item
                        )
                      );
                    }}
                    className="font-mono-tech w-full border border-[#E2E5EA] bg-[#FFFFFF] px-1.5 py-0.5 text-[10px] text-[#111827] focus:border-[#1D4ED8] focus:outline-none"
                  />
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : (
        result &&
        result.camera_poses.some((p) => Boolean(p.thumbnail_base64)) && (
          <section
            aria-label="Vistas registradas en reconstrucción 360"
            className="border border-[#E2E5EA] bg-[#FFFFFF] p-3"
          >
            <div className="mb-2 flex items-center justify-between text-xs">
              <span className="font-semibold text-[#111827]">
                Vistas Orbitales Registradas ({result.view_count})
              </span>
              <span className="font-mono-tech text-[11px] text-[#4B5563]">
                {result.mode}
              </span>
            </div>
            <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-4">
              {result.camera_poses.map((pose) => (
                <div
                  key={`thumb-${pose.view_index}`}
                  className="flex items-center gap-2.5 border border-[#E2E5EA] bg-[#F7F7F5] p-2"
                >
                  {pose.thumbnail_base64 && (
                    <img
                      src={normalizeBase64Image(pose.thumbnail_base64, 'image/png')}
                      alt={pose.label}
                      className="h-14 w-20 shrink-0 border border-[#E2E5EA] object-cover"
                    />
                  )}
                  <div className="min-w-0 flex-1 text-[11px]">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="h-2.5 w-2.5 shrink-0"
                        style={{ backgroundColor: pose.color }}
                      />
                      <span className="truncate font-semibold text-[#111827]">
                        {pose.label}
                      </span>
                    </div>
                    <div className="font-mono-tech mt-0.5 text-[10px] text-[#4B5563]">
                      Az {pose.azimuth_deg.toFixed(1)}° · ×{pose.estimated_scale.toFixed(3)}
                    </div>
                    <div className="font-mono-tech text-[10px] text-[#0F766E]">
                      RMSE {(pose.alignment_rmse_m * 1000).toFixed(1)} mm · {pose.inliers} pts
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )
      )}

      {/* Main 3D Viewport + Polar ActivePose & Volumetric Ledger Split */}
      <div className="grid gap-4 lg:grid-cols-12">
        <div className="lg:col-span-8">
          <Scene3DViewport
            imageBase64=""
            imageMime="image/png"
            width={result?.width ?? 640}
            height={result?.height ?? 480}
            fovHDeg={result?.fov_h_deg ?? 56}
            objects={scaledObjects}
            gtObjects={null}
            occlusionMatrix={[]}
            showGroundTruth={false}
            showOcclusionRays={false}
            showPointCloud={true}
            pointCloud={result?.point_cloud ?? null}
            cameraPoses={result?.camera_poses ?? []}
            hideSingleFocalPlane={true}
            depthStats={result?.depth_stats ?? null}
            scaleFactor={sLin}
            scaleLabel={scaleLabel}
            arucoAnchor={result?.aruco_anchor ?? null}
            unfoldProgress={1}
            cameraMode={cameraMode}
            onChangeCameraMode={setCameraMode}
            selectedObjectId={selectedObjectId}
            hoveredObjectId={hoveredObjectId}
            highlightedIds={null}
            measurement={null}
            modeBadge={result?.mode ?? 'multiview_360'}
            onSelectObject={(id) => setSelectedObjectId((prev) => (prev === id ? null : id))}
            onHoverObject={setHoveredObjectId}
            onCalibrateScale={(nextFactor, label) => {
              setScaleFactor(nextFactor);
              setScaleLabel(label);
            }}
            onResetScale={() => {
              setScaleFactor(1.0);
              setScaleLabel(null);
            }}
          />
        </div>

        {/* Right Editorial Metrology Column */}
        <aside className="space-y-4 lg:col-span-4">
          {/* 1. Polar 360 Coverage Ring & ActivePose Next-Best-View */}
          <section className="border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="flex items-center justify-between border-b border-[#E2E5EA] pb-2">
              <h2 className="flex items-center gap-1.5 text-xs font-semibold text-[#111827]">
                <Compass className="h-3.5 w-3.5 text-[#1D4ED8]" />
                <span>Anillo Polar 360° y ActivePose NBV</span>
              </h2>
              {cov && (
                <span className="font-mono-tech text-xs font-semibold text-[#0F766E]">
                  {cov.angular_coverage_pct.toFixed(0)}%
                </span>
              )}
            </div>

            {cov && (
              <div className="mt-3 flex flex-col items-center gap-3 sm:flex-row lg:flex-col xl:flex-row">
                <svg
                  viewBox="0 0 200 200"
                  className="h-44 w-44 shrink-0"
                  aria-label="Diagrama polar de cobertura angular 360 grados"
                >
                  <circle cx="100" cy="100" r="84" fill="#F7F7F5" stroke="#E2E5EA" strokeWidth="1" />
                  <circle cx="100" cy="100" r="50" fill="#FFFFFF" stroke="#E2E5EA" strokeWidth="1" />

                  {/* 12 azimuthal sectors (30° each) */}
                  {cov.sector_scores.map((score, idx) => {
                    const startAz = idx * 30;
                    const endAz = startAz + 30;
                    const covered = score >= 0.25;
                    return (
                      <path
                        key={`sec-${idx}`}
                        d={buildWedgePath(100, 100, 50, 82, startAz, endAz)}
                        fill={covered ? '#0F766E' : '#FFFFFF'}
                        fillOpacity={covered ? Math.min(0.85, 0.25 + score * 0.6) : 1}
                        stroke="#E2E5EA"
                        strokeWidth="1"
                      />
                    );
                  })}

                  {/* Cardinal labels */}
                  <text x="100" y="11" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono" fill="#4B5563">
                    0°
                  </text>
                  <text x="191" y="103" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono" fill="#4B5563">
                    90°
                  </text>
                  <text x="100" y="196" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono" fill="#4B5563">
                    180°
                  </text>
                  <text x="11" y="103" textAnchor="middle" fontSize="8" fontFamily="JetBrains Mono" fill="#4B5563">
                    270°
                  </text>

                  {/* Recommended NBV direction needle */}
                  {(() => {
                    const [nx, ny] = polarToCartesian(100, 100, 74, cov.recommended_next_azimuth_deg);
                    return (
                      <g>
                        <line
                          x1="100"
                          y1="100"
                          x2={nx}
                          y2={ny}
                          stroke="#B45309"
                          strokeWidth="1.8"
                          strokeDasharray="3 2"
                        />
                        <circle cx={nx} cy={ny} r="4" fill="#B45309" />
                      </g>
                    );
                  })()}

                  {/* Registered cameras */}
                  {result?.camera_poses.map((pose) => {
                    const [px, py] = polarToCartesian(100, 100, 84, pose.azimuth_deg);
                    return (
                      <g key={`dot-${pose.view_index}`}>
                        <circle
                          cx={px}
                          cy={py}
                          r="5"
                          fill={pose.color || '#1D4ED8'}
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                        />
                      </g>
                    );
                  })}

                  <text
                    x="100"
                    y="97"
                    textAnchor="middle"
                    fontSize="15"
                    fontWeight="700"
                    fontFamily="JetBrains Mono"
                    fill="#111827"
                  >
                    {cov.angular_coverage_pct.toFixed(0)}%
                  </text>
                  <text
                    x="100"
                    y="112"
                    textAnchor="middle"
                    fontSize="8"
                    fontFamily="JetBrains Mono"
                    fill="#4B5563"
                  >
                    {cov.is_watertight_360 ? '360° CERRADO' : `GAP ${cov.largest_blind_gap_deg.toFixed(0)}°`}
                  </text>
                </svg>

                <div className="space-y-2 text-xs">
                  <div className="font-mono-tech text-[11px] text-[#111827]">
                    Siguiente ángulo NBV:{' '}
                    <strong className="text-[#B45309]">
                      {cov.recommended_next_azimuth_deg.toFixed(0)}°
                    </strong>
                  </div>
                  <p className="text-[11px] leading-relaxed text-[#4B5563]">
                    {cov.next_best_view_advice_es}
                  </p>
                </div>
              </div>
            )}
          </section>

          {/* 2. Closed 360 Volumetric Metrology Ledger */}
          <section className="border border-[#E2E5EA] bg-[#FFFFFF] p-4">
            <div className="flex items-center justify-between border-b border-[#E2E5EA] pb-2">
              <h2 className="flex items-center gap-1.5 text-xs font-semibold text-[#111827]">
                <Layers className="h-3.5 w-3.5 text-[#0F766E]" />
                <span>Metrología Volumétrica 360°</span>
              </h2>
              {Math.abs(sLin - 1.0) > 0.005 && (
                <span className="font-mono-tech text-[10px] font-semibold text-[#0F766E]">
                  Calibrado ×{sLin.toFixed(3)}
                </span>
              )}
            </div>

            {met ? (
              <dl className="divide-y divide-[#E2E5EA] text-xs">
                <div className="flex items-center justify-between py-2">
                  <dt className="text-[#4B5563]">Dimensiones L × A × H</dt>
                  <dd className="font-mono-tech font-medium text-[#111827]">
                    {(met.bounding_dimensions_m[0] * sLin * 100).toFixed(1)} ×{' '}
                    {(met.bounding_dimensions_m[1] * sLin * 100).toFixed(1)} ×{' '}
                    {(met.bounding_dimensions_m[2] * sLin * 100).toFixed(1)} cm
                  </dd>
                </div>
                <div className="flex items-center justify-between py-2">
                  <dt className="text-[#4B5563]">Volumen Sólido Tomográfico</dt>
                  <dd className="font-mono-tech font-semibold text-[#1D4ED8]">
                    {(met.voxel_solid_volume_m3 * sVol * 1000).toFixed(2)} L (
                    {Math.round(met.voxel_solid_volume_m3 * sVol * 1e6).toLocaleString()} cm³)
                  </dd>
                </div>
                <div className="flex items-center justify-between py-2">
                  <dt className="text-[#4B5563]">Volumen Convex Hull 3D</dt>
                  <dd className="font-mono-tech font-medium text-[#111827]">
                    {(met.convex_hull_volume_m3 * sVol * 1000).toFixed(2)} L
                  </dd>
                </div>
                <div className="flex items-center justify-between py-2">
                  <dt className="text-[#4B5563]">Área Superficial Cerrada</dt>
                  <dd className="font-mono-tech font-medium text-[#111827]">
                    {(met.convex_hull_surface_area_m2 * sArea * 10000).toFixed(0)} cm²
                  </dd>
                </div>
                <div className="flex items-center justify-between py-2">
                  <dt className="text-[#4B5563]">Voxelización · Error ICP</dt>
                  <dd className="font-mono-tech font-medium text-[#111827]">
                    {(met.voxel_resolution_mm * sLin).toFixed(1)} mm ·{' '}
                    <span className="text-[#0F766E]">
                      {(met.mean_icp_rmse_mm * sLin).toFixed(2)} mm RMSE
                    </span>
                  </dd>
                </div>
                {met.gt_volume_m3 !== null && met.gt_volume_m3 !== undefined && (
                  <div className="flex items-center justify-between py-2">
                    <dt className="text-[#4B5563]">Referencia GT Analítica</dt>
                    <dd className="font-mono-tech font-semibold text-[#0F766E]">
                      {(met.gt_volume_m3 * 1000).toFixed(2)} L · Δ{' '}
                      {met.gt_volume_error_pct?.toFixed(1)}%
                    </dd>
                  </div>
                )}
              </dl>
            ) : (
              <div className="py-6 text-center text-xs text-[#4B5563]">
                Calculando reconstrucción volumétrica…
              </div>
            )}
          </section>

          {/* 3. Camera Pose & Registration Table */}
          {result && (
            <section className="border border-[#E2E5EA] bg-[#FFFFFF] p-4">
              <div className="flex items-center justify-between border-b border-[#E2E5EA] pb-2">
                <h2 className="flex items-center gap-1.5 text-xs font-semibold text-[#111827]">
                  <Camera className="h-3.5 w-3.5 text-[#4B5563]" />
                  <span>Registro por Cámara ({result.camera_poses.length})</span>
                </h2>
                <span className="font-mono-tech text-[10px] text-[#4B5563]">
                  {result.point_cloud.count.toLocaleString()} surfels
                </span>
              </div>

              <div className="scroll-affordance-x mt-2">
                <table className="font-mono-tech w-full text-left text-[10px]">
                  <thead className="border-b border-[#E2E5EA] text-[#4B5563]">
                    <tr>
                      <th className="py-1.5 pr-2">Vista</th>
                      <th className="py-1.5 pr-2">Az</th>
                      <th className="py-1.5 pr-2">Método</th>
                      <th className="py-1.5 pr-2">Esc</th>
                      <th className="py-1.5">RMSE</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E2E5EA]">
                    {result.camera_poses.map((p) => (
                      <tr key={`row-${p.view_index}`}>
                        <td className="py-1.5 pr-2 font-semibold text-[#111827]">
                          <span
                            className="mr-1 inline-block h-2 w-2"
                            style={{ backgroundColor: p.color }}
                          />
                          V{p.view_index + 1}
                        </td>
                        <td className="py-1.5 pr-2 text-[#111827]">{p.azimuth_deg.toFixed(0)}°</td>
                        <td className="py-1.5 pr-2 text-[#4B5563]">
                          {p.registration_method
                            .replace('sift_aruco_sim3_icp', 'ArUco+SIFT')
                            .replace('sift_sim3_icp', 'SIFT+ICP')
                            .replace('yaw_hint_icp', 'OrbitHint+ICP')
                            .replace('orbit_icp_fallback', 'Orbit+ICP')
                            .replace('anchor_view', 'Ancla')}
                        </td>
                        <td className="py-1.5 pr-2 text-[#111827]">×{p.estimated_scale.toFixed(2)}</td>
                        <td className="py-1.5 text-[#0F766E]">
                          {(p.alignment_rmse_m * sLin * 1000).toFixed(1)}mm
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </aside>
      </div>
    </div>
  );
};
