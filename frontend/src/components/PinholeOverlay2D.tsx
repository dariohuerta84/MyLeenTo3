import React from 'react';
import { Crosshair, Eye, Layers, Upload } from 'lucide-react';
import type {
  MeasurementState,
  SpatialObject,
} from '../types/spatial';
import { formatMeters, normalizeBase64Image } from '../utils/geometry';

interface PinholeOverlay2DProps {
  imageBase64: string;
  imageMime?: string;
  width: number;
  height: number;
  fovHDeg: number;
  objects: SpatialObject[];
  gtObjects?: SpatialObject[] | null;
  showGroundTruth: boolean;
  selectedObjectId: string | null;
  hoveredObjectId: string | null;
  highlightedIds: string[] | null;
  measureMode: boolean;
  measureFirstId: string | null;
  measurement: MeasurementState | null;
  onSelectObject: (id: string, shiftKey: boolean) => void;
  onHoverObject: (id: string | null) => void;
  onUploadFile: (file: File) => void;
}

const DEFAULT_EDGES: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
];

export const PinholeOverlay2D: React.FC<PinholeOverlay2DProps> = ({
  imageBase64,
  imageMime = 'image/png',
  width,
  height,
  fovHDeg,
  objects,
  gtObjects,
  showGroundTruth,
  selectedObjectId,
  hoveredObjectId,
  highlightedIds,
  measureMode,
  measureFirstId,
  measurement,
  onSelectObject,
  onHoverObject,
  onUploadFile,
}) => {
  const [isDragOver, setIsDragOver] = React.useState(false);

  const safeW = Math.max(1, width || 960);
  const safeH = Math.max(1, height || 640);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onUploadFile(file);
    }
  };

  // Find measurement pair 2D centers if active
  const measureObjA = measurement
    ? objects.find((o) => o.id === measurement.object_a_id)
    : undefined;
  const measureObjB = measurement
    ? objects.find((o) => o.id === measurement.object_b_id)
    : undefined;

  return (
    <section
      aria-label="Proyección Pinhole 2D"
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      className={`flex flex-col overflow-hidden rounded-xl border bg-[#FFFFFF] transition-colors ${
        isDragOver ? 'border-[#1D4ED8] bg-[#1D4ED8]/[0.03]' : 'border-[#E2E5EA]'
      }`}
    >
      {/* Header strip */}
      <div className="flex items-center justify-between gap-2 border-b border-[#E2E5EA] bg-[#F7F7F5] px-3.5 py-2">
        <div className="flex items-center gap-2">
          <Eye className="h-3.5 w-3.5 text-[#1D4ED8]" />
          <h2 className="text-xs font-semibold text-[#111827]">
            Proyección Pinhole 2D
          </h2>
          <span className="font-mono-tech rounded bg-[#FFFFFF] border border-[#E2E5EA] px-1.5 py-0.5 text-[10px] text-[#4B5563]">
            {safeW}×{safeH}
          </span>
        </div>
        <div className="flex items-center gap-2 text-[11px] text-[#4B5563]">
          <span className="hidden sm:inline">
            {measureMode
              ? 'Selecciona 2 objetos para medir'
              : 'Click selecciona · Shift+Click mide par'}
          </span>
        </div>
      </div>

      {/* Interactive Image + High-DPI Vector Wireframe Stage */}
      <div className="relative flex-1 bg-[#F0F2F5] p-3 flex items-center justify-center min-h-[300px]">
        {imageBase64 ? (
          <div className="relative w-full overflow-hidden rounded-lg border border-[#E2E5EA] bg-[#FFFFFF]">
            <img
              src={normalizeBase64Image(imageBase64, imageMime)}
              alt="Imagen RGB de entrada calibrada"
              className="block w-full h-auto select-none"
              draggable={false}
            />

            <svg
              viewBox={`0 0 ${safeW} ${safeH}`}
              className="absolute inset-0 h-full w-full"
              role="img"
              aria-label="Cuboides 3D proyectados sobre imagen 2D"
            >
              {/* Optical principal axes reticle */}
              <line
                x1={safeW / 2}
                y1={0}
                x2={safeW / 2}
                y2={safeH}
                stroke="#111827"
                strokeOpacity="0.1"
                strokeWidth="1"
                strokeDasharray="4 6"
              />
              <line
                x1={0}
                y1={safeH / 2}
                x2={safeW}
                y2={safeH / 2}
                stroke="#111827"
                strokeOpacity="0.1"
                strokeWidth="1"
                strokeDasharray="4 6"
              />

              {/* Ground Truth dashed Emerald wireframes */}
              {showGroundTruth &&
                gtObjects?.map((gt) => {
                  const proj = gt.projected_2d;
                  if (!proj || proj.vertices_2d.length < 8) return null;
                  const edges = proj.edges?.length ? proj.edges : DEFAULT_EDGES;
                  return (
                    <g key={`gt-${gt.id}`} pointerEvents="none">
                      {edges.map(([i, j], idx) => {
                        const p1 = proj.vertices_2d[i];
                        const p2 = proj.vertices_2d[j];
                        if (!p1 || !p2) return null;
                        return (
                          <line
                            key={idx}
                            x1={p1[0]}
                            y1={p1[1]}
                            x2={p2[0]}
                            y2={p2[1]}
                            stroke="#0F766E"
                            strokeWidth="2"
                            strokeDasharray="6 4"
                            strokeOpacity="0.88"
                          />
                        );
                      })}
                    </g>
                  );
                })}

              {/* Predicted 3D wireframes */}
              {objects.map((obj) => {
                const proj = obj.projected_2d;
                if (!proj || proj.vertices_2d.length < 8) return null;
                const edges = proj.edges?.length ? proj.edges : DEFAULT_EDGES;

                const isSelected = selectedObjectId === obj.id;
                const isHovered = hoveredObjectId === obj.id;
                const isMeasureEnd =
                  measureFirstId === obj.id ||
                  measurement?.object_a_id === obj.id ||
                  measurement?.object_b_id === obj.id;
                const isDimmed =
                  highlightedIds !== null && !highlightedIds.includes(obj.id);

                const depth = Math.max(0.5, obj.distance_m ?? obj.box_3d[1] ?? 2.5);
                const depthWeight = Math.max(1.5, Math.min(3.2, 4.6 / Math.sqrt(depth)));

                const strokeColor =
                  isSelected || isMeasureEnd
                    ? '#BE123C'
                    : isHovered
                      ? '#111827'
                      : '#1D4ED8';

                const strokeOpacity = isDimmed ? 0.25 : isSelected || isHovered ? 1 : 0.88;

                // Top face polygon (0..3) and bottom face (4..7) for hover/click hit target
                const xs = proj.vertices_2d.map((p) => p[0]);
                const ys = proj.vertices_2d.map((p) => p[1]);
                const minX = Math.max(0, Math.min(...xs));
                const maxX = Math.min(safeW, Math.max(...xs));
                const minY = Math.max(0, Math.min(...ys));
                const maxY = Math.min(safeH, Math.max(...ys));

                const [anchorX, anchorY] = proj.label_anchor_2d ?? [minX, Math.max(20, minY - 8)];
                const clampedAnchorX = Math.max(8, Math.min(safeW - 150, anchorX));
                const clampedAnchorY = Math.max(22, Math.min(safeH - 12, anchorY));

                const arrow = proj.front_arrow_2d;

                return (
                  <g
                    key={obj.id}
                    className="cursor-pointer"
                    onMouseEnter={() => onHoverObject(obj.id)}
                    onMouseLeave={() => onHoverObject(null)}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectObject(obj.id, e.shiftKey);
                    }}
                  >
                    {/* Transparent hit region */}
                    <rect
                      x={minX}
                      y={minY}
                      width={Math.max(16, maxX - minX)}
                      height={Math.max(16, maxY - minY)}
                      fill={isSelected ? '#BE123C' : '#1D4ED8'}
                      fillOpacity={isSelected ? 0.08 : isHovered ? 0.06 : 0.01}
                    />

                    {/* 12 edges of projected OBB */}
                    {edges.map(([i, j], idx) => {
                      const p1 = proj.vertices_2d[i];
                      const p2 = proj.vertices_2d[j];
                      if (!p1 || !p2) return null;
                      const isTopFace = i < 4 && j < 4;
                      return (
                        <line
                          key={idx}
                          x1={p1[0]}
                          y1={p1[1]}
                          x2={p2[0]}
                          y2={p2[1]}
                          stroke={strokeColor}
                          strokeWidth={isSelected || isHovered ? depthWeight + 0.8 : depthWeight}
                          strokeOpacity={isTopFace ? strokeOpacity : strokeOpacity * 0.82}
                          strokeLinecap="round"
                        />
                      );
                    })}

                    {/* Front-face heading arrow (yaw) */}
                    {arrow && arrow.length === 2 && (
                      <g pointerEvents="none" opacity={isDimmed ? 0.25 : 0.95}>
                        <line
                          x1={arrow[0][0]}
                          y1={arrow[0][1]}
                          x2={arrow[1][0]}
                          y2={arrow[1][1]}
                          stroke="#B45309"
                          strokeWidth={isSelected ? 3 : 2.2}
                          strokeLinecap="round"
                        />
                        <circle
                          cx={arrow[1][0]}
                          cy={arrow[1][1]}
                          r="4"
                          fill="#B45309"
                        />
                      </g>
                    )}

                    {/* Crisp non-cluttered label pill */}
                    <g transform={`translate(${clampedAnchorX}, ${clampedAnchorY - 18})`} opacity={isDimmed ? 0.35 : 1}>
                      <rect
                        x="0"
                        y="0"
                        width={Math.min(185, Math.max(92, (obj.label.length + 7) * 7.2))}
                        height="20"
                        rx="4"
                        fill={isSelected || isMeasureEnd ? '#BE123C' : '#FFFFFF'}
                        fillOpacity="0.94"
                        stroke={strokeColor}
                        strokeWidth="1"
                      />
                      <text
                        x="6"
                        y="13.5"
                        fontFamily="JetBrains Mono, monospace"
                        fontSize="11"
                        fontWeight="600"
                        fill={isSelected || isMeasureEnd ? '#FFFFFF' : '#111827'}
                      >
                        {obj.label} · {depth.toFixed(2)}m
                      </text>
                    </g>
                  </g>
                );
              })}

              {/* Measurement connector between pair centers */}
              {measureObjA?.projected_2d && measureObjB?.projected_2d && measurement && (
                <g pointerEvents="none">
                  <line
                    x1={measureObjA.projected_2d.center_2d[0]}
                    y1={measureObjA.projected_2d.center_2d[1]}
                    x2={measureObjB.projected_2d.center_2d[0]}
                    y2={measureObjB.projected_2d.center_2d[1]}
                    stroke="#BE123C"
                    strokeWidth="2.5"
                    strokeDasharray="6 4"
                  />
                  <circle
                    cx={measureObjA.projected_2d.center_2d[0]}
                    cy={measureObjA.projected_2d.center_2d[1]}
                    r="5"
                    fill="#BE123C"
                  />
                  <circle
                    cx={measureObjB.projected_2d.center_2d[0]}
                    cy={measureObjB.projected_2d.center_2d[1]}
                    r="5"
                    fill="#BE123C"
                  />
                  <g
                    transform={`translate(${
                      (measureObjA.projected_2d.center_2d[0] +
                        measureObjB.projected_2d.center_2d[0]) /
                        2 -
                      56
                    }, ${
                      (measureObjA.projected_2d.center_2d[1] +
                        measureObjB.projected_2d.center_2d[1]) /
                        2 -
                      13
                    })`}
                  >
                    <rect
                      x="0"
                      y="0"
                      width="112"
                      height="24"
                      rx="5"
                      fill="#111827"
                      fillOpacity="0.92"
                    />
                    <text
                      x="56"
                      y="15.5"
                      textAnchor="middle"
                      fontFamily="JetBrains Mono, monospace"
                      fontSize="11.5"
                      fontWeight="600"
                      fill="#FFFFFF"
                    >
                      Δ {formatMeters(measurement.geometry.center_distance_m, 2)}
                    </text>
                  </g>
                </g>
              )}
            </svg>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-[#CBD5E1] bg-[#FFFFFF] p-8 text-center">
            <Upload className="h-5 w-5 text-[#4B5563]" />
            <p className="text-xs font-medium text-[#111827]">
              Arrastra una foto RGB o selecciona un benchmark calibrado
            </p>
          </div>
        )}
      </div>

      {/* Subtle optical legend footer */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[#E2E5EA] bg-[#FFFFFF] px-3.5 py-2 text-[11px] text-[#4B5563]">
        <div className="flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[#1D4ED8]" />
            Predicción ({objects.length})
          </span>
          {showGroundTruth && gtObjects && gtObjects.length > 0 && (
            <span className="inline-flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full bg-[#0F766E]" />
              Ground Truth ({gtObjects.length})
            </span>
          )}
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[#B45309]" />
            Dirección frontal (yaw)
          </span>
        </div>
        <div className="font-mono-tech flex items-center gap-2 text-[11px] text-[#111827]">
          <Crosshair className="h-3 w-3 text-[#4B5563]" />
          <span>FOV_h {fovHDeg.toFixed(1)}°</span>
          {measureFirstId && (
            <span className="inline-flex items-center gap-1 rounded bg-[#BE123C]/10 px-1.5 py-0.5 text-[#BE123C]">
              <Layers className="h-3 w-3" />
              Origen: {measureFirstId}
            </span>
          )}
        </div>
      </div>
    </section>
  );
};
