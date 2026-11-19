import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { Edges, Line, OrbitControls } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import {
  Box,
  Camera,
  Compass,
  Crosshair,
  Download,
  Eye,
  Focus,
  Grid,
  Layers,
  RotateCcw,
  Scale,
  Scissors,
  X,
} from 'lucide-react';
import type {
  ArucoAnchor,
  CameraViewMode,
  DepthStats,
  MeasurementState,
  OcclusionEntry,
  PerObjectMatch,
  PointCloudData,
  SpatialObject,
} from '../types/spatial';
import { computeFovVDeg, formatMeters, normalizeBase64Image } from '../utils/geometry';

interface Scene3DViewportProps {
  imageBase64: string;
  imageMime?: string;
  width: number;
  height: number;
  fovHDeg: number;
  objects: SpatialObject[];
  gtObjects?: SpatialObject[] | null;
  perObjectMatches?: PerObjectMatch[];
  occlusionMatrix: OcclusionEntry[];
  showGroundTruth: boolean;
  showOcclusionRays: boolean;
  showPointCloud?: boolean;
  pointCloud?: PointCloudData | null;
  depthStats?: DepthStats | null;
  scaleFactor?: number;
  scaleLabel?: string | null;
  arucoAnchor?: ArucoAnchor | null;
  unfoldProgress: number; // 0..1
  cameraMode: CameraViewMode;
  onChangeCameraMode: (mode: CameraViewMode) => void;
  selectedObjectId: string | null;
  hoveredObjectId: string | null;
  highlightedIds: string[] | null;
  measurement: MeasurementState | null;
  modeBadge: string;
  onSelectObject: (id: string, shiftKey: boolean) => void;
  onHoverObject: (id: string | null) => void;
  onCalibrateScale?: (nextFactor: number, label: string) => void;
  onResetScale?: () => void;
}

// Reusable scratch objects outside useFrame (zero GC allocations per frame)
const _scratchPos = new THREE.Vector3();
const _scratchTargetPos = new THREE.Vector3();
const _scratchTargetLook = new THREE.Vector3();
const _scratchPlanePos = new THREE.Vector3();
const _scratchQuat = new THREE.Quaternion();
const _scratchInterpQuat = new THREE.Quaternion();
const _identityQuat = new THREE.Quaternion();
const _scratchScale = new THREE.Vector3();

interface CameraPreset {
  pos: [number, number, number];
  target: [number, number, number];
}

function getCameraPreset(
  mode: CameraViewMode,
  focusPos: [number, number, number] | null | undefined,
  sceneScale: number
): CameraPreset {
  const safeScale = Math.min(5.5, Math.max(0.45, sceneScale || 2.5));
  const scaleRatio = Math.min(1.6, Math.max(0.24, safeScale / 2.5));
  const focus: [number, number, number] = focusPos ?? [0, 0, -safeScale];
  switch (mode) {
    case 'camera_aligned':
      return { pos: [0, 0, 0.001], target: [0, 0, -Math.max(0.6, safeScale)] };
    case 'bev':
      return {
        pos: [focus[0], Math.max(1.3, safeScale * 2.4), focus[2] + 0.01],
        target: [focus[0], 0, focus[2]],
      };
    case 'isometric':
      return {
        pos: [
          focus[0] + 3.8 * scaleRatio,
          3.2 * scaleRatio,
          focus[2] + 3.8 * scaleRatio,
        ],
        target: focus,
      };
    case 'orbit':
    default:
      return {
        pos: [
          focus[0] + 2.4 * scaleRatio,
          1.85 * scaleRatio,
          focus[2] + 3.4 * scaleRatio,
        ],
        target: focus,
      };
  }
}

const CameraRig: React.FC<{
  cameraMode: CameraViewMode;
  fovVDeg: number;
  focusTarget: [number, number, number] | null;
  sceneScale: number;
  transitionNonce: number;
}> = ({ cameraMode, fovVDeg, focusTarget, sceneScale, transitionNonce }) => {
  const { camera } = useThree();
  const controlsRef = useRef<OrbitControlsImpl | null>(null);
  const animatingRef = useRef<boolean>(false);
  const userDraggingRef = useRef<boolean>(false);

  // Sync vertical FOV with exact pinhole calculation
  useEffect(() => {
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.fov = Math.max(15, Math.min(120, fovVDeg));
      camera.updateProjectionMatrix();
    }
  }, [camera, fovVDeg]);

  // Trigger smooth camera preset transition on mode, focus, or scale change
  useEffect(() => {
    animatingRef.current = true;
  }, [cameraMode, focusTarget, sceneScale, transitionNonce]);

  useFrame((_, rawDelta) => {
    if (!animatingRef.current || userDraggingRef.current) return;
    const dt = Math.min(rawDelta, 0.1);
    const preset = getCameraPreset(cameraMode, focusTarget, sceneScale);
    _scratchTargetPos.set(...preset.pos);
    _scratchTargetLook.set(...preset.target);

    camera.position.x = THREE.MathUtils.damp(camera.position.x, _scratchTargetPos.x, 7, dt);
    camera.position.y = THREE.MathUtils.damp(camera.position.y, _scratchTargetPos.y, 7, dt);
    camera.position.z = THREE.MathUtils.damp(camera.position.z, _scratchTargetPos.z, 7, dt);

    if (controlsRef.current) {
      controlsRef.current.target.x = THREE.MathUtils.damp(
        controlsRef.current.target.x,
        _scratchTargetLook.x,
        7,
        dt
      );
      controlsRef.current.target.y = THREE.MathUtils.damp(
        controlsRef.current.target.y,
        _scratchTargetLook.y,
        7,
        dt
      );
      controlsRef.current.target.z = THREE.MathUtils.damp(
        controlsRef.current.target.z,
        _scratchTargetLook.z,
        7,
        dt
      );
      controlsRef.current.update();
    } else {
      camera.lookAt(_scratchTargetLook);
    }

    const posDone = camera.position.distanceTo(_scratchTargetPos) < 0.015;
    const lookDone =
      !controlsRef.current ||
      controlsRef.current.target.distanceTo(_scratchTargetLook) < 0.015;

    if (posDone && lookDone) {
      camera.position.copy(_scratchTargetPos);
      if (controlsRef.current) {
        controlsRef.current.target.copy(_scratchTargetLook);
        controlsRef.current.update();
      }
      animatingRef.current = false;
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      minDistance={0.05}
      maxDistance={22}
      onStart={() => {
        userDraggingRef.current = true;
        animatingRef.current = false;
      }}
      onEnd={() => {
        userDraggingRef.current = false;
      }}
    />
  );
};

const CameraFrustumAndFocalPlane: React.FC<{
  imageBase64: string;
  imageMime: string;
  fovHDeg: number;
  fovVDeg: number;
  focalPlaneDepth: number;
  unfoldProgress: number;
  cameraMode: CameraViewMode;
}> = ({
  imageBase64,
  imageMime,
  fovHDeg,
  fovVDeg,
  focalPlaneDepth,
  unfoldProgress,
  cameraMode,
}) => {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);
  const originMarkerRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (originMarkerRef.current) {
      originMarkerRef.current.visible =
        cameraMode !== 'camera_aligned' && state.camera.position.length() > 0.35;
    }
  });

  useEffect(() => {
    if (!imageBase64) {
      setTexture(null);
      return;
    }
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      if (cancelled) return;
      const tex = new THREE.Texture(img);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      setTexture((prev) => {
        prev?.dispose();
        return tex;
      });
    };
    img.src = normalizeBase64Image(imageBase64, imageMime);
    return () => {
      cancelled = true;
    };
  }, [imageBase64, imageMime]);

  useEffect(() => {
    return () => {
      texture?.dispose();
    };
  }, [texture]);

  const depth = Math.max(0.15, focalPlaneDepth);
  const halfW = depth * Math.tan((fovHDeg * Math.PI) / 360);
  const halfH = depth * Math.tan((fovVDeg * Math.PI) / 360);

  const corners: [number, number, number][] = [
    [-halfW, halfH, -depth],
    [halfW, halfH, -depth],
    [halfW, -halfH, -depth],
    [-halfW, -halfH, -depth],
  ];

  const planeOpacity = Math.max(0.22, 0.88 - unfoldProgress * 0.55);
  const resolvedPlaneOpacity =
    cameraMode === 'camera_aligned' ? 0.55 : planeOpacity;

  return (
    <group>
      {/* Camera origin marker + rays hidden when near (0,0,0) or camera_aligned */}
      <group ref={originMarkerRef} visible={cameraMode !== 'camera_aligned'}>
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[0.04, 16, 16]} />
          <meshStandardMaterial color="#111827" roughness={0.4} />
        </mesh>

        {/* Frustum edges from origin to focal plane corners */}
        {corners.map((c, idx) => (
          <Line
            key={`ray-${idx}`}
            points={[
              [0, 0, 0],
              c,
            ]}
            color="#64748B"
            lineWidth={1.1}
            transparent
            opacity={0.75}
          />
        ))}
      </group>

      {/* Focal plane perimeter */}
      <Line
        points={[...corners, corners[0]]}
        color="#1D4ED8"
        lineWidth={1.4}
        transparent
        opacity={0.7}
      />

      {/* Textured RGB focal plane */}
      {texture && (
        <mesh position={[0, 0, -depth]} renderOrder={-1}>
          <planeGeometry args={[halfW * 2, halfH * 2]} />
          <meshBasicMaterial
            map={texture}
            transparent
            opacity={resolvedPlaneOpacity}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      )}
    </group>
  );
};

const CanvasSpriteLabel: React.FC<{
  position: [number, number, number];
  lines: string[];
  accentColor?: string;
  borderColor?: string;
  backgroundColor?: string;
  worldHeight?: number;
  depthTest?: boolean;
}> = ({
  position,
  lines,
  accentColor = '#111827',
  borderColor = '#E2E5EA',
  backgroundColor = 'rgba(255, 255, 255, 0.94)',
  worldHeight = 0.22,
  depthTest = false,
}) => {
  const { texture, aspect } = useMemo(() => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const fontSize = 26;
    const subFontSize = 22;
    const padX = 20;
    const padY = 14;
    const lineGap = 8;

    let maxWidth = 80;
    if (ctx) {
      lines.forEach((line, idx) => {
        ctx.font =
          idx === 0
            ? `600 ${fontSize}px "JetBrains Mono", ui-monospace, monospace`
            : `500 ${subFontSize}px "JetBrains Mono", ui-monospace, monospace`;
        maxWidth = Math.max(maxWidth, ctx.measureText(line).width);
      });
    }

    const totalTextH =
      lines.length === 1
        ? fontSize
        : fontSize + (lines.length - 1) * (subFontSize + lineGap);

    const w = Math.ceil(maxWidth + padX * 2);
    const h = Math.ceil(totalTextH + padY * 2);
    canvas.width = w;
    canvas.height = h;

    if (ctx) {
      const r = 12;
      ctx.beginPath();
      ctx.moveTo(r, 1);
      ctx.lineTo(w - r, 1);
      ctx.quadraticCurveTo(w - 1, 1, w - 1, r);
      ctx.lineTo(w - 1, h - r);
      ctx.quadraticCurveTo(w - 1, h - 1, w - r, h - 1);
      ctx.lineTo(r, h - 1);
      ctx.quadraticCurveTo(1, h - 1, 1, h - r);
      ctx.lineTo(1, r);
      ctx.quadraticCurveTo(1, 1, r, 1);
      ctx.closePath();

      ctx.fillStyle = backgroundColor;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = borderColor;
      ctx.stroke();

      ctx.textBaseline = 'top';
      let y = padY;
      lines.forEach((line, idx) => {
        ctx.font =
          idx === 0
            ? `600 ${fontSize}px "JetBrains Mono", ui-monospace, monospace`
            : `500 ${subFontSize}px "JetBrains Mono", ui-monospace, monospace`;
        ctx.fillStyle = idx === 0 ? accentColor : '#4B5563';
        ctx.fillText(line, padX, y);
        y += (idx === 0 ? fontSize : subFontSize) + lineGap;
      });
    }

    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    return { texture: tex, aspect: w / Math.max(1, h) };
  }, [lines.join('\n'), accentColor, borderColor, backgroundColor]);

  useEffect(() => {
    return () => {
      texture.dispose();
    };
  }, [texture]);

  return (
    <sprite
      position={position}
      scale={[worldHeight * aspect, worldHeight, 1]}
      renderOrder={10}
    >
      <spriteMaterial
        map={texture}
        transparent
        depthTest={depthTest}
        depthWrite={false}
      />
    </sprite>
  );
};

type ParticleRenderMode = 'surfels' | 'mesh' | 'hybrid';
type ParticleColorMode = 'rgb' | 'normals' | 'relief';
type SliceAxisMode = 'off' | 'x' | 'y' | 'z';

function decodeBase64Bytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes;
}

function sliceAxisToNum(axis: SliceAxisMode): number {
  if (axis === 'x') return 1;
  if (axis === 'y') return 2;
  if (axis === 'z') return 3;
  return 0;
}

function exportPointCloudToPly(
  pointCloud: PointCloudData,
  isolateObjIdx: number,
  sliceAxis: SliceAxisMode,
  sliceMin: number,
  sliceMax: number,
  scaleFactor = 1.0
): void {
  if (!pointCloud || pointCloud.count <= 0 || !pointCloud.positions_b64 || !pointCloud.colors_b64) {
    return;
  }
  const posBytes = decodeBase64Bytes(pointCloud.positions_b64);
  const colBytes = decodeBase64Bytes(pointCloud.colors_b64);
  if (posBytes.byteLength !== pointCloud.count * 12 || colBytes.byteLength !== pointCloud.count * 3) {
    return;
  }
  const pos = new Float32Array(posBytes.buffer);
  const nrmBytes = pointCloud.normals_b64 ? decodeBase64Bytes(pointCloud.normals_b64) : null;
  const nrm =
    nrmBytes && nrmBytes.byteLength === pointCloud.count * 3
      ? new Int8Array(nrmBytes.buffer)
      : null;
  const objBytes = pointCloud.object_indices_b64
    ? decodeBase64Bytes(pointCloud.object_indices_b64)
    : null;
  const objIds =
    objBytes && objBytes.byteLength === pointCloud.count ? objBytes : null;

  const lines: string[] = [];
  for (let i = 0; i < pointCloud.count; i += 1) {
    if (isolateObjIdx > 0 && (!objIds || objIds[i] !== isolateObjIdx)) {
      continue;
    }
    const px = pos[i * 3] * scaleFactor;
    const py = pos[i * 3 + 1] * scaleFactor;
    const pz = pos[i * 3 + 2] * scaleFactor;
    if (sliceAxis !== 'off') {
      const c = sliceAxis === 'x' ? px : sliceAxis === 'y' ? py : pz;
      if (c < sliceMin || c > sliceMax) continue;
    }
    const r = colBytes[i * 3];
    const g = colBytes[i * 3 + 1];
    const b = colBytes[i * 3 + 2];
    const nx = nrm ? (nrm[i * 3] / 127.0).toFixed(4) : '0.0000';
    const ny = nrm ? (nrm[i * 3 + 1] / 127.0).toFixed(4) : '0.0000';
    const nz = nrm ? (nrm[i * 3 + 2] / 127.0).toFixed(4) : '1.0000';
    lines.push(`${px.toFixed(5)} ${py.toFixed(5)} ${pz.toFixed(5)} ${nx} ${ny} ${nz} ${r} ${g} ${b}`);
  }

  const header = [
    'ply',
    'format ascii 1.0',
    'comment Generated by MyLeenTo3 Dense 2DGS Surfel Metrology',
    `element vertex ${lines.length}`,
    'property float x',
    'property float y',
    'property float z',
    'property float nx',
    'property float ny',
    'property float nz',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    'end_header',
  ].join('\n');

  const blob = new Blob([`${header}\n${lines.join('\n')}\n`], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = isolateObjIdx > 0 ? 'myleento3_isolated_body.ply' : 'myleento3_body_cloud.ply';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

interface SurfaceCaliperProfile {
  worldA: [number, number, number];
  worldB: [number, number, number];
  euclideanM: number;
  deltaXyzM: [number, number, number];
  slopeDeg: number;
  surfaceArcM: number;
  maxReliefMm: number;
  profilePointsWorld: [number, number, number][];
  elevationsMm: number[];
}

function computeSurfaceProfileFromCloud(
  decodedPositions: Float32Array | null,
  pointCount: number,
  rawA: [number, number, number],
  rawB: [number, number, number],
  scaleFactor: number
): SurfaceCaliperProfile {
  const rawDx = rawB[0] - rawA[0];
  const rawDy = rawB[1] - rawA[1];
  const rawDz = rawB[2] - rawA[2];
  const rawChordLen = Math.hypot(rawDx, rawDy, rawDz);

  const worldA: [number, number, number] = [
    rawA[0] * scaleFactor,
    rawA[1] * scaleFactor,
    rawA[2] * scaleFactor,
  ];
  const worldB: [number, number, number] = [
    rawB[0] * scaleFactor,
    rawB[1] * scaleFactor,
    rawB[2] * scaleFactor,
  ];
  const deltaXyzM: [number, number, number] = [
    rawDx * scaleFactor,
    rawDy * scaleFactor,
    rawDz * scaleFactor,
  ];
  const euclideanM = rawChordLen * scaleFactor;
  const horizSpan = Math.hypot(deltaXyzM[0], deltaXyzM[2]);
  const slopeDeg =
    euclideanM > 1e-6
      ? (Math.atan2(Math.abs(deltaXyzM[1]), Math.max(horizSpan, 1e-7)) * 180) / Math.PI
      : 0;

  if (!decodedPositions || pointCount <= 0 || rawChordLen <= 1e-4) {
    return {
      worldA,
      worldB,
      euclideanM,
      deltaXyzM,
      slopeDeg,
      surfaceArcM: euclideanM,
      maxReliefMm: 0,
      profilePointsWorld: [worldA, worldB],
      elevationsMm: [0, 0],
    };
  }

  const numBins = 18;
  const ux = rawDx / rawChordLen;
  const uy = rawDy / rawChordLen;
  const uz = rawDz / rawChordLen;
  const cylRad = Math.max(0.025, 0.14 * rawChordLen);
  const cylRad2 = cylRad * cylRad;

  const bestDist2 = new Float64Array(numBins).fill(1e12);
  const binX = new Float32Array(numBins);
  const binY = new Float32Array(numBins);
  const binZ = new Float32Array(numBins);
  const binHas = new Uint8Array(numBins);

  for (let i = 0; i < pointCount; i += 1) {
    const px = decodedPositions[i * 3];
    const py = decodedPositions[i * 3 + 1];
    const pz = decodedPositions[i * 3 + 2];
    const vx = px - rawA[0];
    const vy = py - rawA[1];
    const vz = pz - rawA[2];
    const proj = vx * ux + vy * uy + vz * uz;
    if (proj < 0 || proj > rawChordLen) continue;
    const perpX = vx - proj * ux;
    const perpY = vy - proj * uy;
    const perpZ = vz - proj * uz;
    const d2 = perpX * perpX + perpY * perpY + perpZ * perpZ;
    if (d2 > cylRad2) continue;

    const bIdx = Math.min(numBins - 1, Math.max(0, Math.floor((proj / rawChordLen) * numBins)));
    if (d2 < bestDist2[bIdx]) {
      bestDist2[bIdx] = d2;
      binX[bIdx] = px;
      binY[bIdx] = py;
      binZ[bIdx] = pz;
      binHas[bIdx] = 1;
    }
  }

  const profilePointsWorld: [number, number, number][] = [worldA];
  const elevationsMm: number[] = [0];
  let maxReliefMm = 0;

  for (let k = 0; k < numBins; k += 1) {
    if (!binHas[k]) continue;
    const wx = binX[k] * scaleFactor;
    const wy = binY[k] * scaleFactor;
    const wz = binZ[k] * scaleFactor;
    profilePointsWorld.push([wx, wy, wz]);

    // Perpendicular deviation from worldA -> worldB chord in millimeters
    const rx = wx - worldA[0];
    const ry = wy - worldA[1];
    const rz = wz - worldA[2];
    const cx = ry * deltaXyzM[2] - rz * deltaXyzM[1];
    const cy = rz * deltaXyzM[0] - rx * deltaXyzM[2];
    const cz = rx * deltaXyzM[1] - ry * deltaXyzM[0];
    const perpMm = (Math.hypot(cx, cy, cz) / Math.max(euclideanM, 1e-6)) * 1000.0;
    if (perpMm > maxReliefMm) maxReliefMm = perpMm;
    elevationsMm.push(perpMm);
  }
  profilePointsWorld.push(worldB);
  elevationsMm.push(0);

  let surfaceArcM = 0;
  for (let i = 1; i < profilePointsWorld.length; i += 1) {
    const a = profilePointsWorld[i - 1];
    const b = profilePointsWorld[i];
    surfaceArcM += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  }
  surfaceArcM = Math.max(euclideanM, surfaceArcM);

  return {
    worldA,
    worldB,
    euclideanM,
    deltaXyzM,
    slopeDeg,
    surfaceArcM,
    maxReliefMm,
    profilePointsWorld,
    elevationsMm,
  };
}

const SURFEL_VERT = `
  attribute vec3 aColor;
  attribute vec3 aNormal;
  attribute float aObjectId;
  attribute float aReliefMm;

  uniform float uPointSize;
  uniform float uPointScale;
  uniform float uViewportH;
  uniform float uIsolateObjIdx;
  uniform float uSliceAxis;
  uniform float uSliceMin;
  uniform float uSliceMax;

  varying vec3 vColor;
  varying vec3 vNormal;
  varying float vReliefMm;
  varying float vDiscard;

  void main() {
    float keep = 1.0;
    if (uIsolateObjIdx > 0.5 && abs(aObjectId - uIsolateObjIdx) > 0.4) {
      keep = 0.0;
    }
    if (uSliceAxis > 0.5) {
      float coord = uSliceAxis < 1.5 ? position.x : (uSliceAxis < 2.5 ? position.y : position.z);
      if (coord < uSliceMin || coord > uSliceMax) {
        keep = 0.0;
      }
    }

    vDiscard = 1.0 - keep;
    vColor = aColor;
    vNormal = aNormal;
    vReliefMm = aReliefMm;

    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mvPos;
    float camDist = max(-mvPos.z, 0.08);
    float pxSize = uPointSize * uPointScale * (uViewportH * 0.5 * projectionMatrix[1][1]) / camDist;
    gl_PointSize = keep > 0.5 ? clamp(pxSize, 1.6, 26.0) : 0.0;
  }
`;

const SURFEL_FRAG = `
  uniform float uColorMode;
  uniform float uMaxReliefMm;
  uniform float uOpacity;

  varying vec3 vColor;
  varying vec3 vNormal;
  varying float vReliefMm;
  varying float vDiscard;

  vec3 reliefRamp(float t) {
    vec3 lo = vec3(0.118, 0.161, 0.231);
    vec3 mid = vec3(0.059, 0.463, 0.431);
    vec3 hi = vec3(0.706, 0.325, 0.035);
    return t < 0.5 ? mix(lo, mid, t * 2.0) : mix(mid, hi, (t - 0.5) * 2.0);
  }

  void main() {
    if (vDiscard > 0.5) discard;
    vec2 centered = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(centered, centered);
    if (r2 > 0.90) discard;

    vec3 n = normalize(vNormal + vec3(0.0, 0.0, 0.0001));
    vec3 lightDir = normalize(vec3(0.35, 0.65, 0.72));
    float diff = clamp(dot(n, lightDir) * 0.5 + 0.5, 0.0, 1.0);

    vec3 baseCol = vColor;
    if (uColorMode > 0.5 && uColorMode < 1.5) {
      baseCol = n * 0.5 + 0.5;
    } else if (uColorMode >= 1.5) {
      float t = clamp(vReliefMm / max(uMaxReliefMm, 1.0), 0.0, 1.0);
      baseCol = reliefRamp(t);
    }

    vec3 lit = baseCol * (0.76 + 0.24 * diff);
    float edgeAlpha = (1.0 - smoothstep(0.62, 0.90, r2)) * exp(-r2 * 0.45);
    gl_FragColor = vec4(lit, clamp(edgeAlpha * uOpacity, 0.0, 1.0));
  }
`;

const MetricPointCloud: React.FC<{
  pointCloud?: PointCloudData | null;
  visible: boolean;
  pointSize: number;
  pointScale: number;
  colorMode: ParticleColorMode;
  isolateObjIdx: number;
  sliceAxis: SliceAxisMode;
  sliceMin: number;
  sliceMax: number;
  caliperMode?: boolean;
  scaleFactor?: number;
  onPickSurfacePoint?: (rawPt: [number, number, number]) => void;
}> = ({
  pointCloud,
  visible,
  pointSize,
  pointScale,
  colorMode,
  isolateObjIdx,
  sliceAxis,
  sliceMin,
  sliceMax,
  caliperMode = false,
  scaleFactor = 1.0,
  onPickSurfacePoint,
}) => {
  const { size, gl } = useThree();

  const geometry = useMemo(() => {
    if (!pointCloud || pointCloud.count <= 0 || !pointCloud.positions_b64 || !pointCloud.colors_b64) {
      return null;
    }
    try {
      const posBytes = decodeBase64Bytes(pointCloud.positions_b64);
      const colBytes = decodeBase64Bytes(pointCloud.colors_b64);
      if (posBytes.byteLength !== pointCloud.count * 12 || colBytes.byteLength !== pointCloud.count * 3) {
        return null;
      }
      const posF32 = new Float32Array(posBytes.buffer);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(posF32, 3));
      geom.setAttribute('aColor', new THREE.BufferAttribute(colBytes, 3, true));

      const nrmBytes = pointCloud.normals_b64 ? decodeBase64Bytes(pointCloud.normals_b64) : null;
      if (nrmBytes && nrmBytes.byteLength === pointCloud.count * 3) {
        const nrmI8 = new Int8Array(nrmBytes.buffer);
        geom.setAttribute('aNormal', new THREE.BufferAttribute(nrmI8, 3, true));
      } else {
        const fbNrm = new Int8Array(pointCloud.count * 3);
        for (let i = 0; i < pointCloud.count; i += 1) {
          fbNrm[i * 3 + 2] = 127;
        }
        geom.setAttribute('aNormal', new THREE.BufferAttribute(fbNrm, 3, true));
      }

      const objBytes = pointCloud.object_indices_b64
        ? decodeBase64Bytes(pointCloud.object_indices_b64)
        : null;
      const objAttr =
        objBytes && objBytes.byteLength === pointCloud.count
          ? objBytes
          : new Uint8Array(pointCloud.count);
      geom.setAttribute('aObjectId', new THREE.BufferAttribute(objAttr, 1, false));

      const relBytes = pointCloud.relief_cm_b64 ? decodeBase64Bytes(pointCloud.relief_cm_b64) : null;
      if (relBytes && relBytes.byteLength === pointCloud.count * 2) {
        const relU16 = new Uint16Array(relBytes.buffer);
        geom.setAttribute('aReliefMm', new THREE.BufferAttribute(relU16, 1, false));
      } else {
        const fbRel = new Uint16Array(pointCloud.count);
        let maxZ = -1e9;
        for (let i = 0; i < pointCloud.count; i += 1) {
          if (posF32[i * 3 + 2] > maxZ) maxZ = posF32[i * 3 + 2];
        }
        for (let i = 0; i < pointCloud.count; i += 1) {
          fbRel[i] = Math.min(65535, Math.max(0, Math.round((maxZ - posF32[i * 3 + 2]) * 1000)));
        }
        geom.setAttribute('aReliefMm', new THREE.BufferAttribute(fbRel, 1, false));
      }

      geom.computeBoundingSphere();
      return geom;
    } catch {
      return null;
    }
  }, [pointCloud]);

  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: SURFEL_VERT,
      fragmentShader: SURFEL_FRAG,
      transparent: true,
      depthWrite: true,
      depthTest: true,
      uniforms: {
        uPointSize: { value: pointSize },
        uPointScale: { value: pointScale },
        uViewportH: { value: size.height * gl.getPixelRatio() },
        uIsolateObjIdx: { value: isolateObjIdx },
        uSliceAxis: { value: sliceAxisToNum(sliceAxis) },
        uSliceMin: { value: sliceMin },
        uSliceMax: { value: sliceMax },
        uColorMode: { value: colorMode === 'normals' ? 1 : colorMode === 'relief' ? 2 : 0 },
        uMaxReliefMm: { value: Math.max(pointCloud?.max_relief_mm ?? 150, 20) },
        uOpacity: { value: 0.96 },
      },
    });
  }, []);

  useEffect(() => {
    material.uniforms.uPointSize.value = pointSize;
    material.uniforms.uPointScale.value = pointScale;
    material.uniforms.uIsolateObjIdx.value = isolateObjIdx;
    material.uniforms.uSliceAxis.value = sliceAxisToNum(sliceAxis);
    material.uniforms.uSliceMin.value = sliceMin;
    material.uniforms.uSliceMax.value = sliceMax;
    material.uniforms.uColorMode.value = colorMode === 'normals' ? 1 : colorMode === 'relief' ? 2 : 0;
    material.uniforms.uMaxReliefMm.value = Math.max(pointCloud?.max_relief_mm ?? 150, 20);
  }, [
    material,
    pointSize,
    pointScale,
    isolateObjIdx,
    sliceAxis,
    sliceMin,
    sliceMax,
    colorMode,
    pointCloud?.max_relief_mm,
  ]);

  useFrame(() => {
    material.uniforms.uViewportH.value = size.height * gl.getPixelRatio();
  });

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  useEffect(() => {
    return () => {
      material.dispose();
    };
  }, [material]);

  if (!visible || !geometry) return null;

  const handlePointsClick = (e: ThreeEvent<MouseEvent>) => {
    if (!caliperMode || !onPickSurfacePoint) return;
    e.stopPropagation();
    const inv = scaleFactor > 1e-5 ? 1 / scaleFactor : 1;
    onPickSurfacePoint([e.point.x * inv, e.point.y * inv, e.point.z * inv]);
  };

  return <points geometry={geometry} material={material} renderOrder={1} onClick={handlePointsClick} />;
};

const MetricSurfaceMesh: React.FC<{
  pointCloud?: PointCloudData | null;
  visible: boolean;
  colorMode: ParticleColorMode;
  sliceAxis: SliceAxisMode;
  sliceMin: number;
  sliceMax: number;
  caliperMode?: boolean;
  scaleFactor?: number;
  onPickSurfacePoint?: (rawPt: [number, number, number]) => void;
}> = ({
  pointCloud,
  visible,
  colorMode,
  sliceAxis,
  sliceMin,
  sliceMax,
  caliperMode = false,
  scaleFactor = 1.0,
  onPickSurfacePoint,
}) => {
  const surfaceMesh = pointCloud?.surface_mesh;

  const geometry = useMemo(() => {
    if (
      !surfaceMesh ||
      surfaceMesh.vertex_count <= 0 ||
      surfaceMesh.triangle_count <= 0 ||
      !surfaceMesh.positions_b64 ||
      !surfaceMesh.indices_b64
    ) {
      return null;
    }
    try {
      const posBytes = decodeBase64Bytes(surfaceMesh.positions_b64);
      const colBytes = decodeBase64Bytes(surfaceMesh.colors_b64);
      const idxBytes = decodeBase64Bytes(surfaceMesh.indices_b64);
      if (
        posBytes.byteLength !== surfaceMesh.vertex_count * 12 ||
        colBytes.byteLength !== surfaceMesh.vertex_count * 3 ||
        idxBytes.byteLength !== surfaceMesh.triangle_count * 12
      ) {
        return null;
      }
      const posF32 = new Float32Array(posBytes.buffer);
      const idxU32 = new Uint32Array(idxBytes.buffer);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.BufferAttribute(posF32, 3));
      geom.setAttribute('color', new THREE.BufferAttribute(colBytes, 3, true));

      const nrmBytes = surfaceMesh.normals_b64 ? decodeBase64Bytes(surfaceMesh.normals_b64) : null;
      if (nrmBytes && nrmBytes.byteLength === surfaceMesh.vertex_count * 3) {
        const nrmI8 = new Int8Array(nrmBytes.buffer);
        geom.setAttribute('normal', new THREE.BufferAttribute(nrmI8, 3, true));
      } else {
        geom.computeVertexNormals();
      }
      geom.setIndex(new THREE.BufferAttribute(idxU32, 1));
      geom.computeBoundingSphere();
      return geom;
    } catch {
      return null;
    }
  }, [surfaceMesh]);

  const clippingPlanes = useMemo<THREE.Plane[]>(() => {
    if (sliceAxis === 'off') return [];
    const nPos =
      sliceAxis === 'x'
        ? new THREE.Vector3(1, 0, 0)
        : sliceAxis === 'y'
        ? new THREE.Vector3(0, 1, 0)
        : new THREE.Vector3(0, 0, 1);
    const nNeg = nPos.clone().negate();
    return [new THREE.Plane(nPos, -sliceMin), new THREE.Plane(nNeg, sliceMax)];
  }, [sliceAxis, sliceMin, sliceMax]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!visible || !geometry) return null;

  const handleMeshClick = (e: ThreeEvent<MouseEvent>) => {
    if (!caliperMode || !onPickSurfacePoint) return;
    e.stopPropagation();
    const inv = scaleFactor > 1e-5 ? 1 / scaleFactor : 1;
    onPickSurfacePoint([e.point.x * inv, e.point.y * inv, e.point.z * inv]);
  };

  return (
    <mesh geometry={geometry} renderOrder={0} onClick={handleMeshClick}>
      {colorMode === 'normals' ? (
        <meshNormalMaterial side={THREE.DoubleSide} clippingPlanes={clippingPlanes} />
      ) : (
        <meshStandardMaterial
          vertexColors
          roughness={0.72}
          metalness={0.04}
          side={THREE.DoubleSide}
          clippingPlanes={clippingPlanes}
        />
      )}
    </mesh>
  );
};

const MetricOpticalBenchFloor: React.FC<{
  cameraMode: CameraViewMode;
  sceneScale: number;
}> = ({ cameraMode, sceneScale }) => {
  const safeScale = Math.min(5.5, Math.max(0.45, sceneScale || 2.5));
  const scaleRatio = Math.min(1.6, Math.max(0.24, safeScale / 2.5));
  const floorY = -0.58 * scaleRatio;
  const isCloseUp = safeScale < 1.5;

  const rings = useMemo(
    () => (isCloseUp ? [0.25, 0.5, 0.75, 1.0, 1.5, 2.0] : [1, 2, 3, 4, 5, 6]),
    [isCloseUp]
  );

  const ringPointsList = useMemo(() => {
    return rings.map((r) => {
      const pts: [number, number, number][] = [];
      const segments = 64;
      for (let i = 0; i <= segments; i++) {
        // Semicircle in front of camera (-Z)
        const angle = Math.PI + (i / segments) * Math.PI;
        pts.push([Math.cos(angle) * r, floorY, Math.sin(angle) * r]);
      }
      return { r, pts };
    });
  }, [rings, floorY]);

  return (
    <group>
      {/* Subtle matte grid on ground plane */}
      <gridHelper
        args={isCloseUp ? [6, 12, '#94A3B8', '#CBD5E1'] : [16, 16, '#94A3B8', '#CBD5E1']}
        position={[0, floorY - 0.01, isCloseUp ? -1.5 : -4]}
      />

      {/* Concentric metric distance arcs */}
      {ringPointsList.map(({ r, pts }) => (
        <group key={r}>
          <Line
            points={pts}
            color="#94A3B8"
            lineWidth={1}
            dashed
            dashSize={isCloseUp ? 0.05 : 0.14}
            gapSize={isCloseUp ? 0.04 : 0.1}
            transparent
            opacity={0.75}
          />
          {cameraMode !== 'camera_aligned' && (
            <CanvasSpriteLabel
              position={[r * 0.45, floorY + Math.max(0.012, 0.02 * scaleRatio), -r * 0.89]}
              lines={[`${r}m`]}
              accentColor="#4B5563"
              borderColor="#CBD5E1"
              backgroundColor="rgba(240, 242, 245, 0.9)"
              worldHeight={Math.max(0.055, 0.11 * scaleRatio)}
              depthTest
            />
          )}
        </group>
      ))}
    </group>
  );
};

const AnimatedPredictedOBB: React.FC<{
  obj: SpatialObject;
  unfoldProgress: number;
  focalPlaneDepth: number;
  scaleRatio: number;
  isSelected: boolean;
  isHovered: boolean;
  isDimmed: boolean;
  isColliding: boolean;
  onSelect: (id: string, shiftKey: boolean) => void;
  onHover: (id: string | null) => void;
}> = ({
  obj,
  unfoldProgress,
  focalPlaneDepth,
  scaleRatio,
  isSelected,
  isHovered,
  isDimmed,
  isColliding,
  onSelect,
  onHover,
}) => {
  const groupRef = useRef<THREE.Group>(null);

  const pose = obj.threejs_pose;
  const fallbackPos: [number, number, number] = [
    obj.box_3d[0],
    obj.box_3d[2],
    -obj.box_3d[1],
  ];
  const fallbackSize: [number, number, number] = [
    obj.box_3d[3],
    obj.box_3d[5],
    obj.box_3d[4],
  ];

  const targetPos = pose?.position ?? fallbackPos;
  const targetQuat = pose?.quaternion ?? [0, 0, 0, 1];
  const targetSize = pose?.size ?? fallbackSize;
  const forwardDir = pose?.forward_dir_three ?? [0, 0, -1];

  // Projection point onto dynamic focal plane z = -focalPlaneDepth
  const focalZ = -Math.max(0.15, focalPlaneDepth);
  const planeIntersection = useMemo<[number, number, number]>(() => {
    const z = Math.min(-0.08, targetPos[2]);
    const t = focalZ / z;
    return [targetPos[0] * t, targetPos[1] * t, focalZ];
  }, [targetPos, focalZ]);

  // Smoothly damp position, rotation and scale for 2D -> 3D unfold without setState
  useFrame((_, rawDelta) => {
    const g = groupRef.current;
    if (!g) return;
    const dt = Math.min(rawDelta, 0.1);
    const u = Math.max(0, Math.min(1, unfoldProgress));

    _scratchPlanePos.set(...planeIntersection);
    _scratchTargetPos.set(...targetPos);
    _scratchPos.lerpVectors(_scratchPlanePos, _scratchTargetPos, u);

    g.position.x = THREE.MathUtils.damp(g.position.x, _scratchPos.x, 9, dt);
    g.position.y = THREE.MathUtils.damp(g.position.y, _scratchPos.y, 9, dt);
    g.position.z = THREE.MathUtils.damp(g.position.z, _scratchPos.z, 9, dt);

    _scratchQuat.set(targetQuat[0], targetQuat[1], targetQuat[2], targetQuat[3]);
    _scratchInterpQuat.copy(_identityQuat).slerp(_scratchQuat, u);
    g.quaternion.slerp(_scratchInterpQuat, 1 - Math.exp(-9 * dt));

    const zScale = 0.04 + 0.96 * u;
    const xyScale = 0.35 + 0.65 * u;
    _scratchScale.set(xyScale, xyScale, zScale);
    g.scale.x = THREE.MathUtils.damp(g.scale.x, _scratchScale.x, 9, dt);
    g.scale.y = THREE.MathUtils.damp(g.scale.y, _scratchScale.y, 9, dt);
    g.scale.z = THREE.MathUtils.damp(g.scale.z, _scratchScale.z, 9, dt);
  });

  const accentColor =
    isSelected || isColliding ? '#BE123C' : isHovered ? '#111827' : '#1D4ED8';

  const fillOpacity = isDimmed ? 0.05 : isSelected ? 0.24 : isHovered ? 0.2 : 0.14;

  // Forward heading arrow in world space anchored at targetPos
  const minArrow = Math.max(0.06, 0.22 * scaleRatio);
  const arrowLen = Math.max(minArrow, Math.min(0.65 * scaleRatio, Math.max(...targetSize) * 0.65));
  const arrowTip: [number, number, number] = [
    targetPos[0] + forwardDir[0] * arrowLen,
    targetPos[1] + forwardDir[1] * arrowLen,
    targetPos[2] + forwardDir[2] * arrowLen,
  ];

  return (
    <group>
      {/* 2D -> 3D projection ray during unfold */}
      {unfoldProgress < 0.99 && (
        <Line
          points={[[0, 0, 0], planeIntersection, targetPos]}
          color="#1D4ED8"
          lineWidth={1}
          dashed
          dashSize={0.08}
          gapSize={0.06}
          transparent
          opacity={0.45}
        />
      )}

      <group
        ref={groupRef}
        position={targetPos}
        quaternion={targetQuat}
      >
        <mesh
          castShadow
          receiveShadow
          onClick={(e) => {
            e.stopPropagation();
            onSelect(obj.id, e.nativeEvent.shiftKey);
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            onHover(obj.id);
          }}
          onPointerOut={(e) => {
            e.stopPropagation();
            onHover(null);
          }}
        >
          <boxGeometry args={targetSize} />
          <meshStandardMaterial
            color={accentColor}
            transparent
            opacity={fillOpacity}
            roughness={0.65}
            metalness={0.05}
            depthWrite={false}
          />
          <Edges
            threshold={15}
            color={accentColor}
            lineWidth={isSelected || isHovered ? 2.2 : 1.4}
          />
        </mesh>
      </group>

      {/* Local heading arrow */}
      {unfoldProgress > 0.7 && !isDimmed && (
        <Line
          points={[targetPos, arrowTip]}
          color="#B45309"
          lineWidth={2}
        />
      )}

      {/* Pure WebGL sprite label on selected or hovered object */}
      {(isSelected || isHovered) && (
        <CanvasSpriteLabel
          position={[targetPos[0], targetPos[1] + targetSize[1] / 2 + 0.14 * scaleRatio, targetPos[2]]}
          lines={[
            `${obj.label} · ${formatMeters(obj.distance_m ?? obj.box_3d[1], 2)}`,
          ]}
          accentColor={isSelected ? '#BE123C' : '#1D4ED8'}
          borderColor={isSelected ? '#BE123C' : '#E2E5EA'}
          worldHeight={Math.max(0.055, 0.14 * scaleRatio)}
        />
      )}
    </group>
  );
};

const GroundTruthOBB: React.FC<{
  gt: SpatialObject;
  match?: PerObjectMatch;
}> = ({ gt, match }) => {
  const pose = gt.threejs_pose;
  const pos: [number, number, number] = pose?.position ?? [
    gt.box_3d[0],
    gt.box_3d[2],
    -gt.box_3d[1],
  ];
  const quat: [number, number, number, number] = pose?.quaternion ?? [0, 0, 0, 1];
  const size: [number, number, number] = pose?.size ?? [
    gt.box_3d[3],
    gt.box_3d[5],
    gt.box_3d[4],
  ];

  // Build dashed edges from corners_three if available
  const edgeSegments = useMemo<[number, number, number][][]>(() => {
    const c = pose?.corners_three;
    if (!c || c.length < 8) return [];
    const pairs: [number, number][] = [
      [0, 1], [1, 2], [2, 3], [3, 0],
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    return pairs.map(([a, b]) => [c[a], c[b]]);
  }, [pose]);

  // Centroid error vector from GT to matched pred
  const errorTip = useMemo<[number, number, number] | null>(() => {
    if (!match?.centroid_error_vec_three) return null;
    const [ex, ey, ez] = match.centroid_error_vec_three;
    return [pos[0] + ex, pos[1] + ey, pos[2] + ez];
  }, [match, pos]);

  return (
    <group>
      {edgeSegments.length > 0 ? (
        edgeSegments.map((seg, idx) => (
          <Line
            key={idx}
            points={seg}
            color="#0F766E"
            lineWidth={1.6}
            dashed
            dashSize={0.07}
            gapSize={0.05}
          />
        ))
      ) : (
        <mesh position={pos} quaternion={quat}>
          <boxGeometry args={size} />
          <meshBasicMaterial color="#0F766E" wireframe transparent opacity={0.65} />
        </mesh>
      )}

      {errorTip && (
        <Line
          points={[pos, errorTip]}
          color="#0F766E"
          lineWidth={2}
        />
      )}
    </group>
  );
};

const MeasurementRuler3D: React.FC<{ measurement: MeasurementState }> = ({
  measurement,
}) => {
  const { closest_point_a_three: pA, closest_point_b_three: pB } = measurement.geometry;
  const mid: [number, number, number] = [
    (pA[0] + pB[0]) / 2,
    Math.max(pA[1], pB[1]) + 0.36,
    (pA[2] + pB[2]) / 2,
  ];

  const subtitle = [
    `Centro: ${formatMeters(measurement.geometry.center_distance_m, 3)}`,
    `Holgura: ${formatMeters(measurement.geometry.surface_clearance_m, 3)}`,
    measurement.geometry.is_colliding ? 'COLISIÓN' : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <group>
      <Line points={[pA, pB]} color="#BE123C" lineWidth={2.6} />
      <mesh position={pA}>
        <sphereGeometry args={[0.038, 12, 12]} />
        <meshStandardMaterial color="#BE123C" />
      </mesh>
      <mesh position={pB}>
        <sphereGeometry args={[0.038, 12, 12]} />
        <meshStandardMaterial color="#BE123C" />
      </mesh>
      <CanvasSpriteLabel
        position={mid}
        lines={[
          `${measurement.object_a_label} ↔ ${measurement.object_b_label}`,
          subtitle,
        ]}
        accentColor="#BE123C"
        borderColor="#BE123C"
        worldHeight={0.34}
      />
    </group>
  );
};

const SurfaceCaliper3D: React.FC<{
  pointA: [number, number, number] | null;
  profile: SurfaceCaliperProfile | null;
  scaleFactor: number;
  scaleRatio: number;
}> = ({ pointA, profile, scaleFactor, scaleRatio }) => {
  if (!pointA && !profile) return null;

  const worldA: [number, number, number] = profile
    ? profile.worldA
    : [pointA![0] * scaleFactor, pointA![1] * scaleFactor, pointA![2] * scaleFactor];

  const sphereRadius = Math.max(0.012, 0.022 * scaleRatio);

  if (!profile) {
    return (
      <group>
        <mesh position={worldA} renderOrder={8}>
          <sphereGeometry args={[sphereRadius, 14, 14]} />
          <meshStandardMaterial color="#0F766E" />
        </mesh>
        <CanvasSpriteLabel
          position={[worldA[0], worldA[1] + 0.08 * scaleRatio, worldA[2]]}
          lines={['P1 · Origen']}
          accentColor="#0F766E"
          borderColor="#0F766E"
          worldHeight={Math.max(0.08, 0.14 * scaleRatio)}
        />
      </group>
    );
  }

  const mid: [number, number, number] = [
    (profile.worldA[0] + profile.worldB[0]) / 2,
    Math.max(profile.worldA[1], profile.worldB[1]) + Math.max(0.09, 0.15 * scaleRatio),
    (profile.worldA[2] + profile.worldB[2]) / 2,
  ];
  const distCm = (profile.euclideanM * 100).toFixed(1);
  const distMm = Math.round(profile.euclideanM * 1000);

  return (
    <group>
      <mesh position={profile.worldA} renderOrder={8}>
        <sphereGeometry args={[sphereRadius, 14, 14]} />
        <meshStandardMaterial color="#0F766E" />
      </mesh>
      <mesh position={profile.worldB} renderOrder={8}>
        <sphereGeometry args={[sphereRadius, 14, 14]} />
        <meshStandardMaterial color="#BE123C" />
      </mesh>
      <Line points={[profile.worldA, profile.worldB]} color="#BE123C" lineWidth={2.4} />
      {profile.profilePointsWorld.length > 2 && (
        <Line
          points={profile.profilePointsWorld}
          color="#0F766E"
          lineWidth={1.8}
          dashed
          dashSize={Math.max(0.015, 0.025 * scaleRatio)}
          gapSize={Math.max(0.01, 0.015 * scaleRatio)}
        />
      )}
      <CanvasSpriteLabel
        position={mid}
        lines={[
          `${distCm} cm · ${distMm} mm`,
          `Arco ${(profile.surfaceArcM * 100).toFixed(1)} cm · Relieve ${profile.maxReliefMm.toFixed(1)} mm`,
        ]}
        accentColor="#BE123C"
        borderColor="#BE123C"
        worldHeight={Math.max(0.11, 0.22 * scaleRatio)}
      />
    </group>
  );
};

export const Scene3DViewport: React.FC<Scene3DViewportProps> = ({
  imageBase64,
  imageMime = 'image/png',
  width,
  height,
  fovHDeg,
  objects,
  gtObjects,
  perObjectMatches,
  occlusionMatrix,
  showGroundTruth,
  showOcclusionRays,
  showPointCloud = true,
  pointCloud,
  depthStats,
  scaleFactor = 1.0,
  scaleLabel = null,
  arucoAnchor = null,
  unfoldProgress,
  cameraMode,
  onChangeCameraMode,
  selectedObjectId,
  hoveredObjectId,
  highlightedIds,
  measurement,
  modeBadge,
  onSelectObject,
  onHoverObject,
  onCalibrateScale,
  onResetScale,
}) => {
  const stageRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(true);
  const [transitionNonce, setTransitionNonce] = useState(0);

  const fovVDeg = useMemo(
    () => computeFovVDeg(fovHDeg, width || 960, height || 640),
    [fovHDeg, width, height]
  );

  const { sceneScale, scaleRatio, focalPlaneDepth, pointSize } = useMemo(() => {
    let raw = depthStats?.median_m || pointCloud?.median_depth_m || 0;
    if (!raw && objects.length > 0) {
      const ys = objects
        .map((o) => o.box_3d[1])
        .filter((v) => Number.isFinite(v) && v > 0)
        .sort((a, b) => a - b);
      if (ys.length > 0) {
        raw = ys[Math.floor(ys.length / 2)];
      }
    }
    const clampedScale = raw > 0 && Number.isFinite(raw) ? Math.min(5.5, Math.max(0.45, raw)) : 2.5;
    const ratio = Math.min(1.6, Math.max(0.24, clampedScale / 2.5));
    const planeDepth = Math.min(0.85, Math.max(0.26, clampedScale * 0.42));
    const pSize = Math.min(0.02, Math.max(0.0045, clampedScale * 0.0075));
    return {
      sceneScale: clampedScale,
      scaleRatio: ratio,
      focalPlaneDepth: planeDepth,
      pointSize: pSize,
    };
  }, [depthStats, pointCloud, objects]);

  // Off-screen WebGL pausing: contentvisibilityautostatechange + IntersectionObserver
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;

    const handleContentVisibility = (event: Event) => {
      const skipped = (event as Event & { skipped?: boolean }).skipped;
      if (typeof skipped === 'boolean') {
        setIsVisible(!skipped);
      }
    };

    el.addEventListener('contentvisibilityautostatechange', handleContentVisibility);

    let observer: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== 'undefined') {
      observer = new IntersectionObserver(
        (entries) => {
          const entry = entries[0];
          if (entry) {
            setIsVisible(entry.isIntersecting);
          }
        },
        { threshold: 0.05 }
      );
      observer.observe(el);
    }

    return () => {
      el.removeEventListener('contentvisibilityautostatechange', handleContentVisibility);
      observer?.disconnect();
    };
  }, []);

  const focusTarget = useMemo<[number, number, number] | null>(() => {
    if (!selectedObjectId) return null;
    const found = objects.find((o) => o.id === selectedObjectId);
    if (!found) return null;
    return (
      found.threejs_pose?.position ?? [
        found.box_3d[0],
        found.box_3d[2],
        -found.box_3d[1],
      ]
    );
  }, [objects, selectedObjectId]);

  const matchByGtId = useMemo(() => {
    const map = new Map<string, PerObjectMatch>();
    perObjectMatches?.forEach((m) => map.set(m.gt_id, m));
    return map;
  }, [perObjectMatches]);

  const presets: Array<{
    mode: CameraViewMode;
    keyHint: string;
    label: string;
    icon: React.FC<{ className?: string }>;
  }> = [
    { mode: 'camera_aligned', keyHint: '1', label: 'Cámara', icon: Camera },
    { mode: 'orbit', keyHint: '2', label: 'Órbita 3D', icon: Compass },
    { mode: 'bev', keyHint: '3', label: 'BEV', icon: Grid },
    { mode: 'isometric', keyHint: '4', label: 'Isométrica', icon: Box },
  ];

  const [renderMode, setRenderMode] = useState<ParticleRenderMode>('surfels');
  const [colorMode, setColorMode] = useState<ParticleColorMode>('rgb');
  const [pointScale, setPointScale] = useState<number>(1.35);
  const [isolateSelected, setIsolateSelected] = useState<boolean>(false);
  const [sliceAxis, setSliceAxis] = useState<SliceAxisMode>('off');
  const [sliceRangePct, setSliceRangePct] = useState<[number, number]>([0, 100]);

  const [caliperMode, setCaliperMode] = useState<boolean>(false);
  const [caliperRawA, setCaliperRawA] = useState<[number, number, number] | null>(null);
  const [caliperRawB, setCaliperRawB] = useState<[number, number, number] | null>(null);
  const [referenceCmInput, setReferenceCmInput] = useState<string>('8.56');

  const safeScaleFactor = scaleFactor > 0.05 && Number.isFinite(scaleFactor) ? scaleFactor : 1.0;

  useEffect(() => {
    setCaliperRawA(null);
    setCaliperRawB(null);
  }, [pointCloud]);

  const decodedPositions = useMemo(() => {
    if (!pointCloud || pointCloud.count <= 0 || !pointCloud.positions_b64) return null;
    try {
      const bytes = decodeBase64Bytes(pointCloud.positions_b64);
      if (bytes.byteLength !== pointCloud.count * 12) return null;
      return new Float32Array(bytes.buffer);
    } catch {
      return null;
    }
  }, [pointCloud]);

  const caliperProfile = useMemo(() => {
    if (!caliperRawA || !caliperRawB) return null;
    return computeSurfaceProfileFromCloud(
      decodedPositions,
      pointCloud?.count ?? 0,
      caliperRawA,
      caliperRawB,
      safeScaleFactor
    );
  }, [decodedPositions, pointCloud?.count, caliperRawA, caliperRawB, safeScaleFactor]);

  const handlePickSurfacePoint = (rawPt: [number, number, number]) => {
    if (!caliperMode) return;
    if (!caliperRawA || (caliperRawA && caliperRawB)) {
      setCaliperRawA(rawPt);
      setCaliperRawB(null);
    } else {
      setCaliperRawB(rawPt);
    }
  };

  const applyCalibrationWithCm = (refCm: number, refTitle: string) => {
    if (!onCalibrateScale || !caliperProfile || caliperProfile.euclideanM <= 1e-4) return;
    if (!Number.isFinite(refCm) || refCm <= 0.1) return;
    const refMeters = refCm / 100.0;
    const rawChordM = caliperProfile.euclideanM / safeScaleFactor;
    const nextFactor = Math.min(10.0, Math.max(0.1, refMeters / Math.max(rawChordM, 1e-5)));
    onCalibrateScale(Number(nextFactor.toFixed(5)), refTitle);
  };

  const hasSurfaceMesh = Boolean(
    pointCloud?.surface_mesh && pointCloud.surface_mesh.triangle_count > 0
  );
  const effectiveRenderMode: ParticleRenderMode =
    !hasSurfaceMesh && renderMode !== 'surfels' ? 'surfels' : renderMode;

  const selectedPaletteIdx = useMemo(() => {
    if (!selectedObjectId || !pointCloud?.object_id_palette) return 0;
    const idx = pointCloud.object_id_palette.indexOf(selectedObjectId);
    return idx >= 0 ? idx + 1 : 0;
  }, [selectedObjectId, pointCloud?.object_id_palette]);

  const isolateObjIdx = isolateSelected && selectedPaletteIdx > 0 ? selectedPaletteIdx : 0;

  const { sliceMin, sliceMax } = useMemo(() => {
    if (sliceAxis === 'off') return { sliceMin: -100, sliceMax: 100 };
    const axIdx = sliceAxis === 'x' ? 0 : sliceAxis === 'y' ? 1 : 2;
    const rawMin = (pointCloud?.bounds_three?.min?.[axIdx] ?? -3.0) * safeScaleFactor;
    const rawMax = (pointCloud?.bounds_three?.max?.[axIdx] ?? 3.0) * safeScaleFactor;
    const lo = Math.min(rawMin, rawMax);
    const hi = Math.max(rawMin, rawMax);
    const span = Math.max(hi - lo, 0.04);
    return {
      sliceMin: lo + (sliceRangePct[0] / 100) * span - 0.002,
      sliceMax: lo + (sliceRangePct[1] / 100) * span + 0.002,
    };
  }, [sliceAxis, sliceRangePct, pointCloud?.bounds_three, safeScaleFactor]);

  const collidingIds = useMemo(() => {
    const set = new Set<string>();
    if (measurement?.geometry.is_colliding) {
      set.add(measurement.object_a_id);
      set.add(measurement.object_b_id);
    }
    return set;
  }, [measurement]);

  const hasDenseParticles = Boolean(pointCloud && pointCloud.count > 0 && showPointCloud);

  return (
    <section
      aria-label="Escena 3D Calibrada WebGL"
      className="flex flex-col overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#FFFFFF]"
    >
      {/* Top camera preset bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#E2E5EA] bg-[#F7F7F5] px-3.5 py-2">
        <div className="flex items-center gap-2">
          <Layers className="h-3.5 w-3.5 text-[#1D4ED8]" />
          <h2 className="text-xs font-semibold text-[#111827]">
            Escena 3D Métrica (Three.js)
          </h2>
          {!isVisible && (
            <span className="font-mono-tech rounded bg-[#F0F2F5] px-1.5 py-0.5 text-[10px] text-[#4B5563]">
              WebGL en pausa
            </span>
          )}
        </div>

        <div className="flex items-center gap-1">
          {presets.map(({ mode, keyHint, label, icon: Icon }) => {
            const active = cameraMode === mode;
            return (
              <button
                key={mode}
                type="button"
                onClick={() => {
                  onChangeCameraMode(mode);
                  setTransitionNonce((n) => n + 1);
                }}
                className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${
                  active
                    ? 'bg-[#111827] text-white'
                    : 'bg-[#FFFFFF] text-[#4B5563] border border-[#E2E5EA] hover:text-[#111827]'
                }`}
                title={`Atajo de teclado: ${keyHint}`}
              >
                <Icon className="h-3 w-3" />
                <span>{label}</span>
                <kbd
                  className={`font-mono-tech ml-0.5 text-[9px] ${
                    active ? 'text-white/75' : 'text-[#9CA3AF]'
                  }`}
                >
                  {keyHint}
                </kbd>
              </button>
            );
          })}
        </div>
      </div>

      {/* Dense Body Surfel / 2.5D Mesh / Slice Inspector Bar */}
      {hasDenseParticles && pointCloud && (
        <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1.5 border-b border-[#E2E5EA] bg-[#FFFFFF] px-3.5 py-1.5 text-[11px]">
          <div className="flex flex-wrap items-center gap-2">
            {/* Render mode: Surfels / Malla 2.5D / Híbrido */}
            <div className="flex items-center rounded border border-[#E2E5EA] bg-[#F7F7F5] p-0.5">
              {(
                [
                  { id: 'surfels', label: 'Surfels' },
                  { id: 'mesh', label: 'Malla 2.5D' },
                  { id: 'hybrid', label: 'Híbrido' },
                ] as const
              ).map((m) => {
                const disabled = m.id !== 'surfels' && !hasSurfaceMesh;
                const active = effectiveRenderMode === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    disabled={disabled}
                    onClick={() => setRenderMode(m.id)}
                    className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                      active
                        ? 'bg-[#111827] text-white'
                        : disabled
                        ? 'cursor-not-allowed text-[#9CA3AF]'
                        : 'text-[#4B5563] hover:text-[#111827]'
                    }`}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>

            {/* Color map: RGB / Normales / Relieve */}
            <div className="flex items-center rounded border border-[#E2E5EA] bg-[#F7F7F5] p-0.5">
              {(
                [
                  { id: 'rgb', label: 'RGB' },
                  { id: 'normals', label: 'Normales' },
                  { id: 'relief', label: 'Relieve' },
                ] as const
              ).map((c) => {
                const active = colorMode === c.id;
                return (
                  <button
                    key={c.id}
                    type="button"
                    onClick={() => setColorMode(c.id)}
                    className={`rounded px-1.5 py-0.5 text-[10px] font-medium transition-colors ${
                      active
                        ? 'bg-[#1D4ED8] text-white'
                        : 'text-[#4B5563] hover:text-[#111827]'
                    }`}
                  >
                    {c.label}
                  </button>
                );
              })}
            </div>

            {/* Surfel radius multiplier */}
            <div className="flex items-center gap-1.5">
              <span className="font-mono-tech text-[10px] text-[#4B5563]">
                Radio {pointScale.toFixed(1)}×
              </span>
              <input
                type="range"
                aria-label="Tamaño de surfels 2DGS"
                min={0.6}
                max={2.4}
                step={0.1}
                value={pointScale}
                onChange={(e) => setPointScale(Number(e.target.value))}
                className="optical-slider w-16"
              />
            </div>

            {/* Isolate selected body */}
            <button
              type="button"
              disabled={selectedPaletteIdx === 0}
              onClick={() => setIsolateSelected((v) => !v)}
              title={
                selectedPaletteIdx === 0
                  ? 'Selecciona un objeto en la escena para aislar sus partículas'
                  : 'Aislar nube densa del cuerpo seleccionado'
              }
              className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-medium transition-colors ${
                isolateObjIdx > 0
                  ? 'border-[#1D4ED8] bg-[#1D4ED8]/10 text-[#1D4ED8]'
                  : selectedPaletteIdx === 0
                  ? 'cursor-not-allowed border-[#E2E5EA] bg-[#F7F7F5] text-[#9CA3AF]'
                  : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#111827] hover:bg-[#F7F7F5]'
              }`}
            >
              <Focus className="h-3 w-3" />
              <span>Aislar Cuerpo</span>
            </button>

            {/* Surface Caliper 3D (2-click measurement + scale calibration) */}
            <button
              type="button"
              onClick={() => setCaliperMode((v) => !v)}
              title="Medir distancia 3D directa sobre surfels/malla (2 clics) y calibrar escala física"
              className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-[10px] font-medium transition-colors ${
                caliperMode
                  ? 'border-[#0F766E] bg-[#0F766E]/10 text-[#0F766E]'
                  : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#111827] hover:bg-[#F7F7F5]'
              }`}
            >
              <Crosshair className="h-3 w-3" />
              <span>Calibrador 3D</span>
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Cross-section slicer */}
            <div className="flex items-center gap-1 rounded border border-[#E2E5EA] bg-[#F7F7F5] px-1.5 py-0.5">
              <Scissors className="h-3 w-3 text-[#4B5563]" />
              <span className="text-[10px] text-[#4B5563]">Corte</span>
              {(['off', 'z', 'y', 'x'] as const).map((ax) => (
                <button
                  key={ax}
                  type="button"
                  onClick={() => {
                    setSliceAxis(ax);
                    if (ax === 'off') setSliceRangePct([0, 100]);
                  }}
                  className={`font-mono-tech rounded px-1 py-0.5 text-[10px] uppercase ${
                    sliceAxis === ax
                      ? 'bg-[#111827] text-white'
                      : 'text-[#4B5563] hover:text-[#111827]'
                  }`}
                >
                  {ax}
                </button>
              ))}
              {sliceAxis !== 'off' && (
                <>
                  <input
                    type="range"
                    aria-label="Límite inferior de corte"
                    min={0}
                    max={95}
                    step={1}
                    value={sliceRangePct[0]}
                    onChange={(e) => {
                      const v = Math.min(Number(e.target.value), sliceRangePct[1] - 4);
                      setSliceRangePct([Math.max(0, v), sliceRangePct[1]]);
                    }}
                    className="optical-slider w-12"
                  />
                  <input
                    type="range"
                    aria-label="Límite superior de corte"
                    min={5}
                    max={100}
                    step={1}
                    value={sliceRangePct[1]}
                    onChange={(e) => {
                      const v = Math.max(Number(e.target.value), sliceRangePct[0] + 4);
                      setSliceRangePct([sliceRangePct[0], Math.min(100, v)]);
                    }}
                    className="optical-slider w-12"
                  />
                  <span className="font-mono-tech text-[10px] text-[#111827]">
                    [{sliceMin.toFixed(2)},{sliceMax.toFixed(2)}]m
                  </span>
                </>
              )}
            </div>

            {/* Export PLY */}
            <button
              type="button"
              onClick={() =>
                exportPointCloudToPly(
                  pointCloud,
                  isolateObjIdx,
                  sliceAxis,
                  sliceMin,
                  sliceMax,
                  safeScaleFactor
                )
              }
              className="inline-flex items-center gap-1 rounded border border-[#E2E5EA] bg-[#FFFFFF] px-2 py-0.5 text-[10px] font-medium text-[#111827] transition-colors hover:bg-[#F7F7F5]"
              title="Exportar nube métrica con normales y RGB en formato estándar .PLY (Blender/CloudCompare/Omniverse)"
            >
              <Download className="h-3 w-3 text-[#0F766E]" />
              <span>Exportar .PLY</span>
            </button>
          </div>
        </div>
      )}

      {/* Interactive 3D Surface Caliper & Reference Scale Anchor Strip (Zero AI-Slop Editorial) */}
      {(caliperMode ||
        Math.abs(safeScaleFactor - 1.0) > 0.005 ||
        Boolean(arucoAnchor?.detected)) && (
        <div className="border-b border-[#E2E5EA] bg-[#FFFFFF] px-3.5 py-2 text-[11px] space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="inline-flex items-center gap-1.5 font-semibold text-[#111827]">
                <Scale className="h-3.5 w-3.5 text-[#0F766E]" />
                <span>Calibrador y Ancla de Escala</span>
              </span>

              {!caliperRawA && caliperMode && (
                <span className="font-mono-tech text-[#0F766E]">
                  Haz clic en P1 sobre la nube o malla 2.5D…
                </span>
              )}
              {caliperRawA && !caliperRawB && (
                <span className="font-mono-tech text-[#B45309]">
                  P1 fijado · Haz clic en P2 para cerrar medición…
                </span>
              )}
              {caliperProfile && (
                <span className="font-mono-tech text-[#111827]">
                  3D <strong className="text-[#BE123C]">{(caliperProfile.euclideanM * 100).toFixed(2)} cm</strong>{' '}
                  ({Math.round(caliperProfile.euclideanM * 1000)} mm) · Arco{' '}
                  <strong className="text-[#0F766E]">{(caliperProfile.surfaceArcM * 100).toFixed(2)} cm</strong>{' '}
                  · Relieve <strong>{caliperProfile.maxReliefMm.toFixed(1)} mm</strong> · Pendiente{' '}
                  <strong>{caliperProfile.slopeDeg.toFixed(1)}°</strong> · Δ[
                  {(caliperProfile.deltaXyzM[0] * 100).toFixed(1)},{' '}
                  {(caliperProfile.deltaXyzM[1] * 100).toFixed(1)},{' '}
                  {(caliperProfile.deltaXyzM[2] * 100).toFixed(1)}] cm
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              {/* Inline SVG surface elevation sparkline */}
              {caliperProfile && caliperProfile.elevationsMm.length >= 3 && (
                <svg
                  width="110"
                  height="26"
                  viewBox="0 0 110 26"
                  aria-label="Perfil topográfico del corte superficial"
                  className="border border-[#E2E5EA] bg-[#F7F7F5]"
                >
                  {(() => {
                    const els = caliperProfile.elevationsMm;
                    const maxE = Math.max(...els, 2);
                    const ptsStr = els
                      .map((v, i) => {
                        const x = 4 + (i / Math.max(els.length - 1, 1)) * 102;
                        const y = 22 - (v / maxE) * 18;
                        return `${x.toFixed(1)},${y.toFixed(1)}`;
                      })
                      .join(' ');
                    return (
                      <polyline
                        fill="none"
                        stroke="#0F766E"
                        strokeWidth="1.6"
                        points={ptsStr}
                      />
                    );
                  })()}
                </svg>
              )}

              {(caliperRawA || caliperRawB) && (
                <button
                  type="button"
                  onClick={() => {
                    setCaliperRawA(null);
                    setCaliperRawB(null);
                  }}
                  className="inline-flex items-center gap-1 rounded border border-[#E2E5EA] bg-[#F7F7F5] px-2 py-0.5 text-[10px] font-medium text-[#4B5563] hover:text-[#111827]"
                >
                  <X className="h-3 w-3" />
                  <span>Limpiar P1/P2</span>
                </button>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-[#E2E5EA] pt-2">
            <div className="flex flex-wrap items-center gap-1.5">
              <label htmlFor="ref-cm-input" className="text-[10px] font-medium text-[#4B5563]">
                Ancla real (cm):
              </label>
              <input
                id="ref-cm-input"
                type="number"
                min={0.5}
                max={2000}
                step={0.1}
                value={referenceCmInput}
                onChange={(e) => setReferenceCmInput(e.target.value)}
                disabled={!onCalibrateScale}
                className="font-mono-tech w-20 rounded border border-[#E2E5EA] bg-[#F7F7F5] px-2 py-0.5 text-[11px] text-[#111827] focus:border-[#1D4ED8] focus:outline-none"
              />
              {[
                { label: 'Tarjeta 8.56', cm: 8.56 },
                { label: 'Moneda 2.5', cm: 2.5 },
                { label: 'A4 29.7', cm: 29.7 },
              ].map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => {
                    setReferenceCmInput(String(preset.cm));
                    if (caliperProfile && caliperProfile.euclideanM > 1e-4) {
                      applyCalibrationWithCm(preset.cm, `${preset.label} cm`);
                    }
                  }}
                  className="font-mono-tech rounded border border-[#E2E5EA] bg-[#FFFFFF] px-1.5 py-0.5 text-[10px] text-[#4B5563] hover:border-[#1D4ED8] hover:text-[#111827]"
                >
                  {preset.label}
                </button>
              ))}
              <button
                type="button"
                disabled={!caliperProfile || !onCalibrateScale}
                onClick={() => {
                  const val = Number(referenceCmInput);
                  applyCalibrationWithCm(val, `Referencia ${referenceCmInput} cm`);
                }}
                className="rounded border border-[#1D4ED8] bg-[#1D4ED8] px-2 py-0.5 text-[10px] font-medium text-white transition-colors hover:bg-[#1E40AF] disabled:cursor-not-allowed disabled:opacity-45"
              >
                Calibrar Escala
              </button>

              {arucoAnchor?.detected && onCalibrateScale && (
                <button
                  type="button"
                  onClick={() =>
                    onCalibrateScale(
                      arucoAnchor.suggested_scale_factor,
                      `ArUco ${arucoAnchor.dictionary} #${arucoAnchor.marker_id}`
                    )
                  }
                  className="font-mono-tech rounded border border-[#0F766E] bg-[#FFFFFF] px-2 py-0.5 text-[10px] font-semibold text-[#0F766E] hover:bg-[#F7F7F5]"
                >
                  Aplicar ArUco #{arucoAnchor.marker_id} (×
                  {arucoAnchor.suggested_scale_factor.toFixed(2)})
                </button>
              )}
            </div>

            <div className="font-mono-tech flex items-center gap-2 text-[10px] text-[#4B5563]">
              <span>
                Escala:{' '}
                <strong className="text-[#111827]">×{safeScaleFactor.toFixed(3)}</strong>
                {scaleLabel ? ` (${scaleLabel})` : ''}
              </span>
              {Math.abs(safeScaleFactor - 1.0) > 0.005 && onResetScale && (
                <button
                  type="button"
                  onClick={onResetScale}
                  className="inline-flex items-center gap-1 rounded border border-[#E2E5EA] bg-[#F7F7F5] px-1.5 py-0.5 text-[10px] font-medium text-[#111827] hover:bg-[#FFFFFF]"
                >
                  <RotateCcw className="h-2.5 w-2.5" />
                  <span>Restaurar 1.00×</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Off-screen pausing WebGL container */}
      <div
        ref={stageRef}
        className={`webgl-stage-container relative h-[430px] sm:h-[480px] w-full bg-[#F0F2F5] ${
          caliperMode ? 'cursor-crosshair' : ''
        }`}
      >
        <Canvas
          dpr={[1, 2]}
          shadows="percentage"
          frameloop={isVisible ? 'always' : 'demand'}
          camera={{ position: [2.4, 1.85, 0.8], fov: fovVDeg, near: 0.03, far: 80 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          onCreated={({ gl, raycaster }) => {
            gl.localClippingEnabled = true;
            if (raycaster.params.Points) {
              raycaster.params.Points.threshold = 0.035;
            }
          }}
        >
          <color attach="background" args={['#F0F2F5']} />

          {/* Clean architectural daylight illumination */}
          <ambientLight intensity={0.85} />
          <directionalLight
            position={[4, 7, 4]}
            intensity={0.95}
            castShadow
            shadow-mapSize-width={1024}
            shadow-mapSize-height={1024}
          />
          <directionalLight position={[-4, 3, -3]} intensity={0.35} />

          <CameraRig
            cameraMode={cameraMode}
            fovVDeg={fovVDeg}
            focusTarget={focusTarget}
            sceneScale={sceneScale}
            transitionNonce={transitionNonce}
          />

          <MetricOpticalBenchFloor cameraMode={cameraMode} sceneScale={sceneScale} />

          <CameraFrustumAndFocalPlane
            imageBase64={imageBase64}
            imageMime={imageMime}
            fovHDeg={fovHDeg}
            fovVDeg={fovVDeg}
            focalPlaneDepth={focalPlaneDepth}
            unfoldProgress={unfoldProgress}
            cameraMode={cameraMode}
          />

          <group scale={[safeScaleFactor, safeScaleFactor, safeScaleFactor]}>
            <MetricSurfaceMesh
              pointCloud={pointCloud}
              visible={
                Boolean(showPointCloud) &&
                isolateObjIdx === 0 &&
                (effectiveRenderMode === 'mesh' || effectiveRenderMode === 'hybrid')
              }
              colorMode={colorMode}
              sliceAxis={sliceAxis}
              sliceMin={sliceMin}
              sliceMax={sliceMax}
              caliperMode={caliperMode}
              scaleFactor={safeScaleFactor}
              onPickSurfacePoint={handlePickSurfacePoint}
            />

            <MetricPointCloud
              pointCloud={pointCloud}
              visible={
                Boolean(showPointCloud) &&
                (effectiveRenderMode === 'surfels' ||
                  effectiveRenderMode === 'hybrid' ||
                  isolateObjIdx > 0)
              }
              pointSize={pointSize}
              pointScale={effectiveRenderMode === 'hybrid' ? pointScale * 0.8 : pointScale}
              colorMode={colorMode}
              isolateObjIdx={isolateObjIdx}
              sliceAxis={sliceAxis}
              sliceMin={sliceMin / safeScaleFactor}
              sliceMax={sliceMax / safeScaleFactor}
              caliperMode={caliperMode}
              scaleFactor={safeScaleFactor}
              onPickSurfacePoint={handlePickSurfacePoint}
            />
          </group>

          {/* Predicted OBBs */}
          {objects.map((obj) => (
            <AnimatedPredictedOBB
              key={obj.id}
              obj={obj}
              unfoldProgress={unfoldProgress}
              focalPlaneDepth={focalPlaneDepth}
              scaleRatio={scaleRatio}
              isSelected={selectedObjectId === obj.id}
              isHovered={hoveredObjectId === obj.id}
              isDimmed={
                highlightedIds !== null && !highlightedIds.includes(obj.id)
              }
              isColliding={collidingIds.has(obj.id)}
              onSelect={onSelectObject}
              onHover={onHoverObject}
            />
          ))}

          {/* Ground Truth dashed Emerald cuboids + centroid error vectors */}
          {showGroundTruth &&
            gtObjects?.map((gt) => (
              <GroundTruthOBB
                key={`gt-3d-${gt.id}`}
                gt={gt}
                match={matchByGtId.get(gt.id)}
              />
            ))}

          {/* Occlusion rays from camera origin (0,0,0) -> hit -> occluded target */}
          {showOcclusionRays &&
            occlusionMatrix.map((entry, idx) => (
              <group key={`occ-${idx}`}>
                {entry.occluded_rays_three?.map((ray, rIdx) => (
                  <group key={`ray-${rIdx}`}>
                    <Line
                      points={[ray.origin, ray.hit]}
                      color="#B45309"
                      lineWidth={1.3}
                      transparent
                      opacity={0.75}
                    />
                    <Line
                      points={[ray.hit, ray.target]}
                      color="#B45309"
                      lineWidth={1.3}
                      dashed
                      dashSize={0.08}
                      gapSize={0.06}
                      transparent
                      opacity={0.6}
                    />
                    <mesh position={ray.hit}>
                      <sphereGeometry args={[0.025, 8, 8]} />
                      <meshBasicMaterial color="#B45309" />
                    </mesh>
                  </group>
                ))}
              </group>
            ))}

          {/* 3D Measurement ruler */}
          {measurement && <MeasurementRuler3D measurement={measurement} />}

          {/* 3D Direct Surface Caliper */}
          {(caliperRawA || caliperProfile) && (
            <SurfaceCaliper3D
              pointA={caliperRawA}
              profile={caliperProfile}
              scaleFactor={safeScaleFactor}
              scaleRatio={scaleRatio}
            />
          )}
        </Canvas>

        {/* Quiet bottom HUD overlay */}
        <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap items-center justify-between gap-2">
          <div className="font-mono-tech flex flex-wrap items-center gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF]/92 px-2.5 py-1 text-[11px] text-[#111827] backdrop-blur-xs">
            <span>
              FOV {fovHDeg.toFixed(1)}° / {fovVDeg.toFixed(1)}°
            </span>
            <span className="text-[#CBD5E1]">|</span>
            <span>1u = 1.00m</span>
            {Math.abs(safeScaleFactor - 1.0) > 0.005 && (
              <>
                <span className="text-[#CBD5E1]">|</span>
                <span className="font-semibold text-[#0F766E]">
                  Escala ×{safeScaleFactor.toFixed(2)}
                </span>
              </>
            )}
            <span className="text-[#CBD5E1]">|</span>
            <span className="inline-flex items-center gap-1 text-[#0F766E]">
              <Eye className="h-3 w-3" />
              {modeBadge}
            </span>
            {pointCloud && pointCloud.count > 0 && showPointCloud && (
              <>
                <span className="text-[#CBD5E1]">|</span>
                <span className="text-[#1D4ED8]">
                  {pointCloud.count.toLocaleString()} surfels
                  {typeof pointCloud.body_point_count === 'number' && pointCloud.body_point_count > 0
                    ? ` · ${pointCloud.body_point_count.toLocaleString()} cuerpo`
                    : ''}
                  {pointCloud.surface_mesh && pointCloud.surface_mesh.triangle_count > 0
                    ? ` · ${pointCloud.surface_mesh.triangle_count.toLocaleString()} tris`
                    : ''}
                </span>
              </>
            )}
          </div>
          <div className="font-mono-tech hidden sm:flex items-center gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF]/92 px-2.5 py-1 text-[10px] text-[#4B5563]">
            <span>Teclas: 1–4 vistas · G GT · M medir · Esc limpiar</span>
          </div>
        </div>
      </div>
    </section>
  );
};
