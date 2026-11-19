import React, { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Edges, Line, OrbitControls } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { Box, Camera, Compass, Eye, Grid, Layers } from 'lucide-react';
import type {
  CameraViewMode,
  MeasurementState,
  OcclusionEntry,
  PerObjectMatch,
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

const FOCAL_PLANE_Z = -0.85; // Three.js forward is -Z

interface CameraPreset {
  pos: [number, number, number];
  target: [number, number, number];
}

function getCameraPreset(
  mode: CameraViewMode,
  focusPos?: [number, number, number] | null
): CameraPreset {
  const focus = focusPos ?? [0, 0.25, -2.8];
  switch (mode) {
    case 'camera_aligned':
      return { pos: [0, 0, 0.001], target: [0, 0, -2.5] };
    case 'bev':
      return { pos: [focus[0], 7.2, focus[2] + 0.01], target: [focus[0], 0, focus[2]] };
    case 'isometric':
      return { pos: [focus[0] + 3.8, 3.2, focus[2] + 3.8], target: focus };
    case 'orbit':
    default:
      return { pos: [focus[0] + 2.4, 1.85, focus[2] + 3.4], target: focus };
  }
}

const CameraRig: React.FC<{
  cameraMode: CameraViewMode;
  fovVDeg: number;
  focusTarget: [number, number, number] | null;
  transitionNonce: number;
}> = ({ cameraMode, fovVDeg, focusTarget, transitionNonce }) => {
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

  // Trigger smooth camera preset transition on mode or focus change
  useEffect(() => {
    animatingRef.current = true;
  }, [cameraMode, focusTarget, transitionNonce]);

  useFrame((_, rawDelta) => {
    if (!animatingRef.current || userDraggingRef.current) return;
    const dt = Math.min(rawDelta, 0.1);
    const preset = getCameraPreset(cameraMode, focusTarget);
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
  unfoldProgress: number;
}> = ({ imageBase64, imageMime, fovHDeg, fovVDeg, unfoldProgress }) => {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

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

  const depth = Math.abs(FOCAL_PLANE_Z);
  const halfW = depth * Math.tan((fovHDeg * Math.PI) / 360);
  const halfH = depth * Math.tan((fovVDeg * Math.PI) / 360);

  const corners: [number, number, number][] = [
    [-halfW, halfH, -depth],
    [halfW, halfH, -depth],
    [halfW, -halfH, -depth],
    [-halfW, -halfH, -depth],
  ];

  const planeOpacity = Math.max(0.22, 0.88 - unfoldProgress * 0.55);

  return (
    <group>
      {/* Camera origin marker */}
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
        <mesh position={[0, 0, -depth]}>
          <planeGeometry args={[halfW * 2, halfH * 2]} />
          <meshBasicMaterial
            map={texture}
            transparent
            opacity={planeOpacity}
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
}> = ({
  position,
  lines,
  accentColor = '#111827',
  borderColor = '#E2E5EA',
  backgroundColor = 'rgba(255, 255, 255, 0.94)',
  worldHeight = 0.22,
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
        depthTest={false}
        depthWrite={false}
      />
    </sprite>
  );
};

const MetricOpticalBenchFloor: React.FC = () => {
  const rings = [1, 2, 3, 4, 5, 6];

  const ringPointsList = useMemo(() => {
    return rings.map((r) => {
      const pts: [number, number, number][] = [];
      const segments = 64;
      for (let i = 0; i <= segments; i++) {
        // Semicircle in front of camera (-Z)
        const angle = Math.PI + (i / segments) * Math.PI;
        pts.push([Math.cos(angle) * r, -0.58, Math.sin(angle) * r]);
      }
      return { r, pts };
    });
  }, []);

  return (
    <group>
      {/* Subtle matte grid on ground plane */}
      <gridHelper
        args={[16, 16, '#94A3B8', '#CBD5E1']}
        position={[0, -0.59, -4]}
      />

      {/* Concentric metric distance arcs 1m..6m */}
      {ringPointsList.map(({ r, pts }) => (
        <group key={r}>
          <Line
            points={pts}
            color="#94A3B8"
            lineWidth={1}
            dashed
            dashSize={0.14}
            gapSize={0.1}
            transparent
            opacity={0.75}
          />
          <CanvasSpriteLabel
            position={[0.16, -0.52, -r]}
            lines={[`${r}m`]}
            accentColor="#4B5563"
            borderColor="#CBD5E1"
            backgroundColor="rgba(240, 242, 245, 0.9)"
            worldHeight={0.15}
          />
        </group>
      ))}
    </group>
  );
};

const AnimatedPredictedOBB: React.FC<{
  obj: SpatialObject;
  unfoldProgress: number;
  isSelected: boolean;
  isHovered: boolean;
  isDimmed: boolean;
  isColliding: boolean;
  onSelect: (id: string, shiftKey: boolean) => void;
  onHover: (id: string | null) => void;
}> = ({
  obj,
  unfoldProgress,
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

  // Projection point onto focal plane z = FOCAL_PLANE_Z
  const planeIntersection = useMemo<[number, number, number]>(() => {
    const z = Math.min(-0.1, targetPos[2]);
    const t = FOCAL_PLANE_Z / z;
    return [targetPos[0] * t, targetPos[1] * t, FOCAL_PLANE_Z];
  }, [targetPos]);

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
  const arrowLen = Math.max(0.25, Math.min(0.65, Math.max(...targetSize) * 0.65));
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
          position={[targetPos[0], targetPos[1] + targetSize[1] / 2 + 0.18, targetPos[2]]}
          lines={[
            `${obj.label} · ${formatMeters(obj.distance_m ?? obj.box_3d[1], 2)}`,
          ]}
          accentColor={isSelected ? '#BE123C' : '#1D4ED8'}
          borderColor={isSelected ? '#BE123C' : '#E2E5EA'}
          worldHeight={0.2}
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
}) => {
  const stageRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(true);
  const [transitionNonce, setTransitionNonce] = useState(0);

  const fovVDeg = useMemo(
    () => computeFovVDeg(fovHDeg, width || 960, height || 640),
    [fovHDeg, width, height]
  );

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

  const collidingIds = useMemo(() => {
    const set = new Set<string>();
    if (measurement?.geometry.is_colliding) {
      set.add(measurement.object_a_id);
      set.add(measurement.object_b_id);
    }
    return set;
  }, [measurement]);

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

      {/* Off-screen pausing WebGL container */}
      <div
        ref={stageRef}
        className="webgl-stage-container relative h-[430px] sm:h-[480px] w-full bg-[#F0F2F5]"
      >
        <Canvas
          dpr={[1, 2]}
          shadows="percentage"
          frameloop={isVisible ? 'always' : 'demand'}
          camera={{ position: [2.4, 1.85, 0.8], fov: fovVDeg, near: 0.05, far: 80 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
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
            transitionNonce={transitionNonce}
          />

          <MetricOpticalBenchFloor />

          <CameraFrustumAndFocalPlane
            imageBase64={imageBase64}
            imageMime={imageMime}
            fovHDeg={fovHDeg}
            fovVDeg={fovVDeg}
            unfoldProgress={unfoldProgress}
          />

          {/* Predicted OBBs */}
          {objects.map((obj) => (
            <AnimatedPredictedOBB
              key={obj.id}
              obj={obj}
              unfoldProgress={unfoldProgress}
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
        </Canvas>

        {/* Quiet bottom HUD overlay */}
        <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap items-center justify-between gap-2">
          <div className="font-mono-tech flex flex-wrap items-center gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF]/92 px-2.5 py-1 text-[11px] text-[#111827] backdrop-blur-xs">
            <span>
              FOV {fovHDeg.toFixed(1)}° / {fovVDeg.toFixed(1)}°
            </span>
            <span className="text-[#CBD5E1]">|</span>
            <span>1u = 1.00m</span>
            <span className="text-[#CBD5E1]">|</span>
            <span className="inline-flex items-center gap-1 text-[#0F766E]">
              <Eye className="h-3 w-3" />
              {modeBadge}
            </span>
          </div>
          <div className="font-mono-tech hidden sm:flex items-center gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF]/92 px-2.5 py-1 text-[10px] text-[#4B5563]">
            <span>Teclas: 1–4 vistas · G GT · M medir · Esc limpiar</span>
          </div>
        </div>
      </div>
    </section>
  );
};
