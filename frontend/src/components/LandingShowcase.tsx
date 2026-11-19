import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Edges, Line } from '@react-three/drei';
import * as THREE from 'three';
import {
  ArrowDown,
  ArrowRight,
  Compass,
  Eye,
  Film,
  Layers,
  Ruler,
  ScanEye,
  ShieldAlert,
} from 'lucide-react';
import type { AppSection, BenchmarkScene, SpatialObject } from '../types/spatial';
import { normalizeBase64Image } from '../utils/geometry';

interface LandingShowcaseProps {
  benchmarks: BenchmarkScene[];
  onOpenLab: (benchmarkId?: string) => void;
  onNavigate: (section: AppSection) => void;
}

interface DemoBox {
  id: string;
  label: string;
  pos: [number, number, number];
  size: [number, number, number];
  quat: [number, number, number, number];
}

const FALLBACK_DEMO_BOXES: DemoBox[] = [
  {
    id: 'demo_1',
    label: 'oscilloscope',
    pos: [-0.62, -0.26, -2.35],
    size: [0.42, 0.32, 0.28],
    quat: [0, 0.156, 0, 0.988],
  },
  {
    id: 'demo_2',
    label: 'sample_case',
    pos: [0.05, -0.31, -2.1],
    size: [0.34, 0.22, 0.26],
    quat: [0, -0.07, 0, 0.997],
  },
  {
    id: 'demo_3',
    label: 'spectrometer',
    pos: [0.64, -0.22, -2.75],
    size: [0.4, 0.36, 0.3],
    quat: [0, 0.216, 0, 0.976],
  },
  {
    id: 'demo_4',
    label: 'calibration_cube',
    pos: [-0.22, -0.34, -1.72],
    size: [0.22, 0.2, 0.22],
    quat: [0, 0.087, 0, 0.996],
  },
  {
    id: 'demo_5',
    label: 'monitor',
    pos: [0.28, -0.06, -3.25],
    size: [0.58, 0.44, 0.16],
    quat: [0, -0.052, 0, 0.999],
  },
];

const CHAPTERS = [
  {
    step: '01',
    shortLabel: 'Pinhole 2D',
    title: 'Foto 2D y Proyección Óptica Pinhole',
    summary:
      'Toda imagen RGB nace proyectada sobre un plano focal a través de la matriz intrínseca K(FOV_h). Aquí el observador mira exactamente desde el lente virtual hacia la escena.',
    metricTag: 'K: f = W / 2·tan(FOV_h/2)',
    accent: '#1D4ED8',
  },
  {
    step: '02',
    shortLabel: 'Extrusión 3D',
    title: 'Poses 9-DOF de Gemini y Base Quiral det(M)=+1',
    summary:
      'Google AI Studio infiere centro (x,y,z), tamaño (dx,dy,dz) y rotación Euler. Python aplica M = Rₓ(−90°) para proyectar cada volumen en Three.js sin espejar ejes.',
    metricTag: 'det(M) = +1.0 · sin inversión',
    accent: '#1D4ED8',
  },
  {
    step: '03',
    shortLabel: 'Regla & Rayos',
    title: 'Medición Exacta y Oclusiones Kay-Kajiya',
    summary:
      'En lugar de dejar que la IA adivine números, Python calcula distancias centro-a-centro, holguras SAT y rayos desde la cámara para descubrir qué objeto bloquea al del fondo.',
    metricTag: 'Distancia 1.17 m · Oclusión por rayos',
    accent: '#BE123C',
  },
  {
    step: '04',
    shortLabel: 'Planta BEV',
    title: 'Vista Superior (BEV) y Diagnóstico Universal',
    summary:
      'Desde arriba verificamos el calce contra el Ground Truth esmeralda y auditamos bordes Sobel, levitación gravitatoria y deriva multi-frame incluso en fotos sin referencia.',
    metricTag: 'mIoU BEV · Plausibilidad Física',
    accent: '#0F766E',
  },
];

const LATENT_CAPABILITIES = [
  {
    badge: 'PECS Sobel',
    title: 'Coherencia de Bordes 2D↔3D',
    desc: 'Comprueba si las 12 aristas proyectadas caen sobre contornos reales de la foto.',
    color: '#0F766E',
  },
  {
    badge: 'PVI SAT',
    title: 'Interpenetración de Sólidos',
    desc: 'Detecta mediante pruebas SAT de 15 ejes si dos volúmenes ocupan el mismo espacio.',
    color: '#BE123C',
  },
  {
    badge: 'GFI Gravedad',
    title: 'Anomalía de Soporte en Piso',
    desc: 'Mide en metros si un objeto predicho levita o se hunde bajo el plano latente.',
    color: '#B45309',
  },
  {
    badge: 'SDAS Escala',
    title: 'Acoplamiento Tamaño–Distancia',
    desc: 'Expone si el modelo infla artificialmente objetos lejanos para compensar píxeles.',
    color: '#1D4ED8',
  },
  {
    badge: '±σ_y Rango',
    title: 'Incertidumbre Cuadrática',
    desc: 'Modela cómo crece el margen de error con y² y se agrava por oclusión o brillo.',
    color: '#111827',
  },
  {
    badge: 'Multi-Frame',
    title: 'Auditoría Estilo World Model',
    desc: 'Cuantifica deriva de volumen (%) y jitter angular (°) frente a videos tipo Genie/Veo.',
    color: '#0F766E',
  },
];

// Reusable scratch vectors outside React render/frame loops (zero allocations per frame)
const _camTargetPos = new THREE.Vector3();
const _lookAtCurrent = new THREE.Vector3(0, -0.15, -2.4);
const _lookAtTarget = new THREE.Vector3(0, -0.15, -2.4);
const _unfoldState = { value: 0.35 };
const _rulerOpacity = { value: 0 };
const _gtOpacity = { value: 0 };

function toDemoBoxes(gtObjects?: SpatialObject[]): DemoBox[] {
  if (!gtObjects || gtObjects.length === 0) return FALLBACK_DEMO_BOXES;
  return gtObjects.slice(0, 6).map((o) => {
    const pose = o.threejs_pose;
    if (pose) {
      return {
        id: o.id,
        label: o.label,
        pos: pose.position,
        size: pose.size,
        quat: pose.quaternion,
      };
    }
    return {
      id: o.id,
      label: o.label,
      pos: [o.box_3d[0], o.box_3d[2], -o.box_3d[1]],
      size: [o.box_3d[3], o.box_3d[5], o.box_3d[4]],
      quat: [0, 0, 0, 1],
    };
  });
}

const ScrollDrivenCuboid: React.FC<{
  box: DemoBox;
  scrollProgressRef: React.MutableRefObject<number>;
  reducedMotion: boolean;
}> = ({ box, scrollProgressRef, reducedMotion }) => {
  const groupRef = useRef<THREE.Group>(null);
  const gtRef = useRef<THREE.Group>(null);
  const focalZ = -0.88;

  const quat = useMemo(
    () => new THREE.Quaternion(box.quat[0], box.quat[1], box.quat[2], box.quat[3]),
    [box.quat]
  );

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.1);
    const progress = reducedMotion ? 0.5 : scrollProgressRef.current;

    // Stage 1 (0..0.25): compressed near focal plane (0.2 -> 0.55)
    // Stage 2..4 (0.25..1.0): fully extruded metric pose (1.0)
    const targetUnfold =
      progress < 0.28
        ? 0.22 + (progress / 0.28) * 0.78
        : 1.0;

    const u = reducedMotion ? 1.0 : THREE.MathUtils.damp(_unfoldState.value, targetUnfold, 6.5, dt);

    if (groupRef.current) {
      const ratio = focalZ / Math.min(box.pos[2], -0.4);
      const startX = box.pos[0] * ratio;
      const startY = box.pos[1] * ratio;
      groupRef.current.position.set(
        THREE.MathUtils.lerp(startX, box.pos[0], u),
        THREE.MathUtils.lerp(startY, box.pos[1], u),
        THREE.MathUtils.lerp(focalZ, box.pos[2], u)
      );
      const zScale = Math.max(0.08, u);
      groupRef.current.scale.set(
        THREE.MathUtils.lerp(ratio, 1.0, u),
        THREE.MathUtils.lerp(ratio, 1.0, u),
        zScale
      );
      groupRef.current.quaternion.copy(quat);
    }

    if (gtRef.current) {
      const showGt = progress >= 0.66 || reducedMotion;
      gtRef.current.visible = showGt;
      gtRef.current.position.set(
        box.pos[0] + 0.03,
        box.pos[1],
        box.pos[2] - 0.025
      );
      gtRef.current.quaternion.copy(quat);
    }
  });

  return (
    <>
      <group ref={groupRef}>
        <mesh>
          <boxGeometry args={box.size} />
          <meshStandardMaterial
            color="#1D4ED8"
            transparent
            opacity={0.14}
            roughness={0.5}
            depthWrite={false}
          />
          <Edges color="#1D4ED8" threshold={15} lineWidth={1.6} />
        </mesh>
      </group>

      {/* Emerald Ground Truth comparison wireframe revealed in Stage 4 BEV */}
      <group ref={gtRef} visible={false}>
        <mesh>
          <boxGeometry args={box.size} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
          <Edges color="#0F766E" threshold={15} lineWidth={1.5} />
        </mesh>
      </group>
    </>
  );
};

const ScrollRigAndOverlays: React.FC<{
  boxes: DemoBox[];
  scrollProgressRef: React.MutableRefObject<number>;
  reducedMotion: boolean;
}> = ({ boxes, scrollProgressRef, reducedMotion }) => {
  const rulerGroupRef = useRef<THREE.Group>(null);
  const frustumOriginRef = useRef<THREE.Group>(null);

  const focalDepth = 0.88;
  const halfW = focalDepth * Math.tan((60 * Math.PI) / 360);
  const halfH = halfW * (640 / 960);

  const frustumCorners: [number, number, number][] = useMemo(
    () => [
      [-halfW, halfH, -focalDepth],
      [halfW, halfH, -focalDepth],
      [halfW, -halfH, -focalDepth],
      [-halfW, -halfH, -focalDepth],
    ],
    [halfW, halfH]
  );

  const ringPoints = useMemo(() => {
    return [1, 2, 3, 4].map((r) => {
      const pts: [number, number, number][] = [];
      for (let i = 0; i <= 48; i++) {
        const ang = Math.PI + (i / 48) * Math.PI;
        pts.push([Math.cos(ang) * r, -0.5, Math.sin(ang) * r]);
      }
      return { r, pts };
    });
  }, []);

  const boxA = boxes[0] ?? FALLBACK_DEMO_BOXES[0];
  const boxB = boxes[1] ?? FALLBACK_DEMO_BOXES[1];
  const boxFar = boxes[4] ?? boxes[boxes.length - 1] ?? FALLBACK_DEMO_BOXES[4];

  useFrame((state, delta) => {
    const dt = Math.min(delta, 0.1);
    const p = reducedMotion ? 0.42 : scrollProgressRef.current;

    if (p < 0.25) {
      // Stage 1: Pinhole Camera Alignment
      const t = p / 0.25;
      _camTargetPos.set(
        THREE.MathUtils.lerp(0.12, 1.35, t),
        THREE.MathUtils.lerp(0.06, 0.95, t),
        THREE.MathUtils.lerp(0.18, 0.85, t)
      );
      _lookAtTarget.set(0, -0.15, -2.35);
    } else if (p < 0.52) {
      // Stage 2: 3D Extrusion Orbit
      const t = (p - 0.25) / 0.27;
      _camTargetPos.set(
        THREE.MathUtils.lerp(1.35, 2.55, t),
        THREE.MathUtils.lerp(0.95, 1.68, t),
        THREE.MathUtils.lerp(0.85, 0.65, t)
      );
      _lookAtTarget.set(0, -0.18, -2.45);
    } else if (p < 0.76) {
      // Stage 3: Ruler & Occlusion Ray Inspection
      const t = (p - 0.52) / 0.24;
      _camTargetPos.set(
        THREE.MathUtils.lerp(2.55, -2.25, t),
        THREE.MathUtils.lerp(1.68, 1.85, t),
        THREE.MathUtils.lerp(0.65, 0.45, t)
      );
      _lookAtTarget.set(0, -0.18, -2.45);
    } else {
      // Stage 4: Bird's-Eye View (BEV) Top-Down
      const t = (p - 0.76) / 0.24;
      _camTargetPos.set(
        THREE.MathUtils.lerp(-2.25, 0.0, t),
        THREE.MathUtils.lerp(1.85, 5.8, t),
        THREE.MathUtils.lerp(0.45, -2.38, t)
      );
      _lookAtTarget.set(0, -0.2, -2.45);
    }

    if (reducedMotion) {
      state.camera.position.set(2.3, 1.55, 0.65);
      _lookAtCurrent.set(0, -0.18, -2.45);
    } else {
      state.camera.position.x = THREE.MathUtils.damp(state.camera.position.x, _camTargetPos.x, 5.5, dt);
      state.camera.position.y = THREE.MathUtils.damp(state.camera.position.y, _camTargetPos.y, 5.5, dt);
      state.camera.position.z = THREE.MathUtils.damp(state.camera.position.z, _camTargetPos.z, 5.5, dt);
      _lookAtCurrent.x = THREE.MathUtils.damp(_lookAtCurrent.x, _lookAtTarget.x, 5.5, dt);
      _lookAtCurrent.y = THREE.MathUtils.damp(_lookAtCurrent.y, _lookAtTarget.y, 5.5, dt);
      _lookAtCurrent.z = THREE.MathUtils.damp(_lookAtCurrent.z, _lookAtTarget.z, 5.5, dt);
    }
    state.camera.lookAt(_lookAtCurrent);

    // Hide camera origin sphere when camera is near (0,0,0)
    if (frustumOriginRef.current) {
      frustumOriginRef.current.visible = state.camera.position.length() > 0.42;
    }

    // Smoothly scale ruler & occlusion rays in during Stage 3+
    const showRuler = p >= 0.45 || reducedMotion;
    const rScale = THREE.MathUtils.damp(_rulerOpacity.value, showRuler ? 1 : 0.001, 7.0, dt);
    _rulerOpacity.value = rScale;
    if (rulerGroupRef.current) {
      rulerGroupRef.current.visible = rScale > 0.04;
      rulerGroupRef.current.scale.setScalar(rScale);
    }

    // Also update shared unfold scalar smoothly
    const targetU = p < 0.28 ? 0.22 + (p / 0.28) * 0.78 : 1.0;
    _unfoldState.value = THREE.MathUtils.damp(_unfoldState.value, targetU, 6.5, dt);
    _gtOpacity.value = p >= 0.66 ? 1 : 0;
  });

  return (
    <group>
      {/* Clean metric floor grid & distance arcs */}
      <gridHelper args={[14, 14, '#94A3B8', '#CBD5E1']} position={[0, -0.52, -3.5]} />
      {ringPoints.map(({ r, pts }) => (
        <Line
          key={`ring-${r}`}
          points={pts}
          color="#94A3B8"
          lineWidth={1}
          dashed
          dashSize={0.14}
          gapSize={0.1}
          transparent
          opacity={0.75}
        />
      ))}

      {/* Camera origin & pinhole frustum pyramid */}
      <group ref={frustumOriginRef}>
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[0.04, 16, 16]} />
          <meshStandardMaterial color="#111827" roughness={0.4} />
        </mesh>
        {frustumCorners.map((c, idx) => (
          <Line
            key={`fr-${idx}`}
            points={[[0, 0, 0], c]}
            color="#64748B"
            lineWidth={1}
            transparent
            opacity={0.65}
          />
        ))}
      </group>

      {/* Focal plane frame */}
      <Line
        points={[...frustumCorners, frustumCorners[0]]}
        color="#1D4ED8"
        lineWidth={1.5}
        transparent
        opacity={0.75}
      />

      {/* Metric ruler & occlusion ray showcase (Stage 3) */}
      <group ref={rulerGroupRef}>
        <Line points={[boxA.pos, boxB.pos]} color="#BE123C" lineWidth={2.5} />
        <mesh position={boxA.pos}>
          <sphereGeometry args={[0.032, 12, 12]} />
          <meshBasicMaterial color="#BE123C" />
        </mesh>
        <mesh position={boxB.pos}>
          <sphereGeometry args={[0.032, 12, 12]} />
          <meshBasicMaterial color="#BE123C" />
        </mesh>
        <Line
          points={[[0, 0, 0], boxB.pos, boxFar.pos]}
          color="#B45309"
          lineWidth={1.5}
          dashed
          dashSize={0.08}
          gapSize={0.05}
        />
      </group>
    </group>
  );
};

const FocalPlaneTexture: React.FC<{
  imageBase64?: string;
}> = ({ imageBase64 }) => {
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
    img.src = normalizeBase64Image(imageBase64, 'image/png');
    return () => {
      cancelled = true;
    };
  }, [imageBase64]);

  useEffect(() => {
    return () => {
      texture?.dispose();
    };
  }, [texture]);

  if (!texture) return null;

  const focalDepth = 0.88;
  const halfW = focalDepth * Math.tan((60 * Math.PI) / 360);
  const halfH = halfW * (640 / 960);

  return (
    <mesh position={[0, 0, -focalDepth]} renderOrder={-1}>
      <planeGeometry args={[halfW * 2, halfH * 2]} />
      <meshBasicMaterial
        map={texture}
        transparent
        opacity={0.46}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </mesh>
  );
};

export const LandingShowcase: React.FC<LandingShowcaseProps> = ({
  benchmarks,
  onOpenLab,
  onNavigate,
}) => {
  const featured = benchmarks[0];
  const storyRef = useRef<HTMLDivElement>(null);
  const stageBoxRef = useRef<HTMLDivElement>(null);
  const chapterRefs = useRef<Array<HTMLElement | null>>([null, null, null, null]);
  const scrollProgressRef = useRef<number>(0.12);

  const [activeChapter, setActiveChapter] = useState<number>(0);
  const [isStageVisible, setIsStageVisible] = useState<boolean>(true);
  const [reducedMotion, setReducedMotion] = useState<boolean>(false);

  const demoBoxes = useMemo(
    () => toDemoBoxes(featured?.gt_objects),
    [featured?.gt_objects]
  );

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  // Off-screen WebGL pausing: contentvisibilityautostatechange + IntersectionObserver
  useEffect(() => {
    const el = stageBoxRef.current;
    if (!el) return;

    const onCv = (event: Event) => {
      const skipped = (event as Event & { skipped?: boolean }).skipped;
      if (typeof skipped === 'boolean') {
        setIsStageVisible(!skipped);
      }
    };
    el.addEventListener('contentvisibilityautostatechange', onCv);

    let io: IntersectionObserver | null = null;
    if (typeof IntersectionObserver !== 'undefined') {
      io = new IntersectionObserver(
        (entries) => {
          if (entries[0]) setIsStageVisible(entries[0].isIntersecting);
        },
        { rootMargin: '150px' }
      );
      io.observe(el);
    }

    return () => {
      el.removeEventListener('contentvisibilityautostatechange', onCv);
      io?.disconnect();
    };
  }, []);

  // Passive rAF-throttled scroll tracker driving the pinned 3D scene
  useEffect(() => {
    let rafId = 0;

    const syncScroll = () => {
      rafId = 0;
      const container = storyRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      const total = Math.max(rect.height - window.innerHeight * 0.72, 1);
      const raw = (-rect.top + window.innerHeight * 0.16) / total;
      const clamped = Math.min(Math.max(raw, 0), 1);
      scrollProgressRef.current = clamped;

      const idx = Math.min(3, Math.floor(clamped * 3.99));
      setActiveChapter((prev) => (prev === idx ? prev : idx));
    };

    const onScroll = () => {
      if (!rafId) {
        rafId = requestAnimationFrame(syncScroll);
      }
    };

    syncScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      if (rafId) cancelAnimationFrame(rafId);
    };
  }, []);

  const jumpToChapter = (idx: number) => {
    const targetProgress = (idx + 0.35) / 4;
    scrollProgressRef.current = targetProgress;
    setActiveChapter(idx);
    chapterRefs.current[idx]?.scrollIntoView({
      behavior: reducedMotion ? 'auto' : 'smooth',
      block: 'center',
    });
  };

  const handleBenchmarkOpen = (
    sceneId: string,
    event: React.MouseEvent<HTMLButtonElement>
  ) => {
    const card = event.currentTarget.closest('.benchmark-card');
    const thumb = card?.querySelector('.benchmark-thumb');
    thumb?.classList.add('scene-transition-target');
    onOpenLab(sceneId);
  };

  return (
    <div className="bg-[#F7F7F5] text-[#111827]">
      {/* Hero section with CSS scroll-driven parallax layers */}
      <section className="parallax-hero-scope mx-auto max-w-[1360px] px-4 pb-12 pt-10 sm:px-6 lg:pt-14">
        <div className="grid items-center gap-10 lg:grid-cols-12">
          <div className="lg:col-span-6 space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-[#E2E5EA] bg-[#FFFFFF] px-3 py-1 text-xs text-[#4B5563]">
              <span className="h-2 w-2 rounded-full bg-[#0F766E]" />
              <span>Visión Computacional 3D · Google AI Studio + Three.js</span>
            </div>

            <h1 className="font-serif-editorial text-4xl leading-[1.05] tracking-tight text-[#111827] sm:text-5xl lg:text-[3.75rem]">
              Convierte cualquier foto 2D en un espacio 3D medible en metros.
            </h1>

            <p className="max-w-[60ch] text-base leading-relaxed text-[#4B5563]">
              <strong>MyLeenTo3</strong> conecta la visión multimodal de Gemini con geometría
              óptica exacta en Python: reconstruye objetos en 3D, mide distancias reales sin
              alucinaciones y audita debilidades ocultas incluso en fotos sin sensores LiDAR.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button
                type="button"
                onClick={() => onOpenLab(featured?.id)}
                className="inline-flex items-center gap-2 rounded-lg bg-[#1D4ED8] px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#1E40AF] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#1D4ED8]"
              >
                <span>Abrir Laboratorio 3D</span>
                <ArrowRight className="h-4 w-4" />
              </button>
              <a
                href="#scrolly-3d-story"
                className="inline-flex items-center gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-4 py-2.5 text-sm font-medium text-[#111827] transition-colors hover:bg-[#F0F2F5]"
              >
                <span>Recorrido 3D Interactivo</span>
                <ArrowDown className="h-4 w-4 text-[#4B5563]" />
              </a>
              <button
                type="button"
                onClick={() => onNavigate('multiframe')}
                className="inline-flex items-center gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-4 py-2.5 text-sm font-medium text-[#111827] transition-colors hover:bg-[#F0F2F5]"
              >
                <Film className="h-4 w-4 text-[#4B5563]" />
                <span>Multi-Frame</span>
              </button>
            </div>

            <dl className="grid max-w-lg grid-cols-3 gap-4 border-t border-[#E2E5EA] pt-5">
              <div>
                <dt className="text-[11px] text-[#4B5563]">Precisión 3D</dt>
                <dd className="font-mono-tech mt-0.5 text-sm font-semibold text-[#1D4ED8]">
                  69.9% mIoU
                </dd>
              </div>
              <div>
                <dt className="text-[11px] text-[#4B5563]">Error Traslación</dt>
                <dd className="font-mono-tech mt-0.5 text-sm font-semibold text-[#0F766E]">
                  4.6 cm (ATE)
                </dd>
              </div>
              <div>
                <dt className="text-[11px] text-[#4B5563]">Cálculo Óptico</dt>
                <dd className="font-mono-tech mt-0.5 text-sm font-semibold text-[#111827]">
                  &lt; 10 ms / FOV
                </dd>
              </div>
            </dl>
          </div>

          {/* Right Hero Visual with staggered parallax metadata */}
          <div className="relative lg:col-span-6">
            <div
              className="parallax-layer overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#FFFFFF]"
              style={{ '--parallax-y': '-16px' } as React.CSSProperties}
            >
              <div className="flex items-center justify-between border-b border-[#E2E5EA] bg-[#F7F7F5] px-4 py-2.5">
                <div className="flex items-center gap-2 text-xs font-medium text-[#111827]">
                  <ScanEye className="h-4 w-4 text-[#1D4ED8]" />
                  <span>Banco Óptico Calibrado — Referencia 9-DOF</span>
                </div>
                <span className="font-mono-tech text-[11px] text-[#4B5563]">
                  FOV_h 60.0° · 960×640
                </span>
              </div>

              <div className="relative bg-[#F0F2F5] p-4">
                {featured?.image_base64 ? (
                  <div className="relative overflow-hidden rounded-lg border border-[#E2E5EA] bg-[#FFFFFF]">
                    <img
                      src={normalizeBase64Image(featured.image_base64)}
                      alt={featured.title}
                      className="block h-auto w-full object-cover"
                    />
                    <svg
                      viewBox="0 0 960 640"
                      className="pointer-events-none absolute inset-0 h-full w-full"
                      aria-hidden="true"
                    >
                      <line x1="480" y1="0" x2="480" y2="640" stroke="#1D4ED8" strokeOpacity="0.24" strokeDasharray="6 6" />
                      <line x1="0" y1="320" x2="960" y2="320" stroke="#1D4ED8" strokeOpacity="0.24" strokeDasharray="6 6" />
                      <circle cx="480" cy="320" r="14" fill="none" stroke="#1D4ED8" strokeWidth="1.5" strokeOpacity="0.65" />
                      <rect x="20" y="20" width="248" height="32" rx="6" fill="#FFFFFF" fillOpacity="0.94" stroke="#E2E5EA" />
                      <text x="34" y="40" fontFamily="JetBrains Mono, monospace" fontSize="13" fill="#111827">
                        K: f=831.4px · det(M) = +1.0
                      </text>
                    </svg>
                  </div>
                ) : (
                  <div className="flex aspect-[3/2] items-center justify-center rounded-lg border border-dashed border-[#CBD5E1] bg-[#FFFFFF] p-6 text-center">
                    <p className="font-mono-tech text-xs text-[#4B5563]">
                      Cargando escena calibrada desde /api/benchmarks…
                    </p>
                  </div>
                )}

                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[#4B5563]">
                  <span className="font-mono-tech">
                    Gemini Frame (X-Der, Y-Prof, Z-Arr) → Three.js (Y-Arr, −Z-Prof)
                  </span>
                  <button
                    type="button"
                    onClick={() => onOpenLab(featured?.id)}
                    className="inline-flex items-center gap-1 font-medium text-[#1D4ED8] hover:underline"
                  >
                    <span>Probar en Vivo</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Staggered floating parallax telemetry card (matte, zero neon) */}
            <div
              className="parallax-layer pointer-events-none mt-3 hidden rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-3.5 py-2.5 sm:inline-flex items-center gap-4 lg:absolute lg:-bottom-4 lg:right-4"
              style={{ '--parallax-y': '-34px' } as React.CSSProperties}
            >
              <div>
                <div className="text-[10px] font-medium uppercase tracking-wide text-[#4B5563]">
                  Motor Determinístico
                </div>
                <div className="font-mono-tech text-xs font-semibold text-[#111827]">
                  SAT 15-ejes · Kay-Kajiya · Sobel PECS
                </div>
              </div>
              <span className="h-6 w-px bg-[#E2E5EA]" />
              <div className="font-mono-tech text-xs font-semibold text-[#0F766E]">
                0% alucinación
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Scroll-Driven 3D Scrollytelling Section (3d-scroll-web + scrollytelling) */}
      <section
        id="scrolly-3d-story"
        ref={storyRef}
        className="scrolly-narrative-scope border-t border-[#E2E5EA] bg-[#FFFFFF] py-14"
      >
        <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
          <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[#E2E5EA] pb-6">
            <div className="max-w-2xl">
              <span className="font-mono-tech text-[11px] font-semibold uppercase tracking-wider text-[#1D4ED8]">
                Narrativa 3D Sincronizada al Scroll
              </span>
              <h2 className="font-serif-editorial mt-1 text-3xl text-[#111827] sm:text-4xl">
                Cómo pasa una imagen plana a geometría 3D auditable
              </h2>
            </div>

            {/* Direct chapter selector pills */}
            <div className="flex flex-wrap items-center gap-1.5" role="tablist" aria-label="Etapas del pipeline 3D">
              {CHAPTERS.map((ch, idx) => {
                const isAct = activeChapter === idx;
                return (
                  <button
                    key={ch.step}
                    type="button"
                    role="tab"
                    aria-selected={isAct}
                    onClick={() => jumpToChapter(idx)}
                    className={`rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors ${
                      isAct
                        ? 'border-[#111827] bg-[#111827] text-white'
                        : 'border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563] hover:text-[#111827]'
                    }`}
                  >
                    <span className="font-mono-tech mr-1 opacity-75">{ch.step}</span>
                    {ch.shortLabel}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="mt-8 grid items-start gap-8 lg:grid-cols-12">
            {/* Left Column: 4 Named View-Timeline Chapters */}
            <div className="space-y-6 lg:col-span-5">
              {CHAPTERS.map((ch, idx) => {
                const isAct = activeChapter === idx;
                const chapterClass = `scrolly-chapter-${idx + 1}`;
                return (
                  <article
                    key={ch.step}
                    ref={(el) => {
                      chapterRefs.current[idx] = el;
                    }}
                    className={`${chapterClass} min-h-[210px] py-2`}
                  >
                    <div
                      onClick={() => jumpToChapter(idx)}
                      className={`scrolly-card-inner cursor-pointer rounded-xl border p-5 transition-colors ${
                        isAct
                          ? 'border-[#1D4ED8] bg-[#FFFFFF]'
                          : 'border-[#E2E5EA] bg-[#F7F7F5] hover:border-[#CBD5E1]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className="font-mono-tech rounded px-2 py-0.5 text-[11px] font-semibold"
                          style={{
                            color: ch.accent,
                            backgroundColor: `${ch.accent}12`,
                          }}
                        >
                          ETAPA {ch.step}
                        </span>
                        <span className="font-mono-tech text-[11px] text-[#4B5563]">
                          {ch.metricTag}
                        </span>
                      </div>
                      <h3 className="mt-3 text-base font-semibold text-[#111827]">
                        {ch.title}
                      </h3>
                      <p className="mt-2 text-xs leading-relaxed text-[#4B5563]">
                        {ch.summary}
                      </p>
                    </div>
                  </article>
                );
              })}
            </div>

            {/* Right Column: Sticky Live Three.js WebGL Stage (touch-action: pan-y, zero scroll hijack) */}
            <div className="lg:sticky lg:top-20 lg:col-span-7">
              <div className="overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#FFFFFF]">
                <div className="flex items-center justify-between border-b border-[#E2E5EA] bg-[#F7F7F5] px-4 py-2.5">
                  <div className="flex items-center gap-2 text-xs font-medium text-[#111827]">
                    <Compass className="h-3.5 w-3.5 text-[#1D4ED8]" />
                    <span>{CHAPTERS[activeChapter]?.title}</span>
                  </div>
                  <span className="font-mono-tech text-[11px] text-[#4B5563]">
                    {isStageVisible ? 'WebGL Activo' : 'Pausado (Off-screen)'}
                  </span>
                </div>

                <div
                  ref={stageBoxRef}
                  className="webgl-stage-container relative h-[360px] sm:h-[440px] w-full bg-[#F0F2F5]"
                  style={{ touchAction: 'pan-y' }}
                >
                  <Canvas
                    dpr={[1, 2]}
                    shadows="percentage"
                    frameloop={isStageVisible ? 'always' : 'demand'}
                    camera={{ position: [0.3, 0.25, 0.3], fov: 42, near: 0.05, far: 60 }}
                    gl={{ antialias: true, powerPreference: 'high-performance' }}
                  >
                    <color attach="background" args={['#F0F2F5']} />
                    <ambientLight intensity={0.85} />
                    <directionalLight position={[4, 7, 4]} intensity={0.9} />

                    <ScrollRigAndOverlays
                      boxes={demoBoxes}
                      scrollProgressRef={scrollProgressRef}
                      reducedMotion={reducedMotion}
                    />
                    <FocalPlaneTexture imageBase64={featured?.image_base64} />
                    {demoBoxes.map((b) => (
                      <ScrollDrivenCuboid
                        key={b.id}
                        box={b}
                        scrollProgressRef={scrollProgressRef}
                        reducedMotion={reducedMotion}
                      />
                    ))}
                  </Canvas>

                  {/* Quiet bottom caption inside 3D stage */}
                  <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF]/92 px-3 py-1.5 text-[11px] text-[#4B5563]">
                    <span className="font-mono-tech text-[#111827]">
                      Etapa {CHAPTERS[activeChapter]?.step}/04 · Desplázate para animar cámara 3D
                    </span>
                    <div className="flex items-center gap-3">
                      <span className="inline-flex items-center gap-1">
                        <span className="h-2 w-2 rounded-xs bg-[#1D4ED8]" /> Predicción
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <span className="h-2 w-2 rounded-xs bg-[#0F766E]" /> GT Esmeralda
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <span className="h-2 w-2 rounded-xs bg-[#BE123C]" /> Regla 3D
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Horizontal Capabilities Strip with Registered @property Edge Mask Fade */}
      <section className="reveal-on-scroll border-t border-[#E2E5EA] bg-[#F7F7F5] py-12">
        <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="font-serif-editorial text-2xl text-[#111827] sm:text-3xl">
                Métricas universales para todo tipo de imagen (sin Ground Truth)
              </h2>
              <p className="mt-1 text-xs text-[#4B5563]">
                Evalúa fotos de celular, renders artísticos o cuadros de videojuegos buscando variables latentes poco medibles.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('metrics')}
              className="inline-flex items-center gap-1 text-xs font-medium text-[#1D4ED8] hover:underline"
            >
              <span>Abrir panel de diagnóstico</span>
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="scroll-affordance-fade-x mt-6 flex gap-4 pb-2">
            {LATENT_CAPABILITIES.map((cap) => (
              <div
                key={cap.badge}
                className="w-[240px] shrink-0 rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-4"
              >
                <span
                  className="font-mono-tech inline-block rounded px-2 py-0.5 text-[10px] font-semibold"
                  style={{ color: cap.color, backgroundColor: `${cap.color}12` }}
                >
                  {cap.badge}
                </span>
                <h3 className="mt-2.5 text-xs font-semibold text-[#111827]">{cap.title}</h3>
                <p className="mt-1.5 text-[11px] leading-relaxed text-[#4B5563]">{cap.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Editorial Section: Mathematical Rigor Pillars */}
      <section className="reveal-on-scroll border-t border-[#E2E5EA] bg-[#FFFFFF] py-14">
        <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
          <div className="grid gap-6 md:grid-cols-3">
            <article className="scroll-card-lifecycle rounded-xl border border-[#E2E5EA] bg-[#F7F7F5] p-6">
              <div className="flex items-center gap-2 text-[#1D4ED8]">
                <Layers className="h-4 w-4" />
                <h3 className="text-sm font-semibold text-[#111827]">
                  Proyección Pinhole K
                </h3>
              </div>
              <p className="mt-2.5 text-xs leading-relaxed text-[#4B5563]">
                Ajusta el campo de visión{' '}
                <span className="font-mono-tech text-[#111827]">FOV_h ∈ [40°, 110°]</span> en vivo
                en menos de 10 ms sin gastar tokens extra de Google AI Studio.
              </p>
            </article>

            <article className="scroll-card-lifecycle rounded-xl border border-[#E2E5EA] bg-[#F7F7F5] p-6">
              <div className="flex items-center gap-2 text-[#0F766E]">
                <Eye className="h-4 w-4" />
                <h3 className="text-sm font-semibold text-[#111827]">
                  Saliencia Adaptativa Offline
                </h3>
              </div>
              <p className="mt-2.5 text-xs leading-relaxed text-[#4B5563]">
                Si cargas una foto sin API Key, el extractor local combina gradientes Sobel,
                contraste cromático y retroproyección K sobre los píxeles reales.
              </p>
            </article>

            <article className="scroll-card-lifecycle rounded-xl border border-[#E2E5EA] bg-[#F7F7F5] p-6">
              <div className="flex items-center gap-2 text-[#B45309]">
                <Ruler className="h-4 w-4" />
                <h3 className="text-sm font-semibold text-[#111827]">
                  Copiloto con Function Calling
                </h3>
              </div>
              <p className="mt-2.5 text-xs leading-relaxed text-[#4B5563]">
                6 herramientas auditables controlan la cámara 3D, trazan reglas entre objetos y
                exponen qué bloquea a qué en tiempo real.
              </p>
            </article>
          </div>
        </div>
      </section>

      {/* Benchmark Launch Cards with Scroll Lifecycle Entry/Exit & View Transitions */}
      <section className="border-t border-[#E2E5EA] bg-[#F7F7F5] py-16">
        <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="font-serif-editorial text-3xl text-[#111827] sm:text-4xl">
                Tres bancos calibrados reproducibles (SEED = 42)
              </h2>
              <p className="mt-2 text-sm text-[#4B5563]">
                Elige un escenario para abrirlo con transición espacial continua en el Laboratorio 3D.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('metrics')}
              className="inline-flex items-center gap-1 text-xs font-medium text-[#1D4ED8] hover:underline"
            >
              <ShieldAlert className="h-3.5 w-3.5" />
              <span>Ver comparativa multi-modelo →</span>
            </button>
          </div>

          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {benchmarks.map((scene) => (
              <div
                key={scene.id}
                className="benchmark-card scroll-card-lifecycle group flex flex-col overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] transition-colors hover:border-[#1D4ED8]/50"
              >
                <div className="benchmark-thumb aspect-[3/2] overflow-hidden border-b border-[#E2E5EA] bg-[#F0F2F5]">
                  <img
                    src={normalizeBase64Image(scene.image_base64)}
                    alt={scene.title}
                    className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                  />
                </div>
                <div className="flex flex-1 flex-col p-5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono-tech text-[11px] text-[#1D4ED8]">
                      {scene.id}
                    </span>
                    <span className="font-mono-tech text-[11px] text-[#4B5563]">
                      {scene.gt_objects.length} OBBs GT
                    </span>
                  </div>
                  <h3 className="mt-1.5 text-base font-semibold text-[#111827]">
                    {scene.title}
                  </h3>
                  <p className="mt-1.5 flex-1 text-xs leading-relaxed text-[#4B5563]">
                    {scene.description}
                  </p>
                  <button
                    type="button"
                    onClick={(e) => handleBenchmarkOpen(scene.id, e)}
                    className="mt-4 inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] px-3 py-2 text-xs font-medium text-[#111827] transition-colors hover:border-[#1D4ED8] hover:bg-[#1D4ED8] hover:text-white"
                  >
                    <span>Cargar en Laboratorio 3D</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};
