import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Edges, Line } from '@react-three/drei';
import * as THREE from 'three';
import {
  ArrowDown,
  ArrowRight,
  Compass,
  Film,
  Orbit,
} from 'lucide-react';
import type { AppSection, BenchmarkScene, SpatialObject } from '../types/spatial';
import { normalizeBase64Image } from '../utils/geometry';
import { LandingParticleField } from './LandingParticleField';

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
      'Toda imagen RGB nace proyectada sobre un plano focal mediante la matriz intrínseca K(FOV_h) extraída por EXIF 35mm o MoGe-2. Aquí el observador mira exactamente desde el lente virtual hacia la escena.',
    metricTag: 'K: f = W / 2·tan(FOV_h/2)',
    accent: '#1D4ED8',
  },
  {
    step: '02',
    shortLabel: 'Extrusión 3D',
    title: 'Poses 9-DOF y Base Quiral det(M)=+1',
    summary:
      'Sobre profundidad métrica local o inferencia Google AI Studio se estima centro (x,y,z), dimensiones reales y rotación Euler. Python aplica M = Rₓ(−90°) para proyectar en Three.js sin espejar ejes.',
    metricTag: 'det(M) = +1.0 — sin inversión',
    accent: '#1D4ED8',
  },
  {
    step: '03',
    shortLabel: 'Regla y Rayos',
    title: 'Medición Exacta y Oclusiones Kay-Kajiya',
    summary:
      'En lugar de estimaciones visuales al azar, Python calcula distancias euclidianas, holguras SAT de 15 ejes y trazado de rayos desde la cámara para medir qué volumen obstruye al del fondo.',
    metricTag: 'Holgura métrica — trazado SAT',
    accent: '#BE123C',
  },
  {
    step: '04',
    shortLabel: 'Planta BEV',
    title: 'Vista Superior (BEV) y Diagnóstico Universal',
    summary:
      'Desde planta superior se contrasta la nube de puntos 3D, el calce frente al Ground Truth esmeralda y las anomalías sin referencia (bordes Sobel, levitación gravitatoria y deriva multi-frame).',
    metricTag: 'mIoU BEV — Plausibilidad Física',
    accent: '#0F766E',
  },
];

const LATENT_CAPABILITIES = [
  {
    code: 'PECS / SOBEL',
    title: 'Coherencia de Bordes 2D-3D',
    desc: 'Comprueba si las 12 aristas proyectadas caen sobre gradientes de contorno reales dentro de la fotografía.',
    color: '#0F766E',
  },
  {
    code: 'PVI / SAT-15',
    title: 'Interpenetración de Sólidos',
    desc: 'Evalúa intersección exacta de poliedros orientados mediante el Teorema de Ejes Separadores en 15 ejes.',
    color: '#BE123C',
  },
  {
    code: 'GFI / PLANO Z',
    title: 'Anomalía de Soporte en Piso',
    desc: 'Cuantifica en metros si una detección levita sobre el plano de apoyo latente o se hunde bajo la superficie.',
    color: '#B45309',
  },
  {
    code: 'SDAS / ESCALA',
    title: 'Acoplamiento Tamaño-Distancia',
    desc: 'Expone cuando el modelo infla artificialmente las dimensiones físicas de objetos lejanos para compensar píxeles.',
    color: '#1D4ED8',
  },
  {
    code: 'SIGMA / RANGO',
    title: 'Incertidumbre Cuadrática en Profundidad',
    desc: 'Modela el crecimiento cuadrático del margen de error proporcional a y² agravado por baja luz u oclusión.',
    color: '#111827',
  },
  {
    code: 'WORLD / DERIVA',
    title: 'Auditoría Temporal Multi-Cuadro',
    desc: 'Mide deriva porcentual de volumen y oscilación angular cuadro a cuadro en secuencias de video sintéticas o reales.',
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
            color="#3B82F6"
            transparent
            opacity={0.18}
            roughness={0.5}
            depthWrite={false}
          />
          <Edges color="#60A5FA" threshold={15} lineWidth={1.7} />
        </mesh>
      </group>

      <group ref={gtRef} visible={false}>
        <mesh>
          <boxGeometry args={box.size} />
          <meshBasicMaterial transparent opacity={0} depthWrite={false} />
          <Edges color="#14B8A6" threshold={15} lineWidth={1.6} />
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
      const t = p / 0.25;
      _camTargetPos.set(
        THREE.MathUtils.lerp(0.12, 1.35, t),
        THREE.MathUtils.lerp(0.06, 0.95, t),
        THREE.MathUtils.lerp(0.18, 0.85, t)
      );
      _lookAtTarget.set(0, -0.15, -2.35);
    } else if (p < 0.52) {
      const t = (p - 0.25) / 0.27;
      _camTargetPos.set(
        THREE.MathUtils.lerp(1.35, 2.55, t),
        THREE.MathUtils.lerp(0.95, 1.68, t),
        THREE.MathUtils.lerp(0.85, 0.65, t)
      );
      _lookAtTarget.set(0, -0.18, -2.45);
    } else if (p < 0.76) {
      const t = (p - 0.52) / 0.24;
      _camTargetPos.set(
        THREE.MathUtils.lerp(2.55, -2.25, t),
        THREE.MathUtils.lerp(1.68, 1.85, t),
        THREE.MathUtils.lerp(0.65, 0.45, t)
      );
      _lookAtTarget.set(0, -0.18, -2.45);
    } else {
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

    if (frustumOriginRef.current) {
      frustumOriginRef.current.visible = state.camera.position.length() > 0.42;
    }

    const showRuler = p >= 0.45 || reducedMotion;
    const rScale = THREE.MathUtils.damp(_rulerOpacity.value, showRuler ? 1 : 0.001, 7.0, dt);
    _rulerOpacity.value = rScale;
    if (rulerGroupRef.current) {
      rulerGroupRef.current.visible = rScale > 0.04;
      rulerGroupRef.current.scale.setScalar(rScale);
    }

    const targetU = p < 0.28 ? 0.22 + (p / 0.28) * 0.78 : 1.0;
    _unfoldState.value = THREE.MathUtils.damp(_unfoldState.value, targetU, 6.5, dt);
    _gtOpacity.value = p >= 0.66 ? 1 : 0;
  });

  return (
    <group>
      <gridHelper args={[14, 14, '#3B5278', '#243654']} position={[0, -0.52, -3.5]} />
      {ringPoints.map(({ r, pts }) => (
        <Line
          key={`ring-${r}`}
          points={pts}
          color="#475569"
          lineWidth={1}
          dashed
          dashSize={0.14}
          gapSize={0.1}
          transparent
          opacity={0.75}
        />
      ))}

      <group ref={frustumOriginRef}>
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[0.04, 16, 16]} />
          <meshStandardMaterial color="#F8FAFC" roughness={0.4} />
        </mesh>
        {frustumCorners.map((c, idx) => (
          <Line
            key={`fr-${idx}`}
            points={[[0, 0, 0], c]}
            color="#94A3B8"
            lineWidth={1}
            transparent
            opacity={0.65}
          />
        ))}
      </group>

      <Line
        points={[...frustumCorners, frustumCorners[0]]}
        color="#60A5FA"
        lineWidth={1.5}
        transparent
        opacity={0.85}
      />

      <group ref={rulerGroupRef}>
        <Line points={[boxA.pos, boxB.pos]} color="#F43F5E" lineWidth={2.5} />
        <mesh position={boxA.pos}>
          <sphereGeometry args={[0.032, 12, 12]} />
          <meshBasicMaterial color="#F43F5E" />
        </mesh>
        <mesh position={boxB.pos}>
          <sphereGeometry args={[0.032, 12, 12]} />
          <meshBasicMaterial color="#F43F5E" />
        </mesh>
        <Line
          points={[[0, 0, 0], boxB.pos, boxFar.pos]}
          color="#F59E0B"
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
  const rootRef = useRef<HTMLDivElement>(null);
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

  // Universal bidirectional rAF fallback when CSS Scroll-Driven view() timelines are unsupported
  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const cssSupported =
      typeof CSS !== 'undefined' &&
      typeof CSS.supports === 'function' &&
      CSS.supports('(animation-timeline: view()) and (animation-range: entry)');
    if (reducedMotion || cssSupported) return;

    const els = Array.from(root.querySelectorAll<HTMLElement>('[data-scroll-anim]'));
    if (els.length === 0) return;

    let rafId = 0;
    const clamp01 = (v: number) => Math.min(Math.max(v, 0), 1);

    const update = () => {
      rafId = 0;
      const vh = window.innerHeight || 900;
      for (const el of els) {
        const kind = el.dataset.scrollAnim;
        const rect = el.getBoundingClientRect();
        const entry = clamp01((vh - rect.top) / (vh * 0.46));
        const exit = clamp01((-rect.top - rect.height * 0.1) / Math.max(rect.height * 0.9, 1));

        if (kind === 'enter-left' || kind === 'enter-right') {
          const sign = kind === 'enter-left' ? -1 : 1;
          const xIn = sign * 110 * (1 - entry);
          const yIn = 34 * (1 - entry);
          const sIn = 0.93 + 0.07 * entry;
          const xOut = sign * 46 * exit;
          const yOut = -22 * exit;
          const sOut = 1 - 0.04 * exit;
          el.style.opacity = String(entry * (1 - 0.86 * exit));
          el.style.transform = `translate3d(${(xIn + xOut).toFixed(1)}px, ${(yIn + yOut).toFixed(1)}px, 0) scale(${(sIn * sOut).toFixed(4)})`;
        } else if (kind === 'curtain-up') {
          el.style.opacity = String(entry);
          el.style.transform = `translate3d(0, ${(42 * (1 - entry)).toFixed(1)}px, 0)`;
        } else if (kind === 'rule-draw' || kind === 'bar-grow') {
          el.style.transform = `scaleX(${entry.toFixed(4)})`;
        }
      }
    };

    const onScroll = () => {
      if (!rafId) rafId = requestAnimationFrame(update);
    };

    update();
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll, { passive: true });
    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      for (const el of els) {
        el.style.opacity = '';
        el.style.transform = '';
      }
    };
  }, [reducedMotion, benchmarks.length]);

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
    <div ref={rootRef} className="relative bg-transparent text-[#111827]">
      {/* Fixed 3D Particle Depth Field reactive across entire page scroll */}
      <LandingParticleField reducedMotion={reducedMotion} />

      <div className="relative z-10">
        {/* 1. Architectural Editorial Hero */}
        <section className="parallax-hero-scope bg-[radial-gradient(ellipse_at_84%_14%,rgba(29,78,216,0.09),transparent_52%),radial-gradient(ellipse_at_12%_86%,rgba(15,118,110,0.08),transparent_48%),linear-gradient(180deg,rgba(250,248,244,0.88)_0%,rgba(242,245,250,0.82)_100%)] pb-16 pt-6">
          <div className="mx-auto max-w-[1360px] px-4 sm:px-6 lg:pt-4">
            <div className="h-[2px] w-full bg-gradient-to-r from-[#1D4ED8] via-[#0F766E] to-[#B45309]" />
            {/* Unboxed Architectural Masthead Hairline */}
            <div className="mt-3 border-b border-[#15284B]/18 pb-3">
              <div className="font-mono-tech flex flex-wrap items-center justify-between gap-2 text-[11px] uppercase tracking-[0.14em] text-[#4B5563]">
                <span className="text-[#15284B] font-semibold">C8286 Edge 2D → MyLeenTo3 Óptica 3D</span>
                <span>MoGe-2 GPU · Google AI Studio · Sin LiDAR</span>
              </div>
            </div>

            <div className="mt-10 grid items-center gap-12 lg:grid-cols-12">
              <div className="lg:col-span-6 space-y-6">
                <h1 className="font-serif-editorial text-4xl font-medium leading-[1.06] text-[#111827] sm:text-5xl lg:text-[3.75rem]">
                  Convierte una foto 2D en{' '}
                  <span className="text-[#15284B] underline decoration-[#B45309]/55 decoration-2 underline-offset-4">
                    geometría 3D auditable
                  </span>{' '}
                  y sin LiDAR
                </h1>

                <p className="max-w-[56ch] text-base leading-relaxed text-[#4B5563]">
                  <strong className="font-semibold text-[#111827]">MyLeenTo3</strong> combina reconstrucción métrica densa en GPU (<span className="font-mono-tech text-xs text-[#15284B]">EXIF + MoGe-2 ViT-S/B/L + YOLOE</span>), reconstrucción multi-vista 360° y una compuerta determinista en Python que bloquea colisiones y levitaciones físicas antes de decidir.
                </p>

                <div className="flex flex-wrap items-center gap-3 pt-1">
                  <button
                    type="button"
                    onClick={() => onOpenLab(featured?.id)}
                    className="inline-flex items-center gap-2 rounded-sm bg-[#15284B] px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-[#1D4ED8] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#1D4ED8]"
                  >
                    <span>Abrir Laboratorio 3D</span>
                    <ArrowRight className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => onNavigate('multiview')}
                    className="inline-flex items-center gap-2 rounded-sm border border-[#0F766E]/40 bg-[#F4FBF9]/90 px-4 py-2.5 text-sm font-medium text-[#0F766E] transition-colors hover:border-[#0F766E] hover:bg-[#FFFFFF]"
                  >
                    <Orbit className="h-4 w-4" />
                    <span>Reconstrucción 360°</span>
                  </button>
                  <a
                    href="#problematica-scroll"
                    className="inline-flex items-center gap-2 rounded-sm border border-[#111827]/20 bg-[#FFFFFF]/80 px-4 py-2.5 text-sm font-medium text-[#111827] transition-colors hover:border-[#111827]"
                  >
                    <span>Explorar Problemática</span>
                    <ArrowDown className="h-4 w-4 text-[#4B5563]" />
                  </a>
                  <button
                    type="button"
                    onClick={() => onNavigate('multiframe')}
                    className="inline-flex items-center gap-2 rounded-sm border border-[#111827]/20 bg-[#FFFFFF]/80 px-4 py-2.5 text-sm font-medium text-[#111827] transition-colors hover:border-[#111827]"
                  >
                    <Film className="h-4 w-4 text-[#4B5563]" />
                    <span>Multi-Frame</span>
                  </button>
                </div>

                {/* Unboxed Typographic Metrics Ledger */}
                <dl className="grid max-w-lg grid-cols-3 border-t border-[#15284B]/20 pt-5">
                  <div className="pr-4">
                    <dt className="font-mono-tech text-[10px] uppercase tracking-wider text-[#4B5563]">
                      Precisión 3D
                    </dt>
                    <dd className="font-mono-tech mt-1 text-lg font-semibold text-[#1D4ED8]">
                      69.9% mIoU
                    </dd>
                  </div>
                  <div className="border-l border-[#15284B]/15 px-4">
                    <dt className="font-mono-tech text-[10px] uppercase tracking-wider text-[#4B5563]">
                      Error Traslación
                    </dt>
                    <dd className="font-mono-tech mt-1 text-lg font-semibold text-[#0F766E]">
                      4.6 cm ATE
                    </dd>
                  </div>
                  <div className="border-l border-[#15284B]/15 pl-4">
                    <dt className="font-mono-tech text-[10px] uppercase tracking-wider text-[#4B5563]">
                      Cálculo GPU / FOV
                    </dt>
                    <dd className="font-mono-tech mt-1 text-lg font-semibold text-[#B45309]">
                      365 ms
                    </dd>
                  </div>
                </dl>
              </div>

              {/* Right Architectural Specimen Plate with Parallax Shift */}
              <div className="relative lg:col-span-6">
                <div
                  className="parallax-layer relative border border-[#15284B]/22 bg-[#FFFFFF]/92 shadow-[0_14px_34px_-18px_rgba(21,40,75,0.22)] backdrop-blur-[2px]"
                  style={{ '--parallax-y': '-20px' } as React.CSSProperties}
                >
                  <div className="flex items-center justify-between bg-[#15284B] px-4 py-2.5 text-xs text-[#F8FAFC]">
                    <span className="font-mono-tech uppercase tracking-wider">
                      Lámina I — Banco Óptico Calibrado 9-DOF
                    </span>
                    <span className="font-mono-tech text-[11px] text-[#CBD5E1]">
                      FOV_h 60.0° · 960×640
                    </span>
                  </div>

                  <div className="p-4">
                    {featured?.image_base64 ? (
                      <div className="relative overflow-hidden border border-[#111827]/12 bg-[#FFFFFF]">
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
                          <line x1="480" y1="0" x2="480" y2="640" stroke="#1D4ED8" strokeOpacity="0.28" strokeDasharray="6 6" />
                          <line x1="0" y1="320" x2="960" y2="320" stroke="#1D4ED8" strokeOpacity="0.28" strokeDasharray="6 6" />
                          <circle cx="480" cy="320" r="14" fill="none" stroke="#1D4ED8" strokeWidth="1.5" strokeOpacity="0.7" />
                          <rect x="20" y="20" width="248" height="30" fill="#FFFFFF" fillOpacity="0.94" stroke="#15284B" strokeOpacity="0.28" />
                          <text x="34" y="39" fontFamily="JetBrains Mono, monospace" fontSize="12.5" fill="#15284B">
                            K: f=831.4px · det(M) = +1.0
                          </text>
                        </svg>
                      </div>
                    ) : (
                      <div className="flex aspect-[3/2] items-center justify-center border border-dashed border-[#111827]/20 bg-[#F7F7F5]/60 p-6 text-center">
                        <p className="font-mono-tech text-xs text-[#4B5563]">
                          Cargando escena calibrada desde /api/benchmarks…
                        </p>
                      </div>
                    )}

                    <div className="mt-3 flex flex-wrap items-center justify-between gap-3 pt-1 text-xs text-[#4B5563]">
                      <span className="font-mono-tech text-[11px]">
                        SAT 15-ejes · Kay-Kajiya · MoGe-2 ViT-S/B/L
                      </span>
                      <button
                        type="button"
                        onClick={() => onOpenLab(featured?.id)}
                        className="inline-flex items-center gap-1 font-medium text-[#1D4ED8] hover:underline"
                      >
                        <span>Inspeccionar en vivo</span>
                        <ArrowRight className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. Problemática Principal: Editorial Spreads (Zero Cards, Zero Pills, Zero Timeline Pole) */}
        <section
          id="problematica-scroll"
          aria-label="Problemática principal antes de usar modelos 3D"
          className="problem-scroll-section border-t border-[#15284B]/18 bg-gradient-to-b from-[#FAF7F2]/82 via-[#F4F7FA]/80 to-[#F3F9F7]/82 py-14 backdrop-blur-[1px] sm:py-18"
        >
          <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
            {/* Asymmetric Editorial Header */}
            <div className="grid items-end gap-6 pb-6 lg:grid-cols-12">
              <div data-scroll-anim="curtain-up" className="scroll-curtain-up lg:col-span-7">
                <div className="font-mono-tech text-[11px] uppercase tracking-[0.16em] text-[#1D4ED8]">
                  La Problemática — De C8286 2D a 3D Métrico
                </div>
                <h2 className="font-serif-editorial mt-2 text-3xl leading-[1.08] text-[#111827] sm:text-5xl">
                  Cuatro brechas que rompen la visión 2D y los modelos 3D sin control
                </h2>
              </div>
              <p data-scroll-anim="curtain-up" className="scroll-curtain-up scroll-stagger-2 max-w-[54ch] text-sm leading-relaxed text-[#4B5563] lg:col-span-5">
                Antes de abrir el Laboratorio 3D, observa por qué no basta con dibujar cajas o invocar a ciegas una IA sobre una foto común: cada lámina contrapone la falla física frente a su verificación geométrica exacta.
              </p>
            </div>
            <div data-scroll-anim="rule-draw" className="scroll-rule-draw h-[2px] w-full bg-gradient-to-r from-[#BE123C]/55 via-[#B45309]/55 to-[#0F766E]/55" />

            {/* SPREAD I: Text Left / Diagram Right (Crimson accent) */}
            <div className="grid items-center gap-10 py-14 lg:grid-cols-12">
              <article data-scroll-anim="enter-left" className="scroll-enter-left lg:col-span-6 space-y-4 border-l-2 border-[#BE123C]/65 pl-5 pr-0 lg:pr-8">
                <div className="flex items-baseline gap-4">
                  <span className="font-serif-editorial text-4xl font-semibold leading-none text-[#BE123C]/40">
                    I
                  </span>
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#BE123C]">
                    Visión 2D Plana — Herencia C8286
                  </span>
                </div>
                <h3 className="font-serif-editorial text-2xl leading-[1.12] text-[#111827] sm:text-3xl">
                  Ceguera de profundidad y colapso bajo oclusión parcial
                </h3>
                <p className="max-w-[54ch] text-sm leading-relaxed text-[#4B5563]">
                  En imágenes planas, un detector tradicional no distingue si un obstáculo está a <strong className="font-semibold text-[#111827]">1.4 m</strong> o a <strong className="font-semibold text-[#111827]">4.8 m</strong>. Cuando otro cuerpo lo cubre parcialmente, los descriptores clásicos HSV/LBP/HOG colapsan (<span className="font-mono-tech text-xs text-[#111827]">F1 &le; 0.104</span>) y hasta <span className="font-mono-tech text-xs text-[#111827]">MobileNetV2 INT8</span> desciende a <span className="font-mono-tech text-xs text-[#111827]">F1 = 0.715</span>.
                </p>
                <div className="font-mono-tech flex flex-wrap gap-x-6 gap-y-1 border-t border-[#BE123C]/15 pt-3 text-[11px] text-[#0F766E]">
                  <span>Ray-Tracing Kay-Kajiya: hasta 89% de oclusión medida</span>
                  <span className="text-[#4B5563]">Δy real = 1.8 m</span>
                </div>
              </article>

              <div data-scroll-anim="enter-right" className="scroll-enter-right lg:col-span-6 border border-[#BE123C]/22 bg-[#FFF9F8]/92 p-4">
                <svg viewBox="0 0 460 190" className="h-auto w-full" role="img" aria-label="Diagrama brecha I: oclusión 2D vs rayos 3D Kay-Kajiya">
                  <rect x="4" y="4" width="212" height="182" fill="#F7F7F5" stroke="#CBD5E1" />
                  <text x="16" y="24" fontFamily="JetBrains Mono, monospace" fontSize="10" fontWeight="700" fill="#4B5563">FOTO 2D SIN PROFUNDIDAD</text>
                  <rect x="34" y="46" width="118" height="106" fill="#FFFFFF" stroke="#9CA3AF" strokeDasharray="4 3" strokeWidth="1.5" />
                  <rect x="86" y="72" width="102" height="90" fill="#E5E7EB" stroke="#4B5563" strokeWidth="1.5" />
                  <text x="137" y="121" textAnchor="middle" fontFamily="Inter, sans-serif" fontSize="11" fontWeight="600" fill="#111827">Frente ? m</text>
                  <text x="110" y="173" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="9.5" fill="#BE123C">Clásico C8286: F1 &le; 0.104</text>

                  <path d="M222 95 L238 95" stroke="#1D4ED8" strokeWidth="2" />
                  <polygon points="236,90 245,95 236,100" fill="#1D4ED8" />

                  <rect x="248" y="4" width="208" height="182" fill="#F0FDFA" stroke="#0F766E" strokeOpacity="0.35" />
                  <text x="260" y="24" fontFamily="JetBrains Mono, monospace" fontSize="10" fontWeight="700" fill="#0F766E">RAYOS 3D KAY-KAJIYA</text>
                  <circle cx="272" cy="146" r="5.5" fill="#111827" />
                  <text x="272" y="162" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#111827">Cam</text>
                  <line x1="276" y1="142" x2="345" y2="96" stroke="#1D4ED8" strokeWidth="1.6" />
                  <line x1="276" y1="142" x2="412" y2="62" stroke="#BE123C" strokeWidth="1.4" strokeDasharray="4 3" />
                  <line x1="276" y1="142" x2="412" y2="118" stroke="#0F766E" strokeWidth="1.4" />
                  <rect x="330" y="78" width="38" height="54" fill="#1D4ED8" fillOpacity="0.16" stroke="#1D4ED8" strokeWidth="1.6" />
                  <rect x="394" y="44" width="46" height="80" fill="#0F766E" fillOpacity="0.14" stroke="#0F766E" strokeWidth="1.6" />
                  <text x="352" y="173" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="9.5" fontWeight="700" fill="#0F766E">Oclusión medida: 89% · Δy=1.8m</text>
                </svg>
              </div>
            </div>
            <div data-scroll-anim="rule-draw" className="scroll-rule-draw h-px w-full bg-[#111827]/15" />

            {/* SPREAD II: Diagram Left (Animated Scroll Bars) / Text Right (Warm Ochre accent) */}
            <div className="grid items-center gap-10 py-14 lg:grid-cols-12">
              <div data-scroll-anim="enter-left" className="scroll-enter-left order-2 border border-[#B45309]/25 bg-[#FFFDF7]/92 p-6 lg:order-1 lg:col-span-6">
                <div className="font-mono-tech mb-5 text-[10px] font-semibold uppercase tracking-wider text-[#B45309]">
                  Costo de Inferencia y Decisión Adaptativa
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-xs font-medium text-[#111827]">
                      <span>100% Gemini 4 Hydrogen (Sin Compuerta)</span>
                      <span className="font-mono-tech text-[#BE123C]">1,420 ms · costo alto</span>
                    </div>
                    <div className="mt-1.5 h-3 w-full bg-[#E5E7EB]">
                      <div data-scroll-anim="bar-grow" className="scroll-bar-grow h-full w-[88%] bg-[#BE123C]" />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs font-medium text-[#111827]">
                      <span>Compuerta Edge → Argon (Escena clean)</span>
                      <span className="font-mono-tech text-[#1D4ED8]">780 ms · −45% latencia</span>
                    </div>
                    <div className="mt-1.5 h-3 w-full bg-[#E5E7EB]">
                      <div data-scroll-anim="bar-grow" className="scroll-bar-grow h-full w-[54%] bg-[#1D4ED8]" />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs font-medium text-[#111827]">
                      <span>Compuerta Edge → Flash-Lite / Local GPU</span>
                      <span className="font-mono-tech text-[#0F766E]">310–365 ms · −72% latencia</span>
                    </div>
                    <div className="mt-1.5 h-3 w-full bg-[#E5E7EB]">
                      <div data-scroll-anim="bar-grow" className="scroll-bar-grow h-full w-[26%] bg-[#0F766E]" />
                    </div>
                  </div>
                </div>
              </div>

              <article data-scroll-anim="enter-right" className="scroll-enter-right order-1 lg:order-2 lg:col-span-6 space-y-4 border-l-2 border-[#B45309]/65 pl-5">
                <div className="flex items-baseline gap-4">
                  <span className="font-serif-editorial text-4xl font-semibold leading-none text-[#B45309]/45">
                    II
                  </span>
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#B45309]">
                    Costo y Escalamiento — Sin Hardware LiDAR
                  </span>
                </div>
                <h3 className="font-serif-editorial text-2xl leading-[1.12] text-[#111827] sm:text-3xl">
                  El dilema entre sensores LiDAR costosos y llamar modelos pesados en cada foto
                </h3>
                <p className="max-w-[54ch] text-sm leading-relaxed text-[#4B5563]">
                  Los escáneres LiDAR encarecen cualquier cámara, pero invocar modelos VLM profundos en el 100% de cuadros desperdicia latencia y cuota. Heredando la cascada <span className="font-mono-tech text-xs text-[#111827]">64→224 px</span> de <span className="font-mono-tech text-xs text-[#111827]">C8286</span> (que recortó <strong className="font-semibold text-[#111827]">97.5%</strong> de ancho de banda), la compuerta decide en qué escenas basta una ruta rápida y cuándo escalar.
                </p>
                <div className="font-mono-tech flex flex-wrap gap-x-6 gap-y-1 border-t border-[#B45309]/15 pt-3 text-[11px] text-[#0F766E]">
                  <span>APROBADO: −45% / −72% latencia</span>
                  <span className="text-[#4B5563]">GPU local: MoGe-2 ViT-S (~1.3 GB VRAM)</span>
                </div>
              </article>
            </div>
            <div data-scroll-anim="rule-draw" className="scroll-rule-draw h-px w-full bg-[#15284B]/18" />

            {/* SPREAD III: Text Left / Diagram Right (Terracotta/Burgundy accent) */}
            <div className="grid items-center gap-10 py-14 lg:grid-cols-12">
              <article data-scroll-anim="enter-left" className="scroll-enter-left lg:col-span-6 space-y-4 border-l-2 border-[#9F1239]/65 pl-5 pr-0 lg:pr-8">
                <div className="flex items-baseline gap-4">
                  <span className="font-serif-editorial text-4xl font-semibold leading-none text-[#9F1239]/40">
                    III
                  </span>
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#9F1239]">
                    Física Imposible — Auditoría Sin Ground Truth
                  </span>
                </div>
                <h3 className="font-serif-editorial text-2xl leading-[1.12] text-[#111827] sm:text-3xl">
                  Alucinaciones físicas silenciosas: sólidos que levitan o se atraviesan
                </h3>
                <p className="max-w-[54ch] text-sm leading-relaxed text-[#4B5563]">
                  Un VLM sin restricciones físicas puede ubicar un objeto flotando medio metro en el aire (<span className="font-mono-tech text-xs text-[#111827]">GFI</span>), superponer dos volúmenes rígidos en el mismo espacio (<span className="font-mono-tech text-xs text-[#111827]">PVI SAT</span>) o inflar escalas bajo ruido fotométrico (<span className="font-mono-tech text-xs text-[#111827]">±σ_y ∝ y²</span>).
                </p>
                <div className="font-mono-tech flex flex-wrap gap-x-6 gap-y-1 border-t border-[#9F1239]/15 pt-3 text-[11px] text-[#0F766E]">
                  <span>PECS Sobel + SAT 15-ejes + bloqueo RECHAZADO_FISICA</span>
                </div>
              </article>

              <div data-scroll-anim="enter-right" className="scroll-enter-right lg:col-span-6 border border-[#9F1239]/22 bg-[#FDF8F8]/92 p-4">
                <svg viewBox="0 0 460 190" className="h-auto w-full" role="img" aria-label="Diagrama brecha III: colisión SAT PVI y levitación GFI">
                  <rect x="4" y="4" width="218" height="182" fill="#F7F7F5" stroke="#CBD5E1" />
                  <text x="16" y="24" fontFamily="JetBrains Mono, monospace" fontSize="10" fontWeight="700" fill="#BE123C">COLISIÓN SÓLIDA SAT (PVI)</text>
                  <rect x="42" y="64" width="84" height="72" fill="#1D4ED8" fillOpacity="0.16" stroke="#1D4ED8" strokeWidth="1.8" />
                  <rect x="94" y="80" width="88" height="68" fill="#BE123C" fillOpacity="0.16" stroke="#BE123C" strokeWidth="1.8" />
                  <rect x="94" y="80" width="32" height="56" fill="#BE123C" fillOpacity="0.32" stroke="#BE123C" strokeDasharray="3 2" />
                  <text x="113" y="168" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="9.5" fontWeight="700" fill="#BE123C">PVI &ge; 8.0% → Solapamiento</text>

                  <rect x="234" y="4" width="222" height="182" fill="#F7F7F5" stroke="#CBD5E1" />
                  <text x="246" y="24" fontFamily="JetBrains Mono, monospace" fontSize="10" fontWeight="700" fill="#B45309">LEVITACIÓN GRAVITATORIA (GFI)</text>
                  <line x1="252" y1="146" x2="438" y2="146" stroke="#4B5563" strokeWidth="2" />
                  <text x="256" y="161" fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#4B5563">Plano soporte latente Z</text>
                  <rect x="296" y="54" width="96" height="54" fill="#B45309" fillOpacity="0.14" stroke="#B45309" strokeWidth="1.8" />
                  <line x1="344" y1="108" x2="344" y2="146" stroke="#BE123C" strokeWidth="1.6" strokeDasharray="3 2" />
                  <text x="352" y="131" fontFamily="JetBrains Mono, monospace" fontSize="9.5" fontWeight="700" fill="#BE123C">+0.58 m</text>
                  <line x1="252" y1="175" x2="438" y2="175" stroke="#BE123C" strokeWidth="1" strokeOpacity="0.35" />
                  <text x="345" y="171" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="8.5" fontWeight="700" fill="#BE123C">DICTAMEN: RECHAZADO_FISICA</text>
                </svg>
              </div>
            </div>
            <div data-scroll-anim="rule-draw" className="scroll-rule-draw h-px w-full bg-[#15284B]/18" />

            {/* SPREAD IV: Diagram Left / Text Right (Botanical Teal accent) */}
            <div className="grid items-center gap-10 py-14 lg:grid-cols-12">
              <div data-scroll-anim="enter-left" className="scroll-enter-left order-2 border border-[#0F766E]/25 bg-[#F4FBF9]/92 p-4 lg:order-1 lg:col-span-6">
                <svg viewBox="0 0 460 190" className="h-auto w-full" role="img" aria-label="Diagrama brecha IV: sensibilidad FOV y deriva multi-frame">
                  <rect x="4" y="4" width="222" height="182" fill="#FFFFFF" stroke="#CBD5E1" />
                  <text x="16" y="24" fontFamily="JetBrains Mono, monospace" fontSize="10" fontWeight="700" fill="#1D4ED8">DESCALIBRADO FOV_h (PINHOLE)</text>
                  <line x1="26" y1="142" x2="206" y2="142" stroke="#CBD5E1" strokeWidth="1.2" />
                  <line x1="26" y1="48" x2="26" y2="142" stroke="#CBD5E1" strokeWidth="1.2" />
                  <polyline fill="none" stroke="#1D4ED8" strokeWidth="2.2" points="28,124 62,92 94,58 134,98 172,126 202,138" />
                  <circle cx="94" cy="58" r="4.5" fill="#0F766E" />
                  <text x="102" y="54" fontFamily="JetBrains Mono, monospace" fontSize="9.5" fontWeight="700" fill="#0F766E">60°: 80.7%</text>
                  <text x="148" y="114" fontFamily="JetBrains Mono, monospace" fontSize="9" fill="#BE123C">100°: 2.6%</text>
                  <text x="115" y="168" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="9" fill="#4B5563">Reproyección en vivo &lt; 10 ms</text>

                  <rect x="236" y="4" width="220" height="182" fill="#FFFFFF" stroke="#CBD5E1" />
                  <text x="248" y="24" fontFamily="JetBrains Mono, monospace" fontSize="10" fontWeight="700" fill="#0F766E">DERIVA VIDEO (WORLD MODELS)</text>
                  <rect x="256" y="56" width="38" height="44" fill="#FFFFFF" stroke="#1D4ED8" strokeWidth="1.6" />
                  <rect x="314" y="48" width="54" height="58" fill="#FFFFFF" stroke="#B45309" strokeWidth="1.6" />
                  <rect x="384" y="64" width="32" height="36" fill="#FFFFFF" stroke="#BE123C" strokeWidth="1.6" />
                  <text x="275" y="118" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#4B5563">t0</text>
                  <text x="341" y="118" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#B45309">t1 (+17%)</text>
                  <text x="400" y="118" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="8.5" fill="#BE123C">t2 (jitter)</text>
                  <text x="346" y="168" textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize="9.5" fontWeight="700" fill="#0F766E">Estabilidad medida: 58.5 / 100</text>
                </svg>
              </div>

              <article data-scroll-anim="enter-right" className="scroll-enter-right order-1 lg:order-2 lg:col-span-6 space-y-4 border-l-2 border-[#0F766E]/65 pl-5">
                <div className="flex items-baseline gap-4">
                  <span className="font-serif-editorial text-4xl font-semibold leading-none text-[#0F766E]/45">
                    IV
                  </span>
                  <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#0F766E]">
                    Óptica FOV y Deriva Temporal
                  </span>
                </div>
                <h3 className="font-serif-editorial text-2xl leading-[1.12] text-[#111827] sm:text-3xl">
                  Descalibrado de lente (<span className="font-mono-tech text-xl">FOV_h</span>) y deriva en secuencias de video
                </h3>
                <p className="max-w-[54ch] text-sm leading-relaxed text-[#4B5563]">
                  Un error en la focal derrumba el calce 2D-3D desde <strong className="font-semibold text-[#111827]">80.7%</strong> (<span className="font-mono-tech text-xs text-[#111827]">60°</span>) hasta <strong className="font-semibold text-[#111827]">2.6%</strong> (<span className="font-mono-tech text-xs text-[#111827]">100°</span>), mientras fotos verticales en primer plano (<span className="font-mono-tech text-xs text-[#111827]">~36°</span>) distorsionan distancias si se asume un gran angular fijo. En video, un <em>World Model</em> deforma volúmenes cuadro a cuadro (<strong className="font-semibold text-[#111827]">17.0%</strong> en <em>transport crate</em>).
                </p>
                <div className="font-mono-tech flex flex-wrap gap-x-6 gap-y-1 border-t border-[#0F766E]/15 pt-3 text-[11px] text-[#0F766E]">
                  <span>EXIF + Intrínsecos MoGe-2 automáticos + Auditoría Multi-Frame</span>
                </div>
              </article>
            </div>
            <div data-scroll-anim="rule-draw" className="scroll-rule-draw h-[2px] w-full bg-gradient-to-r from-[#0F766E]/50 via-[#1D4ED8]/50 to-[#B45309]/50" />

            <div data-scroll-anim="curtain-up" className="scroll-curtain-up mt-10 flex flex-wrap items-center justify-between gap-4">
              <span className="font-mono-tech text-xs text-[#15284B]">
                Reconstrucción local en vivo + 9 herramientas espaciales + QUBO-MWIS 360°
              </span>
              <div className="flex flex-wrap items-center gap-3">
                <a
                  href="#scrolly-3d-story"
                  className="inline-flex items-center gap-2 rounded-sm border border-[#15284B]/25 bg-[#FFFFFF]/88 px-4 py-2 text-xs font-medium text-[#15284B] transition-colors hover:border-[#15284B]"
                >
                  <span>Pipeline 3D al scroll</span>
                  <ArrowDown className="h-3.5 w-3.5 text-[#4B5563]" />
                </a>
                <button
                  type="button"
                  onClick={() => onOpenLab(featured?.id)}
                  className="inline-flex items-center gap-2 rounded-sm bg-[#15284B] px-5 py-2 text-xs font-medium text-white transition-colors hover:bg-[#1D4ED8]"
                >
                  <span>Ir al Laboratorio 3D</span>
                  <ArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* 3. Scroll-Driven 3D Scrollytelling Section (Deep Prussian-Slate Drafting Band) */}
        <section
          id="scrolly-3d-story"
          ref={storyRef}
          className="scrolly-narrative-scope border-t border-[#15284B] bg-[#101C31]/95 py-16 text-[#F8FAFC] backdrop-blur-[1.5px]"
        >
          <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
            <div className="flex flex-wrap items-end justify-between gap-4 border-b border-[#334155] pb-6">
              <div data-scroll-anim="curtain-up" className="scroll-curtain-up max-w-2xl">
                <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#F59E0B]">
                  Narrativa 3D Sincronizada al Scroll
                </span>
                <h2 className="font-serif-editorial mt-1.5 text-3xl leading-[1.08] text-[#F8FAFC] sm:text-4xl">
                  Cómo pasa una imagen plana a geometría 3D auditable
                </h2>
              </div>

              {/* Minimal Underline Chapter Selector */}
              <div className="flex flex-wrap items-center gap-4 border-b border-[#334155]" role="tablist" aria-label="Etapas del pipeline 3D">
                {CHAPTERS.map((ch, idx) => {
                  const isAct = activeChapter === idx;
                  return (
                    <button
                      key={ch.step}
                      type="button"
                      role="tab"
                      aria-selected={isAct}
                      onClick={() => jumpToChapter(idx)}
                      className={`border-b-2 pb-2 text-xs font-medium transition-colors ${
                        isAct
                          ? 'border-[#F59E0B] text-[#FFFFFF]'
                          : 'border-transparent text-[#94A3B8] hover:text-[#F8FAFC]'
                      }`}
                    >
                      <span className="font-mono-tech mr-1.5 opacity-70">{ch.step}</span>
                      {ch.shortLabel}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="mt-10 grid items-start gap-10 lg:grid-cols-12">
              {/* Left Column: 4 Named View-Timeline Chapters (Border-left Architectural Index) */}
              <div className="space-y-6 lg:col-span-5">
                {CHAPTERS.map((ch, idx) => {
                  const isAct = activeChapter === idx;
                  const chapterClass = `scrolly-chapter-${idx + 1}`;
                  const darkAccents = ['#60A5FA', '#38BDF8', '#FB7185', '#2DD4BF'];
                  const darkAccent = darkAccents[idx % darkAccents.length];
                  return (
                    <article
                      key={ch.step}
                      ref={(el) => {
                        chapterRefs.current[idx] = el;
                      }}
                      className={`${chapterClass} min-h-[195px] py-2`}
                    >
                      <div
                        onClick={() => jumpToChapter(idx)}
                        className={`scrolly-card-inner cursor-pointer border-l-2 py-4 pl-5 pr-4 transition-colors backdrop-blur-[1.5px] ${
                          isAct
                            ? 'bg-[#1B2E4F] opacity-100'
                            : 'bg-[#162540]/88 opacity-85 hover:opacity-100'
                        }`}
                        style={{
                          borderLeftColor: isAct ? darkAccent : 'rgba(148, 163, 184, 0.28)',
                        }}
                      >
                        <div className="font-mono-tech flex items-center justify-between gap-2 text-[11px]">
                          <span style={{ color: darkAccent }} className="font-semibold uppercase tracking-wider">
                            {ch.step} / {ch.shortLabel}
                          </span>
                          <span className="text-[#CBD5E1]">{ch.metricTag}</span>
                        </div>
                        <h3 className="font-serif-editorial mt-2 text-xl leading-snug text-[#F8FAFC]">
                          {ch.title}
                        </h3>
                        <p className="mt-2 text-xs leading-relaxed text-[#CBD5E1]">
                          {ch.summary}
                        </p>
                      </div>
                    </article>
                  );
                })}
              </div>

              {/* Right Column: Sticky Live Three.js WebGL Stage */}
              <div className="lg:sticky lg:top-20 lg:col-span-7">
                <div className="border border-[#334155] bg-[#162540]">
                  <div className="flex items-center justify-between border-b border-[#334155] bg-[#132038] px-4 py-2.5">
                    <div className="flex items-center gap-2 text-xs font-medium text-[#F8FAFC]">
                      <Compass className="h-3.5 w-3.5 text-[#F59E0B]" />
                      <span>{CHAPTERS[activeChapter]?.title}</span>
                    </div>
                    <span className="font-mono-tech text-[11px] text-[#94A3B8]">
                      {isStageVisible ? 'WebGL Activo' : 'Pausado (Off-screen)'}
                    </span>
                  </div>

                  <div
                    ref={stageBoxRef}
                    className="webgl-stage-container relative h-[360px] w-full bg-[#142238] sm:h-[440px]"
                    style={{ touchAction: 'pan-y' }}
                  >
                    <Canvas
                      dpr={[1, 2]}
                      shadows="percentage"
                      frameloop={isStageVisible ? 'always' : 'demand'}
                      camera={{ position: [0.3, 0.25, 0.3], fov: 42, near: 0.05, far: 60 }}
                      gl={{ antialias: true, powerPreference: 'high-performance' }}
                    >
                      <color attach="background" args={['#142238']} />
                      <ambientLight intensity={0.95} />
                      <directionalLight position={[4, 7, 4]} intensity={1.0} />

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

                    <div className="pointer-events-none absolute inset-x-3 bottom-3 flex flex-wrap items-center justify-between gap-2 border border-[#334155] bg-[#101C31]/92 px-3 py-1.5 text-[11px] text-[#CBD5E1]">
                      <span className="font-mono-tech text-[#F8FAFC]">
                        Etapa {CHAPTERS[activeChapter]?.step}/04 · Desplázate para animar cámara 3D
                      </span>
                      <div className="flex items-center gap-3">
                        <span className="inline-flex items-center gap-1">
                          <span className="h-2 w-2 bg-[#60A5FA]" /> Predicción
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="h-2 w-2 bg-[#14B8A6]" /> GT Esmeralda
                        </span>
                        <span className="inline-flex items-center gap-1">
                          <span className="h-2 w-2 bg-[#F43F5E]" /> Regla 3D
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 4. Universal Metrics Architectural Ledger (No Ground Truth) */}
        <section className="border-t border-[#15284B]/18 bg-gradient-to-br from-[#FBF8F3]/88 via-[#FFFFFF]/88 to-[#F1F8F6]/88 py-16 backdrop-blur-[1px]">
          <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
            <div data-scroll-anim="curtain-up" className="scroll-curtain-up flex flex-wrap items-end justify-between gap-4 border-b border-[#15284B]/18 pb-5">
              <div>
                <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#0F766E]">
                  Diagnóstico Intrínseco Sin Referencia
                </span>
                <h2 className="font-serif-editorial mt-1 text-3xl text-[#111827] sm:text-4xl">
                  Métricas universales para todo tipo de imagen
                </h2>
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

            <div className="mt-8 grid gap-x-8 gap-y-10 md:grid-cols-2 lg:grid-cols-3">
              {LATENT_CAPABILITIES.map((cap, idx) => {
                const stagger = `scroll-stagger-${(idx % 3) + 1}`;
                return (
                  <article
                    key={cap.code}
                    data-scroll-anim="curtain-up"
                    className={`scroll-curtain-up ${stagger} border-t-2 pt-4`}
                    style={{ borderTopColor: cap.color }}
                  >
                    <div
                      className="font-mono-tech text-[11px] font-semibold uppercase tracking-[0.12em]"
                      style={{ color: cap.color }}
                    >
                      {cap.code}
                    </div>
                    <h3 className="font-serif-editorial mt-2 text-xl text-[#111827]">
                      {cap.title}
                    </h3>
                    <p className="mt-2 text-xs leading-relaxed text-[#4B5563]">
                      {cap.desc}
                    </p>
                  </article>
                );
              })}
            </div>
          </div>
        </section>

        {/* 5. C8286 Lineage & Rigor Index (Editorial Split, No Box Grid) */}
        <section className="problem-scroll-section border-t border-[#15284B]/15 bg-gradient-to-r from-[#EEF4F8]/86 via-[#F8F9FB]/88 to-[#FBF6EC]/86 py-16 backdrop-blur-[1px]">
          <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
            <div className="grid items-start gap-12 lg:grid-cols-12">
              <div data-scroll-anim="enter-left" className="scroll-enter-left lg:col-span-6 space-y-4">
                <div className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#0F766E]">
                  Evolución Directa — De C8286 2D a MyLeenTo3
                </div>
                <h2 className="font-serif-editorial text-3xl leading-[1.08] text-[#111827] sm:text-4xl">
                  Por qué nace MyLeenTo3 tras comparar Visión Clásica Paralela vs. CNNs
                </h2>
                <p className="max-w-[54ch] text-sm leading-relaxed text-[#4B5563]">
                  En el proyecto previo de Computación Paralela (<span className="font-mono-tech text-xs text-[#111827]">C8286</span>), comprobamos que los descriptores 2D clásicos colapsan bajo ruido u oclusión (<span className="font-mono-tech text-xs text-[#111827]">F1 &le; 0.104</span>), que <span className="font-mono-tech text-xs text-[#111827]">MobileNetV2 INT8</span> alcanza <span className="font-mono-tech text-xs text-[#111827]">F1 = 0.715</span> y que una cascada adaptativa Edge-Servidor recorta <span className="font-mono-tech text-xs text-[#111827]">97.5%</span> de ancho de banda. <strong className="font-semibold text-[#111827]">MyLeenTo3</strong> traslada ese mismo principio al espacio 3D métrico.
                </p>
                <div className="pt-2">
                  <a
                    href="https://github.com/dariohuerta84/C8286-Proyecto1-Vision-Clasica-vs-CNN"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1.5 border-b border-[#1D4ED8]/40 pb-0.5 text-xs font-medium text-[#1D4ED8] transition-colors hover:border-[#1D4ED8]"
                  >
                    <span>Ver repositorio base C8286 en GitHub</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </a>
                </div>
              </div>

              <div data-scroll-anim="enter-right" className="scroll-enter-right lg:col-span-6 divide-y divide-[#15284B]/12 border-y border-[#15284B]/18 bg-[#FFFFFF]/88 px-5 backdrop-blur-[1.5px]">
                <div className="border-l-2 border-[#1D4ED8] py-5 pl-4">
                  <div className="font-mono-tech text-[11px] uppercase tracking-wider text-[#1D4ED8]">
                    01 — Motor GPU Métrico Local (MoGe-2 ViT-S/B/L)
                  </div>
                  <h3 className="font-serif-editorial mt-1 text-xl text-[#111827]">
                    Descubrimiento VLM/YOLOE-PF y hasta 80,000 surfels 2DGS
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-[#4B5563]">
                    Extrae <span className="font-mono-tech text-[#111827]">FOV_h</span> vía EXIF o intrínsecos neuronales, levanta vocabulario abierto (<span className="font-mono-tech text-[#111827]">YOLOE-PF 4,585 cl. / OV</span>) y construye nube 2DGS + malla 2.5D exportable a <span className="font-mono-tech text-[#111827]">.PLY</span>.
                  </p>
                </div>

                <div className="border-l-2 border-[#0F766E] py-5 pl-4">
                  <div className="font-mono-tech text-[11px] uppercase tracking-wider text-[#0F766E]">
                    02 — Metrología Exacta Fase 2 (Calibrador 3D + ArUco)
                  </div>
                  <h3 className="font-serif-editorial mt-1 text-xl text-[#111827]">
                    Calibración física directa y perfil topográfico superficial
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-[#4B5563]">
                    Mide arco superficial en milímetros con 2 clics sobre WebGL, fija escala real con tarjeta/moneda/hoja A4 o marcadores <span className="font-mono-tech text-[#111827]">ArUco</span> y reproyecta <span className="font-mono-tech text-[#111827]">FOV_h</span> en menos de <span className="font-mono-tech text-[#111827]">10 ms</span>.
                  </p>
                </div>

                <div className="border-l-2 border-[#B45309] py-5 pl-4">
                  <div className="font-mono-tech text-[11px] uppercase tracking-wider text-[#B45309]">
                    03 — Reconstrucción 360° Fase 3 y Copiloto 9 Funciones
                  </div>
                  <h3 className="font-serif-editorial mt-1 text-xl text-[#111827]">
                    Registro SIFT + Sim(3) Umeyama + ICP y ActivePose NBV
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-[#4B5563]">
                    Une 2 a 6 vistas en volumen sólido tomográfico / ConvexHull 360° con guía polar de zonas ciegas, junto al copiloto inteligente con 9 funciones geométricas auditables.
                  </p>
                </div>

                <div className="border-l-2 border-[#9F1239] py-5 pl-4">
                  <div className="font-mono-tech text-[11px] uppercase tracking-wider text-[#9F1239]">
                    04 — Motor Combinatorio QUBO-MWIS Fase 4 (Harvard HQI)
                  </div>
                  <h3 className="font-serif-editorial mt-1 text-xl text-[#111827]">
                    Bifurcación Simulada, NMS 3D Rydberg y Puente QuEra Aquila
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-[#4B5563]">
                    Resuelve registro isométrico 3D anti-simetría, supresión de colisiones SAT 15-ejes y planificación multi-disparo NBV (<em>Classical Shadows</em>) con exportación BQM / AWS Braket.
                  </p>
                  <button
                    type="button"
                    onClick={() => onNavigate('quantum')}
                    className="mt-2 inline-flex items-center gap-1 border-b border-[#9F1239]/40 pb-0.5 text-xs font-medium text-[#9F1239] transition-colors hover:border-[#9F1239]"
                  >
                    <span>Explorar Optimizador QUBO</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 6. Benchmarks Specimen Plates with View Transition Continuity */}
        <section className="border-t border-[#15284B]/18 bg-[#F6F2EA]/88 py-16 backdrop-blur-[1px]">
          <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
            <div data-scroll-anim="curtain-up" className="scroll-curtain-up flex flex-wrap items-end justify-between gap-4 border-b border-[#15284B]/18 pb-5">
              <div>
                <span className="font-mono-tech text-[11px] uppercase tracking-[0.14em] text-[#15284B]">
                  Banco Reproducible — SEED = 42
                </span>
                <h2 className="font-serif-editorial mt-1 text-3xl text-[#111827] sm:text-4xl">
                  Tres escenarios calibrados para auditar 2D y 3D
                </h2>
              </div>
              <button
                type="button"
                onClick={() => onNavigate('metrics')}
                className="inline-flex items-center gap-1 text-xs font-medium text-[#1D4ED8] hover:underline"
              >
                <span>Ver comparativa multi-modelo</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="mt-8 grid gap-6 md:grid-cols-3">
              {benchmarks.map((scene, idx) => {
                const stagger = `scroll-stagger-${(idx % 3) + 1}`;
                const plateAccent = ['#1D4ED8', '#B45309', '#0F766E'][idx % 3];
                return (
                  <article
                    key={scene.id}
                    data-scroll-anim="curtain-up"
                    className={`benchmark-card scroll-curtain-up ${stagger} group flex flex-col border border-[#15284B]/16 border-t-2 bg-[#FFFFFF]/92 transition-colors hover:border-[#15284B]`}
                    style={{ borderTopColor: plateAccent }}
                  >
                    <div className="benchmark-thumb aspect-[3/2] overflow-hidden border-b border-[#15284B]/12 bg-[#F0F2F5]">
                      <img
                        src={normalizeBase64Image(scene.image_base64)}
                        alt={scene.title}
                        className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
                      />
                    </div>
                    <div className="flex flex-1 flex-col p-5">
                      <div className="font-mono-tech flex items-center justify-between gap-2 text-[11px]">
                        <span style={{ color: plateAccent }} className="font-semibold">{scene.id}</span>
                        <span className="text-[#4B5563]">
                          {scene.gt_objects.length} OBBs GT
                        </span>
                      </div>
                      <h3 className="font-serif-editorial mt-2 text-xl text-[#111827]">
                        {scene.title}
                      </h3>
                      <p className="mt-1.5 flex-1 text-xs leading-relaxed text-[#4B5563]">
                        {scene.description}
                      </p>
                      <button
                        type="button"
                        onClick={(e) => handleBenchmarkOpen(scene.id, e)}
                        className="mt-4 inline-flex w-full items-center justify-center gap-1.5 border border-[#15284B]/22 bg-[#F7F7F5] px-3 py-2 text-xs font-medium text-[#15284B] transition-colors hover:border-[#15284B] hover:bg-[#15284B] hover:text-white"
                      >
                        <span>Cargar en Laboratorio 3D</span>
                        <ArrowRight className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};
