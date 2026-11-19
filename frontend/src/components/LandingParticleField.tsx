import React, { useEffect, useRef } from 'react';

type ParticleVariant = 'dot' | 'ring' | 'cross';

interface ParticleNode {
  cx: number;
  cy: number;
  cz: number;
  lx: number;
  ly: number;
  lz: number;
  radius: number;
  phase: number;
  color: [number, number, number];
  layerBias: number;
  variant: ParticleVariant;
}

interface ProjectedPoint {
  sx: number;
  sy: number;
  alpha: number;
  radius: number;
  color: [number, number, number];
  variant: ParticleVariant;
}

interface LandingParticleFieldProps {
  reducedMotion: boolean;
}

const PARTICLE_COUNT = 132;
const MIN_DIST_3D_SQ = 0.34 * 0.34;
const LINK_MIN_DIST_SQ = 52 * 52;
const LINK_MAX_DIST_SQ = 148 * 148;
const PALETTE: Array<[number, number, number]> = [
  [51, 65, 85],   // Slate #334155
  [29, 78, 216],  // Cobalt #1D4ED8
  [15, 118, 110], // Emerald #0F766E
  [180, 83, 9],   // Amber #B45309
];

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function buildParticles(): ParticleNode[] {
  const rng = mulberry32(42);
  const cloudCoords: Array<[number, number, number]> = [];
  let attempts = 0;

  while (cloudCoords.length < PARTICLE_COUNT && attempts < 14000) {
    attempts += 1;
    const x = (rng() - 0.5) * 6.6;
    const y = (rng() - 0.5) * 3.8;
    const z = (rng() - 0.5) * 2.9;
    let tooClose = false;
    for (let k = 0; k < cloudCoords.length; k += 1) {
      const [px, py, pz] = cloudCoords[k];
      const d2 = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2;
      if (d2 < MIN_DIST_3D_SQ) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) {
      cloudCoords.push([x, y, z]);
    }
  }

  // Fallback top-up if rare tight packing stops early
  while (cloudCoords.length < PARTICLE_COUNT) {
    cloudCoords.push([
      (rng() - 0.5) * 6.6,
      (rng() - 0.5) * 3.8,
      (rng() - 0.5) * 2.9,
    ]);
  }

  const nx = 11;
  const ny = 4;
  const nz = 3;
  const nodes: ParticleNode[] = [];

  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const [cx, cy, cz] = cloudCoords[i];
    const ix = i % nx;
    const iy = Math.floor(i / nx) % ny;
    const iz = Math.floor(i / (nx * ny)) % nz;

    const lx = ((ix / (nx - 1)) - 0.5) * 5.2 + (rng() - 0.5) * 0.14;
    const ly = ((iy / (ny - 1)) - 0.5) * 3.1 + (rng() - 0.5) * 0.14;
    const lz = ((iz / (nz - 1)) - 0.5) * 2.4 + (rng() - 0.5) * 0.14;

    const paletteRoll = rng();
    const color =
      paletteRoll < 0.40
        ? PALETTE[0]
        : paletteRoll < 0.72
        ? PALETTE[1]
        : paletteRoll < 0.90
        ? PALETTE[2]
        : PALETTE[3];

    const shapeRoll = rng();
    const variant: ParticleVariant =
      shapeRoll < 0.10 ? 'cross' : shapeRoll < 0.40 ? 'ring' : 'dot';

    nodes.push({
      cx,
      cy,
      cz,
      lx,
      ly,
      lz,
      radius: 1.9 + rng() * 2.0,
      phase: rng() * Math.PI * 2,
      color,
      layerBias: 0.7 + rng() * 0.95,
      variant,
    });
  }
  return nodes;
}

function smoothBell(p: number): number {
  const a = Math.min(Math.max((p - 0.28) / 0.25, 0), 1);
  const b = Math.min(Math.max((0.92 - p) / 0.24, 0), 1);
  const sa = a * a * (3 - 2 * a);
  const sb = b * b * (3 - 2 * b);
  return sa * sb;
}

export const LandingParticleField: React.FC<LandingParticleFieldProps> = ({
  reducedMotion,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const nodes = buildParticles();
    const projected: ProjectedPoint[] = new Array(nodes.length);
    let width = window.innerWidth;
    let height = window.innerHeight;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);

    const resize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();

    const readScrollProgress = () => {
      const maxScroll = Math.max(
        document.documentElement.scrollHeight - window.innerHeight,
        1
      );
      return Math.min(Math.max(window.scrollY / maxScroll, 0), 1);
    };

    let targetProgress = readScrollProgress();
    let smoothProgress = targetProgress;
    let lastScrollY = window.scrollY;
    let scrollVelocity = 0;
    let rafId = 0;
    let lastTime = performance.now();

    const drawFrame = (now: number) => {
      const dt = Math.min((now - lastTime) * 0.001, 0.1);
      lastTime = now;
      const elapsed = now * 0.001;

      if (reducedMotion) {
        smoothProgress = 0.25;
        scrollVelocity = 0;
      } else {
        const alphaP = 1 - Math.exp(-8.0 * dt);
        smoothProgress += (targetProgress - smoothProgress) * alphaP;
        scrollVelocity *= Math.exp(-6.2 * dt);
      }

      ctx.clearRect(0, 0, width, height);

      const morph = smoothBell(smoothProgress);
      const rotY = smoothProgress * Math.PI * 1.9 + (reducedMotion ? 0 : elapsed * 0.042);
      const rotX =
        Math.sin(smoothProgress * Math.PI * 1.4) * 0.28 +
        (reducedMotion ? 0 : Math.cos(elapsed * 0.14) * 0.035);

      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);

      const fovScale = Math.min(width, height) * 0.72;
      const halfW = width * 0.5;
      const halfH = height * 0.5;
      const cameraDist = 3.2;
      const driftOffset = -(smoothProgress - 0.5) * height * 0.24 - scrollVelocity * 0.26;

      let activeCount = 0;
      for (let i = 0; i < nodes.length; i += 1) {
        const n = nodes[i];
        const idle = reducedMotion ? 0 : Math.sin(elapsed * 0.55 + n.phase) * 0.025;
        const px = n.cx + (n.lx - n.cx) * morph + idle;
        const py = n.cy + (n.ly - n.cy) * morph;
        const pz = n.cz + (n.lz - n.cz) * morph;

        const rx = px * cosY - pz * sinY;
        const rz1 = px * sinY + pz * cosY;
        const ry = py * cosX - rz1 * sinX;
        const rz = py * sinX + rz1 * cosX;

        const zCam = rz + cameraDist;
        if (zCam <= 0.45) continue;

        const perspective = fovScale / zCam;
        const sx = halfW + rx * perspective;
        const sy = halfH + ry * perspective + driftOffset * n.layerBias;

        if (sx < -48 || sx > width + 48 || sy < -48 || sy > height + 48) continue;

        const depthNorm = Math.min(Math.max((4.8 - zCam) / 2.9, 0), 1);
        const centerDist = Math.abs(sx / Math.max(width, 1) - 0.5);
        const corridorSoftener =
          centerDist < 0.22 ? 0.52 + 0.48 * (centerDist / 0.22) : 1.0;

        const alpha = (0.38 + depthNorm * 0.38) * corridorSoftener;
        const radius = Math.max(1.6, n.radius * (0.78 + depthNorm * 0.48));

        projected[activeCount++] = {
          sx,
          sy,
          alpha,
          radius,
          color: n.color,
          variant: n.variant,
        };
      }

      // Sparse architectural filaments only across medium distances (no dense clumps)
      ctx.lineWidth = 0.85;
      for (let i = 0; i < activeCount; i += 1) {
        const a = projected[i];
        let links = 0;
        for (let j = i + 1; j < activeCount && links < 2; j += 1) {
          const b = projected[j];
          const dx = a.sx - b.sx;
          const dy = a.sy - b.sy;
          const d2 = dx * dx + dy * dy;
          if (d2 >= LINK_MIN_DIST_SQ && d2 <= LINK_MAX_DIST_SQ) {
            const d = Math.sqrt(d2);
            const distFade = 1 - (d - 52) / (148 - 52);
            const lineAlpha = Math.min(0.22, Math.max(0.03, distFade * Math.min(a.alpha, b.alpha) * 0.42));
            const [r, g, bl] = a.color;
            ctx.strokeStyle = `rgba(${r}, ${g}, ${bl}, ${lineAlpha.toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(a.sx, a.sy);
            ctx.lineTo(b.sx, b.sy);
            ctx.stroke();
            links += 1;
          }
        }
      }

      // Crisp architectural particles: solid dots, hollow geometric rings, and optical crosshairs
      for (let i = 0; i < activeCount; i += 1) {
        const p = projected[i];
        const [r, g, b] = p.color;
        if (p.variant === 'ring') {
          ctx.lineWidth = 1.15;
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${p.alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, p.radius + 0.6, 0, Math.PI * 2);
          ctx.stroke();
        } else if (p.variant === 'cross') {
          const arm = Math.max(3.5, p.radius + 1.8);
          ctx.lineWidth = 1.15;
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${p.alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.moveTo(p.sx - arm, p.sy);
          ctx.lineTo(p.sx + arm, p.sy);
          ctx.moveTo(p.sx, p.sy - arm);
          ctx.lineTo(p.sx, p.sy + arm);
          ctx.stroke();
        } else {
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.alpha.toFixed(3)})`;
          ctx.beginPath();
          ctx.arc(p.sx, p.sy, p.radius, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };

    const tick = (now: number) => {
      if (document.hidden || reducedMotion) {
        rafId = 0;
        return;
      }
      drawFrame(now);
      rafId = requestAnimationFrame(tick);
    };

    const startLoop = () => {
      if (reducedMotion) {
        drawFrame(performance.now());
        return;
      }
      if (!rafId && !document.hidden) {
        lastTime = performance.now();
        rafId = requestAnimationFrame(tick);
      }
    };

    const onScroll = () => {
      const y = window.scrollY;
      scrollVelocity = Math.max(-140, Math.min(140, scrollVelocity + (y - lastScrollY) * 0.55));
      lastScrollY = y;
      targetProgress = readScrollProgress();
      if (reducedMotion) drawFrame(performance.now());
    };

    const onVisibilityChange = () => {
      if (document.hidden) {
        if (rafId) {
          cancelAnimationFrame(rafId);
          rafId = 0;
        }
      } else {
        startLoop();
      }
    };

    window.addEventListener('resize', resize, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
    document.addEventListener('visibilitychange', onVisibilityChange);
    startLoop();

    return () => {
      if (rafId) cancelAnimationFrame(rafId);
      window.removeEventListener('resize', resize);
      window.removeEventListener('scroll', onScroll);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [reducedMotion]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 z-0 h-full w-full"
    />
  );
};
