import React, { useEffect, useRef } from 'react';

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
}

interface ProjectedPoint {
  sx: number;
  sy: number;
  alpha: number;
  radius: number;
  color: [number, number, number];
}

interface LandingParticleFieldProps {
  reducedMotion: boolean;
}

const PARTICLE_COUNT = 240;
const CONNECTION_DIST_SQ = 88 * 88;
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
  const nodes: ParticleNode[] = [];
  const nx = 6;
  const ny = 5;
  const nz = 8;

  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const ix = i % nx;
    const iy = Math.floor(i / nx) % ny;
    const iz = Math.floor(i / (nx * ny)) % nz;

    // Cloud coordinates: optical depth frustum scatter
    const cx = (rng() - 0.5) * 3.1;
    const cy = (rng() - 0.5) * 2.0;
    const cz = (rng() - 0.5) * 2.7;

    // Lattice coordinates: ordered 3D metric space with crisp jitter
    const lx = ((ix / (nx - 1)) - 0.5) * 2.25 + (rng() - 0.5) * 0.08;
    const ly = ((iy / (ny - 1)) - 0.5) * 1.55 + (rng() - 0.5) * 0.08;
    const lz = ((iz / (nz - 1)) - 0.5) * 2.05 + (rng() - 0.5) * 0.08;

    const paletteRoll = rng();
    const color =
      paletteRoll < 0.46
        ? PALETTE[0]
        : paletteRoll < 0.72
        ? PALETTE[1]
        : paletteRoll < 0.9
        ? PALETTE[2]
        : PALETTE[3];

    nodes.push({
      cx,
      cy,
      cz,
      lx,
      ly,
      lz,
      radius: 1.25 + rng() * 1.45,
      phase: rng() * Math.PI * 2,
      color,
      layerBias: 0.65 + rng() * 0.8,
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
        const alphaP = 1 - Math.exp(-7.5 * dt);
        smoothProgress += (targetProgress - smoothProgress) * alphaP;
        scrollVelocity *= Math.exp(-6.5 * dt);
      }

      ctx.clearRect(0, 0, width, height);

      const morph = smoothBell(smoothProgress);
      const rotY = smoothProgress * Math.PI * 1.65 + (reducedMotion ? 0 : elapsed * 0.045);
      const rotX =
        Math.sin(smoothProgress * Math.PI * 1.4) * 0.28 +
        (reducedMotion ? 0 : Math.cos(elapsed * 0.14) * 0.04);

      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);

      const fovScale = Math.min(width, height) * 0.72;
      const halfW = width * 0.5;
      const halfH = height * 0.5;
      const cameraDist = 3.1;
      const driftOffset = -(smoothProgress - 0.5) * height * 0.14 - scrollVelocity * 0.16;

      let activeCount = 0;
      for (let i = 0; i < nodes.length; i += 1) {
        const n = nodes[i];
        const idle = reducedMotion ? 0 : Math.sin(elapsed * 0.6 + n.phase) * 0.025;
        const px = n.cx + (n.lx - n.cx) * morph + idle;
        const py = n.cy + (n.ly - n.cy) * morph;
        const pz = n.cz + (n.lz - n.cz) * morph;

        // Rotate around Y then X
        const rx = px * cosY - pz * sinY;
        const rz1 = px * sinY + pz * cosY;
        const ry = py * cosX - rz1 * sinX;
        const rz = py * sinX + rz1 * cosX;

        const zCam = rz + cameraDist;
        if (zCam <= 0.35) continue;

        const perspective = fovScale / zCam;
        const sx = halfW + rx * perspective;
        const sy = halfH + ry * perspective + driftOffset * n.layerBias;

        if (sx < -40 || sx > width + 40 || sy < -40 || sy > height + 40) continue;

        const depthNorm = Math.min(Math.max((4.5 - zCam) / 2.8, 0), 1);
        const alpha = 0.16 + depthNorm * 0.3;
        const radius = Math.max(0.9, n.radius * (0.65 + depthNorm * 0.65));

        projected[activeCount++] = { sx, sy, alpha, radius, color: n.color };
      }

      // Draw crisp architectural constellation hairlines first
      ctx.lineWidth = 0.65;
      for (let i = 0; i < activeCount; i += 1) {
        const a = projected[i];
        let links = 0;
        for (let j = i + 1; j < activeCount && links < 3; j += 1) {
          const b = projected[j];
          const dx = a.sx - b.sx;
          const dy = a.sy - b.sy;
          const d2 = dx * dx + dy * dy;
          if (d2 < CONNECTION_DIST_SQ) {
            const distFade = 1 - Math.sqrt(d2) / 88;
            const lineAlpha = Math.min(0.13, distFade * Math.min(a.alpha, b.alpha) * 0.45);
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

      // Draw crisp particle nodes
      for (let i = 0; i < activeCount; i += 1) {
        const p = projected[i];
        const [r, g, b] = p.color;
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.sx, p.sy, p.radius, 0, Math.PI * 2);
        ctx.fill();
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
      scrollVelocity = Math.max(-120, Math.min(120, scrollVelocity + (y - lastScrollY) * 0.45));
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
