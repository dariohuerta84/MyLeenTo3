import type { SpatialObject } from '../types/spatial';

export function computeFovVDeg(
  fovHDeg: number,
  width: number,
  height: number
): number {
  const safeW = Math.max(1, width);
  const safeH = Math.max(1, height);
  const halfRad = (fovHDeg * Math.PI) / 360;
  const vRad = 2 * Math.atan(Math.tan(halfRad) * (safeH / safeW));
  return (vRad * 180) / Math.PI;
}

export function formatMeters(val: number | undefined | null, digits = 2): string {
  if (val === undefined || val === null || Number.isNaN(val)) return '—';
  return `${val.toFixed(digits)} m`;
}

export function formatDegrees(val: number | undefined | null, digits = 1): string {
  if (val === undefined || val === null || Number.isNaN(val)) return '—';
  return `${val.toFixed(digits)}°`;
}

export function formatPercent(val: number | undefined | null, digits = 1): string {
  if (val === undefined || val === null || Number.isNaN(val)) return '—';
  return `${(val * 100).toFixed(digits)}%`;
}

export function normalizeBase64Image(
  base64: string,
  mimeType = 'image/png'
): string {
  if (!base64) return '';
  if (base64.startsWith('data:')) return base64;
  return `data:${mimeType};base64,${base64}`;
}

export function stripDataUrlPrefix(dataUrl: string): {
  base64: string;
  mimeType: string;
} {
  const match = dataUrl.match(/^data:([^;]+);base64,(.*)$/);
  if (match) {
    return { mimeType: match[1], base64: match[2] };
  }
  return { mimeType: 'image/png', base64: dataUrl };
}

export function exportPosesToJson(
  objects: SpatialObject[],
  meta: { scene: string; model: string; fov_h_deg: number }
): void {
  const payload = {
    exported_at: new Date().toISOString(),
    ...meta,
    objects: objects.map((o) => ({
      id: o.id,
      label: o.label,
      box_3d: o.box_3d,
      confidence: o.confidence,
      volume_m3: o.volume_m3,
      distance_m: o.distance_m,
      threejs_pose: o.threejs_pose,
    })),
  };
  triggerDownload(
    JSON.stringify(payload, null, 2),
    `myleento3_poses_${meta.scene}.json`,
    'application/json'
  );
}

export function exportPosesToCsv(
  objects: SpatialObject[],
  sceneName: string
): void {
  const header = [
    'id',
    'label',
    'x_m',
    'y_depth_m',
    'z_up_m',
    'dx_m',
    'dy_m',
    'dz_m',
    'roll_deg',
    'pitch_deg',
    'yaw_deg',
    'volume_m3',
    'distance_m',
  ];
  const rows = objects.map((o) => {
    const [x, y, z, dx, dy, dz, roll, pitch, yaw] = o.box_3d;
    const vol = o.volume_m3 ?? dx * dy * dz;
    const dist = o.distance_m ?? Math.hypot(x, y, z);
    return [
      o.id,
      `"${o.label.replace(/"/g, '""')}"`,
      x.toFixed(4),
      y.toFixed(4),
      z.toFixed(4),
      dx.toFixed(4),
      dy.toFixed(4),
      dz.toFixed(4),
      roll.toFixed(2),
      pitch.toFixed(2),
      yaw.toFixed(2),
      vol.toFixed(4),
      dist.toFixed(4),
    ].join(',');
  });
  triggerDownload(
    [header.join(','), ...rows].join('\n'),
    `myleento3_poses_${sceneName}.csv`,
    'text/csv;charset=utf-8;'
  );
}

function triggerDownload(content: string, filename: string, mime: string): void {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function runViewTransition(update: () => void): void {
  const doc = document as Document & {
    startViewTransition?: (cb: () => void) => { finished: Promise<void> };
  };
  const prefersReduced =
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (doc.startViewTransition && !prefersReduced) {
    doc.startViewTransition(update);
  } else {
    update();
  }
}
