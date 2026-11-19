import type {
  AnalyzeResponse,
  BenchmarkScene,
  CompareModelsResponse,
  CopilotResponse,
  GeminiModelId,
  HealthResponse,
  MultiFrameAuditResult,
  MultiFrameDemoResponse,
  MultiFrameItem,
  PairwiseGeometry,
  ReprojectResponse,
  SpatialObject,
  ThinkingLevel,
} from '../types/spatial';

const API_BASE = '/api';

async function requestJson<T>(
  path: string,
  options: RequestInit = {},
  apiKey?: string
): Promise<T> {
  const headers = new Headers(options.headers);
  if (!headers.has('Content-Type') && options.body) {
    headers.set('Content-Type', 'application/json');
  }
  if (apiKey && apiKey.trim().length > 0) {
    headers.set('X-Gemini-API-Key', apiKey.trim());
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const payload = await res.json();
      detail = payload?.detail || payload?.message || detail;
    } catch {
      // ignore non-json error bodies
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/health');
}

export async function fetchBenchmarks(): Promise<BenchmarkScene[]> {
  const data = await requestJson<{ benchmarks: BenchmarkScene[] }>('/benchmarks');
  return data.benchmarks ?? [];
}

export interface AnalyzePayload {
  benchmark_id?: string;
  image_base64?: string;
  mime_type?: string;
  width?: number;
  height?: number;
  model: GeminiModelId;
  thinking_level?: ThinkingLevel;
  fov_h_deg: number;
  max_objects?: number;
  custom_prompt?: string;
  force_demo?: boolean;
}

export async function analyzeScene(
  payload: AnalyzePayload,
  apiKey?: string
): Promise<AnalyzeResponse> {
  return requestJson<AnalyzeResponse>(
    '/analyze',
    {
      method: 'POST',
      body: JSON.stringify(payload),
    },
    apiKey
  );
}

export interface ReprojectPayload {
  objects: SpatialObject[];
  gt_objects?: SpatialObject[] | null;
  image_base64?: string | null;
  width: number;
  height: number;
  fov_h_deg: number;
}

function toObjectInput(obj: SpatialObject) {
  return {
    id: obj.id,
    label: obj.label,
    box_3d: obj.box_3d,
    confidence: obj.confidence ?? 0.9,
  };
}

export async function reprojectScene(
  payload: ReprojectPayload
): Promise<ReprojectResponse> {
  return requestJson<ReprojectResponse>('/reproject', {
    method: 'POST',
    body: JSON.stringify({
      ...payload,
      objects: payload.objects.map(toObjectInput),
      gt_objects: payload.gt_objects
        ? payload.gt_objects.map(toObjectInput)
        : undefined,
    }),
  });
}

export async function measurePair(
  objectA: Pick<SpatialObject, 'id' | 'label' | 'box_3d'>,
  objectB: Pick<SpatialObject, 'id' | 'label' | 'box_3d'>
): Promise<PairwiseGeometry> {
  const raw = await requestJson<
    PairwiseGeometry & { geometry?: PairwiseGeometry; measurement?: PairwiseGeometry }
  >('/measure', {
    method: 'POST',
    body: JSON.stringify({
      object_a: { id: objectA.id, label: objectA.label, box_3d: objectA.box_3d },
      object_b: { id: objectB.id, label: objectB.label, box_3d: objectB.box_3d },
    }),
  });
  return raw.geometry ?? raw.measurement ?? raw;
}

export async function compareModels(
  benchmarkId: string,
  fovHDeg: number,
  useLive = false,
  apiKey?: string
): Promise<CompareModelsResponse> {
  return requestJson<CompareModelsResponse>(
    '/compare-models',
    {
      method: 'POST',
      body: JSON.stringify({
        benchmark_id: benchmarkId,
        fov_h_deg: fovHDeg,
        use_live: useLive,
      }),
    },
    apiKey
  );
}

export async function fetchMultiFrameDemo(): Promise<MultiFrameDemoResponse> {
  return requestJson<MultiFrameDemoResponse>('/multiframe-demo');
}

export async function runMultiFrameAudit(
  frames: MultiFrameItem[],
  apiKey?: string
): Promise<MultiFrameAuditResult> {
  const raw = await requestJson<
    MultiFrameAuditResult & { audit?: MultiFrameAuditResult }
  >(
    '/multiframe-audit',
    {
      method: 'POST',
      body: JSON.stringify({ frames }),
    },
    apiKey
  );
  return raw.audit ?? raw;
}

export interface CopilotPayload {
  message: string;
  objects: SpatialObject[];
  gt_objects?: SpatialObject[] | null;
  image_base64?: string;
  width: number;
  height: number;
  fov_h_deg: number;
  model: GeminiModelId;
  previous_interaction_id?: string | null;
}

export async function sendCopilotMessage(
  payload: CopilotPayload,
  apiKey?: string
): Promise<CopilotResponse> {
  return requestJson<CopilotResponse>(
    '/copilot',
    {
      method: 'POST',
      body: JSON.stringify({
        ...payload,
        objects: payload.objects.map(toObjectInput),
        gt_objects: payload.gt_objects
          ? payload.gt_objects.map(toObjectInput)
          : undefined,
      }),
    },
    apiKey
  );
}
