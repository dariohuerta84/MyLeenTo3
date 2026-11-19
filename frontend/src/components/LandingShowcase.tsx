import React from 'react';
import { ArrowRight, CheckCircle2, Film, Layers, Ruler, ScanEye } from 'lucide-react';
import type { AppSection, BenchmarkScene } from '../types/spatial';
import { normalizeBase64Image } from '../utils/geometry';

interface LandingShowcaseProps {
  benchmarks: BenchmarkScene[];
  onOpenLab: (benchmarkId?: string) => void;
  onNavigate: (section: AppSection) => void;
}

export const LandingShowcase: React.FC<LandingShowcaseProps> = ({
  benchmarks,
  onOpenLab,
  onNavigate,
}) => {
  const featured = benchmarks[0];

  return (
    <div className="bg-[#F7F7F5] text-[#111827]">
      {/* Hero section */}
      <section className="mx-auto max-w-[1360px] px-4 pb-16 pt-12 sm:px-6 lg:pt-16">
        <div className="grid items-center gap-10 lg:grid-cols-12">
          <div className="lg:col-span-6 space-y-6">
            <div className="inline-flex items-center gap-2 rounded-full border border-[#E2E5EA] bg-[#FFFFFF] px-3 py-1 text-xs text-[#4B5563]">
              <span className="h-2 w-2 rounded-full bg-[#0F766E]" />
              <span>Geometría 9-DOF Calibrada · Gemini Interactions API v1</span>
            </div>

            <h1 className="font-serif-editorial text-4xl leading-[1.06] tracking-tight text-[#111827] sm:text-5xl lg:text-[3.65rem]">
              De una fotografía RGB monocular a poses 3D métricas y verificables.
            </h1>

            <p className="max-w-[60ch] text-base leading-relaxed text-[#4B5563]">
              MyLeenTo3 combina modelos multimodales Gemini con proyección pinhole exacta,
              transformación de base quiral <span className="font-mono-tech text-[#111827]">M = Rₓ(−90°)</span>{' '}
              y herramientas deterministas en Python para medir distancias, detectar oclusiones y
              auditar estabilidad geométrica en secuencias multi-frame.
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
              <button
                type="button"
                onClick={() => onNavigate('multiframe')}
                className="inline-flex items-center gap-2 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-4 py-2.5 text-sm font-medium text-[#111827] transition-colors hover:bg-[#F0F2F5]"
              >
                <Film className="h-4 w-4 text-[#4B5563]" />
                <span>Auditar Multi-Frame</span>
              </button>
            </div>

            <dl className="grid grid-cols-3 gap-4 border-t border-[#E2E5EA] pt-5 max-w-lg">
              <div>
                <dt className="text-[11px] text-[#4B5563]">Vector de Pose</dt>
                <dd className="font-mono-tech mt-0.5 text-sm font-semibold text-[#111827]">
                  9-DOF (m, °)
                </dd>
              </div>
              <div>
                <dt className="text-[11px] text-[#4B5563]">Determinante M</dt>
                <dd className="font-mono-tech mt-0.5 text-sm font-semibold text-[#0F766E]">
                  det(M) = +1.0
                </dd>
              </div>
              <div>
                <dt className="text-[11px] text-[#4B5563]">Reproyección</dt>
                <dd className="font-mono-tech mt-0.5 text-sm font-semibold text-[#1D4ED8]">
                  &lt; 10 ms / FOV
                </dd>
              </div>
            </dl>
          </div>

          {/* Right visual anchor: Interactive Optical Pinhole Preview Card */}
          <div className="lg:col-span-6">
            <div className="overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#FFFFFF]">
              <div className="flex items-center justify-between border-b border-[#E2E5EA] bg-[#F7F7F5] px-4 py-2.5">
                <div className="flex items-center gap-2 text-xs font-medium text-[#111827]">
                  <ScanEye className="h-4 w-4 text-[#1D4ED8]" />
                  <span>Banco Óptico Calibrado — Escena de Referencia</span>
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
                      className="block w-full h-auto object-cover"
                    />
                    {/* Crisp SVG reticle & pinhole schematic overlay */}
                    <svg
                      viewBox="0 0 960 640"
                      className="pointer-events-none absolute inset-0 h-full w-full"
                      aria-hidden="true"
                    >
                      <line x1="480" y1="0" x2="480" y2="640" stroke="#1D4ED8" strokeOpacity="0.22" strokeDasharray="6 6" />
                      <line x1="0" y1="320" x2="960" y2="320" stroke="#1D4ED8" strokeOpacity="0.22" strokeDasharray="6 6" />
                      <circle cx="480" cy="320" r="14" fill="none" stroke="#1D4ED8" strokeWidth="1.5" strokeOpacity="0.65" />
                      <rect x="20" y="20" width="236" height="32" rx="6" fill="#FFFFFF" fillOpacity="0.92" stroke="#E2E5EA" />
                      <text x="34" y="40" fontFamily="JetBrains Mono, monospace" fontSize="13" fill="#111827">
                        K: f=831.4px · cx=480 · cy=320
                      </text>
                    </svg>
                  </div>
                ) : (
                  <div className="flex aspect-[3/2] items-center justify-center rounded-lg border border-dashed border-[#CBD5E1] bg-[#FFFFFF] p-6 text-center">
                    <div className="space-y-2">
                      <p className="font-mono-tech text-xs text-[#4B5563]">
                        Cargando banco calibrado desde /api/benchmarks…
                      </p>
                      <button
                        type="button"
                        onClick={() => onOpenLab()}
                        className="rounded-md bg-[#1D4ED8] px-3 py-1.5 text-xs font-medium text-white"
                      >
                        Ir al Laboratorio 3D
                      </button>
                    </div>
                  </div>
                )}

                <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[#4B5563]">
                  <span className="font-mono-tech">
                    Gemini Frame (X-R, Y-Fwd, Z-Up) → Three.js (X-R, Y-Up, −Z-Fwd)
                  </span>
                  <button
                    type="button"
                    onClick={() => onOpenLab(featured?.id)}
                    className="inline-flex items-center gap-1 font-medium text-[#1D4ED8] hover:underline"
                  >
                    <span>Inspeccionar en 3D</span>
                    <ArrowRight className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Editorial Scrollytelling Section 1: Mathematical Formulation */}
      <section className="reveal-on-scroll border-t border-[#E2E5EA] bg-[#FFFFFF] py-16">
        <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
          <div className="max-w-2xl">
            <h2 className="font-serif-editorial text-3xl text-[#111827] sm:text-4xl">
              Rigor geométrico sin alucinaciones aritméticas
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-[#4B5563]">
              El modelo de lenguaje estima únicamente parámetros semánticos y poses 9-DOF estructuradas.
              Todas las transformaciones proyectivas, intersecciones SAT, rayos de oclusión Kay-Kajiya y
              métricas IoU se ejecutan en Python determinista.
            </p>
          </div>

          <div className="mt-10 grid gap-6 md:grid-cols-3">
            <article className="rounded-xl border border-[#E2E5EA] bg-[#F7F7F5] p-6">
              <div className="flex items-center gap-2 text-[#1D4ED8]">
                <Layers className="h-4 w-4" />
                <h3 className="text-sm font-semibold text-[#111827]">
                  Proyección Pinhole K
                </h3>
              </div>
              <p className="mt-2.5 text-xs leading-relaxed text-[#4B5563]">
                Reproyección instantánea al modificar el campo visual horizontal{' '}
                <span className="font-mono-tech text-[#111827]">FOV_h ∈ [40°, 110°]</span> sin re-invocar
                el modelo de visión.
              </p>
              <pre className="font-mono-tech mt-4 overflow-x-auto rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] p-3 text-[11px] leading-relaxed text-[#111827]">
{`f = W / (2 · tan(FOV_h / 2))
u = f · (X_cv / Z_cv) + W/2
v = f · (Y_cv / Z_cv) + H/2`}
              </pre>
            </article>

            <article className="rounded-xl border border-[#E2E5EA] bg-[#F7F7F5] p-6">
              <div className="flex items-center gap-2 text-[#0F766E]">
                <CheckCircle2 className="h-4 w-4" />
                <h3 className="text-sm font-semibold text-[#111827]">
                  Base Quiral M = Rₓ(−90°)
                </h3>
              </div>
              <p className="mt-2.5 text-xs leading-relaxed text-[#4B5563]">
                A diferencia de reflexiones ingenuas que invierten la quiralidad, nuestra rotación propia
                preserva orientación y ángulos de Euler ZYX en Three.js.
              </p>
              <pre className="font-mono-tech mt-4 overflow-x-auto rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] p-3 text-[11px] leading-relaxed text-[#111827]">
{`p_three = [ x,  z, -y ]
R_three = M · R_gem · Mᵀ
size    = [ dx, dz, dy ]`}
              </pre>
            </article>

            <article className="rounded-xl border border-[#E2E5EA] bg-[#F7F7F5] p-6">
              <div className="flex items-center gap-2 text-[#B45309]">
                <Ruler className="h-4 w-4" />
                <h3 className="text-sm font-semibold text-[#111827]">
                  Copiloto con Function Calling
                </h3>
              </div>
              <p className="mt-2.5 text-xs leading-relaxed text-[#4B5563]">
                Cada consulta métrica invoca funciones auditables: distancia centro-centro, holgura
                superficial SAT, matriz de oclusión y control directo de la cámara WebGL.
              </p>
              <pre className="font-mono-tech mt-4 overflow-x-auto rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] p-3 text-[11px] leading-relaxed text-[#111827]">
{`measure_distance(a, b)
analyze_occlusions_and_collisions()
set_camera_view("bev")`}
              </pre>
            </article>
          </div>
        </div>
      </section>

      {/* Editorial Scrollytelling Section 2: Implicit World Models vs Explicit 3D */}
      <section className="reveal-on-scroll border-t border-[#E2E5EA] bg-[#F7F7F5] py-16">
        <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
          <div className="grid gap-8 lg:grid-cols-12">
            <div className="lg:col-span-5">
              <h2 className="font-serif-editorial text-3xl text-[#111827] sm:text-4xl">
                World Models Implícitos vs. Percepción 3D Explícita
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-[#4B5563]">
                Los modelos de mundo generativos como Google Genie o Veo sintetizan dinámica visual en
                espacio latente de píxeles, pero no exponen coordenadas métricas ni garantizan invarianza
                de escala. MyLeenTo3 actúa como una capa explícita y auditable sobre secuencias RGB.
              </p>
              <button
                type="button"
                onClick={() => onNavigate('multiframe')}
                className="mt-5 inline-flex items-center gap-2 rounded-lg border border-[#111827] bg-[#FFFFFF] px-4 py-2 text-xs font-medium text-[#111827] transition-colors hover:bg-[#111827] hover:text-white"
              >
                <span>Explorar Auditoría Multi-Frame</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="lg:col-span-7">
              <div className="scroll-affordance-x rounded-xl border border-[#E2E5EA] bg-[#FFFFFF]">
                <table className="w-full min-w-[520px] border-collapse text-left text-xs">
                  <thead>
                    <tr className="border-b border-[#E2E5EA] bg-[#F7F7F5] text-[#4B5563]">
                      <th className="px-4 py-3 font-medium">Propiedad</th>
                      <th className="px-4 py-3 font-medium">World Models Implícitos (Genie / Veo)</th>
                      <th className="px-4 py-3 font-semibold text-[#1D4ED8]">
                        MyLeenTo3 (Explícito 9-DOF)
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E2E5EA]">
                    <tr>
                      <td className="px-4 py-3 font-medium text-[#111827]">Representación</td>
                      <td className="px-4 py-3 text-[#4B5563]">Tokens latentes / video RGB</td>
                      <td className="px-4 py-3 text-[#111827]">OBBs métricos [x,y,z,dx,dy,dz,r,p,y]</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-medium text-[#111827]">Medición física</td>
                      <td className="px-4 py-3 text-[#4B5563]">No expuesta directamente</td>
                      <td className="px-4 py-3 text-[#111827]">Distancias en metros, SAT y IoU 3D/BEV</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-medium text-[#111827]">Sensibilidad FOV</td>
                      <td className="px-4 py-3 text-[#4B5563]">Implícita en el entrenamiento</td>
                      <td className="px-4 py-3 text-[#111827]">Calibrable en tiempo real (40°–110°)</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 font-medium text-[#111827]">Auditoría temporal</td>
                      <td className="px-4 py-3 text-[#4B5563]">Susceptible a deriva de escala</td>
                      <td className="px-4 py-3 text-[#111827]">Cuantifica drift de volumen y jitter de yaw</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Editorial Scrollytelling Section 3: 3 Calibrated Benchmark Scenes */}
      <section className="reveal-on-scroll border-t border-[#E2E5EA] bg-[#FFFFFF] py-16">
        <div className="mx-auto max-w-[1360px] px-4 sm:px-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="font-serif-editorial text-3xl text-[#111827] sm:text-4xl">
                Tres bancos calibrados reproducibles (SEED = 42)
              </h2>
              <p className="mt-2 text-sm text-[#4B5563]">
                Selecciona un escenario para abrirlo directamente en el espacio sincronizado 2D ↔ 3D con Ground Truth exacto.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('metrics')}
              className="text-xs font-medium text-[#1D4ED8] hover:underline"
            >
              Ver comparativa multi-modelo →
            </button>
          </div>

          <div className="mt-8 grid gap-6 md:grid-cols-3">
            {benchmarks.map((scene) => (
              <div
                key={scene.id}
                className="group flex flex-col overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#F7F7F5] transition-colors hover:border-[#1D4ED8]/50"
              >
                <div className="aspect-[3/2] overflow-hidden border-b border-[#E2E5EA] bg-[#F0F2F5]">
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
                    onClick={() => onOpenLab(scene.id)}
                    className="mt-4 inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-3 py-2 text-xs font-medium text-[#111827] transition-colors hover:border-[#1D4ED8] hover:text-[#1D4ED8]"
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
