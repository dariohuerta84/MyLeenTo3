import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Scale, Sparkles } from 'lucide-react';
import type { SceneIntelligence } from '../types/spatial';

interface SceneIntelligencePanelProps {
  intelligence?: SceneIntelligence | null;
}

const DISCOVERY_MODE_LABEL: Record<string, string> = {
  vlm_gemini: 'Híbrido VLM → YOLOE-OV → MoGe-2',
  local_yoloe_pf_adaptive: 'Local Autónomo YOLOE-PF 4585 → YOLOE-OV → MoGe-2',
};

const DIAGNOSIS_SOURCE_LABEL: Record<string, string> = {
  vlm_3d_grounded: 'LLM Multimodal + Telemetría 3D',
  deterministic_3d_grounded: 'Motor Determinista 3D',
};

function severityAccentBorder(severity: string): string {
  switch (severity) {
    case 'high':
      return 'border-l-[#BE123C]';
    case 'medium':
      return 'border-l-[#B45309]';
    case 'low':
      return 'border-l-[#0F766E]';
    default:
      return 'border-l-[#1D4ED8]';
  }
}

export const SceneIntelligencePanel: React.FC<SceneIntelligencePanelProps> = ({
  intelligence,
}) => {
  const [expanded, setExpanded] = useState(true);

  if (!intelligence || !intelligence.diagnosis) {
    return null;
  }

  const { diagnosis } = intelligence;
  const discoveryLabel =
    DISCOVERY_MODE_LABEL[intelligence.discovery_mode] ?? intelligence.discovery_mode;
  const sourceLabel =
    DIAGNOSIS_SOURCE_LABEL[diagnosis.diagnosis_source ?? 'deterministic_3d_grounded'] ??
    'Motor 3D';
  const scaleChecks = diagnosis.scale_prior_checks ?? [];
  const pfDiscoveries = intelligence.prompt_free_discoveries ?? [];
  const vocabList = intelligence.active_vocabulary ?? [];

  return (
    <section
      aria-label="Diagnóstico adaptativo de escena fundado en 3D"
      className="rounded-xl border border-[#E2E5EA] bg-[#FFFFFF]"
    >
      {/* Editorial Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h2 className="font-serif-editorial text-lg text-[#111827]">
              {diagnosis.diagnosis_title || intelligence.domain_label_es}
            </h2>
            <span className="font-mono-tech text-[11px] text-[#4B5563]">
              {discoveryLabel}
            </span>
          </div>
          <p className="mt-0.5 text-xs text-[#4B5563]">
            {intelligence.scene_context_es}
          </p>
        </div>

        <div className="flex items-center gap-4">
          <div className="font-mono-tech text-right text-[11px] text-[#4B5563]">
            <div>
              Fuente: <strong className="text-[#111827]">{sourceLabel}</strong>
            </div>
            <div>
              Confianza:{' '}
              <strong
                className={
                  diagnosis.confidence_level === 'alta'
                    ? 'text-[#0F766E]'
                    : diagnosis.confidence_level === 'media'
                      ? 'text-[#B45309]'
                      : 'text-[#BE123C]'
                }
              >
                {diagnosis.confidence_level.toUpperCase()}
              </strong>
              {typeof intelligence.vlm_guided_count === 'number' &&
                intelligence.vlm_guided_count > 0 && (
                  <span className="ml-2 text-[#1D4ED8]">
                    · {intelligence.vlm_guided_count} guías VLM
                  </span>
                )}
            </div>
          </div>

          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="inline-flex items-center gap-1 border border-[#E2E5EA] bg-[#F7F7F5] px-2.5 py-1.5 text-[11px] font-medium text-[#111827] hover:bg-[#FFFFFF]"
            aria-expanded={expanded}
          >
            {expanded ? (
              <>
                <ChevronDown className="h-3.5 w-3.5" />
                <span>Contraer</span>
              </>
            ) : (
              <>
                <ChevronRight className="h-3.5 w-3.5" />
                <span>Expandir</span>
              </>
            )}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-[#E2E5EA]">
          {/* Executive Summary + Prediction */}
          <div className="grid gap-4 px-4 py-3.5 lg:grid-cols-12">
            <div className="lg:col-span-7 space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-[#4B5563]">
                Síntesis y Predicción Espacial
              </div>
              <p className="text-xs leading-relaxed text-[#111827]">
                {diagnosis.executive_summary}
              </p>
              <p className="border-l-2 border-l-[#1D4ED8] pl-3 text-xs leading-relaxed text-[#4B5563]">
                {diagnosis.model_predictions_es}
              </p>
            </div>

            {/* Active Open-Vocab & PF Discovery Readout */}
            <div className="lg:col-span-5 border-t border-[#E2E5EA] pt-3 lg:border-l lg:border-t-0 lg:pl-4 lg:pt-0 space-y-2">
              <div className="flex items-center justify-between text-[11px] font-semibold uppercase tracking-wider text-[#4B5563]">
                <span className="inline-flex items-center gap-1.5">
                  <Sparkles className="h-3 w-3 text-[#1D4ED8]" />
                  Vocabulario Open-Vocab Activo
                </span>
                <span className="font-mono-tech font-normal text-[#111827]">
                  {vocabList.length} clases
                </span>
              </div>
              <p className="font-mono-tech text-[11px] leading-relaxed text-[#111827]">
                {vocabList.length > 0 ? vocabList.join(' · ') : 'Vocabulario estándar'}
              </p>

              {pfDiscoveries.length > 0 && (
                <div className="border-t border-[#E2E5EA] pt-2">
                  <span className="text-[11px] font-medium text-[#4B5563]">
                    Pistas Prompt-Free (4,585 cat.):{' '}
                  </span>
                  <span className="font-mono-tech text-[11px] text-[#0F766E]">
                    {pfDiscoveries
                      .slice(0, 5)
                      .map((d) => `${d.label} (${(d.confidence * 100).toFixed(0)}%)`)
                      .join(' · ')}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Grounded Key Findings */}
          {diagnosis.key_findings && diagnosis.key_findings.length > 0 && (
            <div className="border-t border-[#E2E5EA]">
              <div className="bg-[#F7F7F5] px-4 py-2 text-[11px] font-semibold uppercase tracking-wider text-[#4B5563]">
                Hallazgos Geométricos Fundados en 3D
              </div>
              <div className="divide-y divide-[#E2E5EA]">
                {diagnosis.key_findings.map((finding, idx) => (
                  <div
                    key={`${finding.title}-${idx}`}
                    className={`grid gap-2 border-l-2 px-4 py-2.5 sm:grid-cols-12 sm:items-baseline ${severityAccentBorder(
                      finding.severity
                    )}`}
                  >
                    <div className="sm:col-span-4 text-xs font-semibold text-[#111827]">
                      {finding.title}
                    </div>
                    <div className="font-mono-tech sm:col-span-3 text-[11px] font-medium text-[#1D4ED8]">
                      {finding.metric_evidence}
                    </div>
                    <div className="sm:col-span-5 text-xs leading-relaxed text-[#4B5563]">
                      {finding.interpretation}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Scale Priors Consistency + Recommended Actions */}
          <div className="grid border-t border-[#E2E5EA] lg:grid-cols-12">
            <div className="px-4 py-3 lg:col-span-7 space-y-2">
              <div className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-[#4B5563]">
                <Scale className="h-3.5 w-3.5 text-[#0F766E]" />
                <span>Verificación contra Priors de Escala Física</span>
              </div>
              <p className="text-xs text-[#111827]">{diagnosis.scale_consistency_note}</p>

              {scaleChecks.length > 0 && (
                <div className="overflow-x-auto border-t border-[#E2E5EA] pt-2">
                  <table className="w-full text-left text-[11px]">
                    <thead>
                      <tr className="border-b border-[#E2E5EA] text-[#4B5563]">
                        <th className="py-1 pr-3 font-medium">Objeto</th>
                        <th className="py-1 pr-3 font-medium">Medido máx.</th>
                        <th className="py-1 pr-3 font-medium">Prior típico</th>
                        <th className="py-1 pr-3 font-medium">Ratio</th>
                        <th className="py-1 font-medium">Estado</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono-tech divide-y divide-[#E2E5EA]">
                      {scaleChecks.map((chk) => (
                        <tr key={`${chk.object_id}-${chk.label}`}>
                          <td className="py-1 pr-3 font-sans text-[#111827]">{chk.label}</td>
                          <td className="py-1 pr-3 text-[#111827]">
                            {chk.measured_max_m.toFixed(2)} m
                          </td>
                          <td className="py-1 pr-3 text-[#4B5563]">
                            {chk.expected_typical_m.toFixed(2)} m
                          </td>
                          <td className="py-1 pr-3 text-[#1D4ED8]">{chk.ratio.toFixed(2)}×</td>
                          <td
                            className={`py-1 font-semibold ${
                              chk.status === 'consistente'
                                ? 'text-[#0F766E]'
                                : 'text-[#B45309]'
                            }`}
                          >
                            {chk.status}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="border-t border-[#E2E5EA] px-4 py-3 lg:col-span-5 lg:border-l lg:border-t-0 space-y-2">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-[#4B5563]">
                Acciones Recomendadas
              </div>
              <ul className="space-y-1.5 text-xs leading-relaxed text-[#4B5563]">
                {(diagnosis.recommended_actions ?? []).map((act, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="font-mono-tech text-[11px] text-[#1D4ED8]">
                      0{i + 1}
                    </span>
                    <span>{act}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </section>
  );
};
