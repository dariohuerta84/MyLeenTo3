import React from 'react';
import { BarChart3, Box, Compass, Film, KeyRound, Orbit } from 'lucide-react';
import type { AppSection, GeminiModelId } from '../types/spatial';

interface HeaderNavProps {
  activeSection: AppSection;
  onNavigate: (section: AppSection) => void;
  selectedModel: GeminiModelId;
  onSelectModel: (model: GeminiModelId) => void;
  hasApiKey: boolean;
  backendEnvConfigured: boolean;
  onOpenApiKeyModal: () => void;
}

const NAV_ITEMS: Array<{
  id: AppSection;
  label: string;
  icon: React.FC<{ className?: string }>;
}> = [
  { id: 'landing', label: 'Visión Pinhole', icon: Compass },
  { id: 'lab', label: 'Laboratorio 3D', icon: Box },
  { id: 'multiview', label: 'Reconstrucción 360°', icon: Orbit },
  { id: 'multiframe', label: 'Auditoría Multi-Frame', icon: Film },
  { id: 'metrics', label: 'Métricas', icon: BarChart3 },
];

const MODEL_OPTIONS: Array<{ id: GeminiModelId; label: string }> = [
  { id: 'gemini-4-hydrogen', label: 'gemini-4-hydrogen' },
  { id: 'gemini-4-argon', label: 'gemini-4-argon' },
  { id: 'gemini-3.5-flash-lite', label: 'gemini-3.5-flash-lite' },
];

export const HeaderNav: React.FC<HeaderNavProps> = ({
  activeSection,
  onNavigate,
  selectedModel,
  onSelectModel,
  hasApiKey,
  backendEnvConfigured,
  onOpenApiKeyModal,
}) => {
  const liveReady = hasApiKey || backendEnvConfigured;

  return (
    <header className="optical-header-bar sticky top-0 z-40 border-b border-[#E2E5EA] bg-[#FFFFFF]/95 backdrop-blur-sm">
      <div className="optical-header-inner mx-auto flex h-14 max-w-[1600px] items-center justify-between gap-3 px-4 sm:px-6">
        {/* Wordmark */}
        <button
          type="button"
          onClick={() => onNavigate('landing')}
          className="flex items-center gap-2.5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-[#1D4ED8] rounded-md"
        >
          <span
            aria-hidden="true"
            className="flex h-7 w-7 items-center justify-center rounded-md border border-[#111827] bg-[#F7F7F5]"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 stroke-[#111827]" fill="none" strokeWidth="1.75">
              <path d="M12 3L4 7.5v9L12 21l8-4.5v-9L12 3z" />
              <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" stroke="#1D4ED8" />
            </svg>
          </span>
          <span className="flex flex-col leading-none">
            <span className="font-serif-editorial text-base font-semibold tracking-tight text-[#111827]">
              MyLeenTo3
            </span>
            <span className="font-mono-tech text-[10px] text-[#4B5563]">
              9-DOF Optical Lab
            </span>
          </span>
        </button>

        {/* Navigation tabs */}
        <nav aria-label="Navegación principal" className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
            const active = activeSection === id;
            return (
              <button
                key={id}
                type="button"
                aria-current={active ? 'page' : undefined}
                onClick={() => onNavigate(id)}
                className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#1D4ED8] ${
                  active
                    ? 'bg-[#111827] text-white'
                    : 'text-[#4B5563] hover:bg-[#F7F7F5] hover:text-[#111827]'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span>{label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right controls: model selector + API key + quick CTA */}
        <div className="flex items-center gap-2">
          <div className="hidden sm:flex items-center gap-1.5 rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] px-2.5 py-1">
            <label htmlFor="header-model-select" className="sr-only">
              Modelo Gemini
            </label>
            <span className="text-[11px] text-[#4B5563]">Modelo</span>
            <select
              id="header-model-select"
              value={selectedModel}
              onChange={(e) => onSelectModel(e.target.value as GeminiModelId)}
              className="font-mono-tech bg-transparent text-xs font-medium text-[#111827] focus:outline-none cursor-pointer"
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            onClick={onOpenApiKeyModal}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-[#1D4ED8] ${
              liveReady
                ? 'border-[#0F766E]/35 bg-[#0F766E]/[0.06] text-[#0F766E]'
                : 'border-[#E2E5EA] bg-[#FFFFFF] text-[#4B5563] hover:bg-[#F7F7F5] hover:text-[#111827]'
            }`}
          >
            <KeyRound className="h-3.5 w-3.5" />
            <span className="hidden lg:inline">
              {hasApiKey ? 'Key en RAM' : backendEnvConfigured ? 'Env Activa' : 'API Key'}
            </span>
          </button>

          {activeSection === 'landing' && (
            <button
              type="button"
              onClick={() => onNavigate('lab')}
              className="rounded-lg bg-[#1D4ED8] px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-[#1E40AF] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#1D4ED8]"
            >
              Abrir Laboratorio 3D
            </button>
          )}
        </div>
      </div>

      {/* Mobile navigation strip */}
      <nav
        aria-label="Navegación móvil"
        className="flex md:hidden items-center gap-1 overflow-x-auto border-t border-[#E2E5EA] px-3 py-1.5"
      >
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => {
          const active = activeSection === id;
          return (
            <button
              key={id}
              type="button"
              aria-current={active ? 'page' : undefined}
              onClick={() => onNavigate(id)}
              className={`flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium ${
                active
                  ? 'bg-[#111827] text-white'
                  : 'text-[#4B5563] hover:bg-[#F7F7F5]'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{label}</span>
            </button>
          );
        })}
      </nav>
    </header>
  );
};
