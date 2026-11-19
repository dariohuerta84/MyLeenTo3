import React, { useEffect, useRef, useState } from 'react';
import { Eye, EyeOff, KeyRound, ShieldCheck, X } from 'lucide-react';

interface ApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  apiKey: string;
  onSaveApiKey: (key: string) => void;
  backendEnvConfigured: boolean;
}

export const ApiKeyModal: React.FC<ApiKeyModalProps> = ({
  isOpen,
  onClose,
  apiKey,
  onSaveApiKey,
  backendEnvConfigured,
}) => {
  const [draft, setDraft] = useState(apiKey);
  const [revealed, setRevealed] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setDraft(apiKey);
  }, [apiKey, isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const timer = window.setTimeout(() => inputRef.current?.focus(), 30);
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('keydown', onKeyDown);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSaveApiKey(draft.trim());
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-[#111827]/35 px-4 backdrop-blur-[2px]"
      role="dialog"
      aria-modal="true"
      aria-labelledby="apikey-modal-title"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-lg rounded-xl border border-[#E2E5EA] bg-[#FFFFFF] p-6">
        <div className="flex items-start justify-between gap-4 border-b border-[#E2E5EA] pb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] text-[#1D4ED8]">
              <KeyRound className="h-4 w-4" />
            </div>
            <div>
              <h2
                id="apikey-modal-title"
                className="text-base font-semibold text-[#111827]"
              >
                Credencial Google AI Studio (Opcional)
              </h2>
              <p className="text-xs text-[#4B5563]">
                Habilita inferencia en vivo con Gemini Interactions API v1
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Cerrar modal de API Key"
            className="rounded-lg p-1.5 text-[#4B5563] transition-colors hover:bg-[#F7F7F5] hover:text-[#111827]"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          <div className="rounded-lg border border-[#0F766E]/25 bg-[#0F766E]/[0.04] p-3.5 text-xs leading-relaxed text-[#111827]">
            <div className="flex items-center gap-2 font-medium text-[#0F766E]">
              <ShieldCheck className="h-4 w-4 shrink-0" />
              <span>Solo en memoria RAM — Nunca persistida en disco</span>
            </div>
            <p className="mt-1.5 text-[#4B5563]">
              La clave se mantiene únicamente en el estado de React de esta pestaña y se envía en la cabecera HTTP{' '}
              <code className="font-mono-tech rounded bg-[#FFFFFF] px-1 py-0.5 text-[11px] text-[#111827] border border-[#E2E5EA]">
                X-Gemini-API-Key
              </code>
              . No se guarda en localStorage, cookies ni registros del servidor.
            </p>
          </div>

          {backendEnvConfigured && (
            <p className="text-xs text-[#0F766E] font-medium">
              El backend ya detectó una variable de entorno GEMINI_API_KEY activa. Puedes dejar este campo vacío para usarla o sobrescribirla temporalmente.
            </p>
          )}

          <div>
            <label
              htmlFor="gemini-api-key-input"
              className="block text-xs font-medium text-[#111827]"
            >
              Google AI Studio API Key
            </label>
            <div className="relative mt-1.5">
              <input
                ref={inputRef}
                id="gemini-api-key-input"
                type={revealed ? 'text' : 'password'}
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                placeholder="AIzaSy..."
                autoComplete="off"
                spellCheck={false}
                className="font-mono-tech w-full rounded-lg border border-[#E2E5EA] bg-[#F7F7F5] px-3 py-2.5 pr-10 text-xs text-[#111827] placeholder:text-[#9CA3AF] focus:border-[#1D4ED8] focus:bg-[#FFFFFF] focus:outline-none"
              />
              <button
                type="button"
                onClick={() => setRevealed((v) => !v)}
                aria-label={revealed ? 'Ocultar clave' : 'Mostrar clave'}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded p-1 text-[#4B5563] hover:text-[#111827]"
              >
                {revealed ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <p className="mt-1.5 text-[11px] text-[#4B5563]">
              Sin clave configurada, MyLeenTo3 opera en modo <span className="font-mono-tech text-[#111827]">calibrated_demo</span> con los 3 benchmarks reproducibles (SEED = 42) y el copiloto geométrico determinista.
            </p>
          </div>

          <div className="flex items-center justify-between gap-3 border-t border-[#E2E5EA] pt-4">
            <button
              type="button"
              onClick={() => {
                setDraft('');
                onSaveApiKey('');
                onClose();
              }}
              className="rounded-lg border border-[#E2E5EA] px-3 py-2 text-xs font-medium text-[#4B5563] transition-colors hover:bg-[#F7F7F5] hover:text-[#111827]"
            >
              Limpiar clave
            </button>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg px-3 py-2 text-xs font-medium text-[#4B5563] hover:text-[#111827]"
              >
                Cancelar
              </button>
              <button
                type="submit"
                className="rounded-lg bg-[#1D4ED8] px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-[#1E40AF]"
              >
                Guardar en memoria
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
};
