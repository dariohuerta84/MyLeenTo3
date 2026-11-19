import React, { useRef, useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Cpu,
  Send,
  Sparkles,
  Terminal,
  Trash2,
} from 'lucide-react';
import type {
  ChatMessage,
  CopilotToolTrace,
  GeminiModelId,
  SceneAction,
  SpatialObject,
} from '../types/spatial';
import { sendCopilotMessage } from '../services/api';

interface SpatialCopilotPanelProps {
  objects: SpatialObject[];
  gtObjects?: SpatialObject[] | null;
  imageBase64?: string;
  width: number;
  height: number;
  fovHDeg: number;
  selectedModel: GeminiModelId;
  apiKey: string;
  onExecuteSceneActions: (actions: SceneAction[]) => void;
}

const QUICK_PROMPTS = [
  'Mide la distancia entre los 2 objetos más cercanos',
  '¿Conviene escalar el modelo 3D o es confiable esta escena?',
  'Analiza oclusiones y colisiones',
  'Cambia a vista BEV y enfoca el objeto más lejano',
  'Ejecuta barrido de sensibilidad FOV',
];

function renderFormattedInline(text: string): React.ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      return (
        <strong key={i} className="font-semibold text-[#111827]">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
      return (
        <code
          key={i}
          className="font-mono-tech rounded border border-[#E2E5EA] bg-[#F7F7F5] px-1 py-0.5 text-[11px] text-[#1D4ED8]"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return <React.Fragment key={i}>{part}</React.Fragment>;
  });
}

const ToolTraceBlock: React.FC<{ trace: CopilotToolTrace[] }> = ({ trace }) => {
  const [open, setOpen] = useState(false);

  if (!trace || trace.length === 0) return null;

  return (
    <div className="mt-2 rounded-lg border border-[#E2E5EA] bg-[#F7F7F5]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-2.5 py-1.5 text-left text-[11px] font-medium text-[#4B5563] hover:text-[#111827]"
      >
        <span className="inline-flex items-center gap-1.5">
          <Terminal className="h-3 w-3 text-[#1D4ED8]" />
          <span>
            Traza determinista ({trace.length}{' '}
            {trace.length === 1 ? 'función' : 'funciones'})
          </span>
        </span>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
      </button>

      {open && (
        <div className="space-y-2 border-t border-[#E2E5EA] p-2.5">
          {trace.map((item, idx) => {
            const fnName = item.name || item.tool_name || 'tool';
            return (
              <div
                key={`${fnName}-${idx}`}
                className="rounded border border-[#E2E5EA] bg-[#FFFFFF] p-2"
              >
                <div className="font-mono-tech text-[11px] font-semibold text-[#1D4ED8]">
                  {fnName}()
                </div>
                <div className="mt-1 grid gap-1.5 sm:grid-cols-2">
                  <div>
                    <span className="text-[10px] font-medium text-[#4B5563]">
                      Argumentos JSON
                    </span>
                    <pre className="font-mono-tech mt-0.5 max-h-28 overflow-auto rounded bg-[#F7F7F5] p-1.5 text-[10px] text-[#111827]">
                      {JSON.stringify(item.arguments ?? {}, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <span className="text-[10px] font-medium text-[#4B5563]">
                      Salida Python
                    </span>
                    <pre className="font-mono-tech mt-0.5 max-h-28 overflow-auto rounded bg-[#F7F7F5] p-1.5 text-[10px] text-[#0F766E]">
                      {JSON.stringify(item.result ?? {}, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export const SpatialCopilotPanel: React.FC<SpatialCopilotPanelProps> = ({
  objects,
  gtObjects,
  imageBase64,
  width,
  height,
  fovHDeg,
  selectedModel,
  apiKey,
  onExecuteSceneActions,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        'Copiloto Espacial listo. Puedo medir distancias métricas y holguras SAT, auditar oclusiones, filtrar objetos por profundidad o controlar la cámara 3D mediante herramientas deterministas.',
      mode: 'ready',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [interactionId, setInteractionId] = useState<string | null>(null);
  const scrollEndRef = useRef<HTMLDivElement>(null);

  const sendPrompt = async (promptText: string) => {
    const trimmed = promptText.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
      }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await sendCopilotMessage(
        {
          message: trimmed,
          objects,
          gt_objects: gtObjects,
          image_base64: imageBase64,
          width,
          height,
          fov_h_deg: fovHDeg,
          model: selectedModel,
          previous_interaction_id: interactionId,
        },
        apiKey
      );

      if (res.interaction_id) {
        setInteractionId(res.interaction_id);
      }

      const assistantMsg: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: res.reply,
        mode: res.mode,
        tool_trace: res.tool_trace,
        scene_actions: res.scene_actions,
        timestamp: new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
        }),
      };

      setMessages((prev) => [...prev, assistantMsg]);

      if (res.scene_actions && res.scene_actions.length > 0) {
        onExecuteSceneActions(res.scene_actions);
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : 'Error al consultar copiloto';
      setMessages((prev) => [
        ...prev,
        {
          id: `error-${Date.now()}`,
          role: 'assistant',
          content: `No se pudo completar la consulta: ${errMsg}`,
          mode: 'error',
          timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
          }),
        },
      ]);
    } finally {
      setLoading(false);
      window.setTimeout(() => {
        scrollEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }, 40);
    }
  };

  return (
    <section
      aria-label="Copiloto Espacial Determinista"
      className="flex flex-col overflow-hidden rounded-xl border border-[#E2E5EA] bg-[#FFFFFF]"
    >
      {/* Header */}
      <div className="flex items-center justify-between gap-2 border-b border-[#E2E5EA] bg-[#F7F7F5] px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-[#1D4ED8]" />
          <div>
            <h2 className="text-xs font-semibold text-[#111827]">
              Copiloto Espacial · Function Calling
            </h2>
          </div>
          <span className="font-mono-tech rounded border border-[#E2E5EA] bg-[#FFFFFF] px-1.5 py-0.5 text-[10px] text-[#4B5563]">
            {selectedModel}
          </span>
        </div>

        <button
          type="button"
          onClick={() => {
            setMessages((prev) => prev.slice(0, 1));
            setInteractionId(null);
          }}
          title="Reiniciar conversación"
          className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] text-[#4B5563] hover:bg-[#FFFFFF] hover:text-[#111827]"
        >
          <Trash2 className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Limpiar</span>
        </button>
      </div>

      {/* Quick prompt chips */}
      <div className="flex flex-wrap gap-1.5 border-b border-[#E2E5EA] bg-[#FFFFFF] px-4 py-2.5">
        {QUICK_PROMPTS.map((chip) => (
          <button
            key={chip}
            type="button"
            disabled={loading || objects.length === 0}
            onClick={() => sendPrompt(chip)}
            className="inline-flex items-center gap-1 rounded-full border border-[#E2E5EA] bg-[#F7F7F5] px-2.5 py-1 text-[11px] text-[#111827] transition-colors hover:border-[#1D4ED8] hover:text-[#1D4ED8] disabled:opacity-50"
          >
            <Sparkles className="h-3 w-3 text-[#1D4ED8]" />
            <span>{chip}</span>
          </button>
        ))}
      </div>

      {/* Message log */}
      <div className="scroll-affordance-y max-h-[290px] min-h-[200px] flex-1 space-y-3 p-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${
              msg.role === 'user' ? 'items-end' : 'items-start'
            }`}
          >
            <div
              className={`max-w-[92%] rounded-xl px-3.5 py-2.5 text-xs leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-[#111827] text-white'
                  : 'border border-[#E2E5EA] bg-[#FFFFFF] text-[#111827]'
              }`}
            >
              <p className="whitespace-pre-wrap">
                {msg.role === 'assistant'
                  ? renderFormattedInline(msg.content)
                  : msg.content}
              </p>

              {msg.scene_actions && msg.scene_actions.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {msg.scene_actions.map((act, i) => (
                    <span
                      key={i}
                      className="font-mono-tech rounded bg-[#0F766E]/10 px-1.5 py-0.5 text-[10px] font-medium text-[#0F766E]"
                    >
                      acción: {act.type}
                    </span>
                  ))}
                </div>
              )}

              {msg.tool_trace && <ToolTraceBlock trace={msg.tool_trace} />}
            </div>

            <span className="font-mono-tech mt-1 text-[10px] text-[#9CA3AF]">
              {msg.role === 'user' ? 'Investigador' : `Copiloto (${msg.mode ?? 'demo'})`} ·{' '}
              {msg.timestamp}
            </span>
          </div>
        ))}
        {loading && (
          <div className="font-mono-tech text-xs text-[#4B5563]">
            Ejecutando herramientas geométricas en Python…
          </div>
        )}
        <div ref={scrollEndRef} />
      </div>

      {/* Composer */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendPrompt(input);
        }}
        className="flex items-center gap-2 border-t border-[#E2E5EA] bg-[#F7F7F5] px-3.5 py-2.5"
      >
        <label htmlFor="copilot-prompt-input" className="sr-only">
          Consulta espacial al copiloto
        </label>
        <input
          id="copilot-prompt-input"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
          placeholder="Pregunta sobre distancias, oclusiones, filtros o vistas 3D…"
          className="flex-1 rounded-lg border border-[#E2E5EA] bg-[#FFFFFF] px-3 py-2 text-xs text-[#111827] placeholder:text-[#9CA3AF] focus:border-[#1D4ED8] focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="inline-flex items-center gap-1.5 rounded-lg bg-[#1D4ED8] px-3.5 py-2 text-xs font-medium text-white transition-colors hover:bg-[#1E40AF] disabled:opacity-50"
        >
          <Send className="h-3.5 w-3.5" />
          <span>Enviar</span>
        </button>
      </form>
    </section>
  );
};
