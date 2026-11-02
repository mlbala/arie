import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, PlugZap, X } from "lucide-react";
import { useState } from "react";
import { getPluginSlots, setPlugin, testPlugin } from "../api/client";

interface SettingsDialogProps { onClose: () => void }
const SLOT_LABELS: Record<string, string> = {
  llm: "Models", vector_store: "Vector", database: "DB", embedder: "Embedder",
  source: "Source", file_storage: "Files", parser: "Parser",
};

export function SettingsDialog({ onClose }: SettingsDialogProps) {
  const queryClient = useQueryClient();
  const pluginsQuery = useQuery({ queryKey: ["plugins"], queryFn: getPluginSlots });
  const [activeSlot, setActiveSlot] = useState("llm");
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [models, setModels] = useState<Record<string, string>>({});
  const active = pluginsQuery.data?.plugins.find((item) => item.slot === activeSlot);
  const name = selections[activeSlot] ?? String(active?.configured.use ?? active?.available[0] ?? "");
  const provider = active?.providers?.find((item) => item.name === name);
  const model = models[name] ?? provider?.model ?? "";
  const llm = activeSlot === "llm";
  const missingKey = llm && provider?.key_configured === false;
  const currentModel = String(active?.configured.model ?? active?.providers?.find((item) => item.name === active?.configured.use)?.model ?? "");
  const unchanged = name === active?.configured.use && model === currentModel;
  const config = llm ? { model } : Object.fromEntries(
    Object.entries(active?.configured ?? {}).filter(([key]) => key !== "use"),
  );
  const test = useMutation({
    mutationFn: () => testPlugin(activeSlot, name, config),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["plugins"] }),
  });
  const save = useMutation({
    mutationFn: () => setPlugin(activeSlot, name, config),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["plugins"] }),
  });
  const reset = () => { test.reset(); save.reset(); };
  const busy = test.isPending || save.isPending;
  const error = test.error?.message ?? save.error?.message;
  return (
    <div className="modal-backdrop" onMouseDown={(event) => {
      if (event.target === event.currentTarget) onClose();
    }}>
      <section aria-labelledby="settings-title" aria-modal="true" className="settings-modal" role="dialog">
        <header className="modal-heading">
          <div><p className="eyebrow">Workspace configuration</p><h2 className="modal-title" id="settings-title">Settings</h2></div>
          <button aria-label="Close settings" className="icon-button" onClick={onClose} type="button"><X size={15} /></button>
        </header>
        <div className="settings-content">
          <p className="settings-copy">Choose your model provider. Saving verifies the connection and remembers the choice in arie.yaml.</p>
          {pluginsQuery.isPending ? <div className="loading-state">Loading configured plugins…</div> :
            pluginsQuery.isError ? <div className="error-state">{pluginsQuery.error.message}</div> : <>
              <div className="plugin-tabs" role="tablist" aria-label="Plugin slots">
                {pluginsQuery.data.plugins.map((item) => <button
                  aria-selected={activeSlot === item.slot} className={`plugin-tab${activeSlot === item.slot ? " active" : ""}`}
                  key={item.slot} disabled={busy} role="tab" type="button"
                  onClick={() => { setActiveSlot(item.slot); reset(); }}
                >{SLOT_LABELS[item.slot] ?? item.slot}</button>)}
              </div>
              <label className="field-label" htmlFor="plugin-select">{llm ? "Model provider" : `Active ${SLOT_LABELS[activeSlot] ?? activeSlot} provider`}</label>
              <select className="select-control" id="plugin-select" value={name} disabled={busy}
                onChange={(event) => { setSelections((current) => ({ ...current, [activeSlot]: event.target.value })); reset(); }}>
                {(llm ? active?.providers?.map((item) => ({ name: item.name, label: item.label })) : active?.available.map((item) => ({ name: item, label: item })))?.map((item) => <option key={item.name} value={item.name}>{item.label}</option>)}
                {!active?.available.length && <option value="">No providers available</option>}
              </select>
              {llm && <>
                <label className="field-label model-field" htmlFor="model-name">Model name</label>
                <input className="select-control" id="model-name" value={model} maxLength={128} disabled={busy}
                  onChange={(event) => { setModels((current) => ({ ...current, [name]: event.target.value })); reset(); }} />
                <div className={`provider-status${missingKey || provider?.status === "error" ? " error" : ""}`} role={missingKey ? "alert" : "status"}>
                  <strong>{missingKey ? "API key not configured" : test.isSuccess && !error ? "Connection verified" : model !== provider?.model ? "Connection not tested for this model" : provider?.status === "verified" ? "Connection verified" : provider?.status === "error" ? "Connection needs attention" : "API key configured · connection not tested"}</strong>
                  <p>{missingKey ? `${provider?.key_env} is not configured. Add it to the server's .env and restart ARIE.` : test.isSuccess ? "A live request to this provider and model succeeded." : provider?.message ?? "Provider credential information is unavailable. Restart the API."}</p>
                </div>
              </>}
              {!llm && <p className="settings-copy">Change this adapter in arie.yaml and restart ARIE to apply it.</p>}
              <div className="settings-actions">
                <button className="button" disabled={!name || busy || missingKey || (llm && !model.trim())} onClick={() => test.mutate()} type="button">
                  <PlugZap size={12} />{test.isPending ? "Testing…" : "Test connection"}
                </button>
                {llm && <button className="button button-primary" disabled={!name || !model.trim() || busy || missingKey || unchanged} onClick={() => save.mutate()} type="button">
                  <Check size={12} />{save.isPending ? "Checking and saving…" : "Use this model"}
                </button>}
              </div>
              {error && <p className="settings-feedback error" role="alert">{error}</p>}
              {!error && (save.isSuccess || test.isSuccess) && <p className="settings-feedback" role="status">{save.isSuccess ? "Model selection saved to arie.yaml." : "Connection test passed."}</p>}
            </>}
        </div>
      </section>
    </div>
  );
}
