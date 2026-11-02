import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Check, PlugZap, X } from 'lucide-react'
import { useState } from 'react'
import { getPluginSlots, setPlugin, testPlugin } from '../api/client'

interface SettingsDialogProps {
  onClose: () => void
}

const SLOT_LABELS: Record<string, string> = {
  llm: 'LLM',
  vector_store: 'Vector',
  database: 'DB',
  embedder: 'Embedder',
  source: 'Source',
  file_storage: 'Files',
  parser: 'Parser',
}

export function SettingsDialog({ onClose }: SettingsDialogProps) {
  const queryClient = useQueryClient()
  const pluginsQuery = useQuery({
    queryKey: ['plugins'],
    queryFn: getPluginSlots,
  })
  const [activeSlot, setActiveSlot] = useState('llm')
  const [pluginSelections, setPluginSelections] = useState<Record<string, string>>({})
  const activePlugin = pluginsQuery.data?.plugins.find((plugin) => plugin.slot === activeSlot)
  const configuredPlugin = activePlugin?.configured.use
  const pluginName =
    pluginSelections[activeSlot] ??
    (typeof configuredPlugin === 'string' ? configuredPlugin : activePlugin?.available[0] ?? '')

  const pluginConfig = Object.fromEntries(
    Object.entries(activePlugin?.configured ?? {}).filter(([key]) => key !== 'use'),
  )
  const testMutation = useMutation({
    mutationFn: () => testPlugin(activeSlot, pluginName, pluginConfig),
  })
  const switchMutation = useMutation({
    mutationFn: () => setPlugin(activeSlot, pluginName, pluginConfig),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['plugins'] })
    },
  })

  const feedback = switchMutation.error?.message ?? testMutation.error?.message
  const hasError = Boolean(switchMutation.error || testMutation.error)

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <section
        aria-labelledby="settings-title"
        aria-modal="true"
        className="settings-modal"
        role="dialog"
      >
        <header className="modal-heading">
          <div>
            <p className="eyebrow">Workspace configuration</p>
            <h2 className="modal-title" id="settings-title">Settings</h2>
          </div>
          <button aria-label="Close settings" className="icon-button" onClick={onClose} type="button">
            <X size={15} />
          </button>
        </header>
        <div className="settings-content">
          <p className="settings-copy">
            Test a connection before switching its active slot. Changes take effect immediately.
          </p>
          {pluginsQuery.isPending ? (
            <div className="loading-state">Loading configured plugins…</div>
          ) : pluginsQuery.isError ? (
            <div className="error-state">{pluginsQuery.error.message}</div>
          ) : (
            <>
              <div className="plugin-tabs" role="tablist" aria-label="Plugin slots">
                {pluginsQuery.data.plugins.map((plugin) => (
                  <button
                    aria-selected={activeSlot === plugin.slot}
                    className={`plugin-tab${activeSlot === plugin.slot ? ' active' : ''}`}
                    key={plugin.slot}
                    onClick={() => {
                      setActiveSlot(plugin.slot)
                      testMutation.reset()
                      switchMutation.reset()
                    }}
                    role="tab"
                    type="button"
                  >
                    {SLOT_LABELS[plugin.slot] ?? plugin.slot}
                  </button>
                ))}
              </div>
              <label className="field-label" htmlFor="plugin-select">
                Active {SLOT_LABELS[activeSlot] ?? activeSlot} provider
              </label>
              <select
                className="select-control"
                id="plugin-select"
                onChange={(event) => {
                  setPluginSelections((current) => ({
                    ...current,
                    [activeSlot]: event.target.value,
                  }))
                  testMutation.reset()
                  switchMutation.reset()
                }}
                value={pluginName}
              >
                {activePlugin?.available.length === 0 && (
                  <option value="">No providers available</option>
                )}
                {activePlugin?.available.map((plugin) => (
                  <option key={plugin} value={plugin}>{plugin}</option>
                ))}
                {pluginName &&
                  !activePlugin?.available.includes(pluginName) && (
                    <option value={pluginName}>{pluginName} (configured)</option>
                  )}
              </select>
              <div className="settings-actions">
                <button
                  className="button"
                  disabled={!pluginName || testMutation.isPending || switchMutation.isPending}
                  onClick={() => testMutation.mutate()}
                  type="button"
                >
                  <PlugZap size={12} />
                  {testMutation.isPending ? 'Testing…' : 'Test connection'}
                </button>
                <button
                  className="button button-primary"
                  disabled={
                    !pluginName ||
                    switchMutation.isPending ||
                    pluginName === activePlugin?.configured.use
                  }
                  onClick={() => switchMutation.mutate()}
                  type="button"
                >
                  <Check size={12} />
                  {switchMutation.isPending ? 'Switching…' : 'Hot-switch'}
                </button>
              </div>
              <div className={`settings-feedback${hasError ? ' error' : ''}`} role="status">
                {switchMutation.isSuccess
                  ? `${SLOT_LABELS[activeSlot] ?? activeSlot} slot switched to ${pluginName}.`
                  : testMutation.isSuccess
                    ? `${pluginName} connection test passed.`
                    : feedback ?? ''}
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  )
}
