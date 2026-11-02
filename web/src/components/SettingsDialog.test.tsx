import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { getPluginSlots, setPlugin, testPlugin, type PluginResponse } from "../api/client";
import { SettingsDialog } from "./SettingsDialog";

vi.mock("../api/client", () => ({ getPluginSlots: vi.fn(), setPlugin: vi.fn(), testPlugin: vi.fn() }));
const options: PluginResponse = {
  plugins: [{ slot: "llm", configured: { use: "deepseek", model: "deepseek-chat" },
    available: ["deepseek", "openai", "gemini"], providers: [
      { name: "deepseek", label: "DeepSeek", model: "deepseek-chat", key_env: "DEEPSEEK_API_KEY", key_configured: false, status: "missing_key", message: "Key missing" },
      { name: "openai", label: "OpenAI", model: "gpt-4.1-mini", key_env: "OPENAI_API_KEY", key_configured: true, status: "not_tested", message: "Connection has not been tested." },
      { name: "gemini", label: "Gemini", model: "gemini-3.5-flash-lite", key_env: "GEMINI_API_KEY", key_configured: false, status: "missing_key", message: "Key missing" },
    ] }],
};

function mount() {
  vi.mocked(getPluginSlots).mockResolvedValue(options);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}><SettingsDialog onClose={vi.fn()} /></QueryClientProvider>);
  return client;
}

it("shows missing-key errors immediately and uses provider-specific model defaults", async () => {
  const client = mount();
  const provider = await screen.findByLabelText("Model provider");
  expect(screen.getByRole("alert")).toHaveTextContent("DEEPSEEK_API_KEY is not configured");
  expect(screen.getByRole("button", { name: "Test connection" })).toBeDisabled();
  fireEvent.change(provider, { target: { value: "gemini" } });
  expect(screen.getByLabelText("Model name")).toHaveValue("gemini-3.5-flash-lite");
  expect(screen.getByRole("alert")).toHaveTextContent("GEMINI_API_KEY is not configured");
  fireEvent.change(provider, { target: { value: "openai" } });
  expect(screen.getByLabelText("Model name")).toHaveValue("gpt-4.1-mini");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  vi.mocked(testPlugin).mockResolvedValue({ ok: true });
  fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
  await screen.findByText("Connection test passed.");
  expect(testPlugin).toHaveBeenCalledWith("llm", "openai", { model: "gpt-4.1-mini" });
  vi.mocked(setPlugin).mockResolvedValue({ slot: "llm", plugin: "openai" });
  fireEvent.click(screen.getByRole("button", { name: "Use this model" }));
  await waitFor(() => expect(setPlugin).toHaveBeenCalledWith("llm", "openai", { model: "gpt-4.1-mini" }));
  expect(await screen.findByText("Model selection saved to arie.yaml.")).toBeInTheDocument();
  client.clear();
});

it("shows rejected credentials without claiming the connection passed", async () => {
  const client = mount();
  fireEvent.change(await screen.findByLabelText("Model provider"), { target: { value: "openai" } });
  vi.mocked(testPlugin).mockRejectedValue(new Error("OpenAI rejected the API key (HTTP 401). Check OPENAI_API_KEY."));
  fireEvent.click(screen.getByRole("button", { name: "Test connection" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("HTTP 401");
  expect(screen.queryByText("Connection test passed.")).not.toBeInTheDocument();
  client.clear();
});
