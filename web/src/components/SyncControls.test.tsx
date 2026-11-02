import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { DEFAULT_SYNC, saveSyncSettings, syncSource } from "../api/client";
import { SyncControls } from "./SyncControls";

vi.mock("../api/client", async (original) => ({
  ...await original<typeof import("../api/client")>(),
  syncSource: vi.fn(), saveSyncSettings: vi.fn(), pauseSync: vi.fn(),
}));

it("sends an inclusive custom range and persists the same defaults", async () => {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  vi.mocked(saveSyncSettings).mockResolvedValue({ settings: DEFAULT_SYNC });
  render(<QueryClientProvider client={client}><SyncControls status={{ job: null, initial_months: 6, settings: DEFAULT_SYNC, sources: ["arxiv"] }} error={null} /></QueryClientProvider>);
  fireEvent.click(screen.getByText("Sync options"));
  fireEvent.change(screen.getByLabelText("Publication window"), { target: { value: "range" } });
  fireEvent.change(screen.getByLabelText("From month year"), { target: { value: "2025" } });
  fireEvent.change(screen.getByLabelText("From month month"), { target: { value: "1" } });
  fireEvent.change(screen.getByLabelText("Through month year"), { target: { value: "2025" } });
  fireEvent.change(screen.getByLabelText("Through month month"), { target: { value: "6" } });
  fireEvent.change(screen.getByLabelText("Paper limit per run"), { target: { value: "10" } });
  fireEvent.click(screen.getByRole("button", { name: "Save defaults to arie.yaml" }));
  const selected = { mode: "range", initial_months: 6, start_month: "2025-01", end_month: "2025-06", max_papers: 10, batch_size: 20 };
  await waitFor(() => expect(saveSyncSettings).toHaveBeenCalledWith(selected));
  fireEvent.click(screen.getByRole("button", { name: "Sync papers" }));
  await waitFor(() => expect(syncSource).toHaveBeenCalledWith("all", selected, undefined));
  client.clear();
});
