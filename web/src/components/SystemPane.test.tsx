import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { deletePaper, getStorageHealth, type PaperSummary } from "../api/client";
import { SelectedDocument, StorageHealth } from "./SystemPane";

vi.mock("../api/client", () => ({ deletePaper: vi.fn(), getStorageHealth: vi.fn() }));

const paper: PaperSummary = {
  id: "selected", title: "LLM agents", abstract: "Research", source: "arxiv",
  status: "INDEXED", published_at: "2025-06-01", relevance_gate: true,
  overridden_by_user: 0, page_count: 3, chunk_count: 5,
  tags: [{ tag: "llm", label: "LLMs", basis: "title", quote: "LLM" }],
};

it("shows measured storage and deletes only the selected PDF after confirmation", async () => {
  vi.mocked(getStorageHealth).mockResolvedValue({
    checked_at: "2026-11-02T12:00:00Z", active_jobs: [],
    database: { provider: "sqlite", status: "healthy", size_bytes: 1048576, papers: 1, evidence_chunks: 5 },
    vector_store: { provider: "exact", status: "indexing", size_bytes: 4096, vectors: 12, detail: "1 of 2 papers embedded with fastembed:BAAI/bge-small-en-v1.5" },
    file_storage: { provider: "local", status: "healthy", size_bytes: 2097152, files: 1, disk_free_bytes: 1000, disk_total_bytes: 2000 },
  });
  vi.mocked(deletePaper).mockResolvedValue(undefined);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const onDeleted = vi.fn();
  render(<QueryClientProvider client={client}><StorageHealth /><SelectedDocument paper={paper} onDeleted={onDeleted} /></QueryClientProvider>);
  fireEvent.click(screen.getByText("Storage & health"));
  await screen.findByText("sqlite · 1.0 MB");
  expect(screen.getByText("1 of 2 papers embedded with fastembed:BAAI/bge-small-en-v1.5")).toBeInTheDocument();
  expect(screen.getByText("exact · 12 vectors · 4.0 KB")).toBeInTheDocument();
  expect(screen.getByText("indexing")).toHaveClass("health-pill", "indexing");
  fireEvent.click(screen.getByRole("button", { name: "Delete PDF" }));
  expect(deletePaper).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Confirm delete" }));
  await waitFor(() => expect(deletePaper).toHaveBeenCalledWith("selected", true));
  await waitFor(() => expect(onDeleted).toHaveBeenCalledWith("selected", true));
  client.clear();
});


it("confirms full paper deletion separately and disables an already deleted PDF", async () => {
  vi.mocked(deletePaper).mockClear();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const onDeleted = vi.fn();
  render(<QueryClientProvider client={client}><SelectedDocument paper={{ ...paper, pdf_deleted: true }} onDeleted={onDeleted} /></QueryClientProvider>);
  await waitFor(() => expect(screen.getByRole("button", { name: "Delete paper" })).toBeEnabled());
  expect(screen.getByRole("button", { name: "Delete PDF" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Delete paper" }));
  expect(deletePaper).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Confirm delete" }));
  await waitFor(() => expect(deletePaper).toHaveBeenCalledWith("selected", false));
  await waitFor(() => expect(onDeleted).toHaveBeenCalledWith("selected", false));
  client.clear();
});

it("shows a degraded answer model with its reason and keeps the header from saying Healthy", async () => {
  const healthy = {
    checked_at: "2026-11-02T12:00:00Z", active_jobs: [],
    database: { provider: "sqlite", status: "healthy", size_bytes: 1024, papers: 1, evidence_chunks: 5 },
    vector_store: { provider: "exact", status: "healthy", size_bytes: 4096, vectors: 12, detail: "All papers embedded" },
    file_storage: { provider: "local", status: "healthy", size_bytes: 2048, files: 1, disk_free_bytes: 1000, disk_total_bytes: 2000 },
  };
  const detail = "nli-onnx failed its last 3 calls (last: timed out after 3.0 s); answers use exact source excerpts.";
  vi.mocked(getStorageHealth).mockResolvedValue({
    ...healthy,
    entailment: { plugin: "nli-onnx", status: "degraded", detail, consecutive_failures: 3, last_error: "timed out after 3.0 s", last_latency_seconds: 3.04 },
    reranker: { plugin: "onnx-cross-encoder", status: "healthy", detail: "Passages are reranked by relevance to each question.", consecutive_failures: 0, last_error: "", last_latency_seconds: 0.25 },
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><StorageHealth /></QueryClientProvider>);
  fireEvent.click(screen.getByText("Storage & health"));
  expect(await screen.findByText(detail)).toBeInTheDocument();
  expect(screen.getByText("degraded")).toHaveClass("health-pill", "degraded");
  expect(screen.getByText("nli-onnx · Tier 3 entailment · last call 3.0 s")).toBeInTheDocument();
  expect(screen.getByText("onnx-cross-encoder · question relevance · last call 0.3 s")).toBeInTheDocument();
  expect(screen.getByText("Model degraded")).toBeInTheDocument();
  expect(screen.queryByText("Healthy")).not.toBeInTheDocument();
  client.clear();
});
