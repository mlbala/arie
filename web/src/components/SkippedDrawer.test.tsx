import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SkippedDrawer } from "./SkippedDrawer";
import { getSkippedPapers, overridePaper, type SkippedPaper } from "../api/client";

vi.mock("../api/client", () => ({ getSkippedPapers: vi.fn(), overridePaper: vi.fn() }));

const paper = (paper_id: string, status: string): SkippedPaper => ({
  paper_id, status, title: `Paper ${paper_id}`, source: "arxiv", reason: `${status} reason`, overridden_by_user: 0,
});

function renderDrawer(onClose = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><SkippedDrawer onClose={onClose} /></QueryClientProvider>);
  return onClose;
}

describe("SkippedDrawer", () => {
  it("retries failed papers but keeps safety-gated papers held", async () => {
    vi.mocked(getSkippedPapers).mockResolvedValue({ items: [paper("failed-1", "FAILED"), paper("gated-1", "QUARANTINED")] });
    vi.mocked(overridePaper).mockResolvedValue(undefined);
    renderDrawer();

    expect(await screen.findByText("FAILED reason")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Held by safety gate/ })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /Retry sync/ }));
    await waitFor(() => expect(overridePaper).toHaveBeenCalledWith("failed-1", expect.anything()));
    expect(getSkippedPapers).toHaveBeenCalledTimes(2);
  });

  it("shows an empty queue and closes from the backdrop", async () => {
    vi.mocked(getSkippedPapers).mockResolvedValue({ items: [] });
    const onClose = renderDrawer();
    expect(await screen.findByText("No papers are waiting for audit.")).toBeInTheDocument();
    fireEvent.mouseDown(screen.getByRole("dialog").parentElement!);
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("reports why the queue could not load", async () => {
    vi.mocked(getSkippedPapers).mockRejectedValue(new Error("Audit service unavailable"));
    renderDrawer();
    expect(await screen.findByText("Audit service unavailable")).toBeInTheDocument();
  });
});
