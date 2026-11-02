import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ImportPaper } from "./ImportPaper";
import { getImportStatus, importArxivPaper, type SyncJob } from "../api/client";

vi.mock("../api/client", () => ({
  importArxivPaper: vi.fn(),
  getImportStatus: vi.fn(),
}));

const receipt: SyncJob = {
  job_id: "import-old",
  kind: "import",
  status: "running",
  source: "arxiv",
  sources: ["arxiv"],
  since: "2017-06-12",
  until: "2026-11-02",
  phase: "extraction",
  message: "Extracting PDF pages",
  discovered: 1,
  processed: 0,
  indexed: 0,
  skipped: 0,
  duplicates: 0,
  failed: 0,
  current_paper: "Attention Is All You Need",
  source_errors: [],
  paper_id: "1706.03762",
};

function mount() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const onImported = vi.fn();
  const onProgress = vi.fn();
  const view = render(
    <QueryClientProvider client={client}>
      <ImportPaper onImported={onImported} onProgress={onProgress} />
    </QueryClientProvider>,
  );
  return { client, onImported, onProgress, ...view };
}

describe("ImportPaper", () => {
  it("starts extraction from an older PDF link and opens the completed paper once", async () => {
    vi.mocked(importArxivPaper).mockResolvedValue(receipt);
    vi.mocked(getImportStatus).mockResolvedValue({
      ...receipt,
      status: "completed",
      indexed: 1,
      phase: "complete",
      message: "Paper ready to chat",
    });
    const { onImported, onProgress, client } = mount();
    fireEvent.change(screen.getByRole("textbox", { name: "arXiv PDF URL" }), {
      target: { value: "https://arxiv.org/pdf/1706.03762.pdf" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Import paper" }));
    await waitFor(() =>
      expect(importArxivPaper).toHaveBeenCalledWith(
        "https://arxiv.org/pdf/1706.03762.pdf",
        expect.anything(),
      ),
    );
    await waitFor(() => expect(onImported).toHaveBeenCalledWith("1706.03762"));
    expect(onImported).toHaveBeenCalledTimes(1);
    expect(onProgress).toHaveBeenCalledWith(
      expect.objectContaining({ phase: "extraction" }),
    );
    expect(
      screen.getByRole("button", { name: /Paper ready/ }),
    ).toBeInTheDocument();
    client.clear();
  });

  it("shows a rejected URL error without opening a paper", async () => {
    vi.mocked(importArxivPaper).mockRejectedValue(
      new Error("Use an arXiv PDF URL"),
    );
    const { onImported, client } = mount();
    fireEvent.change(screen.getByRole("textbox", { name: "arXiv PDF URL" }), {
      target: { value: "https://evil.example/test.pdf" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Import paper" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Use an arXiv PDF URL",
    );
    expect(onImported).not.toHaveBeenCalled();
    client.clear();
  });
});
