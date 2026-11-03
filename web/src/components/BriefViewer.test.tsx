import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BriefViewer } from "./BriefViewer";
import { getPaper, type Paper } from "../api/client";

vi.mock("../api/client", () => ({ getPaper: vi.fn() }));

/** Workers do not exist in jsdom; record messages and let tests answer them. */
class FakeWorker {
  static instances: FakeWorker[] = [];
  messages: unknown[] = [];
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  constructor() { FakeWorker.instances.push(this); }
  postMessage(message: unknown) { this.messages.push(message); }
  terminate() {}
}

const citation = { page_start: 3, char_start: 10, char_end: 40, label: "p. 3" };
const paper = {
  id: "1706.03762", title: "Attention Is All You Need", abstract: "Transformers.", source: "arxiv",
  status: "BRIEFED", published_at: "2017-06-12", relevance_gate: true, overridden_by_user: 0, outline: [],
  pdf_url: "https://arxiv.org/pdf/1706.03762", brief: { parts: [] },
  verified_5_part_brief: [{ label: "method", title: "Method", content: "Uses **attention** only.", citations: [citation] }],
} satisfies Paper;

function renderViewer(paperId: string | null) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><BriefViewer paperId={paperId} citation={null} /></QueryClientProvider>);
}

describe("BriefViewer", () => {
  beforeEach(() => { FakeWorker.instances = []; vi.stubGlobal("Worker", FakeWorker); });
  afterEach(() => vi.unstubAllGlobals());

  it("asks for a paper before showing a brief", () => {
    renderViewer(null);
    expect(screen.getByText("Select a paper to open its verified brief.")).toBeInTheDocument();
    expect(getPaper).not.toHaveBeenCalled();
  });

  it("renders brief parts off the main thread and falls back to plain text on worker errors", async () => {
    vi.mocked(getPaper).mockResolvedValue(paper);
    renderViewer(paper.id);
    expect(await screen.findByText("Attention Is All You Need")).toBeInTheDocument();
    const markdown = FakeWorker.instances[0]!;
    expect(markdown.messages).toEqual([{ id: "1706.03762:0", content: "Uses **attention** only." }]);
    markdown.onerror?.({ message: "render failed" });
    expect(await screen.findByText("Uses **attention** only.")).toBeInTheDocument();
  });

  it("opens a citation as an exact highlighted source span", async () => {
    vi.mocked(getPaper).mockResolvedValue(paper);
    renderViewer(paper.id);
    fireEvent.click(await screen.findByRole("button", { name: "p. 3" }));
    const pdf = FakeWorker.instances[1]!;
    expect(pdf.messages).toEqual([{ type: "highlight", pdfUrl: paper.pdf_url, citation: { page_start: 3, char_start: 10, char_end: 40 } }]);
    pdf.onmessage?.({ data: { type: "highlight", page: 3, before: "The ", highlighted: "attention span", after: " here" } });
    expect(await screen.findByText("attention span")).toContainHTML("attention span");
    expect(screen.getByText("Page 3")).toBeInTheDocument();
  });
});
