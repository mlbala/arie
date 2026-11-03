import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PaperFeed } from "./PaperFeed";
import { getPaper, getPapers, type PaperSummary } from "../api/client";

vi.mock("../api/client", () => ({ getPapers: vi.fn(), getPaper: vi.fn() }));
// jsdom has no layout, so render every row instead of a measured window.
vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 98,
    getVirtualItems: () => Array.from({ length: count }, (_, index) => ({ index, start: index * 98 })),
    scrollToIndex: vi.fn(),
  }),
}));

const summary = (id: string, title: string, status = "INDEXED"): PaperSummary => ({
  id, title, status, source: "arxiv", published_at: "2026-10-01T00:00:00Z", abstract: `${title} abstract`,
  relevance_gate: true, overridden_by_user: 0,
});

function renderFeed(selectedPaperId: string | null = null) {
  const onSelect = vi.fn<(paperId: string) => void>();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><PaperFeed selectedPaperId={selectedPaperId} onSelectPaper={onSelect} /></QueryClientProvider>);
  return onSelect;
}

describe("PaperFeed", () => {
  it("filters by status and search text and selects a paper", async () => {
    vi.mocked(getPapers).mockResolvedValue({
      items: [summary("a", "Attention Is All You Need"), summary("b", "Gated Memory", "SKIPPED")],
      next_cursor: null, has_more: false,
    });
    vi.mocked(getPaper).mockResolvedValue({} as never);
    const onSelect = renderFeed();

    expect(await screen.findByText("2 papers")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Skipped" }));
    expect(screen.queryByRole("button", { name: /Attention Is All You Need/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "All papers" }));
    fireEvent.change(screen.getByLabelText("Search papers"), { target: { value: "attention" } });
    fireEvent.click(screen.getByRole("button", { name: /Attention Is All You Need/ }));
    expect(onSelect).toHaveBeenCalledWith("a");
    fireEvent.change(screen.getByLabelText("Search papers"), { target: { value: "no such paper" } });
    expect(screen.getByText("No papers match this search.")).toBeInTheDocument();
  });

  it("moves the selection with j and focuses search with slash", async () => {
    vi.mocked(getPapers).mockResolvedValue({ items: [summary("a", "First"), summary("b", "Second")], next_cursor: null, has_more: false });
    const onSelect = renderFeed("a");
    await screen.findByText("2 papers");
    fireEvent.keyDown(window, { key: "j" });
    expect(onSelect).toHaveBeenCalledWith("b");
    fireEvent.keyDown(window, { key: "/" });
    expect(screen.getByLabelText("Search papers")).toHaveFocus();
  });

  it("offers a retry when the library cannot load", async () => {
    vi.mocked(getPapers).mockRejectedValueOnce(new Error("Library offline"));
    renderFeed();
    expect(await screen.findByText(/Library offline/)).toBeInTheDocument();
    vi.mocked(getPapers).mockResolvedValue({ items: [], next_cursor: null, has_more: false });
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText("No papers match this search.")).toBeInTheDocument();
  });
});
