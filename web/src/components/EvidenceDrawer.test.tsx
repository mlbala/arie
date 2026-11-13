import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { EvidenceDrawer } from "./EvidenceDrawer";

it("keeps extracted pages readable after PDF deletion and hides unavailable PDF links", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => new Response(JSON.stringify(
    String(input).endsWith("/pages/1")
      ? { text: "Retained source evidence.", page_number: 1 }
      : { id: "retained", title: "Retained paper", page_count: 1, pdf_deleted: true },
  ), { headers: { "content-type": "application/json" } })));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><EvidenceDrawer paperId="retained" citation={null} onClose={vi.fn()} /></QueryClientProvider>);
  expect(await screen.findByText("Retained source evidence.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Original PDF" })).toBeDisabled();
  expect(screen.queryByRole("link", { name: "Open PDF in new tab" })).not.toBeInTheDocument();
  expect(screen.getByText("Original PDF removed. Extracted content remains available.")).toBeInTheDocument();
  client.clear();
});

it("highlights a cited chunk without a quote on its own page, cut at the page end", async () => {
  const pageText = "Header line. Workers read isolated processes and the run ends here.";
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => new Response(JSON.stringify(
    String(input).endsWith("/pages/2")
      ? { text: pageText, page_number: 2 }
      : { id: "paper", title: "Paper", page_count: 3, pdf_deleted: false },
  ), { headers: { "content-type": "application/json" } })));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const citation = { page_start: 2, char_start: 13, char_end: 400, chunk_id: "paper:3" };
  const { container } = render(<QueryClientProvider client={client}><EvidenceDrawer paperId="paper" citation={citation} onClose={vi.fn()} /></QueryClientProvider>);
  expect(await screen.findByText("Page 2 of 3")).toBeInTheDocument();
  await vi.waitFor(() => expect(container.querySelector("mark")).toHaveTextContent("Workers read isolated processes and the run ends here."));
  expect(container.querySelector("mark")?.textContent).toBe(pageText.slice(13));
  client.clear();
});

it("hides picture-text markers but highlights the cited words at their stored offsets", async () => {
  const pageText = "Parts.\n\n<!-- Start of picture text -->\nEncoder Retriever Decoder<br><!-- End of picture text -->\n";
  const start = pageText.indexOf("Encoder");
  const citation = { page_start: 1, char_start: start, char_end: start + 7, chunk_id: "paper:1", quote: "Encoder" };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => new Response(JSON.stringify(
    String(input).endsWith("/pages/1")
      ? { text: pageText, page_number: 1 }
      : { id: "paper", title: "Paper", page_count: 1, pdf_deleted: false },
  ), { headers: { "content-type": "application/json" } })));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const { container } = render(<QueryClientProvider client={client}><EvidenceDrawer paperId="paper" citation={citation} onClose={vi.fn()} /></QueryClientProvider>);
  await vi.waitFor(() => expect(container.querySelector("mark")).toHaveTextContent("Encoder"));
  const shown = container.querySelector(".extracted-page pre")?.textContent ?? "";
  expect(shown).not.toContain("picture text");
  expect(shown).toContain("Encoder Retriever Decoder");
  client.clear();
});

const FIGURE_PAPER = {
  id: "paper", title: "Paper", page_count: 4, pdf_deleted: false,
  figures: [
    { number: "2", page: 3, caption: "Figure 2: Accuracy by model size.", has_crop: true, crop_url: "/v1/papers/paper/figures/2.png" },
    { number: "3", page: 4, caption: "Figure 3: Ablation.", has_crop: false, crop_url: null },
  ],
};

function renderFigureDrawer(citation: { page_start: number; char_start: number; char_end: number; figures?: string[] }) {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => new Response(JSON.stringify(
    String(input).includes("/pages/") ? { text: "Figure 2: Accuracy by model size. More text.", page_number: 3 } : FIGURE_PAPER,
  ), { headers: { "content-type": "application/json" } })));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(<QueryClientProvider client={client}><EvidenceDrawer paperId="paper" citation={citation} onClose={vi.fn()} /></QueryClientProvider>);
  return { client, ...view };
}

it("shows the paper's own figure crop with its caption for a figure chunk and opens its PDF page", async () => {
  const { client } = renderFigureDrawer({ page_start: 3, char_start: 0, char_end: 33, figures: ["2", "3"] });
  const image = await screen.findByRole("img", { name: "Figure 2 as drawn in the paper" });
  expect(image.getAttribute("src")).toContain("/v1/papers/paper/figures/2.png");
  expect(screen.getByText("Figure 2: Accuracy by model size. · page 3")).toBeInTheDocument();
  // Figure 3 has no crop, so nothing is shown for it.
  expect(screen.getAllByRole("img")).toHaveLength(1);
  fireEvent.click(screen.getByRole("button", { name: "Open Figure 2 on page 3 of the PDF" }));
  expect(screen.getByRole("button", { name: "Original PDF" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByTitle("Original PDF, page 3")).toBeInTheDocument();
  client.clear();
});

it("shows no figure for a text chunk", async () => {
  const { client } = renderFigureDrawer({ page_start: 3, char_start: 34, char_end: 44 });
  expect(await screen.findByText("Page 3 of 4")).toBeInTheDocument();
  await screen.findByText("Paper");
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
  expect(document.querySelector(".evidence-figure")).toBeNull();
  client.clear();
});
