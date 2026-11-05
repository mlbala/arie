import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
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
