import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { CitationPopover } from "./CitationPopover";

const anchor = { top: 400, bottom: 420, left: 100, right: 120, width: 20, height: 20, x: 100, y: 400, toJSON: () => ({}) } as DOMRect;

it("shows a table row's caption and header above the quote, apart from it", () => {
  render(<CitationPopover id="c1" onDismiss={vi.fn()} hover={{
    anchor, page: 4, quote: "| BERT | 85.1 |", table: true, context: "Table 1: F1.\n| Model | F1 |",
  }} />);
  const card = screen.getByRole("tooltip");
  expect(card.querySelector(".quote-context pre")?.textContent).toBe("Table 1: F1.\n| Model | F1 |");
  expect(card.querySelector("blockquote")?.textContent).toBe("| BERT | 85.1 |");
});

it("shows no context for a plain quote", () => {
  render(<CitationPopover id="c2" onDismiss={vi.fn()} hover={{ anchor, page: 2, quote: "We propose Cedar." }} />);
  expect(screen.getByRole("tooltip").querySelector(".quote-context")).toBeNull();
});
