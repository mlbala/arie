import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { clearAllChats, type PaperSummary } from "../api/client";
import { HistorySection } from "./HistorySection";

vi.mock("../api/client", () => ({ clearAllChats: vi.fn() }));

const paper: PaperSummary = {
  id: "p1", title: "LLM agents", abstract: "Research", source: "arxiv",
  status: "INDEXED", published_at: "2025-06-01", relevance_gate: true, overridden_by_user: 0,
};

function setup(papers: PaperSummary[]) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const props = { onSelect: vi.fn(), onClearList: vi.fn(), onChatsCleared: vi.fn() };
  render(<QueryClientProvider client={client}><HistorySection papers={papers} max={10} {...props} /></QueryClientProvider>);
  return { client, ...props };
}

it("keeps the title and an icon-only trash button when the list is empty", () => {
  const { client } = setup([]);
  const history = screen.getByRole("region", { name: "History" });
  expect(within(history).getByText("History")).toBeInTheDocument();
  expect(within(history).queryByRole("button", { name: "Clear list" })).not.toBeInTheDocument();
  const trash = within(history).getByRole("button", { name: "Delete all chat history" });
  expect(trash).toHaveTextContent("");
  client.clear();
});

it("shows Clear list next to the trash button while papers are listed", () => {
  const { client, onClearList } = setup([paper]);
  fireEvent.click(screen.getByRole("button", { name: "Clear list" }));
  expect(onClearList).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("button", { name: "Delete all chat history" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "LLM agents" })).toBeInTheDocument();
  client.clear();
});

it("deletes all saved chats only after confirmation and can be cancelled", async () => {
  vi.mocked(clearAllChats).mockResolvedValue({ deleted: 3 });
  const { client, onChatsCleared } = setup([]);
  fireEvent.click(screen.getByRole("button", { name: "Delete all chat history" }));
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.queryByRole("group", { name: "Confirm chat deletion" })).not.toBeInTheDocument();
  expect(clearAllChats).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Delete all chat history" }));
  fireEvent.click(screen.getByRole("button", { name: "Confirm delete" }));
  await waitFor(() => expect(clearAllChats).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(onChatsCleared).toHaveBeenCalledTimes(1));
  expect(screen.queryByRole("group", { name: "Confirm chat deletion" })).not.toBeInTheDocument();
  client.clear();
});
