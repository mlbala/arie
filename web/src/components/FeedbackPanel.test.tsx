import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { FeedbackAgreement, FeedbackItem, ImplicitFeedback } from "../api/client";
import { FeedbackPanel } from "./FeedbackPanel";

const rated: FeedbackItem = {
  turn_id: "turn-1", question: "How is Cedar evaluated?", rating: "down", note: "Missed the benchmark",
  created_at: "2026-11-04T10:00:00Z", reasons: ["wrong", "missing_citation"], mode: "explain",
  task: "explain", prompt_version: "explain.v1", trace_id: "explain:abc", key_facts: [], source_sentence: "",
};
const implicit: ImplicitFeedback = {
  turn_id: "turn-2", question: "What is the accuracy?", signal: "said_wrong",
  follow_up: "That's wrong", created_at: "2026-11-04T10:05:00Z",
};
const agreement: FeedbackAgreement = {
  rated: 4, judged: 3, agreed: 2, rate: 0.667, ready: false, implicit: 1,
  by_task: [
    { task: "explain", rated: 3, judged: 3, agreed: 2, rate: 0.667 },
    { task: "fact", rated: 1, judged: 0, agreed: 0, rate: null },
  ],
};

afterEach(() => vi.unstubAllGlobals());

it("shows why an answer was rated down and what it recorded about itself", () => {
  render(<FeedbackPanel paperId="paper-1" feedback={[rated, { ...rated, turn_id: "t3", rating: "up", question: "Liked" }]}
    implicit={[]} agreement={undefined} onChanged={vi.fn()} />);
  expect(screen.getByText("Answers you rated down (1)")).toBeInTheDocument();
  expect(screen.getByText("How is Cedar evaluated?")).toBeInTheDocument();
  expect(screen.getByText("Wrong")).toBeInTheDocument();
  expect(screen.getByText("Missing citation")).toBeInTheDocument();
  expect(screen.getByText("“Missed the benchmark”")).toBeInTheDocument();
  expect(screen.getByText("explain · explain.v1 · explain:abc")).toBeInTheDocument();
  // A thumbs-up is not a case and is not listed.
  expect(screen.queryByText("Liked")).not.toBeInTheDocument();
});

it("shows the judge's agreement overall and per task, and that gate 4 is not ready", () => {
  render(<FeedbackPanel paperId="paper-1" feedback={[]} implicit={[]} agreement={agreement} onChanged={vi.fn()} />);
  expect(screen.getByText("2 of 3")).toBeInTheDocument();
  expect(screen.getByText("67%")).toBeInTheDocument();
  expect(screen.getByText(/1 of 4 rated\s+answers had no judge verdict/)).toBeInTheDocument();
  expect(screen.getByText(/Gate 4 is read at 30 rated answers; 4 so far\./)).toBeInTheDocument();
  const tasks = within(screen.getByRole("list", { name: "Agreement per task" })).getAllByRole("listitem");
  expect(tasks.map((row) => row.textContent)).toEqual([
    "explain: 2 of 3 judged (67%), 3 rated",
    "fact: 0 of 0 judged (–), 1 rated",
  ]);
});

it("renders nothing when there is no feedback and no rated answer", () => {
  const { container } = render(<FeedbackPanel paperId="paper-1" feedback={[]} implicit={[]}
    agreement={{ ...agreement, rated: 0, judged: 0, agreed: 0, rate: null, implicit: 0, by_task: [] }} onChanged={vi.fn()} />);
  expect(container).toBeEmptyDOMElement();
});

it("lists answers a follow-up questioned without a form", () => {
  render(<FeedbackPanel paperId="paper-1" feedback={[]} implicit={[implicit]} agreement={undefined} onChanged={vi.fn()} />);
  expect(screen.getByText("Answers a follow-up questioned (1)")).toBeInTheDocument();
  expect(screen.getByText("What is the accuracy?")).toBeInTheDocument();
  expect(screen.getByText(/said the answer was wrong: “That's wrong”/)).toBeInTheDocument();
  expect(screen.queryByLabelText("Key fact 1")).not.toBeInTheDocument();
});

it("turns a rated-down answer into a scored case with key facts and the paper's sentence", async () => {
  const fetcher = vi.fn().mockImplementation(async () => new Response(JSON.stringify({
    turn_id: "turn-1", key_facts: ["Cedar reaches 92%"], source_sentence: "Cedar reaches 92% accuracy.",
  })));
  vi.stubGlobal("fetch", fetcher);
  const onChanged = vi.fn();
  render(<FeedbackPanel paperId="paper-1" feedback={[rated]} implicit={[]} agreement={undefined} onChanged={onChanged} />);
  fireEvent.change(screen.getByLabelText("Key fact 1"), { target: { value: "  Cedar reaches 92%  " } });
  fireEvent.change(screen.getByLabelText("Paper sentence"), { target: { value: "Cedar reaches 92% accuracy." } });
  fireEvent.click(screen.getByRole("button", { name: "Save key facts" }));
  await vi.waitFor(() => expect(onChanged).toHaveBeenCalledTimes(1));
  const [url, init] = fetcher.mock.calls[0]!;
  expect(String(url)).toContain("/v1/papers/paper-1/chat/turn-1/feedback/facts");
  expect(init.method).toBe("PUT");
  // Blank slots are dropped and the fact is trimmed.
  expect(JSON.parse(String(init.body))).toEqual({
    key_facts: ["Cedar reaches 92%"], source_sentence: "Cedar reaches 92% accuracy.",
  });
});

it("needs a key fact, and shows the API's reason when it refuses the facts", async () => {
  const fetcher = vi.fn().mockImplementation(async () => new Response(
    JSON.stringify({ detail: "Key fact not found in the sentence" }), { status: 400 }));
  vi.stubGlobal("fetch", fetcher);
  const onChanged = vi.fn();
  render(<FeedbackPanel paperId="paper-1" feedback={[rated]} implicit={[]} agreement={undefined} onChanged={onChanged} />);
  fireEvent.click(screen.getByRole("button", { name: "Save key facts" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Add at least one key fact.");
  expect(fetcher).not.toHaveBeenCalled();
  fireEvent.change(screen.getByLabelText("Key fact 1"), { target: { value: "Cedar reaches 92%" } });
  fireEvent.click(screen.getByRole("button", { name: "Save key facts" }));
  await vi.waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Key fact not found in the sentence"));
  expect(onChanged).not.toHaveBeenCalled();
});

it("shows a scored case's key facts and lets the reader change them", () => {
  render(<FeedbackPanel paperId="paper-1" feedback={[{ ...rated, key_facts: ["Cedar reaches 92%"], source_sentence: "Cedar reaches 92% accuracy." }]}
    implicit={[]} agreement={undefined} onChanged={vi.fn()} />);
  expect(screen.getByText("Scored case: Cedar reaches 92%")).toBeInTheDocument();
  expect(screen.getByText("Change the key facts")).toBeInTheDocument();
  expect(screen.getByLabelText("Key fact 1")).toHaveValue("Cedar reaches 92%");
  expect(screen.getByLabelText("Paper sentence")).toHaveValue("Cedar reaches 92% accuracy.");
});
