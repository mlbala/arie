import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ChatWorkspace } from "./ChatWorkspace";
import { getChat, type PaperSummary } from "../api/client";

// Saved history is fetched separately from the answer stream these tests script.
vi.mock("../api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("../api/client")>(),
  getChat: vi.fn(async () => ({ turns: [] })),
}));

const papers: PaperSummary[] = [
  {
    id: "paper-1",
    title: "Evidence and Verification",
    source: "arxiv",
    status: "INDEXED",
    published_at: "2025-01-12T00:00:00Z",
    abstract: "A test paper.",
    relevance_gate: true,
    overridden_by_user: 0,
  },
  {
    id: "paper-2",
    title: "Retrieval Methods",
    source: "openreview",
    status: "INDEXED",
    published_at: "2025-01-10T00:00:00Z",
    abstract: "Another test paper.",
    relevance_gate: true,
    overridden_by_user: 0,
  },
];

describe("ChatWorkspace", () => {
  it("retains the hidden Activity pane state when switching papers", () => {
    const onSelectPaper = vi.fn();
    const props = {
      papers,
      activePaperId: "paper-1",
      onSelectPaper,
      onClosePaper: vi.fn(),
      onCitation: vi.fn(),
    };
    const { rerender } = render(<ChatWorkspace {...props} />);

    fireEvent.change(
      screen.getByLabelText("Ask about Evidence and Verification"),
      {
        target: { value: "Keep this draft" },
      },
    );
    rerender(<ChatWorkspace {...props} activePaperId="paper-2" />);
    rerender(<ChatWorkspace {...props} activePaperId="paper-1" />);

    expect(
      screen.getByLabelText("Ask about Evidence and Verification"),
    ).toHaveValue("Keep this draft");
  });

  it("aborts an in-flight question when switching papers", async () => {
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(new DOMException("Aborted", "AbortError"));
        });
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const props = {
      papers,
      activePaperId: "paper-1",
      onSelectPaper: vi.fn(),
      onClosePaper: vi.fn(),
      onCitation: vi.fn(),
    };
    const { rerender } = render(<ChatWorkspace {...props} />);

    fireEvent.change(
      screen.getByLabelText("Ask about Evidence and Verification"),
      {
        target: { value: "What is the method?" },
      },
    );
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const signal = fetchMock.mock.calls[0]?.[1]?.signal;

    rerender(<ChatWorkspace {...props} activePaperId="paper-2" />);

    await waitFor(() => expect(signal?.aborted).toBe(true));
  });
});

describe("verified answer stream", () => {
  function streamResponse(events: string[]) {
    const encoder = new TextEncoder();
    // Split events across arbitrary network chunks, including CRLF delimiters.
    const wire = events.join("").replace(/\n/g, "\r\n");
    return new Response(
      new ReadableStream({
        start(controller) {
          for (let i = 0; i < wire.length; i += 17)
            controller.enqueue(encoder.encode(wire.slice(i, i + 17)));
          controller.close();
        },
      }),
      { status: 200 },
    );
  }

  it("shows real progress and only accepts verified claims from the active paper", async () => {
    const quote = "The method uses isolated worker processes.";
    const claim = {
      text: quote,
      quote,
      citation: { page_start: 2, char_start: 20, char_end: 20 + quote.length },
    };
    const event = (type: string, data: object) =>
      `event: ${type}\ndata: ${JSON.stringify(data)}\n\n`;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          event("progress", {
            paper_id: "paper-1",
            status: "retrieval",
            message: "Finding passages",
          }),
          event("claim", {
            paper_id: "paper-2",
            verified: true,
            claim: { ...claim, text: "Foreign paper content" },
          }),
          event("claim", {
            paper_id: "paper-1",
            verified: false,
            claim: { ...claim, text: "Unverified claim" },
          }),
          event("progress", {
            paper_id: "paper-1",
            status: "verification",
            message: "Checking quotes",
          }),
          event("claim", { paper_id: "paper-1", verified: true, claim }),
          event("done", { paper_id: "paper-1", mode: "source_excerpts" }),
        ]),
      ),
    );
    const onProgress = vi.fn();
    const onCitation = vi.fn();
    render(
      <ChatWorkspace
        papers={papers}
        activePaperId="paper-1"
        onSelectPaper={vi.fn()} onClosePaper={vi.fn()}
        onCitation={onCitation}
        onProgress={onProgress}
      />,
    );
    fireEvent.change(
      screen.getByLabelText("Ask about Evidence and Verification"),
      { target: { value: "What is the method?" } },
    );
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await screen.findByText(quote, { selector: "p" });
    expect(screen.queryByText("Foreign paper content")).not.toBeInTheDocument();
    expect(screen.queryByText("Unverified claim")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(onProgress).toHaveBeenCalledWith(
        "paper-1",
        expect.objectContaining({ stage: "complete" }),
      ),
    );
    expect(onProgress).toHaveBeenCalledWith("paper-1", {
      stage: "verification",
      message: "Checking quotes",
    });
    fireEvent.click(screen.getByRole("button", { name: /Page 2/ }));
    expect(onCitation).toHaveBeenCalledWith(
      expect.objectContaining({ page_start: 2, char_start: 20, quote }),
    );
  });

  it("renders an explicit abstention when the paper has no supporting evidence", async () => {
    const message = "I could not find verified evidence in this paper.";
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          streamResponse([
            `event: abstention\ndata: ${JSON.stringify({ paper_id: "paper-1", message })}\n\n`,
            'event: done\ndata: {"paper_id":"paper-1"}\n\n',
          ]),
        ),
    );
    render(
      <ChatWorkspace
        papers={papers}
        activePaperId="paper-1"
        onSelectPaper={vi.fn()} onClosePaper={vi.fn()}
        onCitation={vi.fn()}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: "What is this paper about?" }),
    );
    expect(await screen.findByText(message)).toBeInTheDocument();
  });
});

it("renders reviewed summary text with its source and discards empty supporting quotes", async () => {
  const event = (kind: string, data: object) => `event: ${kind}\ndata: ${JSON.stringify(data)}\n\n`;
  const quote = "We propose a retrieval model using a gated memory bank.";
  const text = "The retrieval model uses a gated memory bank.";
  const citation = { page_start: 1, char_start: 0, char_end: quote.length };
  const wire = [
    event("answer", { paper_id: "paper-1", mode: "grounded_summary", heading: "Paper summary", evaluation: { reviewed_claims: 1, rejected_claims: 0, answer_relevant: true, required_aspects: ["study contribution"], covered_aspects: 1 } }),
    event("claim", { paper_id: "paper-1", verified: true, claim: { text, quote, label: "Method", citation } }),
    event("claim", { paper_id: "paper-1", verified: true, claim: { text: "Unbacked text", quote: "", citation } }),
    event("done", { paper_id: "paper-1", mode: "grounded_summary" }),
  ].join("");
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(new ReadableStream({
    start(controller) { controller.enqueue(new TextEncoder().encode(wire)); controller.close(); },
  }))));
  render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()} onCitation={vi.fn()} />);
  fireEvent.click(screen.getByRole("button", { name: "What is this paper about?" }));
  expect(await screen.findByRole("heading", { name: "Paper summary" })).toBeInTheDocument();
  expect(await screen.findByText(text, { exact: false, selector: "p" })).toBeInTheDocument();
  fireEvent.click(screen.getByText("Supporting passages", { selector: "summary" }));
  expect(screen.getByText(quote, { selector: "blockquote" })).toBeInTheDocument();
  expect(screen.queryByText("Unbacked text")).not.toBeInTheDocument();
  expect(screen.getByText("Model evaluation")).toBeInTheDocument();
  expect(screen.getByText("Question relevance: passed · Evidence support: passed")).toBeInTheDocument();
  expect(screen.getByText("Requested points covered: 1/1")).toBeInTheDocument();
});


it("combines three verified statements into one paragraph before grouped sources", async () => {
  const claims = [
    { text: "Cedar studies evidence retrieval.", quote: "We study evidence retrieval using the Cedar model." },
    { text: "It uses a gated memory bank.", quote: "The method uses a gated memory bank for retrieving evidence." },
    { text: "The evaluation covers English documents.", quote: "We evaluated the method on English documents." },
  ];
  const event = (kind: string, data: object) => `event: ${kind}\ndata: ${JSON.stringify(data)}\n\n`;
  const wire = [
    event("answer", { paper_id: "paper-1", mode: "reviewed_answer", heading: "Answer", evaluation: { reviewed_claims: 4, rejected_claims: 1, answer_relevant: true } }),
    ...claims.map((claim, i) => event("claim", { paper_id: "paper-1", verified: true, claim: { ...claim, citation: { page_start: i + 1, char_start: 0, char_end: claim.quote.length } } })),
    event("done", { paper_id: "paper-1", mode: "reviewed_answer" }),
  ].join("");
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(wire)));
  const onCitation = vi.fn();
  render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()} onCitation={onCitation} />);
  fireEvent.click(screen.getByRole("button", { name: "What is this paper about?" }));
  await screen.findByRole("heading", { name: "Answer" });
  const paragraph = document.querySelector(".answer-summary")!;
  expect(document.querySelectorAll(".answer-summary")).toHaveLength(1);
  for (const claim of claims) expect(paragraph).toHaveTextContent(claim.text);
  const sources = screen.getByRole("region", { name: "Verified sources" });
  expect(paragraph.compareDocumentPosition(sources) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(screen.getByText("4 statements checked · 3 accepted · 1 rejected")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Source 2, page 2" }));
  expect(onCitation).toHaveBeenCalledWith(expect.objectContaining({ page_start: 2, quote: claims[1]!.quote }));
});


it("sends paper-specific question history for follow-ups and renders detailed paragraphs", async () => {
  const event = (kind: string, data: object) => `event: ${kind}\ndata: ${JSON.stringify(data)}\n\n`;
  const quote = "The method uses isolated worker processes to read evidence.";
  const response = (paperId: string, detailed = false) => new Response([
    event("answer", { paper_id: paperId, mode: "source_excerpts", context_question: "What is this paper about?", heading: detailed ? "Detailed explanation" : "Answer" }),
    event("claim", { paper_id: paperId, verified: true, claim: { text: quote, quote, paragraph: 0, label: "Method", citation: { page_start: 1, char_start: 0, char_end: quote.length, chunk_id: `${paperId}:first` } } }),
    ...(detailed ? [event("claim", { paper_id: paperId, verified: true, claim: { text: "The workers keep each paper isolated.", quote: "The workers keep each paper isolated throughout extraction.", paragraph: 1, label: "Findings", citation: { page_start: 2, char_start: 0, char_end: 58, chunk_id: `${paperId}:second` } } })] : []),
    event("done", { paper_id: paperId, mode: "source_excerpts" }),
  ].join(""));
  const fetcher = vi.fn().mockResolvedValueOnce(response("paper-1")).mockResolvedValueOnce(response("paper-1", true)).mockResolvedValueOnce(response("paper-1", true)).mockResolvedValueOnce(response("paper-1", true)).mockResolvedValueOnce(response("paper-1", true)).mockResolvedValueOnce(response("paper-2"));
  vi.stubGlobal("fetch", fetcher);
  const props = { papers, activePaperId: "paper-1", onSelectPaper: vi.fn(), onClosePaper: vi.fn(), onCitation: vi.fn() };
  const { rerender } = render(<ChatWorkspace {...props} />);
  fireEvent.click(screen.getByRole("button", { name: "What is this paper about?" }));
  await screen.findByRole("heading", { name: "Answer", level: 3 });
  // A single-paragraph answer has no theme lead-in.
  expect(document.querySelectorAll(".answer-theme")).toHaveLength(0);
  fireEvent.change(screen.getByLabelText("Ask about Evidence and Verification"), { target: { value: "explain. more detail" } });
  fireEvent.click(screen.getByRole("button", { name: "Send question" }));
  await screen.findByRole("heading", { name: "Detailed explanation" });
  const body = JSON.parse(String(fetcher.mock.calls[1]![1].body));
  expect(body).toEqual({ question: "explain. more detail", history: [{ question: "What is this paper about?", source_ids: ["paper-1:first"] }] });
  expect(JSON.stringify(body)).not.toContain(quote);
  const lastTurn = screen.getAllByRole("article").at(-1)!;
  expect(lastTurn.querySelectorAll(".answer-summary")).toHaveLength(2);
  expect(Array.from(lastTurn.querySelectorAll(".answer-theme")).map((item) => item.textContent)).toEqual(["Method. ", "Findings. "]);
  expect(lastTurn.querySelectorAll(".supporting-passages blockquote")).toHaveLength(2);
  for (let followUp = 0; followUp < 3; followUp++) {
    fireEvent.change(screen.getByLabelText("Ask about Evidence and Verification"), { target: { value: "continue" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(3 + followUp));
    await vi.waitFor(() => expect(screen.getByRole("button", { name: "Send question" })).toBeInTheDocument());
  }
  const continued = JSON.parse(String(fetcher.mock.calls[4]![1].body));
  expect(continued.history).toHaveLength(3);
  expect(continued.history.every((turn: { question: string }) => turn.question === "What is this paper about?")).toBe(true);
  rerender(<ChatWorkspace {...props} activePaperId="paper-2" />);
  fireEvent.click(screen.getByRole("button", { name: "What is this paper about?" }));
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(6));
  expect(JSON.parse(String(fetcher.mock.calls[5]![1].body))).toEqual({ question: "What is this paper about?" });
});

it("restores saved turns as verified answers and uses them as follow-up context", async () => {
  const quote = "The method uses isolated worker processes to read evidence.";
  const citation = { page_start: 3, char_start: 10, char_end: 10 + quote.length, chunk_id: "paper-1:7" };
  vi.mocked(getChat).mockResolvedValueOnce({ turns: [{
    question: "How does the method work?",
    events: [
      { event: "answer", data: { paper_id: "paper-1", mode: "reviewed_answer", heading: "Detailed explanation", context_question: "How does the method work?", evaluation: { reviewed_claims: 1, rejected_claims: 0, answer_relevant: true } } },
      { event: "claim", data: { paper_id: "paper-1", verified: true, claim: { text: "Isolated workers read the evidence.", quote, paragraph: 0, citation } } },
      { event: "claim", data: { paper_id: "other-paper", verified: true, claim: { text: "Foreign claim.", quote: "Foreign claim text here.", citation } } },
      { event: "done", data: { paper_id: "paper-1", mode: "reviewed_answer" } },
    ],
  }] });
  const event = (kind: string, data: object) => `event: ${kind}\ndata: ${JSON.stringify(data)}\n\n`;
  const fetcher = vi.fn().mockResolvedValue(new Response([
    event("answer", { paper_id: "paper-1", mode: "source_excerpts", heading: "Answer" }),
    event("done", { paper_id: "paper-1", mode: "source_excerpts" }),
  ].join("")));
  vi.stubGlobal("fetch", fetcher);
  render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()} onCitation={vi.fn()} />);
  expect(await screen.findByRole("heading", { name: "Detailed explanation" })).toBeInTheDocument();
  const turn = screen.getAllByRole("article")[0]!;
  expect(within(turn).getByText("How does the method work?")).toBeInTheDocument();
  expect(within(turn).getByText("Isolated workers read the evidence.", { exact: false })).toBeInTheDocument();
  expect(screen.queryByText("Foreign claim.", { exact: false })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Ask about Evidence and Verification"), { target: { value: "continue" } });
  fireEvent.click(screen.getByRole("button", { name: "Send question" }));
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  expect(JSON.parse(String(fetcher.mock.calls[0]![1].body)).history).toEqual([
    { question: "How does the method work?", source_ids: ["paper-1:7"] },
  ]);
});

it("rates a saved answer and restores the rating, with a note for a down rating", async () => {
  const quote = "The method uses isolated worker processes to read evidence.";
  const citation = { page_start: 3, char_start: 10, char_end: 10 + quote.length, chunk_id: "paper-1:7" };
  vi.mocked(getChat).mockResolvedValueOnce({ turns: [{
    turn_id: "turn-1",
    question: "How does the method work?",
    rating: "up",
    events: [
      { event: "answer", data: { paper_id: "paper-1", mode: "source_excerpts", heading: "Answer" } },
      { event: "claim", data: { paper_id: "paper-1", verified: true, claim: { text: "Isolated workers read the evidence.", quote, citation } } },
      { event: "done", data: { paper_id: "paper-1", mode: "source_excerpts", turn_id: "turn-1" } },
    ],
  }] });
  const fetcher = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ turn_id: "turn-1", rating: "down" })));
  vi.stubGlobal("fetch", fetcher);
  render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()} onCitation={vi.fn()} />);
  expect(await screen.findByRole("button", { name: "Helpful" })).toHaveAttribute("aria-pressed", "true");
  fireEvent.click(screen.getByRole("button", { name: "Not helpful" }));
  expect(screen.getByRole("button", { name: "Not helpful" })).toHaveAttribute("aria-pressed", "true");
  fireEvent.change(screen.getByLabelText("What was wrong?"), { target: { value: "Missed the method" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
  const [url, init] = fetcher.mock.calls[1]!;
  expect(String(url)).toContain("/v1/papers/paper-1/chat/turn-1/feedback");
  expect(init.method).toBe("PUT");
  expect(JSON.parse(String(init.body))).toEqual({ rating: "down", note: "Missed the method" });
});
