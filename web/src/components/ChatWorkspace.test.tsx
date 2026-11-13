import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useReducer } from "react";
import { describe, expect, it, vi } from "vitest";
import { ChatWorkspace } from "./ChatWorkspace";
import { getChat, getExtractedPage, getGuide, type CitationRange, type PaperSummary } from "../api/client";
import { EMPTY_SESSIONS, paperSessionsReducer } from "../paperSessions";

// Saved history is fetched separately from the answer stream these tests script.
vi.mock("../api/client", async (importOriginal) => ({
  ...await importOriginal<typeof import("../api/client")>(),
  getChat: vi.fn(async () => ({ turns: [] })),
  getGuide: vi.fn(async () => ({ paper_id: "paper-1", status: "missing" })),
  getExtractedPage: vi.fn(),
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

  it("names the figure an abstention points to and links its crop without showing it inline", async () => {
    const message = "I could not find verified evidence in this paper.";
    const fact = "The closest passage refers to Figure 2 (p. 5).";
    const hint = { paper_id: "paper-1", number: "2", page: 5, message: fact, crop_url: "/v1/papers/paper-1/figures/2.png" };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        streamResponse([
          `event: abstention\ndata: ${JSON.stringify({ paper_id: "paper-1", message })}\n\n`,
          `event: figure_hint\ndata: ${JSON.stringify(hint)}\n\n`,
          'event: done\ndata: {"paper_id":"paper-1"}\n\n',
        ]),
      ),
    );
    const { container } = render(
      <ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()} onCitation={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "What is this paper about?" }));
    expect(await screen.findByText(message)).toBeInTheDocument();
    // The abstention keeps its text; the hint adds where the closest passage points, and a link.
    expect(screen.getByText(fact, { exact: false })).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "Open Figure 2 (p. 5)" });
    expect(link.getAttribute("href")).toContain("/v1/papers/paper-1/figures/2.png");
    expect(container.querySelector(".chat-thread img, .answer-card img")).toBeNull();
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
  expect(body).toEqual({ question: "explain. more detail", mode: "strict", history: [{ question: "What is this paper about?", source_ids: ["paper-1:first"] }] });
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
  expect(JSON.parse(String(fetcher.mock.calls[5]![1].body))).toEqual({ question: "What is this paper about?", mode: "strict" });
});

it("shows a source chip's page and quote on hover or focus, and still opens the source on click", async () => {
  const quote = "The method uses isolated worker processes to read evidence.";
  const citation = { page_start: 3, char_start: 10, char_end: 10 + quote.length, chunk_id: "paper-1:7" };
  vi.mocked(getChat).mockResolvedValueOnce({ turns: [{
    question: "How does the method work?",
    events: [
      { event: "answer", data: { paper_id: "paper-1", mode: "reviewed_answer", heading: "Detailed explanation", evaluation: { reviewed_claims: 1, rejected_claims: 0, answer_relevant: true } } },
      { event: "claim", data: { paper_id: "paper-1", verified: true, claim: { text: "Isolated workers read the evidence.", quote, paragraph: 0, citation } } },
      { event: "done", data: { paper_id: "paper-1", mode: "reviewed_answer" } },
    ],
  }] });
  const onCitation = vi.fn();
  render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()} onCitation={onCitation} />);
  const chip = await screen.findByRole("button", { name: "Source 1, page 3" });
  expect(chip).toHaveTextContent(/^1$/);
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

  fireEvent.mouseEnter(chip);
  const card = screen.getByRole("tooltip");
  expect(card).toHaveTextContent("Page 3");
  expect(card).toHaveTextContent(quote);
  expect(chip).toHaveAttribute("aria-describedby", card.id);
  fireEvent.mouseLeave(chip);
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

  fireEvent.focus(chip);
  expect(screen.getByRole("tooltip")).toHaveTextContent(quote);
  fireEvent.keyDown(document, { key: "Escape" });
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

  fireEvent.mouseEnter(chip);
  fireEvent.click(chip);
  expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  expect(onCitation).toHaveBeenCalledWith(expect.objectContaining({ page_start: 3, quote }));
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

it("saves a down rating at once, then sends the reasons the reader picks with the note", async () => {
  vi.mocked(getChat).mockResolvedValueOnce({ turns: [{
    turn_id: "turn-1",
    question: "How does the method work?",
    rating: null,
    events: [
      { event: "answer", data: { paper_id: "paper-1", mode: "source_excerpts", heading: "Answer" } },
      { event: "done", data: { paper_id: "paper-1", mode: "source_excerpts", turn_id: "turn-1" } },
    ],
  }] });
  const fetcher = vi.fn().mockImplementation(async () => new Response(JSON.stringify({ turn_id: "turn-1", rating: "down" })));
  vi.stubGlobal("fetch", fetcher);
  render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()} onCitation={vi.fn()} />);
  fireEvent.click(await screen.findByRole("button", { name: "Not helpful" }));
  // The rating is saved before any reason is chosen, so a reader who stops here still counts.
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
  expect(JSON.parse(String(fetcher.mock.calls[0]![1].body))).toEqual({ rating: "down", note: "" });

  const chips = within(screen.getByRole("group", { name: "What was wrong with it?" }));
  expect(chips.getAllByRole("button").map((chip) => chip.textContent)).toEqual([
    "Wrong", "Not in the paper", "Too shallow", "Too long", "Missing citation", "Did not answer",
  ]);
  fireEvent.click(chips.getByRole("button", { name: "Wrong" }));
  fireEvent.click(chips.getByRole("button", { name: "Too long" }));
  fireEvent.click(chips.getByRole("button", { name: "Wrong" }));
  fireEvent.click(chips.getByRole("button", { name: "Missing citation" }));
  expect(chips.getByRole("button", { name: "Wrong" })).toHaveAttribute("aria-pressed", "false");
  expect(chips.getByRole("button", { name: "Too long" })).toHaveAttribute("aria-pressed", "true");
  fireEvent.change(screen.getByLabelText("What was wrong?"), { target: { value: "Too much" } });
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
  expect(JSON.parse(String(fetcher.mock.calls[1]![1].body))).toEqual({
    rating: "down", note: "Too much", reasons: ["too_long", "missing_citation"],
  });
});

describe("explain answers", () => {
  const event = (kind: string, data: object) => `event: ${kind}\ndata: ${JSON.stringify(data)}\n\n`;
  const text = "**Method.** Workers read isolated processes [c3]. Transformers are common. The gain is 9.1 points [c3].";
  const at = (part: string) => ({ start: text.indexOf(part), end: text.indexOf(part) + part.length, text: part });
  const tail = [
    event("answer", { paper_id: "paper-1", mode: "explained_answer", heading: "Explanation", evaluation: { complete: true, fallback_reason: "" } }),
    event("citations", { paper_id: "paper-1", citations: [{ marker: 3, chunk_id: "paper-1:3", page: 2, char_start: 10, char_end: 90, checked: false }] }),
    event("checks", { paper_id: "paper-1", sentences: [
      { ...at("**Method.** Workers read isolated processes [c3]."), status: "cited_ok", markers: [3], reasons: [] },
      { ...at("Transformers are common."), status: "background", markers: [], reasons: [] },
      { ...at("The gain is 9.1 points [c3]."), status: "flagged", markers: [3], reasons: ["number_not_in_source"] },
    ] }),
    // The judge's verdicts (S13) are not shown yet; they must not be read as sentence checks.
    event("progress", { paper_id: "paper-1", status: "judging", message: "Checking the citations" }),
    event("judge", { paper_id: "paper-1", status: "complete", supported: 1, partial: 0, unsupported: 1, verdicts: [
      { number: 1, ...at("**Method.** Workers read isolated processes [c3]."), verdict: "unsupported", reason: "Not in the chunk." },
      { number: 2, ...at("The gain is 9.1 points [c3]."), verdict: "supported", reason: "Stated." },
    ] }),
    event("done", { paper_id: "paper-1", mode: "explained_answer", status: "complete", turn_id: "turn-9" }),
  ];

  /** The chat with its modes held by the real session reducer, as App holds them. */
  function Chat({ onCitation }: { onCitation: (citation: CitationRange) => void }) {
    const [sessions, dispatch] = useReducer(paperSessionsReducer, EMPTY_SESSIONS);
    return <ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()} onClosePaper={vi.fn()}
      onCitation={onCitation} modes={sessions.modes} onModeChange={(id, mode) => dispatch({ type: "setMode", id, mode })} />;
  }

  it("streams Markdown as it arrives, then opens cited chunks and styles checked sentences", async () => {
    const encoder = new TextEncoder();
    let stream!: ReadableStreamDefaultController<Uint8Array>;
    const push = (chunk: string) => stream.enqueue(encoder.encode(chunk));
    const fetcher = vi.fn().mockResolvedValue(new Response(new ReadableStream<Uint8Array>({ start(controller) { stream = controller; } })));
    vi.stubGlobal("fetch", fetcher);
    const onCitation = vi.fn();
    const { container } = render(<Chat onCitation={onCitation} />);
    expect(screen.getByRole("button", { name: "Strict" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "Explain" }));
    expect(screen.getByRole("button", { name: "Explain" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.change(screen.getByLabelText("Ask about Evidence and Verification"), { target: { value: "How does it work?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send question" }));
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
    expect(JSON.parse(String(fetcher.mock.calls[0]![1].body))).toEqual({ question: "How does it work?", mode: "explain" });

    push(event("progress", { paper_id: "paper-1", status: "reading", message: "Reading the whole paper" }));
    push(event("delta", { paper_id: "paper-1", text: text.slice(0, 24) }));
    expect(await screen.findByText("Method.")).toBeInTheDocument();
    expect(screen.queryByText(/isolated processes/)).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Reading the whole paper");
    push(event("delta", { paper_id: "paper-1", text: text.slice(24) }));
    expect(await screen.findByText(/isolated processes/)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Source c3, page not known yet" })[0]).toBeDisabled();
    for (const item of tail) push(item);
    stream.close();

    const chips = await screen.findAllByRole("button", { name: "Source c3, page 2" });
    expect(chips.map((chip) => chip.textContent)).toEqual(["p. 2", "p. 2"]);
    fireEvent.click(chips[1]!);
    expect(onCitation).toHaveBeenCalledWith({ page_start: 2, char_start: 10, char_end: 90, chunk_id: "paper-1:3" });
    expect(container.querySelector(".status-flagged")).toHaveTextContent("The gain is 9.1 points");
    expect(container.querySelector(".status-flagged")).toHaveAttribute("title", "A number is not in the cited passage");
    expect(container.querySelector(".status-background")).toHaveTextContent("Transformers are common.");
    expect(await screen.findByRole("button", { name: "Helpful" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Explanation" })).toBeInTheDocument();
  });

  it("re-renders a saved explanation from its events and escapes raw HTML in it", async () => {
    const saved = `${text} <img src=x onerror="alert(1)"><script>alert(2)</script>`;
    const events = [`event: delta\ndata: ${JSON.stringify({ paper_id: "paper-1", text: saved })}\n\n`, ...tail].map((block) => {
      const [kind, data] = block.trim().split("\n");
      return { event: kind!.slice(7), data: JSON.parse(data!.slice(6)) as Record<string, unknown> };
    });
    vi.mocked(getChat).mockResolvedValueOnce({ turns: [{ turn_id: "turn-9", question: "How does it work?", rating: "down", events }] });
    const onCitation = vi.fn();
    const { container } = render(<Chat onCitation={onCitation} />);
    const chips = await screen.findAllByRole("button", { name: "Source c3, page 2" });
    fireEvent.click(chips[0]!);
    expect(onCitation).toHaveBeenCalledWith(expect.objectContaining({ page_start: 2, chunk_id: "paper-1:3" }));
    expect(container.querySelector(".status-flagged")).toHaveTextContent("The gain is 9.1 points");
    expect(container.querySelector(".explanation-text img, .explanation-text script")).toBeNull();
    expect(screen.getByText(/<img src=x onerror="alert\(1\)">/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Not helpful" })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows the cited page and passage when a chip is hovered, read from the page text", async () => {
    // Characters 10 to 90 of page 2 are the cited chunk; the text around it must not appear.
    const chunk = "Isolated workers read one paper each and never see another paper. ".padEnd(80, "-").slice(0, 80);
    vi.mocked(getExtractedPage).mockResolvedValue({ number: 2, text: `BEFORE>>> ${chunk}AFTERWARDS` });
    const events = [`event: delta\ndata: ${JSON.stringify({ paper_id: "paper-1", text })}\n\n`, ...tail].map((block) => {
      const [kind, data] = block.trim().split("\n");
      return { event: kind!.slice(7), data: JSON.parse(data!.slice(6)) as Record<string, unknown> };
    });
    vi.mocked(getChat).mockResolvedValueOnce({ turns: [{ turn_id: "turn-9", question: "How does it work?", events }] });
    const onCitation = vi.fn();
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<QueryClientProvider client={client}><Chat onCitation={onCitation} /></QueryClientProvider>);
    const chips = await screen.findAllByRole("button", { name: "Source c3, page 2" });
    fireEvent.mouseOver(chips[0]!);
    const card = screen.getByRole("tooltip");
    expect(card).toHaveTextContent("Page 2");
    expect(card).toHaveTextContent("Not fully checked");
    await waitFor(() => expect(card).toHaveTextContent("Isolated workers read one paper each"));
    expect(card).not.toHaveTextContent("BEFORE");
    expect(card).not.toHaveTextContent("AFTERWARDS");
    expect(getExtractedPage).toHaveBeenCalledWith("paper-1", 2);
    fireEvent.mouseOut(chips[0]!);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    fireEvent.mouseOver(chips[1]!);
    fireEvent.click(chips[1]!);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    expect(onCitation).toHaveBeenCalledWith(expect.objectContaining({ page_start: 2, chunk_id: "paper-1:3" }));
  });

  it("says when an explanation stopped early", async () => {
    vi.mocked(getChat).mockResolvedValueOnce({ turns: [{ question: "Explain it", events: [
      { event: "delta", data: { paper_id: "paper-1", text: "Part of an answer [c3]." } },
      { event: "answer", data: { paper_id: "paper-1", mode: "explained_answer", heading: "Explanation", evaluation: { complete: false, fallback_reason: "stall_timeout" } } },
      { event: "done", data: { paper_id: "paper-1", mode: "explained_answer", status: "partial" } },
    ] }] });
    render(<Chat onCitation={vi.fn()} />);
    expect(await screen.findByText("This explanation stopped before it finished.")).toBeInTheDocument();
    expect(screen.getByText(/Part of an answer/)).toBeInTheDocument();
  });
});

describe("paper guide", () => {
  const guide = {
    tldr: "Cedar gates memory writes [c2].",
    problem: "",
    method_steps: ["Score each write [c4].", "Keep the best ones [c4]."],
    key_equations: [{ equation: "s = W x", meaning: "The write score [c4].", symbols: [{ term: "W", meaning: "A weight matrix [c4]." }] }],
    results: ["Accuracy rose to 68% [c2]."],
    limitations: [],
    glossary: [{ term: "write", meaning: "Storing an item [c2]." }],
    section_summaries: [{ section: "Results", summary: "Cedar beats the baseline [c2]." }],
    suggested_questions: ["How is the score trained?"],
  };
  const citations = [
    { marker: 2, chunk_id: "paper-1:2", page: 3, char_start: 5, char_end: 60, checked: true },
    { marker: 4, chunk_id: "paper-1:4", page: 4, char_start: 0, char_end: 40, checked: true },
  ];

  it("opens the chat with the cited guide and asks a suggested question", async () => {
    vi.mocked(getGuide).mockResolvedValueOnce({ paper_id: "paper-1", status: "ready", guide, citations });
    const fetcher = vi.fn().mockResolvedValue(new Response(""));
    vi.stubGlobal("fetch", fetcher);
    const onCitation = vi.fn();
    render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()}
      onClosePaper={vi.fn()} onCitation={onCitation} />);

    const opening = await screen.findByRole("article", { name: "Reading guide" });
    expect(within(opening).getByText(/Cedar gates memory writes/)).toBeInTheDocument();
    expect(within(opening).getByRole("heading", { name: "How it works" })).toBeInTheDocument();
    expect(within(opening).getByText("s = W x")).toBeInTheDocument();
    expect(within(opening).getByText("Results", { selector: "li strong" })).toBeInTheDocument();
    // Other papers' panes stay mounted (hidden), so look only in this paper's log.
    const log = opening.closest(".conversation-log") as HTMLElement;
    expect(within(log).queryByText("What would you like to understand?")).not.toBeInTheDocument();

    fireEvent.click(within(opening).getAllByRole("button", { name: "Source c4, page 4" })[0]!);
    expect(onCitation).toHaveBeenCalledWith({ page_start: 4, char_start: 0, char_end: 40, chunk_id: "paper-1:4" });

    fireEvent.click(within(opening).getByRole("button", { name: /How is the score trained\?/ }));
    await vi.waitFor(() => expect(fetcher).toHaveBeenCalledTimes(1));
    expect(JSON.parse(String(fetcher.mock.calls[0]![1].body))).toMatchObject({ question: "How is the score trained?" });
  });

  it("keeps the generic start while the guide is still being written", async () => {
    vi.mocked(getGuide).mockResolvedValueOnce({ paper_id: "paper-1", status: "pending" });
    render(<ChatWorkspace papers={papers} activePaperId="paper-1" onSelectPaper={vi.fn()}
      onClosePaper={vi.fn()} onCitation={vi.fn()} />);
    expect(await screen.findByText("The reading guide for this paper is being written.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /What is this paper about\?/ })).toBeInTheDocument();
  });
});
