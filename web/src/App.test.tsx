import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import App from "./App";

it("opens an imported older paper even when it is outside the first library page", async () => {
  const older = {
    id: "1706.03762",
    title: "Attention Is All You Need",
    source: "arxiv",
    status: "INDEXED",
    published_at: "2017-06-12",
    abstract: "A paper about attention.",
    relevance_gate: true,
    overridden_by_user: 0,
    page_count: 15,
    outline: [],
    revision: "old-paper-version",
  };
  const recent = Array.from({ length: 20 }, (_, index) => ({
    ...older,
    id: `new-${index}`,
    title: `Recent paper ${index}`,
    published_at: "2026-10-01",
  }));
  const job = {
    job_id: "import-old",
    kind: "import",
    paper_id: older.id,
    source: "arxiv",
    status: "completed",
    sources: ["arxiv"],
    since: older.published_at,
    until: "2026-11-02",
    phase: "complete",
    message: "Paper ready to chat",
    discovered: 1,
    indexed: 1,
    processed: 1,
    failed: 0,
    skipped: 0,
    duplicates: 0,
    source_errors: [],
    current_paper: null,
  };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      let body: unknown;
      if (url.startsWith("/v1/papers?"))
        body = { items: recent, next_cursor: "20", has_more: true };
      else if (url === "/v1/plugins") body = { plugins: [] };
      else if (url === "/v1/sync") body = { job: null, initial_months: 6 };
      else if (url === "/v1/storage") body = {
        checked_at: "2026-11-02T00:00:00Z", active_jobs: [],
        database: { provider: "sqlite", status: "healthy", size_bytes: 1024, papers: 21, evidence_chunks: 0 },
        vector_store: { provider: "exact", status: "inactive", size_bytes: null, detail: "Not populated" },
        file_storage: { provider: "local", status: "healthy", size_bytes: 0, files: 0, disk_free_bytes: null, disk_total_bytes: null },
      };
      else if (url.startsWith("/v1/imports")) body = job;
      else if (url === "/v1/papers/1706.03762") body = older;
      else throw new Error(`Unexpected request: ${url}`);
      return new Response(JSON.stringify(body), {
        headers: { "content-type": "application/json" },
      });
    }),
  );
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  );
  await screen.findByRole("heading", { name: "Recent paper 0" }, { timeout: 3000 });
  fireEvent.click(screen.getByRole("button", { name: "Import & sync" }));
  fireEvent.change(screen.getByRole("textbox", { name: "arXiv PDF URL" }), {
    target: { value: "https://arxiv.org/pdf/1706.03762" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Import paper" }));
  expect(
    await screen.findByRole("heading", { name: older.title, level: 1 }),
  ).toBeInTheDocument();
  expect(
    screen.getByRole("textbox", { name: `Ask about ${older.title}` }),
  ).toBeEnabled();
  client.clear();
});

it("remembers panel placement and sends month/domain filters to the library API", async () => {
  localStorage.removeItem("arie.ui.layout");
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    let body: unknown;
    if (url.startsWith("/v1/papers")) body = { items: [], next_cursor: null, has_more: false };
    else if (url === "/v1/plugins") body = { plugins: [] };
    else if (url === "/v1/sync") body = { job: null, initial_months: 6 };
    else if (url === "/v1/storage") body = {
      checked_at: "2026-11-02T00:00:00Z", active_jobs: [],
      database: { provider: "sqlite", status: "healthy", size_bytes: 0, papers: 0, evidence_chunks: 0 },
      vector_store: { provider: "exact", status: "inactive", size_bytes: null, detail: "Not populated" },
      file_storage: { provider: "local", status: "healthy", size_bytes: 0, files: 0, disk_free_bytes: null, disk_total_bytes: null },
    };
    else throw new Error(`Unexpected request: ${url}`);
    return new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetcher);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
  fireEvent.click(screen.getByText("Layout"));
  fireEvent.change(screen.getByLabelText("Library position"), { target: { value: "right" } });
  fireEvent.change(screen.getByLabelText("Import form position"), { target: { value: "library" } });
  expect(JSON.parse(localStorage.getItem("arie.ui.layout")!)).toEqual({ library: "right", import: "library", libraryWidth: 280, activityWidth: 300, sizing: "auto" });
  expect(document.querySelector(".research-shell")).toHaveClass("library-right");
  fireEvent.change(screen.getByLabelText("Panel sizing"), { target: { value: "custom" } });
  fireEvent.change(screen.getByLabelText("Library width"), { target: { value: "340" } });
  expect(document.querySelector(".research-shell")).toHaveStyle({ "--library-width": "340px" });
  expect(JSON.parse(localStorage.getItem("arie.ui.layout")!).libraryWidth).toBe(340);
  expect(screen.getByRole("complementary", { name: "Research library and pipeline" })).toContainElement(screen.getByRole("textbox", { name: "arXiv PDF URL" }));
  await vi.waitFor(() => expect(fetcher.mock.calls.some(([url]) => String(url).includes("months=6"))).toBe(true));
  fireEvent.change(screen.getByLabelText("Activity pane width"), { target: { value: "320" } });
  expect(document.querySelector(".research-shell")).toHaveStyle({ "--activity-width": "320px" });
  const library = within(screen.getByRole("region", { name: "Library papers" }));
  expect(library.queryByLabelText("Publication window")).not.toBeInTheDocument();
  fireEvent.click(library.getByRole("button", { name: "Filter & sort" }));
  fireEvent.change(library.getByLabelText("Publication window"), { target: { value: "month" } });
  fireEvent.change(library.getByLabelText("Publication year"), { target: { value: "2026" } });
  fireEvent.click(library.getByRole("button", { name: "June" }));
  fireEvent.change(screen.getByLabelText("Research domain"), { target: { value: "agents" } });
  await vi.waitFor(() => expect(fetcher.mock.calls.some(([url]) => String(url).includes("month=2026-06&domain=agents"))).toBe(true));
  fireEvent.change(library.getByLabelText("Publication window"), { target: { value: "year" } });
  await vi.waitFor(() => expect(fetcher.mock.calls.some(([url]) => String(url).includes("year=2026&domain=agents"))).toBe(true));
  fireEvent.change(library.getByLabelText("Publication window"), { target: { value: "all" } });
  await vi.waitFor(() => expect(String(fetcher.mock.calls.at(-1)?.[0])).toBe("/v1/papers?domain=agents"));
  fireEvent.change(library.getByLabelText("Number of months"), { target: { value: "24" } });
  await vi.waitFor(() => expect(String(fetcher.mock.calls.at(-1)?.[0])).toBe("/v1/papers?months=24&domain=agents"));
  fireEvent.change(library.getByLabelText("Number of months"), { target: { value: "all" } });
  expect(library.getByLabelText("Number of months")).toHaveValue("all");
  expect(library.getByLabelText("Publication window")).toHaveValue("all");
  await vi.waitFor(() => expect(String(fetcher.mock.calls.at(-1)?.[0])).toBe("/v1/papers?domain=agents"));
  fireEvent.change(library.getByLabelText("Number of months"), { target: { value: "12" } });
  await vi.waitFor(() => expect(String(fetcher.mock.calls.at(-1)?.[0])).toBe("/v1/papers?months=12&domain=agents"));
  fireEvent.click(library.getByRole("button", { name: "Reset filters" }));
  expect(library.getByText("Last 6 months", { selector: ".library-filter-summary" })).toBeInTheDocument();
  expect(library.getByRole("button", { name: "Filter & sort" })).not.toHaveClass("is-active");
  fireEvent.click(library.getByRole("button", { name: "Show results" }));
  expect(library.queryByLabelText("Publication window")).not.toBeInTheDocument();
  expect(library.getByRole("button", { name: "Filter & sort" })).toHaveAttribute("aria-expanded", "false");
  client.clear();
  localStorage.removeItem("arie.ui.layout");
});

it("shows relevant search choices without automatically opening the first paper", async () => {
  const base = {
    source: "arxiv", status: "INDEXED", published_at: "2017-06-12",
    abstract: "A Transformer model for sequence transduction.",
    relevance_gate: true, overridden_by_user: 0, page_count: 15,
  };
  const match = { ...base, id: "1706.03762", title: "Attention Is All You Need", match_reason: "Similar title" };
  const recent = { ...base, id: "recent", title: "Recent Research" };
  const fetcher = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    let body: unknown;
    if (url.startsWith("/v1/papers")) body = {
      items: url.includes("search=") ? [match] : [recent], next_cursor: null, has_more: false,
    };
    else if (url === "/v1/plugins") body = { plugins: [] };
    else if (url === "/v1/sync") body = { job: null, initial_months: 6 };
    else throw new Error(`Unexpected request: ${url}`);
    return new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetcher);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
  await screen.findByRole("heading", { name: "Recent Research" });
  fireEvent.change(screen.getByRole("textbox", { name: "Search library" }), {
    target: { value: "attenti all you need" },
  });
  const choice = await screen.findByRole("button", { name: /Attention Is All You Need/ }, { timeout: 3000 });
  expect(screen.queryByRole("heading", { name: match.title })).not.toBeInTheDocument();
  expect(screen.getByText("Choose a paper from the library results to start asking questions.")).toBeInTheDocument();
  fireEvent.click(choice);
  expect(await screen.findByRole("heading", { name: match.title })).toBeInTheDocument();
  expect(screen.getByRole("textbox", { name: `Ask about ${match.title}` })).toBeEnabled();
  client.clear();
});


it("automatically adjusts panels when the window resizes and preserves choices within a layout", async () => {
  const originalWidth = window.innerWidth;
  localStorage.removeItem("arie.ui.layout");
  const resize = (width: number) => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: width });
    fireEvent(window, new Event("resize"));
  };
  resize(1440);
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const body = url.startsWith("/v1/papers") ? { items: [], next_cursor: null }
      : url === "/v1/sync" ? { job: null, initial_months: 6 }
      : url === "/v1/plugins" ? { plugins: [] }
      : { active_jobs: [], checked_at: "2026-11-02", database: { status: "healthy" }, file_storage: { status: "healthy" }, vector_store: {} };
    return new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  try {
    render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
    const toggle = screen.getByRole("button", { name: "Import & sync" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(document.querySelector(".system-pane")).toHaveClass("is-docked");
    fireEvent.click(toggle);
    resize(1500);
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    resize(900);
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(document.querySelector(".system-pane")).toHaveClass("is-overlay");
    fireEvent.click(toggle);
    expect(screen.getByRole("button", { name: "Close system pane" })).toBeInTheDocument();
    resize(1600);
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(document.querySelector(".system-pane")).toHaveClass("is-docked");
    expect(screen.queryByRole("button", { name: "Close system pane" })).not.toBeInTheDocument();
    resize(390);
    fireEvent.click(screen.getByRole("button", { name: "Open library" }));
    expect(document.querySelector(".library-sidebar")).toHaveClass("is-open");
    resize(1000);
    resize(390);
    expect(document.querySelector(".library-sidebar")).not.toHaveClass("is-open");
    resize(1280);
    fireEvent.click(screen.getByText("Layout"));
    fireEvent.change(screen.getByLabelText("Panel sizing"), { target: { value: "custom" } });
    fireEvent.change(screen.getByLabelText("Library width"), { target: { value: "400" } });
    fireEvent.change(screen.getByLabelText("Activity pane width"), { target: { value: "360" } });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(document.querySelector(".system-pane")).toHaveClass("is-overlay");
    fireEvent.change(screen.getByLabelText("Panel sizing"), { target: { value: "auto" } });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    expect(document.querySelector(".system-pane")).toHaveClass("is-docked");
  } finally {
    resize(originalWidth);
    client.clear();
    localStorage.removeItem("arie.ui.layout");
  }
});

it("pins five paper chats and shifts older ones into a clearable sidebar history", async () => {
  const papers = Array.from({ length: 7 }, (_, index) => ({
    id: `paper-${index}`,
    title: `Pinned candidate ${index}`,
    source: "arxiv",
    status: "INDEXED",
    published_at: "2026-10-01",
    abstract: "A paper.",
    relevance_gate: true,
    overridden_by_user: 0,
  }));
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const body = url.startsWith("/v1/papers?") ? { items: papers, next_cursor: null, has_more: false }
      : url === "/v1/plugins" ? { plugins: [] }
      : url === "/v1/sync" ? { job: null, initial_months: 6 }
      : { checked_at: "2026-11-02", active_jobs: [], database: { status: "healthy" }, file_storage: { status: "healthy" }, vector_store: {} };
    return new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><App /></QueryClientProvider>);
  await screen.findByRole("heading", { name: "Pinned candidate 0" });
  const library = screen.getByRole("region", { name: "Library papers" });
  const tabs = () => within(screen.getByRole("navigation", { name: "Active paper conversations" }));
  expect(tabs().getAllByRole("button", { name: /^Close / })).toHaveLength(1);

  for (const paper of papers.slice(0, 6))
    fireEvent.click(within(library).getByRole("button", { name: new RegExp(paper.title) }));

  expect(tabs().getAllByRole("button", { name: /^Close / })).toHaveLength(5);
  expect(tabs().queryByRole("button", { name: "Pinned candidate 0" })).not.toBeInTheDocument();
  const history = screen.getByRole("region", { name: "Recently viewed papers" });
  expect(within(history).getByRole("button", { name: "Pinned candidate 0" })).toBeInTheDocument();

  fireEvent.click(tabs().getByRole("button", { name: "Close Pinned candidate 3" }));
  expect(tabs().getAllByRole("button", { name: /^Close / })).toHaveLength(4);
  expect(within(history).getAllByRole("listitem")).toHaveLength(2);

  fireEvent.click(within(history).getByRole("button", { name: "Clear history" }));
  expect(screen.queryByRole("region", { name: "Recently viewed papers" })).not.toBeInTheDocument();
  client.clear();
});
