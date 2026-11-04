/** Explain-mode answers: parse their events and render the streamed Markdown safely (SPEC 11). */
import MarkdownIt from "markdown-it";

/** One sentence verdict; `start` and `end` index the answer text, markers included. */
export interface SentenceStatus {
  start: number;
  end: number;
  text: string;
  status: string;
  reasons: string[];
}

/** A `[cN]` marker resolved to its chunk; offsets are on the chunk's first page. */
export interface ExplainCitation {
  marker: number;
  chunkId: string;
  page: number;
  charStart: number;
  charEnd: number;
  checked: boolean;
}

// No raw HTML, and nothing that loads or links out: a paper could steer the model into
// writing an image or link that leaks text to another host.
const markdown = new MarkdownIt({ html: false, linkify: false, typographer: false })
  .disable(["link", "image", "autolink", "reference"]);

// Sentence wrappers are placed in the source before Markdown runs, as two punctuation
// characters so emphasis next to them still opens and closes.
const OPEN = "⸜";
const CLOSE = "⸝";
const SENTINELS = /[⸜⸝]/g;

const REASONS: Record<string, string> = {
  invalid_marker: "Cites a passage the model was not given",
  number_not_in_source: "A number is not in the cited passage",
  lost_qualifier: "Drops a qualifier the cited passage has",
  uncited_number: "States a number without a citation",
};
const STATUS_TITLES: Record<string, string> = {
  background: "Background, not cited from the paper",
};

/** Reader-facing text for a check's reason code; unknown codes are shown as words. */
export function reasonLabel(code: string): string {
  return REASONS[code] ?? code.replace(/_/g, " ");
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => `&#${char.charCodeAt(0)};`);
}

/** The parts of a sentence to wrap: table cells separately, without `|`, `>` or spaces. */
function segments(text: string, start: number, end: number): [number, number][] {
  const pieces: [number, number][] = [];
  let from = start;
  for (let index = start; index <= end; index += 1) {
    if (index < end && text[index] !== "|") continue;
    let a = from;
    let b = index;
    while (a < b && /[\s>|]/.test(text[a]!)) a += 1;
    while (b > a && /\s/.test(text[b - 1]!)) b -= 1;
    if (b > a) pieces.push([a, b]);
    from = index + 1;
  }
  return pieces;
}

/**
 * Render an explanation as HTML: Markdown with raw HTML escaped, each checked sentence in a
 * `status-<status>` span (reasons in its title) and each `[cN]` as a page chip. A chip is a
 * `button[data-marker=N]`; `citations` undefined means they have not arrived yet.
 */
export function renderExplanation(
  text: string,
  sentences: SentenceStatus[],
  citations: ExplainCitation[] | undefined,
): string {
  const clean = text.replace(SENTINELS, "�");
  const wrapped: SentenceStatus[] = [];
  let source = "";
  let cursor = 0;
  for (const sentence of [...sentences].sort((a, b) => a.start - b.start)) {
    const start = clean.slice(sentence.start, sentence.end) === sentence.text
      ? sentence.start : clean.indexOf(sentence.text, cursor);
    if (start < cursor || !sentence.text) continue;
    for (const [a, b] of segments(clean, start, start + sentence.text.length)) {
      source += `${clean.slice(cursor, a)}${OPEN}${clean.slice(a, b)}${CLOSE}`;
      wrapped.push(sentence);
      cursor = b;
    }
  }
  source += clean.slice(cursor);

  let next = 0;
  let open = 0;
  const html = markdown.render(source).replace(SENTINELS, (char) => {
    if (char === CLOSE) {
      if (!open) return "";
      open -= 1;
      return "</span>";
    }
    const sentence = wrapped[next++];
    if (!sentence) return "";
    open += 1;
    const status = sentence.status.replace(/[^\w-]/g, "");
    const title = sentence.reasons.length
      ? sentence.reasons.map(reasonLabel).join("; ") : STATUS_TITLES[sentence.status];
    return `<span class="answer-sentence status-${status}" data-status="${escapeHtml(sentence.status)}"`
      + `${title ? ` title="${escapeHtml(title)}"` : ""}>`;
  });

  const byMarker = new Map(citations?.map((citation) => [citation.marker, citation]));
  return html.replace(/\[c(\d+)\]/g, (_, digits: string) => {
    const marker = Number(digits);
    const cited = byMarker.get(marker);
    if (cited) return `<button type="button" class="citation-chip${cited.checked ? "" : " unchecked"}"`
      + ` data-marker="${marker}" aria-label="Source c${marker}, page ${cited.page}">p. ${cited.page}</button>`;
    const label = citations ? "not a passage of this paper" : "page not known yet";
    return `<button type="button" class="citation-chip missing" data-marker="${marker}" disabled`
      + ` aria-label="Source c${marker}, ${label}">c${marker}</button>`;
  });
}

/** Sentences from any event that carries them (`checks`, later checkers too). */
export function normalizeSentences(value: unknown): SentenceStatus[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item: unknown) => {
    if (typeof item !== "object" || item === null) return [];
    const { start, end, text, status, reasons } = item as Record<string, unknown>;
    if (typeof start !== "number" || typeof end !== "number" || typeof text !== "string"
      || typeof status !== "string" || !Number.isInteger(start) || start < 0 || end < start) return [];
    const codes = Array.isArray(reasons) ? reasons.filter((code) => typeof code === "string") : [];
    return [{ start, end, text, status, reasons: codes }];
  });
}

/** A later verdict replaces the status of the same sentence, except that a flag stays. */
export function mergeSentences(current: SentenceStatus[], incoming: SentenceStatus[]): SentenceStatus[] {
  const merged = new Map(current.map((sentence) => [`${sentence.start}:${sentence.end}`, sentence]));
  for (const sentence of incoming) {
    const key = `${sentence.start}:${sentence.end}`;
    const previous = merged.get(key);
    merged.set(key, previous ? {
      ...sentence,
      status: previous.status === "flagged" ? "flagged" : sentence.status,
      reasons: [...new Set([...previous.reasons, ...sentence.reasons])],
    } : sentence);
  }
  return [...merged.values()].sort((a, b) => a.start - b.start);
}

/** The `citations` event's list; entries without a page or valid offsets are dropped. */
export function normalizeCitations(value: unknown): ExplainCitation[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item: unknown) => {
    if (typeof item !== "object" || item === null) return [];
    const { marker, chunk_id, page, char_start, char_end, checked } = item as Record<string, unknown>;
    if (typeof marker !== "number" || typeof chunk_id !== "string" || typeof page !== "number"
      || typeof char_start !== "number" || typeof char_end !== "number" || !Number.isInteger(page)
      || page < 1 || char_start < 0 || char_end <= char_start) return [];
    return [{ marker, chunkId: chunk_id, page, charStart: char_start, charEnd: char_end, checked: checked === true }];
  });
}
