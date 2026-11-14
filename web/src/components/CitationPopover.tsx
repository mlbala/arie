import { useQuery } from "@tanstack/react-query";
import { FileText } from "lucide-react";
import { useEffect } from "react";
import { createPortal } from "react-dom";
import { getExtractedPage } from "../api/client";
import type { ExplainCitation } from "../explanation";
import { withoutPictureMarkers } from "../pictureText";

/** What a citation chip shows on hover. `quote` is undefined while it loads, null if it cannot be shown. */
export interface CitationHover {
  anchor: DOMRect;
  page: number;
  quote: string | null | undefined;
  table?: boolean;
  /** A table row's caption and header (S27), shown above the quote for reading only. */
  context?: string | undefined;
  unchecked?: boolean;
}

const WIDTH = 340;
const MARGIN = 12;
const GAP = 8;
const EXCERPT_CHARS = 600;

/**
 * A hover card with a citation's page and quote. It sits in `document.body` at fixed
 * coordinates so no scroll container clips it, and it goes away on Escape or scroll.
 */
export function CitationPopover({ id, hover, onDismiss }: { id: string; hover: CitationHover; onDismiss: () => void }) {
  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onDismiss();
    };
    document.addEventListener("keydown", escape);
    // The card is fixed, so a scroll would leave it behind the chip it describes.
    window.addEventListener("scroll", onDismiss, { capture: true, passive: true });
    return () => {
      document.removeEventListener("keydown", escape);
      window.removeEventListener("scroll", onDismiss, { capture: true });
    };
  }, [onDismiss]);
  const { anchor } = hover;
  const width = Math.min(WIDTH, window.innerWidth - 2 * MARGIN);
  const centred = anchor.left + anchor.width / 2 - width / 2;
  const left = Math.min(Math.max(centred, MARGIN), window.innerWidth - width - MARGIN);
  // Above the chip, unless it is near the top of the window.
  const style = anchor.top > 220
    ? { left, bottom: window.innerHeight - anchor.top + GAP }
    : { left, top: anchor.bottom + GAP };
  return createPortal(
    <div role="tooltip" id={id} className="citation-popover" style={style}>
      <header>
        <FileText size={11} /> Page {hover.page}
        {hover.unchecked && <span className="citation-popover-flag">Not fully checked</span>}
      </header>
      {hover.quote === undefined
        ? <p>Loading passage…</p>
        : hover.quote === null
          ? <p>Passage preview unavailable. Click to open it.</p>
          : <>
            {hover.context && <QuoteContext text={hover.context} />}
            <blockquote className={hover.table ? "table-quote" : undefined}>{hover.quote}</blockquote>
          </>}
      <footer>Click to open in the paper</footer>
    </div>,
    document.body,
  );
}

/** A table row's caption and header, set apart from the quote: context to read it by, not quoted. */
export function QuoteContext({ text }: { text: string }) {
  return <div className="quote-context">
    <span>Table header · context, not quoted</span>
    <pre>{text.trim()}</pre>
  </div>;
}

/** Explain-mode citations carry no quote, so the hover card reads the cited chunk from its page. */
export function ExplainCitationPopover({ id, paperId, citation, anchor, onDismiss }: {
  id: string;
  paperId: string;
  citation: ExplainCitation;
  anchor: DOMRect;
  onDismiss: () => void;
}) {
  // The evidence drawer reads the same key, so opening the chip afterwards is a cache hit.
  const page = useQuery({
    queryKey: ["page", paperId, citation.page],
    queryFn: () => getExtractedPage(paperId, citation.page),
  });
  const text = page.data?.text;
  const quote = text === undefined
    ? page.isError ? null : undefined
    : excerpt(text.slice(citation.charStart, Math.min(citation.charEnd, text.length)));
  return <CitationPopover id={id} onDismiss={onDismiss}
    hover={{ anchor, page: citation.page, quote, unchecked: !citation.checked }} />;
}

/** One readable run of the chunk, cut at a word when it is long. */
function excerpt(text: string): string | null {
  const flat = withoutPictureMarkers(text).replace(/\s+/g, " ").trim();
  if (!flat) return null;
  if (flat.length <= EXCERPT_CHARS) return flat;
  return `${flat.slice(0, EXCERPT_CHARS).replace(/\s+\S*$/, "")}…`;
}
