import { useQuery } from "@tanstack/react-query";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  FileText,
  Quote,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  apiUrl,
  getExtractedPage,
  getPaper,
  type CitationRange,
} from "../api/client";
import { withoutPictureMarkers as shown } from "../pictureText";

interface Props {
  paperId: string;
  citation: CitationRange | null;
  onClose: () => void;
}

export function EvidenceDrawer({ paperId, citation, onClose }: Props) {
  const [page, setPage] = useState(citation?.page_start ?? 1);
  const [tab, setTab] = useState<"extracted" | "pdf">("extracted");
  const mark = useRef<HTMLElement | null>(null);
  const closeButton = useRef<HTMLButtonElement | null>(null);
  const paperQuery = useQuery({
    queryKey: ["paper", paperId],
    queryFn: () => getPaper(paperId),
  });
  const pageQuery = useQuery({
    queryKey: ["page", paperId, page],
    queryFn: () => getExtractedPage(paperId, page),
  });
  useEffect(() => {
    mark.current?.scrollIntoView?.({ block: "center" });
  }, [pageQuery.data, citation]);
  useEffect(() => {
    const previous = document.activeElement;
    closeButton.current?.focus();
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab") {
        const elements = closeButton.current
          ?.closest("aside")
          ?.querySelectorAll<HTMLElement>(
            "button:not(:disabled), a[href], iframe",
          );
        const first = elements?.[0];
        const last = elements?.[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        }
        if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("keydown", close);
      if (previous instanceof HTMLElement) previous.focus();
    };
  }, [onClose]);
  const text = pageQuery.data?.text ?? "";
  // A quote must match the page exactly. An explain citation has no quote: it marks its whole
  // chunk, cut at the end of the page when the chunk runs onto the next one.
  const end = citation ? Math.min(citation.char_end, text.length) : 0;
  const selected =
    citation?.page_start === page &&
    (citation.quote === undefined
      ? citation.char_start < end
      : citation.char_end <= text.length &&
        text.slice(citation.char_start, citation.char_end) === citation.quote);
  const pdf = apiUrl(`/v1/papers/${encodeURIComponent(paperId)}/pdf`);
  // The paper's own figure crops for the cited chunk's figures (S24); shown, never described.
  const cited = new Set(citation?.figures ?? []);
  const crops = (paperQuery.data?.figures ?? [])
    .filter((figure) => figure.has_crop && figure.crop_url && cited.has(figure.number))
    .filter((figure, i, all) => all.findIndex((f) => f.number === figure.number) === i)
    .slice(0, 3);
  return (
    <div
      className="evidence-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <aside
        className="evidence-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="evidence-title"
      >
        <header>
          <div>
            <p className="eyebrow">FOLLOW THE EVIDENCE</p>
            <h2 id="evidence-title">
              <FileText size={17} /> Source paper
            </h2>
          </div>
          <button
            ref={closeButton}
            className="icon-button"
            aria-label="Close source paper"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </header>
        <div className="evidence-paper-title">
          {paperQuery.data?.title ?? "Loading paper…"}
        </div>
        {citation?.quote && (
          <blockquote className="citation-selection">
            <Quote size={16} />
            <p>{shown(citation.quote)}</p>
            <span>Verified excerpt · page {citation.page_start}</span>
          </blockquote>
        )}
        {crops.map((figure) => (
          <figure className="evidence-figure" key={figure.number}>
            <button
              type="button"
              aria-label={`Open Figure ${figure.number} on page ${figure.page} of the PDF`}
              onClick={() => {
                setPage(figure.page);
                if (!paperQuery.data?.pdf_deleted) setTab("pdf");
              }}
            >
              <img src={apiUrl(figure.crop_url ?? "")} alt={`Figure ${figure.number} as drawn in the paper`} loading="lazy" />
            </button>
            <figcaption>{figure.caption || `Figure ${figure.number}`} · page {figure.page}</figcaption>
          </figure>
        ))}
        <div className="evidence-tabs">
          <button
            aria-pressed={tab === "extracted"}
            onClick={() => setTab("extracted")}
          >
            Extracted content
          </button>
          <button disabled={paperQuery.data?.pdf_deleted} aria-pressed={tab === "pdf"} onClick={() => setTab("pdf")}>
            Original PDF
          </button>
          {!paperQuery.data?.pdf_deleted && <a
            href={`${pdf}#page=${page}`}
            target="_blank"
            rel="noreferrer"
            aria-label="Open PDF in new tab"
          >
            <ArrowUpRight size={15} />
          </a>}
        </div>
        {paperQuery.data?.pdf_deleted && <p className="pdf-removed-notice" role="status">Original PDF removed. Extracted content remains available.</p>}
        <div className="page-toolbar">
          <button
            className="icon-button"
            aria-label="Previous page"
            disabled={page <= 1}
            onClick={() => setPage((value) => value - 1)}
          >
            <ChevronLeft size={16} />
          </button>
          <span>
            Page {page} of {paperQuery.data?.page_count ?? "…"}
          </span>
          <button
            className="icon-button"
            aria-label="Next page"
            disabled={page >= (paperQuery.data?.page_count ?? 1)}
            onClick={() => setPage((value) => value + 1)}
          >
            <ChevronRight size={16} />
          </button>
        </div>
        {tab === "pdf" && !paperQuery.data?.pdf_deleted ? (
          <iframe
            key={`${pdf}:${page}`}
            className="source-pdf"
            title={`Original PDF, page ${page}`}
            src={`${pdf}#page=${page}`}
          />
        ) : (
          <div className="extracted-page">
            {pageQuery.isPending ? (
              <p>Loading extracted page…</p>
            ) : pageQuery.error ? (
              <p role="alert">{pageQuery.error.message}</p>
            ) : (
              <pre>
                {selected && citation ? (
                  <>
                    {shown(text.slice(0, citation.char_start))}
                    <mark ref={mark}>
                      {shown(text.slice(citation.char_start, end))}
                    </mark>
                    {shown(text.slice(end))}
                  </>
                ) : (
                  shown(text)
                )}
              </pre>
            )}
          </div>
        )}
        <footer>
          Page locations use the original PDF. Highlights use the saved
          extraction.
        </footer>
      </aside>
    </div>
  );
}
