import { ArrowUpRight, BookOpen } from "lucide-react";
import type { MouseEvent } from "react";
import type { CitationRange, PaperGuide } from "../api/client";
import { renderExplanation, type ExplainCitation } from "../explanation";
import { guideMarkdown } from "../guide";

/** A paper's opening message: its cited guide and the questions it suggests. */
export function PaperGuideMessage({ guide, citations, disabled, onAsk, onCitation }: {
  guide: PaperGuide;
  citations: ExplainCitation[];
  disabled: boolean;
  onAsk: (question: string) => void;
  onCitation: (citation: CitationRange) => void;
}) {
  const html = renderExplanation(guideMarkdown(guide), [], citations);
  const open = (event: MouseEvent<HTMLDivElement>) => {
    const chip = event.target instanceof Element ? event.target.closest("button[data-marker]") : null;
    const cited = citations.find((citation) => String(citation.marker) === chip?.getAttribute("data-marker"));
    if (cited) onCitation({
      page_start: cited.page,
      char_start: cited.charStart,
      char_end: cited.charEnd,
      chunk_id: cited.chunkId,
    });
  };
  return <article className="conversation-turn paper-guide" aria-label="Reading guide">
    <div className="assistant-turn">
      <div className="answer-heading">
        <span className="assistant-avatar"><BookOpen size={13} /></span>
        <strong>ARIE</strong>
        <span>Reading guide</span>
      </div>
      <div className="explained-answer">
        <div className="explanation-text" onClick={open} dangerouslySetInnerHTML={{ __html: html }} />
      </div>
      {guide.suggested_questions.length > 0 && <div className="question-suggestions">
        {guide.suggested_questions.map((question) => (
          <button key={question} onClick={() => onAsk(question)} disabled={disabled}>
            {question}
            <ArrowUpRight size={14} />
          </button>
        ))}
      </div>}
    </div>
  </article>;
}
