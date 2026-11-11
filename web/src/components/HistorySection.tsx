import { useMutation } from "@tanstack/react-query";
import { BookOpen, History, Trash2 } from "lucide-react";
import { useState } from "react";
import { clearAllChats, type PaperSummary } from "../api/client";

interface HistorySectionProps {
  /** Recently viewed papers, most recent first. */
  papers: PaperSummary[];
  max: number;
  onSelect: (paperId: string) => void;
  /** Empty the recently viewed list only; saved chats are untouched. */
  onClearList: () => void;
  /** Every saved chat was deleted on the server. */
  onChatsCleared: () => void;
}

/**
 * The sidebar's History: the recently viewed papers, plus the one control that deletes every
 * saved chat. The title and the trash icon stay when the list is empty.
 */
export function HistorySection({ papers, max, onSelect, onClearList, onChatsCleared }: HistorySectionProps) {
  const [confirming, setConfirming] = useState(false);
  const clear = useMutation({
    mutationFn: clearAllChats,
    onSuccess: () => {
      setConfirming(false);
      onChatsCleared();
    },
  });
  return (
    <section className="history-section" aria-label="History">
      <div className="library-heading">
        <span>
          <History size={15} /> History
        </span>
        <span className="history-actions">
          {papers.length > 0 && (
            <button className="text-button history-clear" title="Empties this list only; saved chats are kept" onClick={onClearList}>
              Clear list
            </button>
          )}
          <button
            className="icon-button history-delete"
            aria-label="Delete all chat history"
            title="Delete all chat history"
            disabled={clear.isPending}
            onClick={() => { setConfirming(true); clear.reset(); }}
          >
            <Trash2 size={13} />
          </button>
        </span>
      </div>
      {confirming && (
        <div className="delete-confirm" role="group" aria-label="Confirm chat deletion">
          <p>Delete the saved chat of every paper? This cannot be undone. Papers and your ratings stay.</p>
          <button className="button danger-button" disabled={clear.isPending} onClick={() => clear.mutate()}>
            {clear.isPending ? "Deleting…" : "Confirm delete"}
          </button>
          <button className="text-button" onClick={() => setConfirming(false)}>Cancel</button>
        </div>
      )}
      {clear.error && <p className="sync-error" role="alert">{clear.error.message}</p>}
      {papers.length > 0 && (
        <>
          <ul className="history-list">
            {papers.map((paper) => (
              <li key={paper.id}>
                <button title={paper.title} onClick={() => onSelect(paper.id)}>
                  <BookOpen size={12} />
                  <span>{paper.title}</span>
                </button>
              </li>
            ))}
          </ul>
          <p className="history-caption">{papers.length} of {max} · reopen to pin again</p>
        </>
      )}
    </section>
  );
}
