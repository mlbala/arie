import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Download, X } from "lucide-react";
import {
  getSkippedPapers,
  overridePaper,
  type SkippedPaper,
} from "../api/client";

interface SkippedDrawerProps {
  onClose: () => void;
}

export function SkippedDrawer({ onClose }: SkippedDrawerProps) {
  const queryClient = useQueryClient();
  const skippedQuery = useQuery({
    queryKey: ["skipped"],
    queryFn: getSkippedPapers,
  });
  const overrideMutation = useMutation({
    mutationFn: overridePaper,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["skipped"] }),
        queryClient.invalidateQueries({ queryKey: ["papers"] }),
      ]);
    },
  });

  return (
    <div
      className="drawer-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <aside
        aria-labelledby="skipped-title"
        aria-modal="true"
        className="drawer"
        role="dialog"
      >
        <header className="drawer-heading">
          <div>
            <p className="eyebrow">Review queue</p>
            <h2 className="drawer-title" id="skipped-title">
              Skipped audit
            </h2>
          </div>
          <button
            aria-label="Close skipped audit"
            className="icon-button"
            onClick={onClose}
            type="button"
          >
            <X size={15} />
          </button>
        </header>
        <div className="drawer-content">
          <p className="settings-copy">
            Papers below could not be indexed. Review the reason before retrying
            a failed sync. Source and PDF safety gates remain enforced.
          </p>
          {skippedQuery.isPending ? (
            <div className="loading-state">Loading skipped papers…</div>
          ) : skippedQuery.isError ? (
            <div className="error-state">{skippedQuery.error.message}</div>
          ) : skippedQuery.data.items.length === 0 ? (
            <div className="empty-state">No papers are waiting for audit.</div>
          ) : (
            skippedQuery.data.items.map((paper) => (
              <AuditCard
                key={paper.paper_id}
                onOverride={(paperId) => overrideMutation.mutate(paperId)}
                paper={paper}
                pending={
                  overrideMutation.isPending &&
                  overrideMutation.variables === paper.paper_id
                }
              />
            ))
          )}
          {overrideMutation.error && (
            <p className="chat-error" role="alert">
              {overrideMutation.error.message}
            </p>
          )}
        </div>
      </aside>
    </div>
  );
}

interface AuditCardProps {
  paper: SkippedPaper;
  pending: boolean;
  onOverride: (paperId: string) => void;
}

function AuditCard({ paper, pending, onOverride }: AuditCardProps) {
  return (
    <article className="audit-card">
      <h3>{paper.title}</h3>
      <div className="audit-card-meta">
        <span>{paper.source}</span>
        <span>
          <AlertTriangle size={10} /> {paper.status}
        </span>
      </div>
      <p>{paper.reason}</p>
      <button
        className="audit-action"
        disabled={
          pending ||
          paper.status === "SKIPPED" ||
          paper.status === "QUARANTINED"
        }
        onClick={() => onOverride(paper.paper_id)}
        type="button"
      >
        <Download size={11} />
        {pending
          ? "Queueing sync…"
          : paper.status === "SKIPPED" || paper.status === "QUARANTINED"
            ? "Held by safety gate"
            : "Retry sync"}
      </button>
    </article>
  );
}
