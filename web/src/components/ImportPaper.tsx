import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Check, Link2, LoaderCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { getImportStatus, importArxivPaper, type SyncJob } from "../api/client";

interface Props {
  onImported: (paperId: string) => void;
  onProgress: (job: SyncJob | null) => void;
}

export function ImportPaper({ onImported, onProgress }: Props) {
  const [url, setUrl] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const completed = useRef<string | null>(null);
  const queryClient = useQueryClient();
  const status = useQuery({
    queryKey: ["import", jobId],
    queryFn: () => getImportStatus(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      ["queued", "running"].includes(query.state.data?.status ?? "")
        ? 1000
        : false,
  });
  const mutation = useMutation({
    mutationFn: importArxivPaper,
    onSuccess: (job) => {
      queryClient.setQueryData(["import", job.job_id], job);
      setJobId(job.job_id);
    },
  });
  const job = status.data;
  const busy =
    mutation.isPending || job?.status === "queued" || job?.status === "running";
  useEffect(() => {
    onProgress(job ?? null);
    if (
      job?.status === "completed" &&
      job.paper_id &&
      completed.current !== job.job_id
    ) {
      completed.current = job.job_id;
      void queryClient.invalidateQueries({ queryKey: ["papers"] });
      void queryClient.invalidateQueries({ queryKey: ["paper", job.paper_id] });
      void queryClient.invalidateQueries({ queryKey: ["page", job.paper_id] });
      onImported(job.paper_id);
    }
  }, [job, onImported, onProgress, queryClient]);
  const error =
    mutation.error?.message ??
    status.error?.message ??
    (job?.status === "failed" ? job.message : null);
  return (
    <section className="import-paper" aria-label="Import an arXiv paper">
      <div className="import-heading">
        <Link2 size={14} />
        <strong>Read a specific paper</strong>
      </div>
      <p>Paste an arXiv PDF link. Papers from any year are welcome.</p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy && url.trim()) {
            setJobId(null);
            mutation.mutate(url.trim());
          }
        }}
      >
        <input
          aria-label="arXiv PDF URL"
          type="url"
          required
          placeholder="https://arxiv.org/pdf/1706.03762"
          value={url}
          onChange={(event) => setUrl(event.target.value)}
          disabled={busy}
        />
        <button
          type="submit"
          className="import-submit"
          disabled={busy || !url.trim()}
        >
          {busy ? (
            <LoaderCircle className="spin" size={12} />
          ) : (
            <ArrowRight size={12} />
          )}
          {busy ? "Extracting paper…" : "Import paper"}
        </button>
      </form>
      {job && ["queued", "running"].includes(job.status) && (
        <p className="import-progress" role="status">
          {job.message}
        </p>
      )}
      {job?.status === "completed" && job.paper_id && (
        <button
          className="import-ready"
          onClick={() => onImported(job.paper_id!)}
        >
          <Check size={12} />
          Paper ready · Open chat
          <ArrowRight size={11} />
        </button>
      )}
      {error && (
        <p className="import-error" role="alert">
          {error}
        </p>
      )}
      {job?.status === "interrupted" && (
        <p className="import-progress">
          Import interrupted. Submit the link again to resume.
        </p>
      )}
    </section>
  );
}
