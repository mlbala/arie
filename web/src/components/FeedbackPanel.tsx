import { useState } from "react";
import {
  FEEDBACK_REASONS,
  addFeedbackFacts,
  type FeedbackAgreement,
  type FeedbackItem,
  type ImplicitFeedback,
} from "../api/client";

const REASON_LABEL: Record<string, string> = Object.fromEntries(FEEDBACK_REASONS);
const SIGNAL_LABEL: Record<ImplicitFeedback["signal"], string> = {
  said_wrong: "said the answer was wrong",
  asked_again: "asked for it again",
  repeated_question: "asked the same question again",
};
const FACT_SLOTS = 3;

function percent(value: number | null): string {
  return value === null ? "–" : `${Math.round(value * 100)}%`;
}

/**
 * What the reader's ratings found: the judge's agreement with them, the answers rated down
 * (each can become a scored case with 1-3 key facts) and the answers a follow-up questioned.
 * Ratings find failures; they are not a quality score.
 */
export function FeedbackPanel({
  paperId,
  feedback,
  implicit,
  agreement,
  onChanged,
}: {
  paperId: string;
  feedback: FeedbackItem[];
  implicit: ImplicitFeedback[];
  agreement: FeedbackAgreement | undefined;
  onChanged: () => void;
}) {
  const down = feedback.filter((item) => item.rating === "down");
  if (!down.length && !implicit.length && !agreement?.rated) return null;
  return (
    <section className="feedback-panel" aria-label="Feedback on answers">
      {agreement && agreement.rated > 0 && <AgreementSummary agreement={agreement} />}
      {down.length > 0 && (
        <>
          <h3>Answers you rated down ({down.length})</h3>
          <ul className="feedback-list">
            {down.map((item) => (
              <FeedbackRow key={item.turn_id} paperId={paperId} item={item} onSaved={onChanged} />
            ))}
          </ul>
        </>
      )}
      {implicit.length > 0 && (
        <details>
          <summary>Answers a follow-up questioned ({implicit.length})</summary>
          <ul className="feedback-list">
            {implicit.map((item) => (
              <li key={item.turn_id} className="feedback-item">
                <strong>{item.question}</strong>
                <p>
                  You then {SIGNAL_LABEL[item.signal] ?? "questioned it"}: “{item.follow_up}”
                </p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function AgreementSummary({ agreement }: { agreement: FeedbackAgreement }) {
  const tasks = agreement.by_task ?? [];
  return (
    <div className="feedback-agreement">
      <h3>Judge agreement</h3>
      {agreement.judged > 0 ? (
        <p>
          The judge matched your rating on <strong>{agreement.agreed} of {agreement.judged}</strong> judged answers (
          <strong>{percent(agreement.rate)}</strong>). {agreement.rated - agreement.judged} of {agreement.rated} rated
          answers had no judge verdict; only explain answers are judged.
        </p>
      ) : (
        <p>No rated answer has a judge verdict yet; only explain answers are judged.</p>
      )}
      <p>
        {agreement.ready
          ? `${agreement.rated} answers are rated: enough to read gate 4.`
          : `Gate 4 is read at 30 rated answers; ${agreement.rated} so far.`}
        {agreement.implicit > 0 && ` ${agreement.implicit} unrated answers were questioned by a follow-up.`}
      </p>
      {tasks.length > 1 && (
        <ul aria-label="Agreement per task">
          {tasks.map((row) => (
            <li key={row.task}>
              {row.task}: {row.agreed} of {row.judged} judged ({percent(row.rate)}), {row.rated} rated
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function FeedbackRow({ paperId, item, onSaved }: { paperId: string; item: FeedbackItem; onSaved: () => void }) {
  const scored = Boolean(item.key_facts?.length);
  const [facts, setFacts] = useState<string[]>(() =>
    Array.from({ length: FACT_SLOTS }, (_, index) => item.key_facts?.[index] ?? ""),
  );
  const [sentence, setSentence] = useState(item.source_sentence ?? "");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const keyFacts = facts.map((fact) => fact.trim()).filter(Boolean);
    if (!keyFacts.length) {
      setError("Add at least one key fact.");
      return;
    }
    setError("");
    setSaving(true);
    addFeedbackFacts(paperId, item.turn_id, keyFacts, sentence.trim())
      .then(onSaved)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Could not save the key facts."))
      .finally(() => setSaving(false));
  };

  const meta = [item.mode, item.prompt_version, item.trace_id].filter(Boolean).join(" · ");
  return (
    <li className={`feedback-item${scored ? " scored" : ""}`}>
      <strong>{item.question}</strong>
      {item.reasons?.length ? (
        <div className="reasons">
          {item.reasons.map((reason) => (
            <span key={reason}>{REASON_LABEL[reason] ?? reason}</span>
          ))}
        </div>
      ) : null}
      {item.note && <p>“{item.note}”</p>}
      {meta && <p className="meta">{meta}</p>}
      {scored && <p>Scored case: {item.key_facts?.join("; ")}</p>}
      <details>
        <summary>{scored ? "Change the key facts" : "Make it a scored case"}</summary>
        <form className="feedback-facts" onSubmit={submit}>
          {facts.map((fact, index) => (
            <input
              key={index}
              aria-label={`Key fact ${index + 1}`}
              placeholder={`Key fact ${index + 1}${index ? " (optional)" : ""}`}
              maxLength={300}
              value={fact}
              onChange={(event) => setFacts((current) => current.map((value, at) => (at === index ? event.target.value : value)))}
            />
          ))}
          <textarea
            aria-label="Paper sentence"
            placeholder="The paper's own sentence that holds these facts (optional)"
            maxLength={600}
            value={sentence}
            onChange={(event) => setSentence(event.target.value)}
          />
          {error && <span className="answer-error" role="alert">{error}</span>}
          <div className="actions">
            <button className="button" type="submit" disabled={saving}>
              {saving ? "Saving…" : "Save key facts"}
            </button>
          </div>
        </form>
      </details>
    </li>
  );
}
