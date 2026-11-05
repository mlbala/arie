# 0010: Explain mode puts the whole paper in the prompt and checks citations after streaming

**Status:** Accepted; amends [ADR 0003](0003-quote-verifier-ladder.md) (explain mode does not pass every claim through the verifier ladder) and [ADR 0008](0008-model-led-planning-and-conditional-review.md) (explain mode has no LLM planner or review call)

## Context

Strict mode answers from at most 12,000 input tokens of retrieved excerpts, writes at most
700 output tokens and must finish in 25 s (SPEC section 8.1). Every shown claim has to pass
the verifier or the answer falls back to excerpts. The S05 baseline on 30 Qasper dev papers
(49 cases, DeepSeek, reranker on) passed 22 of 49 cases and only 15 of 46 fact cases when
the written answer alone is scored. 9 of 49 answers fell back to excerpts: 5
`entailment_rejected`, 2 `incomplete_coverage`, 2 `question_not_answered`. A reader asking
"explain this paper" gets quotes, not an explanation.

A paper is one document that fits a modern context window, and the providers ARIE
supports are expected to offer windows of 64,000 tokens or more (not yet checked against
their documentation). Putting the paper in the prompt removes
the retrieval step that loses context, and prompt caching makes the repeated prefix cheap
across the turns of one conversation. How many library papers fit under the fit rule below
has not been measured yet; `pack_fits` decides per paper at run time.

## Decision

ARIE gets a second answer mode. Request field `mode`: `strict` (today's pipeline, unchanged)
or `explain`. The default stays `strict` until explain mode passes its gate.

- **Paper pack.** The whole paper is one text: title, abstract, section outline, then every
  kept chunk as `[cN]` plus its text under its section (`core/arie_core/pipeline/pack.py`).
  `N` comes from the chunk id `<paper_id>:<N>`. Contents listings, credits, references and
  acknowledgements are left out, by the rules strict mode already uses. A literal `[c12]` in
  the paper is escaped to `[\c12]`. The pack is stored in `arie_paper_packs` per paper
  revision. `paper_id` never enters model input (I1).
- **Cacheable prefix.** The system prompt is one shared header for writer, judge and guide.
  The user prompt is pack, role instructions, recent turns, then the question. Nothing
  dynamic comes before the question, so all roles reuse one cached pack.
- **Fit rule.** The whole pack is used when its estimated tokens are at most
  `explain.max_pack_share` (default 0.6) of the provider's `llm_providers.<name>.context_tokens`.
  Otherwise a paged pack is used: abstract, introduction, conclusion, then whole sections
  ranked by strict-mode retrieval until the limit. Defaults: DeepSeek 64,000, OpenAI and
  Gemini 128,000, unknown plugin 32,000. They are conservative guesses, not checked against
  the providers' documented limits; `context_tokens` overrides them. A provider with a
  `base_url` and no `context_tokens` also gets 32,000, because that endpoint may serve any
  model.
- **Write, then check.** One streaming writer call cites sources inline as `[cN]`. First
  token within 10 s, stall timeout 15 s, total 90 s. Output limits: brief 600, standard
  1,200, detailed 2,500 tokens. Depth comes from the keyword planner; there is no LLM planner
  call. After streaming: deterministic citation checks, then the grounding checker in the
  background, then an LLM judge on flagged sentences only, on the writer's provider so it
  hits the cached pack. Results stream as SSE events and are saved with the turn.
- **I2, amended.** Strict mode keeps I2 as written. In explain mode every sentence is either
  cited or marked as background, and every citation is checked against the quoted chunk.
  An unsupported sentence is flagged to the reader, not removed. I3 (bounded execution)
  holds with the explain budgets above, through `ModelSession`.

## Consequences

- Explain answers can be longer and read as prose, at 1 call instead of up to 3. Input per
  answer grows from at most 12,000 tokens to the pack size, up to 60% of the context window.
  Cached input is billed lower on the providers that support it; that depends on S07
  reporting cache hits.
- A sentence can reach the reader before it is checked. The reader sees the check results
  as they arrive. This is why I2 changes: strict mode never shows an unverified claim,
  explain mode shows it with its status. Choose `strict` when that is not acceptable.
- Papers whose pack exceeds the share lose whole sections to the paged pack, so those
  answers can miss content that strict mode would have retrieved.
- Packs cost one SQLite row per paper (about the paper's text size). Old papers build
  theirs on first use. Pack format changes bump `PACK_FORMAT`, which forces a rebuild.
- To undo: leave `mode` at `strict`. The pack table and config keys can stay unused.
- Not decided here: the explain prompts (`prompts/explain.v1.yaml`, `judge.v1.yaml`,
  `guide.v1.yaml`), the SSE event names, and the gate that makes `explain` the default.
