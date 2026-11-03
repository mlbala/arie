# 0008: The planner decides the question type, and review runs when it adds a check

**Status:** Accepted

## Context

On the fixed evaluation set, 6 of 16 questions passed. Tracing the model traffic
showed four causes:

- The NLI model scored "The paper proposes Cedar" against "We propose Cedar" at 0.02:
  it does not know a paper's "we" is "the paper". Every overview lost its
  contribution sentence, and the reviewer then judged it as not answering.
- Only one claim per source passage was kept, so a passage could not support two
  separately stated points.
- A plan with one section too many was rejected whole, so "Describe this paper" fell
  back to keyword rules and found no passages.
- Keyword patterns overrode the planner ("recognized wording wins"), so "What are the
  key contributions?" became a definition question.

The reviewer's prompt also mixed support with relevance ("entailed AND helps answer"),
so it rejected faithful claims it judged off-topic.

## Decision

- Tier 3 premises restate "we/our/us" as "the authors" before scoring
  (`attributed_premise`). The displayed quote is unchanged. Changed numbers (0.009) and
  negations (0.000) are still rejected.
- One passage may support up to 3 distinct claims.
- Planner output is trimmed, not discarded: up to 3 distinct queries, aspects and known
  section names are kept. Unknown sections are dropped, so a plan can only narrow
  reading within the bound paper. Extra fields and unknown tasks are still rejected.
- The planner's task label wins. Keyword patterns only fill in when its label is broad
  (`answer`, `overview`, `synthesis`). The patterns remain the fallback without a model.
- A writer that returns no claims abstains (`not_in_paper`) instead of showing excerpts
  that do not answer.
- The reviewer judges support per claim and relevance only for the whole answer.
- The LLM reviewer is skipped only for a direct answer: task `lookup`, `definition` or
  `answer`, not an overview, at most one aspect, not detailed, no claim rejected, and
  every claim at Tier 3 >= 0.9 (`REVIEW_OPTIONAL_ENTAILMENT`). Such answers report
  mode `entailed_answer`, with checks `tier_1` and `tier_3_entailment`.

## Consequences

- The same evaluation set passed 14 and 13 of 16 on two runs. All overview wordings
  pass. Two cases still fail: a comparison whose source never names a baseline, and
  "Why did Cedar achieve 99%?", where a correction cannot cite 99% from any quote.
- I2 holds: every shown claim passes Tiers 1-2 and Tier 3. Skipping the reviewer
  removes only the whole-answer relevance check, for single-point answers whose
  claims Tier 3 already scores as near-certain.
- A skipped review uses 2 model calls instead of 3. The UI states that the separate
  review was not needed. Overviews and multi-part answers still make 3 calls.
