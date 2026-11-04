# 0003: Three-tier quote verifier with a local entailment model

**Status:** Accepted; amended by [ADR 0010](0010-explain-mode-paper-in-the-prompt.md) (explain mode does not pass every claim through the verifier ladder; strict mode is unchanged)

## Context

Invariant I2 forbids unverified claims: every claim shown to a user must cite a quote
that exists in the saved PDF text and supports the claim, or ARIE must abstain. Exact
quote matching proves that a quote exists, not that a paraphrase is faithful. A
model-written "the model performs badly", citing "the model performs well", shares
almost every word with its quote. A word-overlap check cannot catch that, and an LLM
reviewer is a second opinion from the same kind of model.

## Decision

The verifier in `arie_core.harness.verifier` is a ladder:

1. **Tier 1, exact:** the quote is a byte-exact substring of the cited chunk.
2. **Tier 2, normalized:** NFKC, ligature, hyphenation and table normalization, with
   offsets mapped back to the saved text, so citations highlight the real span.
3. **Tier 3, entailment:** a local NLI cross-encoder
   (`cross-encoder/nli-deberta-v3-xsmall`, ONNX, `entailment` slot) scores
   P(quote entails claim). A claim below `min_score` (default 0.5) is dropped.

Deterministic anchor checks (numbers, negations, hedges, absolutes) and a lexical
coverage check run first, as cheap pre-filters. They are never sufficient alone.
Tier 3 runs after drafting and before the LLM reviewer, so the reviewer judges
coverage only on claims that are already entailed. The model runs in an `arie-worker`
process (ADR 0004) and is warmed at startup.

If the model is loading, fails or times out, model-written claims cannot pass, and
the answer abstains to exact source excerpts with a stated reason
(`entailment_unavailable`, `entailment_timeout`, `entailment_rejected`). Extractive
excerpts restate their own quote, so Tiers 1-2 suffice for them.

## Consequences

- Calibration: on paraphrases of source sentences the model scores 0.90-0.996, and on
  polarity flips and contradictions 0.000-0.001. Inference for a full answer takes
  about 0.1 s on CPU.
- The model (about 280 MB) downloads on first start into `HF_HOME`. Until it is
  ready, answers are extractive. The Storage & health pane shows the checker status.
- The model is not MiniCheck. MiniCheck has no maintained ONNX build, and the PyTorch
  versions are too heavy for the worker's memory cap. Another model can be used by
  setting `plugins.entailment` and recalibrating `min_score`.
