# 0011: Explain mode adds no local grounding model; the LLM judge checks flagged sentences

**Status:** Accepted; amends [ADR 0010](0010-explain-mode-paper-in-the-prompt.md) (no grounding checker runs between the deterministic citation checks and the judge)

## Context

ADR 0010 planned three checks after an explanation streams: S10's deterministic citation
checks, then a small local model that scores each cited sentence against its cited chunks
in the background, then an LLM judge on flagged sentences. The local model was to get its
own slot (`grounding`) and its own worker pool. The bar was about 5 s for 15 pairs and
under 1 GB of worker memory on the 2-vCPU host.

S12 measured both candidates, HHEM-2.1-Open (flan-t5-base, Apache-2.0) and LettuceDetect
base (ModernBERT, MIT), on 20 real (sentence, cited chunks) pairs from 3 saved explain
answers on paper 2610.40325. The measurements ran on a Mac with 2 threads, because the
Docker daemon was off; nothing was timed on the host.

- Both need PyTorch and transformers. The image has neither (only onnxruntime and
  tokenizers). The CPU-only torch wheel is 210 MB and unpacks to 753 MB; plain PyPI
  resolves about 3 GB of CUDA wheels instead.
- Time for 15 pairs, one pair per call, and peak RSS: HHEM torch 3.1 s and 901 MB,
  ONNX fp32 10.3 s and 836 MB. LettuceDetect torch 4.5 s and 943 MB, ONNX fp32 13.3 s
  and 819 MB. The fp32 ONNX exports match torch within 0.0001.
- Dynamic int8 ONNX took 3.3 s and 5.9 s, but moved scores by up to 0.45 and 0.52, and
  HHEM int8 missed one of the two unsupported sentences.
- At a 0.5 threshold, both models caught the 2 unsupported sentences, which S10 already
  flags. LettuceDetect also flagged 4 of the 18 supported sentences (table rows, a formula
  in PDF markup, a two-chunk summary); HHEM flagged 7, including plain paraphrases.

So the only builds that met the speed bar cost about 0.8 GB of image, and on this sample
the model caught nothing the deterministic checks missed.

## Decision

- No `grounding` slot, no `plugins/ground-*` package, no grounding worker pool, no
  `grounding.*` config keys and no `grounding` SSE event.
- After an explanation streams, the checks are S10's deterministic citation checks
  (`check_answer`, the `checks` event), then the LLM judge (S13) on the sentences those
  checks flag. The judge runs on the writer's provider so it reads the cached pack.
- Strict mode's Tier 3 entailment model (`entailment` slot, ADR 0003) is unchanged.

## Consequences

- The image and the worker memory stay as they are, and no CPU is spent after each answer.
- A sentence that passes the deterministic checks but misstates its source is not checked
  by any model. Today a wrong claim with the right numbers and stems passes.
- The judge costs one LLM call per flagged sentence, at cache-hit prices for the pack.
  With S10's valid rate of about 0.88, that is about 1 in 9 cited sentences.
- The sample is small: 20 pairs and 2 positives, on one paper, on a Mac. To revisit,
  save `checks` per eval case, collect a larger set of judged sentences, and time a
  candidate in the image with `--cpus=2`. Adding the slot then needs a new ADR that
  amends this one.
