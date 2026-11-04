# Explain mode: work plan

Goal: NotebookLM-style explanations of one paper, with checked citations, at lower cost
on the 2-vCPU Docker host. The research behind it is in the "ARIE research: quality,
cost, evals and feedback" doc. This folder is the plan Claude Code works from.

## How to run a session

1. Open Claude Code in the repo root.
2. Type `/arie-session S01` (the next session whose Needs are done).
3. Let it finish. It runs the checks, updates `PROGRESS.md`, commits, and tells you the next id.
4. Type `/clear`, then start the next session. One session per conversation.

If a session stops halfway (usage limit, failing test), start a new conversation with the
same `/arie-session SNN`. It reads `PROGRESS.md` and `git status` and continues.

## Sessions

Size: S fits easily in one sitting, M is a normal session, L may need a second sitting.

| ID | Session | Needs | Size | Notes | Status |
| --- | --- | --- | --- | --- | --- |
| S00 | Install this kit and record a baseline | — | S | Partly manual | done |
| S01 | Answer trace and logging | S00 | M | | done |
| S02 | Remove dead code listed in SPEC §12 | S00 | S | | todo |
| S03 | Honest model health and eval visibility | S01 | M | API + web | todo |
| S04 | Answer-level eval metrics | S01 | M | | todo |
| S05 | Qasper slice and strict baseline | S04 | M | Needs network and an LLM key | todo |
| S06 | ADR 0010, paper pack, context limits | S04 | M | **plan first** | todo |
| S07 | Streaming usage and cache stats in LLM plugins | S01 | M | | todo |
| S08 | Explain harness core | S06, S07 | L | **plan first** | todo |
| S09 | Explain route, SSE events, eval variant | S08 | M | | todo |
| S10 | Deterministic citation checks | S09 | M | | todo |
| S11 | Explain mode in the web chat, gate 1 | S10 | L | web | todo |
| S12 | Background grounding checker | S10 | L | Measure first | todo |
| S13 | LLM judge, eval metrics, default switch, gate 2 | S12, S05 | M | | todo |
| S14 | Paper guide and section summaries, gate 3 | S10 | L | API + web | todo |
| S15 | Feedback loop, gate 4 | S13 | M | API + web | todo |
| S16 | Foreground lane and answer workers | S03 | M | Optional | todo |
| S17 | Prompt optimization with GEPA | S15 | M | Optional | todo |

Sessions whose Needs are done can run in any order. S02, S03 and S07 are good picks
when you have little usage left.

## Gates

| Gate | Checked in | Pass when (suggested; tune after the S05 baseline) |
| --- | --- | --- |
| 1 | S11 | Explain fact recall on prose ≥ strict; excerpt fallbacks < 10%; first token ≤ 8 s on the 2-vCPU host; cache hit rate > 80% from the second question |
| 2 | S13 | Citation support ≥ 90%; then the default mode switches to explain |
| 3 | S14 | The judge prefers the new overview answers more than half the time |
| 4 | S15 | The judge agrees with your ratings on at least 80% of rated answers |

## Fixed decisions

Sessions follow these instead of re-deciding them. Changing one is a user decision.

- **Two modes.** `strict` is today's claim pipeline, kept as is. `explain` is new. The request field is `mode`, default `strict` until gate 2 passes.
- **Citation markers.** The pack starts each chunk with `[cN]`, where N is the chunk number from its id `<paper_id>:<N>`. The model cites with the same `[cN]`, several allowed (`[c3][c9]`). Literal `[c<digits>]` in paper text is escaped when the pack is built. `paper_id` never enters model input (I1).
- **Prompt prefix, byte-stable across turns and roles.** `system_prompt` = one shared header, identical for writer, judge and guide. `prompt` = paper pack → role instructions → recent turns → the new question. Nothing dynamic (dates, ids, settings) before the question. This lets the writer, the judge and the guide reuse one cached pack.
- **History.** The last 6 turns, question plus answer text with markers stripped. Saved answers are never evidence.
- **Fit rule.** Whole pack if its estimated tokens ≤ 60% of the provider's `context_tokens`. Otherwise a paged pack: abstract, introduction, conclusion, then whole sections ranked by today's retrieval until the limit.
- **Explain budgets.** One streaming writer call. First token within 10 s, stall timeout 15 s, total 90 s. Output tokens: brief 600, standard 1,200, detailed 2,500. Depth comes from the free keyword planner (`plan_answer`); no LLM planner call in explain mode.
- **Checks after streaming.** Deterministic citation checks first (instant), then the background grounding checker, then the LLM judge on flagged sentences only. Results stream as SSE events while the connection is open and are saved with the turn.
- **Judge models.** The in-chat judge uses the writer's provider, so it hits the cached pack. The eval pairwise judge is configurable and defaults to a different provider when a key exists.
- **Storage.** New runtime tables follow the `arie_*` pattern: `arie_paper_packs`, `arie_paper_guides`. New columns are added only if missing, so old databases keep working.
- **Prompts.** `prompts/explain.v1.yaml`, `prompts/judge.v1.yaml`, `prompts/guide.v1.yaml`. Each answer records its prompt version.
