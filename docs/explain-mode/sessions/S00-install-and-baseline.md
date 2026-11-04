# S00: Install this kit and record a baseline

**Needs:** nothing · **Size:** S

## You do (before Claude Code)

1. Copy `CLAUDE.md`, `AGENTS.md`, `.claude/` and `docs/explain-mode/` into the repo root.
2. `git add CLAUDE.md AGENTS.md .claude docs/explain-mode && git commit -m "S00: add Claude Code kit"`
3. Start Claude Code in the repo root. Run `/context` and check that `CLAUDE.md` and `AGENTS.md` are listed under Memory files. Type `/arie` and check that `arie-session` appears.

## Claude does (`/arie-session S00`)

- Run `make check` and `make web-check`. Record pass or fail per target in PROGRESS. Don't fix anything in this session; list failures under Open issues.
- Ask the user once: "Which ingested paper id should be the baseline paper, and is an LLM key configured?" If yes, run the `arie-eval` skill on it with `ARGS="--compare-reranker"` and record the summary as the strict baseline.
- If no paper or key is available, record "baseline pending" and move on.

## Done when

- PROGRESS "Current state" shows S00 done, the check results, and either baseline numbers or "baseline pending".
