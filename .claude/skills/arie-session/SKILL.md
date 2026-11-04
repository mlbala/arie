---
name: arie-session
description: Run one ARIE explain-mode work session from docs/explain-mode/sessions, e.g. /arie-session S03
disable-model-invocation: true
argument-hint: "[session id, e.g. S03]"
---

# Run session $0

## 1. Orient, cheaply

- Read `docs/explain-mode/PROGRESS.md`: only the "Current state" block and the last entry.
- Read the one file matching `docs/explain-mode/sessions/$0-*.md`. It says what to read, what to change and when the session is done.
- Check its **Needs** line against PROGRESS. If a needed session is not done, stop and say which one.
- Run `git status --short`. Uncommitted changes mean an earlier attempt at this session stopped midway; continue from them instead of starting over.
- To find where something lives, open `reference.md` in this skill's folder before searching the repo.

## 2. Plan

- Write a plan of at most 8 bullets: files to change, tests to add, commands to run.
- If the session file is marked **plan first**, stop and wait for approval. Otherwise continue.

## 3. Build

- Make the smallest change that meets the session's **Done when** list. Anything outside its scope goes under "Open issues" in PROGRESS, not into the code.
- Grep, then read line ranges. Don't read whole long files or the whole SPEC.
- Every new behaviour gets a test in the same change.

## 4. Verify

- Narrowest test first: `uv run pytest <file>::<test> -q --tb=short`.
- Then `make check` once. Also `make web-check` if `web/` changed, and `cd web && npm run generate:api` if a route or response model changed.
- If the same failure survives three fix attempts, stop, record it in PROGRESS and tell the user.

## 5. Hand off

- Update only the `docs/SPEC.md` sections whose behaviour changed.
- Append an entry to `docs/explain-mode/PROGRESS.md` with its template (12 lines at most) and update "Current state".
- Set the session's Status to `done` in the table in `docs/explain-mode/README.md`.
- If all checks pass: `git add -A && git commit -m "$0: <one-line summary>"`. Never push.
- Final message: up to 3 bullets on what changed, the check results, then `Next: /clear, then /arie-session <next id>`.

## Stop early

Write PROGRESS first, then stop, if the work is clearly bigger than the session file says, if you have read more than about 15 files, or if you need a decision the session file doesn't settle. In the last case ask the user one question.
