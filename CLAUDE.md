@AGENTS.md

## Claude Code

- One Claude Code session = one session file in `docs/explain-mode/sessions/`. Start it with `/arie-session SNN`. Don't start the next session in the same conversation; the user runs `/clear` first.
- Session files marked **plan first** need plan mode: propose the plan, wait for approval, then edit.
- Don't spawn subagents unless the user asks. Search with Grep and Glob yourself.
- Don't re-read a file you just edited, and don't paste large tool output back to the user.
- Write ADRs with the `arie-adr` skill and run evals with the `arie-eval` skill.

## Compact instructions

When compacting, keep: the session id, the files changed so far, failing tests with their error lines, decisions made, and what is left in the session file's "Done when" list. Drop file contents that were only read for orientation.
