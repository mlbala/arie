---
name: arie-adr
description: Write or amend an ARIE architecture decision record in docs/adr when a change alters an invariant, a budget, a slot or a process boundary
argument-hint: "[short decision title]"
---

# Write an ADR

1. Find the next number: the highest `docs/adr/NNNN-*.md` plus one. Don't read the other ADRs in full; read only one an amendment points at.
2. Create `docs/adr/NNNN-<kebab-title>.md`, 40–80 lines, in this shape:

```markdown
# NNNN: <the decision as one sentence>

**Status:** Accepted[; amends [ADR XXXX](XXXX-....md) (what changes)]

## Context

What forced the decision: the measured problem with numbers, and the constraint.

## Decision

What the code now does, concretely: names, budgets, defaults, config keys.

## Consequences

What gets better, what gets worse or costs more, what is now forbidden, and how to undo it.
```

3. Add a row to the table in `docs/adr/README.md`: `| [NNNN](NNNN-....md) | <decision> | Accepted |`. If it amends an older ADR, change that row's Status to `Accepted, amended by NNNN`. Don't rewrite the older ADR's text.
4. If an invariant changed, update its row in `docs/SPEC.md` section 1 and the summary in `AGENTS.md`.
5. Plain sentences, numbers with units, no marketing words.
