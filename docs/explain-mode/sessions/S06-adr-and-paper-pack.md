# S06: ADR 0010, paper pack and context limits

**Needs:** S04 · **Size:** M · **plan first** (use plan mode; Opus is worth it here if you have headroom)

## Goal

Record the explain-mode decision and build the paper pack: the whole paper as one cache-friendly text with `[cN]` markers.

## Read first

- `docs/explain-mode/README.md` "Fixed decisions"
- SPEC §1, §5, §6.2 (chunking)
- How `select_rows` and `reader.py` exclude credits, references and contents listings (grep for those words), so the pack reuses the same rules
- `ArieConfig` and the `llm_providers` section in `core/arie_core/registry/manager.py`

## Change

- ADR 0010 via the `arie-adr` skill: explain mode (paper in the prompt with a cached prefix, cite while writing, checks after streaming), strict mode kept, I2 amended as in `AGENTS.md`, explain budgets from Fixed decisions. Amends ADR 0003 and ADR 0008.
- `core/arie_core/pipeline/pack.py`: `build_pack(document) -> PaperPack` (frozen struct: `text`, `chunk_numbers` in order, `token_estimate`, `revision`). Text = title, abstract, section outline, then every kept chunk as `[cN]` + its text under its section heading. Escape literal `[c<digits>]` from the paper. Pure and deterministic: same input, same bytes.
- Storage: runtime table `arie_paper_packs(paper_id, revision, text, tokens)`. Build when a paper reaches `INDEXED`; build lazily for older papers on first use.
- Config: `llm_providers.<name>.context_tokens` (int, with a conservative default per plugin) and `explain.max_pack_share` (default 0.6). Add both to `ArieConfig` and `arie.yaml`.
- `pack_fits(pack, context_tokens, share) -> bool`.

## Tests

- Every `[cN]` maps to a real chunk id and its offsets; references and credits are absent; escaping works; two builds give identical bytes; a pack contains only the bound paper's chunks.

## Done when

- ADR 0010 and its README row exist; `AGENTS.md` and SPEC §1 updated for I2; SPEC §5 lists the new table. `make check` passes.
