"""S27: the writer gets the table rows that match the question, checked with their header."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import cast

import pytest
from arie_api.runtime.base import CallContext
from arie_core.budget import BudgetHandle
from arie_core.harness.answer import (
    _source_windows,  # pyright: ignore[reportPrivateUsage]
    compose_answer,
    table_excerpt,
    table_row_units,
)
from arie_core.harness.intent import plan_answer
from arie_core.harness.reader import EvidenceChunk, query_terms
from arie_core.plugin_api import LLM
from conftest import ScriptedModel

CAPTION = "Table 3: F1 of every model on the dev set."
HEADER = "| Model | F1 | Note |"
ROWS = [f"| model-{i:02d} | {50 + i}.0 | baseline run with seed {i} |" for i in range(30)]
TABLE = f"{CAPTION}\n\n{HEADER}\n| --- | --- | --- |\n" + "\n".join(ROWS)
BASE = 200  # the chunk starts this far into its page
PAGE = "x" * BASE + TABLE


def chunk(text: str = TABLE) -> EvidenceChunk:
    return EvidenceChunk("p1", "p1:4", text, 5, BASE, BASE + len(text), "Results")


def ctx(seconds: float = 60.0) -> CallContext:
    budget = BudgetHandle(max_tokens=10_000_000, max_wall_sec=seconds)
    return CallContext(time.monotonic() + seconds, "table-rows", budget)


def test_a_value_in_row_14_of_30_gets_row_14_as_an_exact_unit() -> None:
    question = "What F1 does model-14 reach?"
    units = table_row_units(chunk(), query_terms(question))
    assert units and units[0].text == ROWS[14]
    assert PAGE[units[0].char_start : units[0].char_end] == ROWS[14]
    assert units[0].context == f"{CAPTION}\n{HEADER}"
    assert all(ROWS[15] not in unit.context for unit in units)
    windows = _source_windows([chunk()], plan_answer(question))
    texts = [unit.text for unit in windows]
    assert ROWS[14] in texts and windows[texts.index(ROWS[14])].context
    # Only matching rows are offered; the first rows are not shown as a block any more.
    assert table_excerpt(TABLE) not in texts


def test_a_short_table_is_offered_whole_as_before() -> None:
    short = f"{CAPTION}\n\n{HEADER}\n| --- | --- | --- |\n" + "\n".join(ROWS[:3])
    question = "What F1 does model-01 reach?"
    assert table_row_units(chunk(short), query_terms(question)) == []
    windows = _source_windows([chunk(short)], plan_answer(question))
    assert table_excerpt(short) in [unit.text for unit in windows]
    assert all(unit.context == "" for unit in windows)


def test_no_matching_row_falls_back_to_the_first_rows() -> None:
    assert table_row_units(chunk(), query_terms("Which optimizer is used?")) == []


@pytest.mark.asyncio
async def test_row_claims_are_checked_with_caption_and_header_but_not_other_rows() -> None:
    premises: list[str] = []

    async def entail(pairs: Sequence[tuple[str, str]]) -> list[float]:
        premises.extend(premise for premise, _ in pairs)
        return [0.99 for _ in pairs]

    question = "What F1 does model-14 reach?"
    texts = [unit.text for unit in _source_windows([chunk()], plan_answer(question))]
    row = texts.index(ROWS[14])
    model = ScriptedModel(
        [
            {
                "claims": [
                    {"text": "On the dev set, model-14 reaches an F1 of 64.0.", "source": row},
                    # Row 15's number, cited to row 14: Tier 1 rejects it.
                    {"text": "On the dev set, model-14 reaches an F1 of 65.0.", "source": row},
                ]
            },
            {"supported": [0], "answers_question": True, "adequate_detail": True, "covered": [0]},
        ]
    )
    result = await compose_answer(ctx(), "p1", question, [chunk()], cast(LLM, model), entail=entail)
    assert [excerpt.text for excerpt in result.excerpts] == [
        "On the dev set, model-14 reaches an F1 of 64.0."
    ]
    (excerpt,) = result.excerpts
    assert excerpt.quote == ROWS[14]
    assert PAGE[excerpt.char_start : excerpt.char_end] == ROWS[14]
    assert excerpt.context == f"{CAPTION}\n{HEADER}"
    # Tier 3 read the caption, the header and that one row, never another row.
    assert premises == [f"{CAPTION}\n{HEADER}\n{ROWS[14]}"]
    writer = model.requests[0].prompt
    assert '"context"' in writer and ROWS[14] in writer
