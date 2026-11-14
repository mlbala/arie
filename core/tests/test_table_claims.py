"""S28: a table claim is checked against its row written as a sentence, not the raw pipes."""

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
)
from arie_core.harness.citations import (
    CITED_OK,
    FLAGGED,
    LOST_QUALIFIER,
    NUMBER_NOT_IN_SOURCE,
    check_answer,
)
from arie_core.harness.intent import plan_answer
from arie_core.harness.reader import EvidenceChunk, Excerpt, premise_text
from arie_core.harness.verifier import anchor_numbers, canonical_numbers, preserves_anchors
from arie_core.pipeline.tables import (
    linearize_quote,
    linearize_row,
    table_cells,
    table_premise,
)
from arie_core.plugin_api import LLM
from conftest import ScriptedModel

CAPTION = "**Table 2:** Results on SQuAD. Higher is better."
HEADER = "| Model | F1 | EM |"
SMALL = (
    f"{CAPTION}\n\n{HEADER}\n| --- | --- | --- |\n| BERT | 85.3 | 79.1 |\n| ELMo | 80.1 | 71.4 |"
)


def test_linearize_row_lines_up_header_and_cells() -> None:
    assert linearize_row(CAPTION, HEADER, "| BERT | 85.3 | 79.1 |") == (
        "Table 2 (Results on SQuAD; Higher is better): Model = BERT, F1 = 85.3, EM = 79.1."
    )


def test_linearize_row_skips_an_empty_cell() -> None:
    assert linearize_row("", HEADER, "| BERT |  | 79.1 |") == "Model = BERT, EM = 79.1."


def test_linearize_row_without_a_caption_starts_at_the_cells() -> None:
    assert linearize_row("", "| | F1 |", "| **BERT** | 85.3<sup>†</sup> |") == "BERT, F1 = 85.3 †."


def test_a_multi_row_header_names_columns_by_its_last_row() -> None:
    header = "| | SQuAD | SQuAD |\n| Model | F1 | EM |"
    assert (
        linearize_row("", header, "| BERT | 85.3 | 79.1 |") == "Model = BERT, F1 = 85.3, EM = 79.1."
    )
    context = f"Table 4: Dev.\n{header}"
    assert linearize_quote(context, "| BERT | 85.3 | 79.1 |") == (
        "Table 4 (Dev): Model = BERT, F1 = 85.3, EM = 79.1."
    )


def test_table_cells_strip_emphasis_and_tags() -> None:
    assert table_cells("| **BERT**<br>large | `85.3` |") == ["BERT large", "85.3"]


SMALL_PREMISE = (
    "**Table 2:** Results on SQuAD. Higher is better.\n\n"
    "Table 2: Model = BERT, F1 = 85.3, EM = 79.1.\n\n"
    "Table 2: Model = ELMo, F1 = 80.1, EM = 71.4."
)


def test_table_premise_states_the_caption_once_and_keeps_every_row() -> None:
    assert table_premise(SMALL) == SMALL_PREMISE
    text = f"Notes before.\n\n{SMALL}\n\nNotes after."
    assert table_premise(text) == f"Notes before.\n\n{SMALL_PREMISE}\n\nNotes after."
    bare = "| Model | F1 |\n| --- | --- |\n| BERT | 85.3 |"
    assert table_premise(bare) == "Model = BERT, F1 = 85.3."
    assert table_premise("No table here.") is None


# Numeric rule and number forms.


def test_numbers_compare_in_canonical_form_without_scale_conversion() -> None:
    row = linearize_row(CAPTION, HEADER, "| BERT | 85.30 | 79.1 |")
    assert preserves_anchors("BERT reaches an F1 of 85.3.", row, table=True)
    # 85.30 and 85.3 are one number; 0.853 is not 85.3 (no percent scaling).
    assert canonical_numbers("85.30%") == canonical_numbers("85.3")
    assert not preserves_anchors("BERT reaches an F1 of 0.853.", row, table=True)
    # The row's other cells and the caption's numbers are allowed, another row's are not.
    assert preserves_anchors("In Table 2, BERT reaches an EM of 79.1.", row, table=True)
    assert not preserves_anchors("BERT reaches an F1 of 80.1.", row, table=True)


def test_premise_text_is_the_row_sentence_for_a_row_quote_else_the_quote() -> None:
    row = Excerpt(
        "BERT reaches 85.3 F1.", "| BERT | 85.3 | 79.1 |", 1, 0, 23, "p:1",
        context=f"Table 2: Results on SQuAD.\n{HEADER}",
    )  # fmt: skip
    assert premise_text(row) == "Table 2 (Results on SQuAD): Model = BERT, F1 = 85.3, EM = 79.1."
    prose = Excerpt("We use BERT.", "We use BERT as the encoder.", 1, 0, 27, "p:1")
    assert premise_text(prose) == "We use BERT as the encoder."


def ctx(seconds: float = 60.0) -> CallContext:
    budget = BudgetHandle(max_tokens=10_000_000, max_wall_sec=seconds)
    return CallContext(time.monotonic() + seconds, "table-claims", budget)


@pytest.mark.asyncio
async def test_strict_reads_a_whole_table_as_row_sentences_and_rejects_a_planted_number() -> None:
    chunk = EvidenceChunk("p1", "p1:2", SMALL, 3, 0, len(SMALL), "Results")
    pairs_seen: list[tuple[str, str]] = []

    async def entail(pairs: Sequence[tuple[str, str]]) -> list[float]:
        pairs_seen.extend(pairs)
        # The model reads the rows: ELMo's score given to BERT is not entailed.
        return [0.02 if "80.1" in claim else 0.99 for _, claim in pairs]

    question = "What F1 does BERT reach on SQuAD?"
    texts = [unit.text for unit in _source_windows([chunk], plan_answer(question))]
    whole = texts.index(SMALL)  # a small table is one unit, quoted whole
    model = ScriptedModel(
        [
            {
                "claims": [
                    {"text": "BERT reaches an F1 of 85.3 on SQuAD.", "source": whole},
                    # Planted: a number in no cell, header or caption. Rejected before Tier 3.
                    {"text": "BERT reaches an F1 of 91.2 on SQuAD.", "source": whole},
                    # Another row's number is in the table, so Tier 3 decides it.
                    {"text": "BERT reaches an F1 of 80.1 on SQuAD.", "source": whole},
                ]
            },
            {"supported": [0], "answers_question": True, "adequate_detail": True, "covered": [0]},
        ]
    )
    result = await compose_answer(ctx(), "p1", question, [chunk], cast(LLM, model), entail=entail)
    assert [item.text for item in result.excerpts] == ["BERT reaches an F1 of 85.3 on SQuAD."]
    assert result.excerpts[0].quote == SMALL  # the quote shown is still the exact text
    assert [claim for _, claim in pairs_seen] == [
        "BERT reaches an F1 of 85.3 on SQuAD.",
        "BERT reaches an F1 of 80.1 on SQuAD.",
    ]
    assert {premise for premise, _ in pairs_seen} == {SMALL_PREMISE}


# Explain checks.


def test_explain_flags_a_number_in_no_row_and_passes_the_row_numbers() -> None:
    sources = {1: SMALL}
    right = check_answer("BERT reaches an F1 of 85.3 on SQuAD [c1].", sources)
    absent = check_answer("BERT reaches an F1 of 91.2 on SQuAD [c1].", sources)
    assert right.sentences[0].status == CITED_OK and right.sentences[0].table
    assert absent.sentences[0].status == FLAGGED
    assert absent.sentences[0].reasons == (NUMBER_NOT_IN_SOURCE,)
    both = check_answer("BERT reaches 85.3 F1 and ELMo reaches 80.1 F1 [c1].", sources)
    assert both.flagged == 0


def test_explain_does_not_repeat_a_hedged_caption_into_every_row() -> None:
    table = (
        "Table 3: Accuracy measured only on the test split.\n\n"
        "| Model | Accuracy | Params | Speed |\n| --- | --- | --- | --- |\n"
        "| ResNet | 76.4 | 25M | 300 |\n| ViT | 81.2 | 86M | 120 |"
    )
    # The sentence aligns to the ViT row through its column names; the row sentence does not
    # carry the caption's "only", so the true sentence is not flagged.
    sentence = "The ViT model has accuracy 81.2, 86M params and speed 120 [c4]."
    checks = check_answer(sentence, {4: table})
    assert checks.sentences[0].status == CITED_OK
    assert (table_premise(table) or "").count("only") == 1
    # A sentence restating the caption still keeps its qualifier check.
    hedged = "Accuracy was measured on the test split [c4]."
    assert check_answer(hedged, {4: table}).sentences[0].reasons == (LOST_QUALIFIER,)


def test_explain_checks_of_prose_sources_are_unchanged() -> None:
    sources = {
        1: "We train on 3 datasets. The model may not converge with small batches.",
        2: "Our method improves accuracy by 4.2 points over the baseline.",
    }
    text = (
        "The model converges with small batches [c1]. Accuracy improves by 4.2 points [c2]. "
        "It uses 5 datasets [c1]. Training is fast."
    )
    assert check_answer(text, sources) == check_answer(text, sources, tables=False)
    assert all(not item.table for item in check_answer(text, sources).sentences)


# S29: numbers as PDF-to-Markdown writes them.


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("|Densely-connected 4-layer QRNN|150|91_._4|", {"4", "150", "91.4"}),
        ("|UTCNN (full)|.698|**.755***|", {"0.698", "0.755"}),
        ("|M-_F_1|.602|**.251**|", {"1", "0.602", "0.251"}),
        ("See Section 3.2 and v1.2.", {"3.2", "1.2"}),
        ("About 1,000 papers, e.g. 5 of them, and 3*4 cells.", {"1000", "5", "3", "4"}),
    ],
)
def test_anchor_numbers_read_pdf_emphasis_and_bare_decimals(text: str, numbers: set[str]) -> None:
    assert anchor_numbers(text) == numbers


def test_a_bare_decimal_in_a_table_supports_its_leading_zero_form_in_both_modes() -> None:
    row = "| UTCNN (full) | .698 | **.755*** |"
    assert preserves_anchors("UTCNN reaches 0.755.", row, table=True)
    assert preserves_anchors("UTCNN reaches 0.755.", row)
    assert not preserves_anchors("UTCNN reaches 0.756.", row, table=True)
    table = "| Model | Acc |\n| --- | --- |\n| Densely-connected 4-layer QRNN | 91_._4 |"
    ok = check_answer("The 4-layer QRNN reached 91.4% accuracy [c2].", {2: table})
    wrong = check_answer("The 4-layer QRNN reached 91.5% accuracy [c2].", {2: table})
    assert ok.sentences[0].status == CITED_OK
    assert wrong.sentences[0].reasons == (NUMBER_NOT_IN_SOURCE,)
