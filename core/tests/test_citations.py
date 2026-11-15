"""Deterministic citation checks: sentence splitting and the verdict on each sentence."""

from __future__ import annotations

import pytest
from arie_core.harness.citations import (
    BACKGROUND,
    CITED_OK,
    FLAGGED,
    INVALID_MARKER,
    LOST_QUALIFIER,
    NUMBER_NOT_IN_SOURCE,
    UNCITED_NUMBER,
    check_answer,
    sentence_spans,
)
from arie_core.harness.verifier import anchor_numbers, lost_qualifier, preserves_anchors

SOURCES = {
    1: "The model reaches 92.1% accuracy on the test set. Training took 3 days on 8 GPUs.",
    2: "The method does not improve accuracy on the benchmark. It may help on small corpora.",
    3: "Ablations show the gate matters (see Fig. 4). Results are reported in Table 2.",
}


def sentences(text: str) -> list[str]:
    return [text[a:b] for a, b in sentence_spans(text)]


# (name, answer, status, reasons)
CASES = [
    (
        "a correct citation",
        "The model reaches 92.1% accuracy on the test set [c1].",
        CITED_OK,
        (),
    ),
    (
        "a marker after the full stop",
        "The model reaches 92.1% accuracy on the test set. [c1]",
        CITED_OK,
        (),
    ),
    (
        "a marker that names no chunk in the pack",
        "The model reaches 92.1% accuracy on the test set [c9].",
        FLAGGED,
        (INVALID_MARKER,),
    ),
    (
        "a good marker beside a bad one",
        "The model reaches 92.1% accuracy [c1][c9].",
        FLAGGED,
        (INVALID_MARKER,),
    ),
    (
        "an altered number",
        "The model reaches 95.1% accuracy on the test set [c1].",
        FLAGGED,
        (NUMBER_NOT_IN_SOURCE,),
    ),
    (
        "the same number written differently",
        "The model reaches 92.10% accuracy on the test set [c1].",
        CITED_OK,
        (),
    ),
    (
        "a number found in the second of two cited chunks",
        "Training took 3 days and reached 92.1% accuracy [c2][c1].",
        CITED_OK,
        (),
    ),
    (
        "a dropped negation",
        "The method improves accuracy on the benchmark [c2].",
        FLAGGED,
        (LOST_QUALIFIER,),
    ),
    (
        "a negation kept",
        "The method does not improve accuracy on the benchmark [c2].",
        CITED_OK,
        (),
    ),
    (
        "a dropped hedge",
        "It helps on small corpora [c2].",
        FLAGGED,
        (LOST_QUALIFIER,),
    ),
    (
        "a negation elsewhere in the chunk is not this claim's",
        "Training took 3 days on 8 GPUs [c2].",
        FLAGGED,
        (NUMBER_NOT_IN_SOURCE,),
    ),
    (
        "an abbreviation does not end the sentence",
        "The gate matters, e.g. in ablations (Fig. 4) [c3].",
        CITED_OK,
        (),
    ),
    (
        "a number with no citation",
        "The dataset holds 4,000 examples.",
        FLAGGED,
        (UNCITED_NUMBER,),
    ),
    (
        "a pure background sentence",
        "Attention lets a model weigh each token by its relevance.",
        BACKGROUND,
        (),
    ),
]


@pytest.mark.parametrize(
    ("answer", "status", "reasons"),
    [pytest.param(a, s, r, id=name) for name, a, s, r in CASES],
)
def test_each_sentence_gets_one_status_and_its_reasons(
    answer: str, status: str, reasons: tuple[str, ...]
) -> None:
    [only] = check_answer(answer, SOURCES).sentences
    assert (only.status, only.reasons) == (status, reasons)
    assert answer[only.start : only.end] == only.text


def test_a_list_item_is_a_sentence_and_code_blocks_and_headings_are_ignored() -> None:
    answer = (
        "# Results in 2024\n"
        "\n"
        "Two things matter:\n"
        "\n"
        "- It reaches 92.1% accuracy [c1]\n"
        "- It trains in 5 days [c1]\n"
        "1. Background only\n"
        "\n"
        "```python\n"
        "x = 99. y = 100\n"
        "```\n"
        "~~~\n"
        "z = 7.\n"
        "~~~\n"
        "Done.\n"
    )
    checks = check_answer(answer, SOURCES)
    assert [(s.text, s.status, s.reasons) for s in checks.sentences] == [
        ("Two things matter:", BACKGROUND, ()),
        ("It reaches 92.1% accuracy [c1]", CITED_OK, ()),
        ("It trains in 5 days [c1]", FLAGGED, (NUMBER_NOT_IN_SOURCE,)),
        ("Background only", BACKGROUND, ()),
        ("Done.", BACKGROUND, ()),
    ]


def test_sentences_in_one_paragraph_split_on_full_stops_with_offsets() -> None:
    answer = (
        "It reaches 92.1% accuracy [c1]. Training took 3 days\non 8 GPUs [c1].  "
        "Is it fast? Yes, see Fig. 4 [c3]!"
    )
    assert sentences(answer) == [
        "It reaches 92.1% accuracy [c1].",
        "Training took 3 days\non 8 GPUs [c1].",
        "Is it fast?",
        "Yes, see Fig. 4 [c3]!",
    ]
    checks = check_answer(answer, SOURCES)
    assert [s.status for s in checks.sentences] == [CITED_OK, CITED_OK, BACKGROUND, CITED_OK]
    assert [s.markers for s in checks.sentences] == [(1,), (1,), (), (3,)]


def test_table_rows_are_sentences_and_their_separator_is_not() -> None:
    answer = "| model | acc |\n|---|---|\n| ours | 92.1 [c1] |\n"
    assert sentences(answer) == ["| model | acc |", "| ours | 92.1 [c1] |"]


def test_counts_and_rates_cover_the_whole_answer() -> None:
    answer = (
        "It reaches 92.1% accuracy [c1]. It trains in 5 days [c1]. "
        "The method improves accuracy on the benchmark [c2]. "
        "Attention weighs tokens. The data has 4,000 rows."
    )
    checks = check_answer(answer, SOURCES)
    assert (checks.cited_ok, checks.flagged, checks.background, checks.cited) == (1, 3, 1, 3)
    assert checks.valid_rate == pytest.approx(0.333, abs=0.001)
    assert checks.background_share == 0.2
    assert checks.flagged_markers() == {1, 2}


def test_an_answer_with_no_cited_sentence_has_no_valid_rate() -> None:
    checks = check_answer("Attention weighs tokens.", SOURCES)
    assert checks.valid_rate is None and checks.background_share == 1.0
    empty = check_answer("", SOURCES)
    assert empty.sentences == () and empty.valid_rate is None and empty.background_share is None
    assert check_answer("```\ncode\n```", SOURCES).sentences == ()


def test_a_marker_is_invalid_when_the_pack_did_not_hold_its_chunk() -> None:
    # Chunk 3 exists in the paper but the model saw a paged pack without it.
    seen = {number: text for number, text in SOURCES.items() if number != 3}
    [flagged] = check_answer("The gate matters (see Fig. 4) [c3].", seen).sentences
    assert flagged.reasons == (INVALID_MARKER,)


def test_the_verifier_helpers_keep_their_meaning() -> None:
    assert anchor_numbers("Rose 1,200 to 3.5") == {"1200", "3.5"}
    assert lost_qualifier("It works", "It does not work") and not lost_qualifier("a", "b")
    assert preserves_anchors("It does not work on 3 sets", "It does not work on 3 sets.")
    assert not preserves_anchors("It works on 3 sets", "It does not work on 3 sets.")
    assert not preserves_anchors("It works on 4 sets", "It works on 3 sets.")
    assert not preserves_anchors("It always works", "It works.")


def _qualifier_flag(claim: str, source: str) -> bool:
    [sentence] = check_answer(f"{claim} [c1].", {1: source}).sentences
    return LOST_QUALIFIER in sentence.reasons


def test_explain_skips_restriction_words_and_strict_keeps_them() -> None:
    # S30, 1605.06083: "a small sample" restated without "small" is not a qualification.
    source = "Here is a small sample of the phrases used to describe beards: a thick beard."
    claim = "The paper gives a sample of the phrases used to describe beards"
    assert not _qualifier_flag(claim, source)
    assert lost_qualifier(claim, source)  # strict is unchanged
    # A dropped hedge in the same shape is still flagged.
    hedged = "Here a sample of the phrases may describe beards: a thick beard."
    assert _qualifier_flag("A sample of the phrases describes beards", hedged)


def test_explain_reads_a_qualifier_only_in_the_clause_the_claim_restates() -> None:
    # S30, 1604.03114: the "not" belongs to the clause before the colon.
    source = (
        "However, this does not mean debaters are giving monologues: the interaction shows "
        "a sharp drop in self-coverage and a rise in opponent-coverage once discussion begins."
    )
    claim = "Once discussion begins, self-coverage shows a sharp drop and opponent-coverage a rise"
    assert not _qualifier_flag(claim, source)
    # S30, 1610.07809: ", so as to not hinder" starts a clause of its own.
    noisy = (
        "Most documents contain irrelevant text such as muddled sentences and footnotes, "
        "so as to not hinder keyphrase extraction systems."
    )
    assert not _qualifier_flag("Documents still contain muddled sentences and footnotes", noisy)
    # The planted shapes stay flagged: a hedge in the restated clause, and a negation in a
    # second clause that names the claim's subject.
    assert _qualifier_flag("The method improves accuracy", "The method may improve accuracy.")
    split = "The base model improves accuracy on the benchmark; the pruned model does not."
    assert _qualifier_flag("The pruned model improves accuracy on the benchmark", split)


def test_not_only_and_or_not_are_not_negations_and_past_forms_count() -> None:
    # S30, 1601.02403 and 1605.06083.
    additive = "Features are extracted not only for the current sentence but also for neighbours."
    assert not _qualifier_flag("Features are extracted for the current sentence", additive)
    either = "Dark-skinned people are called African-American whether in the USA or not."
    assert not _qualifier_flag("Dark-skinned people are called African-American", either)
    # S30, 1611.01576: "removed" in the claim keeps the source's "remove".
    removal = "We remove training sentences with more than 300 characters in English or German."
    assert not _qualifier_flag("Training sentences over 300 characters were removed", removal)
    assert _qualifier_flag("Training sentences over 300 characters were kept", removal)
    failed = "The pruned model failed to converge on the larger dataset."
    assert _qualifier_flag("The pruned model converged on the larger dataset", failed)
    assert not _qualifier_flag("The pruned model fails to converge on the larger dataset", failed)
