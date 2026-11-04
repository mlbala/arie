"""Qasper conversion: spans must be in the paper's text and only some answers become cases."""

from __future__ import annotations

import io
import json
import tarfile
from collections import Counter

import pytest
from arie_core.harness.paper_eval import MAX_FEEDBACK_FACTS
from arie_core.harness.qasper import (
    DEV_MEMBER,
    QasperPaper,
    arxiv_pdf_url,
    convert_questions,
    extract_dev,
    first_papers,
    parse_dev,
)
from arie_core.harness.reader import EvidenceChunk

TEXTS = (
    "We use labeled features as prior knowledge for the GE-FL method.",
    "The model reaches an F1 of 83.5 on\nthe  test set, versus 79.1 for the baseline.",
    "Training used a batch size of 32, a learning rate of 0.001 and 5 epochs.",
)
ROWS = [
    EvidenceChunk("p1", f"p1:{i}", text, i + 1, 100 * i, 100 * i + len(text), "Body")
    for i, text in enumerate(TEXTS)
]
# Another paper's text must never make a span count as present (I1).
OTHER = [EvidenceChunk("p2", "p2:0", "We use an LSTM encoder.", 1, 0, 23, "Body")]


def _answer(
    *,
    spans: tuple[str, ...] = (),
    unanswerable: bool = False,
    yes_no: bool | None = None,
    free_form: str = "",
) -> dict[str, object]:
    return {
        "answer": {
            "unanswerable": unanswerable,
            "extractive_spans": list(spans),
            "yes_no": yes_no,
            "free_form_answer": free_form,
            "evidence": [],
            "highlighted_evidence": [],
        },
        "annotation_id": "a",
        "worker_id": "w",
    }


def _question(number: int, *answers: dict[str, object]) -> dict[str, object]:
    return {
        "question": f"Question {number}?",
        "question_id": f"{number:03d}abcdef0123456789" * 2,
        "nlp_background": "five",
        "answers": list(answers),
    }


def _paper(*questions: dict[str, object]) -> QasperPaper:
    record = {"title": "A paper", "abstract": "", "full_text": [], "qas": list(questions)}
    return parse_dev(json.dumps({"p1": record}).encode())["p1"]


def test_spans_are_kept_when_they_are_in_the_text_and_other_answers_are_mapped_or_skipped() -> None:
    paper = _paper(
        _question(1, _answer(spans=("labeled features",))),
        _question(2, _answer(spans=("an LSTM encoder",))),
        _question(3, _answer(unanswerable=True), _answer(unanswerable=True)),
        _question(4, _answer(unanswerable=True), _answer(spans=("GE-FL method",))),
        _question(5, _answer(yes_no=False)),
        _question(6, _answer(free_form="They use a gated memory.")),
        _question(
            7,
            _answer(
                spans=("batch size of 32", "learning rate of 0.001", "5 epochs", "x y", "z")
            ),
        ),
        _question(8, _answer(spans=("5",))),
        _question(9),
        _question(10, _answer(unanswerable=True), _answer(yes_no=True)),
        _question(11, _answer(spans=("F1 of 83.5 on the test set",))),
    )
    cases, counts = convert_questions("p1", paper, [*ROWS, *OTHER])

    assert counts == Counter(
        questions=11,
        fact=4,
        abstain=1,
        skipped_yes_no=2,
        skipped_free_form=1,
        skipped_empty=1,
        dropped_span_too_short=1,
        dropped_span_not_in_text=1,
    )
    by_question = {case.question: case for case in cases}
    assert set(by_question) == {f"Question {n}?" for n in (1, 3, 4, 7, 11)}
    first = by_question["Question 1?"]
    assert (first.kind, first.origin, first.key_facts) == ("fact", "qasper", ["labeled features"])
    assert first.case_id == "qasper-001abcdef0123456"
    # Annotators who disagree do not make a question unanswerable; an extractive one wins.
    assert by_question["Question 4?"].key_facts == ["GE-FL method"]
    abstain = by_question["Question 3?"]
    assert (abstain.kind, abstain.origin, abstain.key_facts) == ("abstain", "qasper", [])
    # At most three key facts; the rest of the spans are not checked.
    assert by_question["Question 7?"].key_facts == [
        "batch size of 32",
        "learning rate of 0.001",
        "5 epochs",
    ]
    assert len(by_question["Question 7?"].key_facts) == MAX_FEEDBACK_FACTS
    # Whitespace differences do not hide a span, and the fact keeps the span as written.
    assert by_question["Question 11?"].key_facts == ["F1 of 83.5 on the test set"]
    assert len({case.case_id for case in cases}) == len(cases)


def test_a_span_matches_ignoring_case_punctuation_and_spacing_and_stores_the_papers_text() -> None:
    texts = (
        "The encoder is trained with the GE-FL method, as in prior work.",
        "Results use exam-\nple sentences from the \ufb01ne-tuned model (see Table 2).",
    )
    rows = [
        EvidenceChunk("p1", f"p1:{i}", text, 1, 100 * i, 100 * i + len(text), "Body")
        for i, text in enumerate(texts)
    ]
    paper = _paper(
        _question(1, _answer(spans=("ge fl METHOD",))),
        _question(2, _answer(spans=("example sentences",))),
        _question(3, _answer(spans=("fine tuned model, see table 2",))),
        _question(4, _answer(spans=("-- ,, ..",))),
    )
    cases, counts = convert_questions("p1", paper, rows)

    facts = {case.question: case.key_facts for case in cases}
    assert facts == {
        "Question 1?": ["GE-FL method"],
        "Question 2?": ["exam- ple sentences"],
        # The match ends at the last letter or digit, so the closing bracket is not included.
        "Question 3?": ["\ufb01ne-tuned model (see Table 2"],
    }
    assert counts["dropped_span_too_short"] == 1 and counts["fact"] == 3
    # Every key fact is the paper's own text, so the scorer's normalisation finds it there.
    squashed = [" ".join(text.split()).lower() for text in texts]
    for fact_list in facts.values():
        assert all(any(fact.lower() in text for text in squashed) for fact in fact_list)


def test_a_span_split_across_chunks_is_not_found() -> None:
    paper = _paper(_question(1, _answer(spans=("method. The model reaches",))))
    cases, counts = convert_questions("p1", paper, ROWS)
    assert cases == [] and counts["dropped_span_not_in_text"] == 1


def _archive(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_only_the_dev_file_is_read_from_the_archive() -> None:
    dev = json.dumps({"p1": {"title": "T", "qas": []}}).encode()
    archive = _archive({"qasper-train-v0.3.json": b"{}", DEV_MEMBER: dev, "README.md": b"x"})
    assert extract_dev(archive) == dev
    with pytest.raises(ValueError, match="not in the Qasper archive"):
        extract_dev(_archive({"../" + DEV_MEMBER: dev, "README.md": b"x"}))


def test_the_first_papers_are_taken_in_id_order() -> None:
    papers = parse_dev(json.dumps({"1909.00694": {}, "1503.00841": {}, "1601.02403": {}}).encode())
    assert first_papers(papers, 2) == ["1503.00841", "1601.02403"]
    assert first_papers(papers, 10) == ["1503.00841", "1601.02403", "1909.00694"]
    assert first_papers(papers, 0) == []
    assert arxiv_pdf_url("1503.00841") == "https://arxiv.org/pdf/1503.00841"
