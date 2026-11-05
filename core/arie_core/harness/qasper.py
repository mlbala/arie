"""Turn Qasper questions into evaluation cases for ARIE.

Qasper (Dasigi et al., NAACL 2021) has questions that NLP practitioners wrote from a
paper's title and abstract, answered by other practitioners who read the full text. That
makes it an outside check next to ARIE's self-generated questions. Everything here is pure
so it runs without network: the download lives in ``arie_api.eval_qasper``.

Per question, looking at every annotator's answer:

* all say unanswerable: an ``abstain`` case (ARIE must give no claims);
* some annotator gave extractive spans: a ``fact`` case whose key facts are the first
  three spans. Each span is looked for in one chunk of the paper as ARIE extracted it,
  comparing letters and digits only (case, punctuation, whitespace and ligatures are
  ignored, since Qasper's text and a PDF's differ in those). The key fact stored is the
  paper's own text for the match, so scoring finds it verbatim; a span with no match drops
  the case, as for generated cases;
* anything else (yes/no, free-form) is skipped and counted.
"""

from __future__ import annotations

import io
import tarfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

import msgspec

from arie_core.harness.paper_eval import MAX_FEEDBACK_FACTS, PaperEvalCase
from arie_core.harness.reader import EvidenceChunk

DEV_MEMBER = "qasper-dev-v0.3.json"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024 * 1024
# Letters and digits a span needs: one shorter ("5", "of") would be "recalled" by almost any answer.
MIN_SPAN_CHARS = 3
ORIGIN = "qasper"


class QasperAnswer(msgspec.Struct, frozen=True):
    unanswerable: bool = False
    extractive_spans: list[str] = []
    yes_no: bool | None = None
    free_form_answer: str = ""


class QasperAnnotation(msgspec.Struct, frozen=True):
    answer: QasperAnswer


class QasperQuestion(msgspec.Struct, frozen=True):
    question: str
    question_id: str
    answers: list[QasperAnnotation] = []


class QasperPaper(msgspec.Struct, frozen=True):
    title: str = ""
    qas: list[QasperQuestion] = []


def arxiv_pdf_url(paper_id: str) -> str:
    """The import URL of a Qasper paper: its id is an arXiv id."""
    return f"https://arxiv.org/pdf/{paper_id}"


def extract_dev(archive: bytes) -> bytes:
    """Return the dev-split JSON from Qasper's ``.tgz``, reading only that one member.

    The member is read by name and never written to a path taken from the archive.
    """
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("The Qasper archive is larger than expected")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = next((item for item in tar.getmembers() if item.name == DEV_MEMBER), None)
        if member is None or not member.isfile():
            raise ValueError(f"{DEV_MEMBER} is not in the Qasper archive")
        if member.size > MAX_JSON_BYTES:
            raise ValueError("The Qasper dev file is larger than expected")
        stream = tar.extractfile(member)
        if stream is None:
            raise ValueError(f"{DEV_MEMBER} could not be read")
        return stream.read()


def parse_dev(raw: bytes) -> dict[str, QasperPaper]:
    """Decode the dev split: paper id to its title and annotated questions."""
    return msgspec.json.decode(raw, type=dict[str, QasperPaper])


def first_papers(papers: Mapping[str, QasperPaper], count: int) -> list[str]:
    """The first ``count`` paper ids in sorted order, so a run can be repeated."""
    return sorted(papers)[: max(count, 0)]


def _kind(answer: QasperAnswer) -> str:
    if answer.unanswerable:
        return "unanswerable"
    if answer.yes_no is not None:
        return "yes_no"
    if answer.extractive_spans:
        return "extractive"
    return "free_form" if answer.free_form_answer.strip() else "empty"


def _loose(text: str) -> tuple[str, list[int]]:
    """The lowercase letters and digits of ``text`` (ligatures expanded), with their positions."""
    letters: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(text):
        for part in unicodedata.normalize("NFKC", char).lower():
            if part.isalnum():
                letters.append(part)
                origin.append(index)
    return "".join(letters), origin


def _find(span: str, chunks: Sequence[tuple[str, list[int], str]]) -> str | None:
    """The paper's own text for ``span`` (whitespace collapsed), or ``None`` if it is absent."""
    target = _loose(span)[0]
    for letters, origin, text in chunks:
        found = letters.find(target)
        if found >= 0:
            start, end = origin[found], origin[found + len(target) - 1]
            return " ".join(text[start : end + 1].split())
    return None


def convert_questions(
    paper_id: str, paper: QasperPaper, rows: Sequence[EvidenceChunk]
) -> tuple[list[PaperEvalCase], Counter[str]]:
    """Convert one paper's questions to cases, checking spans against ``rows``.

    Returns the cases and a count per outcome: ``questions``, ``fact``, ``abstain``,
    ``skipped_yes_no``, ``skipped_free_form``, ``skipped_empty``, ``dropped_span_too_short``
    and ``dropped_span_not_in_text``. Chunks of other papers are ignored (I1).
    """
    chunks = [(*_loose(row.text), row.text) for row in rows if row.paper_id == paper_id]
    counts: Counter[str] = Counter()
    cases: list[PaperEvalCase] = []
    for item in paper.qas:
        counts["questions"] += 1
        kinds = [_kind(annotation.answer) for annotation in item.answers]
        case_id = f"qasper-{item.question_id[:16]}"
        if kinds and all(kind == "unanswerable" for kind in kinds):
            counts["abstain"] += 1
            cases.append(
                PaperEvalCase(
                    case_id=case_id,
                    question=item.question,
                    kind="abstain",
                    origin=ORIGIN,
                    note="unanswerable",
                )
            )
            continue
        extractive = next(
            (a.answer for a in item.answers if _kind(a.answer) == "extractive"), None
        )
        if extractive is None:
            other = next((kind for kind in kinds if kind != "unanswerable"), "empty")
            counts[f"skipped_{other}"] += 1
            continue
        spans = list(dict.fromkeys(" ".join(s.split()) for s in extractive.extractive_spans))
        spans = spans[:MAX_FEEDBACK_FACTS]
        if any(len(_loose(span)[0]) < MIN_SPAN_CHARS for span in spans):
            counts["dropped_span_too_short"] += 1
            continue
        found = [_find(span, chunks) for span in spans]
        if None in found:
            counts["dropped_span_not_in_text"] += 1
        else:
            facts = list(dict.fromkeys(fact for fact in found if fact))
            counts["fact"] += 1
            cases.append(
                PaperEvalCase(
                    case_id=case_id,
                    question=item.question,
                    kind="fact",
                    origin=ORIGIN,
                    answer_quote="; ".join(facts),
                    key_facts=facts,
                )
            )
    return cases, counts


__all__ = [
    "DEV_MEMBER",
    "ORIGIN",
    "QasperPaper",
    "arxiv_pdf_url",
    "convert_questions",
    "extract_dev",
    "first_papers",
    "parse_dev",
]
