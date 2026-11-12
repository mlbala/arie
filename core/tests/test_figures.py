"""Figure records and mention linking (S22, ADR 0015): captions and mentions are paper text."""

from __future__ import annotations

import msgspec
import pytest
from arie_api.runtime import context
from arie_core.contracts import Figure
from arie_core.harness.injection import evidence_chunks
from arie_core.harness.intent import AnswerPlan, plan_answer
from arie_core.harness.reader import EvidenceChunk, retrieve
from arie_core.harness.research import link_figures, read_for_answer
from arie_core.pipeline.figures import attach_figures, chunk_figure_tags, locate_caption
from arie_core.plugin_api.figures import (
    PICTURE_TEXT_END,
    PICTURE_TEXT_START,
    caption_number,
    figure_mentions,
    match_captions,
    overlap_area,
    picture_text_spans,
)


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("As Figure 3 shows, accuracy rises.", ["3"]),
        ("see Fig. 3 for details", ["3"]),
        ("Figs. 2 and 3 compare both runs.", ["2", "3"]),
        ("The appendix plot (Figure A1) repeats it.", ["A1"]),
        ("Figure 3(b) zooms in, and Figure 3c too.", ["3"]),
        ("Figures 2-4 show the ablations.", ["2", "3", "4"]),
        ("Figures 2, 5 and 7 differ.", ["2", "5", "7"]),
        ("A figure of merit is reported in Table 2.", []),
        ("We configure 3 layers.", []),
    ],
)
def test_figure_mentions(text: str, numbers: list[str]) -> None:
    assert figure_mentions(text) == numbers


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("Figure 3: Accuracy by size.", "3"),
        ("**Figure 2.** Overview of Cedar.", "2"),
        ("Fig. 4. Loss curves.", "4"),
        ("**Figure A1** Extra results.", "A1"),
        ("Figure 3 shows that accuracy rises.", ""),
        ("Table 1: Results.", ""),
    ],
)
def test_caption_number(text: str, number: str) -> None:
    assert caption_number(text) == number


def test_caption_below_above_none_and_two_figures_on_one_page() -> None:
    below = (100.0, 300.0, 400.0, 315.0)
    above = (100.0, 60.0, 400.0, 75.0)
    # Caption below wins over one above; a caption above is used when none is below.
    boxes, missing = match_captions([(100.0, 100.0, 400.0, 290.0)], [above, below])
    assert boxes == {1: (100.0, 100.0, 400.0, 290.0)} and missing == 0
    boxes, missing = match_captions([(100.0, 80.0, 400.0, 250.0)], [above])
    assert boxes == {0: (100.0, 80.0, 400.0, 250.0)} and missing == 0
    # No caption overlapping the picture: skipped and counted.
    boxes, missing = match_captions([(100.0, 100.0, 300.0, 290.0)], [(320.0, 300.0, 500.0, 315.0)])
    assert boxes == {} and missing == 1
    # Two columns, two figures, two captions; sub-panels sharing a caption merge.
    pictures = [
        (50.0, 100.0, 140.0, 200.0),
        (150.0, 100.0, 290.0, 200.0),
        (320.0, 100.0, 560.0, 200.0),
    ]
    captions = [(50.0, 210.0, 290.0, 230.0), (320.0, 210.0, 560.0, 230.0)]
    boxes, missing = match_captions(pictures, captions)
    assert boxes == {0: (50.0, 100.0, 290.0, 200.0), 1: (320.0, 100.0, 560.0, 200.0)}
    assert missing == 0


PAGE = "Intro text names Figure 2 here.\n\nFigure 2: Accuracy by size.\n\nFig. 2 and Figure 5."
FIGURE_2 = Figure("2", 1, (0.0, 0.0, 10.0, 10.0), "Figure 2: Accuracy by size.", 33, 60)


def test_chunk_tags_and_caption_location() -> None:
    assert locate_caption(PAGE, "2", FIGURE_2.caption) == (33, 60)
    assert locate_caption(PAGE, "2", "Figure 2: reworded") == (33, 60)
    assert locate_caption(PAGE, "9", "Figure 9: gone") == (-1, -1)
    assert chunk_figure_tags(PAGE[33:60], 1, 33, 60, [FIGURE_2]) == {"figure": "2"}
    assert chunk_figure_tags(PAGE[0:31], 1, 0, 31, [FIGURE_2]) == {"mentions": ["2"]}
    # A caption on another page is not this chunk's; plain text gets no keys at all.
    assert chunk_figure_tags(PAGE[33:60], 2, 33, 60, [FIGURE_2]) == {"mentions": ["2"]}
    assert chunk_figure_tags("No figures here.", 1, 0, 16, [FIGURE_2]) == {}


def test_attach_figures_keeps_pages_and_chunks() -> None:
    chunks: list[dict[str, object]] = [
        dict(
            paper_id="p", chunk_id=f"p:{i}", page_start=1, char_start=s, char_end=e, text=PAGE[s:e]
        )
        for i, (s, e) in enumerate([(0, 31), (33, 60), (62, len(PAGE))])
    ]
    document: dict[str, object] = {"pages": [{"number": 1, "text": PAGE}], "chunks": chunks}
    before = [(c["chunk_id"], c["text"], c["char_start"]) for c in chunks]
    parsed = msgspec.structs.replace(FIGURE_2, caption_start=5, caption_end=7)
    missing = attach_figures(document, [parsed, Figure("4", 1, (0, 0, 1, 1), "Figure 4: x")])
    assert missing == 1
    assert [(c["chunk_id"], c["text"], c["char_start"]) for c in chunks] == before
    stored = document["figures"]
    assert isinstance(stored, list) and stored[0]["caption_start"] == 33
    assert [c.get("figure") for c in chunks] == [None, "2", None]
    assert chunks[2]["mentions"] == ["2", "5"]


def figure_rows() -> list[EvidenceChunk]:
    stored = [
        dict(
            paper_id="p",
            chunk_id="c0",
            text="Cedar uses a gated memory bank.",
            page_start=1,
            char_start=0,
            char_end=31,
            section="Methods",
        ),
        dict(
            paper_id="p",
            chunk_id="c1",
            text="Accuracy rises with size, see Figure 2.",
            page_start=2,
            char_start=0,
            char_end=39,
            section="Results",
            mentions=["2"],
        ),
        dict(
            paper_id="p",
            chunk_id="c2",
            text="Training uses four CPU workers.",
            page_start=2,
            char_start=40,
            char_end=71,
            section="Methods",
        ),
        dict(
            paper_id="p",
            chunk_id="c3",
            text="Figure 2: Accuracy by model size on Cedar.",
            page_start=3,
            char_start=0,
            char_end=42,
            section="Results",
            figure="2",
        ),
        dict(
            paper_id="p",
            chunk_id="c4",
            text="The gain in Fig. 2 holds for both datasets.",
            page_start=4,
            char_start=0,
            char_end=43,
            section="Discussion",
            mentions=["2"],
        ),
    ]
    return evidence_chunks("p", stored, None)


def plan(question: str, overview: bool = False) -> AnswerPlan:
    return AnswerPlan(question=question, depth="standard", overview=overview)


def test_a_question_or_caption_reaching_a_figure_pulls_in_its_own_words() -> None:
    scoped = figure_rows()
    c0, c1, c2, c3, c4 = scoped
    assert c1.mentions == ("2",) and c3.figure == "2"
    # The question names Figure 2: the row mentioning it is followed by the caption and the
    # other mention.
    assert link_figures(plan("What does Figure 2 show?"), [c1, c2], scoped) == [c1, c3, c4, c2]
    # No selected row refers to it: the caption and mentions follow the top row.
    assert link_figures(plan("What does Figure 2 show?"), [c0, c2], scoped) == [c0, c3, c1, c4, c2]
    # A selected caption brings in the chunks that mention its figure.
    assert link_figures(plan("How does accuracy scale?"), [c3, c2], scoped) == [c3, c1, c4, c2]
    # A passing mention alone links nothing, and overviews keep their rows.
    assert link_figures(plan("How does accuracy scale?"), [c1, c2], scoped) == [c1, c2]
    assert link_figures(plan("What is it about?", overview=True), [c3], scoped) == [c3]


def test_no_figures_selects_the_same_rows() -> None:
    plain = [msgspec.structs.replace(row, figure="", mentions=()) for row in figure_rows()]
    question = plan_answer("How does accuracy change with model size?")
    assert link_figures(question, plain[:3], plain) == plain[:3]
    assert link_figures(question, plain[1:4], plain) == plain[1:4]


@pytest.mark.parametrize("depth", ["standard", "detailed"])
def test_strict_reading_adds_the_caption_chunk(depth: str) -> None:
    question = msgspec.structs.replace(
        plan_answer("How does accuracy rise with size in Figure 2?"), depth=depth, overview=False
    )
    linked = [
        row.chunk_id for row in read_for_answer(context("f"), "p", question, figure_rows()).rows
    ]
    assert "c1" in linked and "c3" in linked and linked.index("c3") == linked.index("c1") + 1


# A page as pymupdf4llm writes it: prose, the words drawn inside Figure 1, then its caption.
DRAWN_PAGE = (
    "The system has three parts.\n\n"
    f"{PICTURE_TEXT_START}\nEncoder Retriever<br>Decoder<br>{PICTURE_TEXT_END}\n\n"
    "Figure 1: The retrieval pipeline.\n\n"
    f"{PICTURE_TEXT_START}\n<br>{PICTURE_TEXT_END}\n"
)
DRAWN_START = DRAWN_PAGE.index("Encoder")
DRAWN_END = DRAWN_PAGE.index("<br>" + PICTURE_TEXT_END)
CAPTION_START = DRAWN_PAGE.index("Figure 1:")
FIGURE_1 = Figure(
    "1",
    1,
    (100, 120, 450, 300),
    "Figure 1: The retrieval pipeline.",
    CAPTION_START,
    CAPTION_START + 33,
    ((DRAWN_START, DRAWN_END),),
)


def test_picture_text_spans_are_the_words_between_the_markers() -> None:
    # Edge whitespace and <br> are cut; a block with no words gives no span.
    assert picture_text_spans(DRAWN_PAGE) == [(DRAWN_START, DRAWN_END)]
    assert DRAWN_PAGE[DRAWN_START:DRAWN_END] == "Encoder Retriever<br>Decoder"
    # Only blocks wholly inside the range count.
    assert picture_text_spans(DRAWN_PAGE, CAPTION_START) == []
    assert overlap_area((0, 0, 10, 10), (5, 5, 20, 20)) == 25
    assert overlap_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0


def test_a_chunk_of_drawn_words_is_tagged_with_its_figure_but_not_as_its_caption() -> None:
    block = DRAWN_PAGE.index(PICTURE_TEXT_START)
    end = DRAWN_PAGE.index(PICTURE_TEXT_END) + len(PICTURE_TEXT_END)
    assert chunk_figure_tags(DRAWN_PAGE[block:end], 1, block, end, [FIGURE_1]) == {
        "figure": "1",
        "figure_text": True,
    }
    caption = DRAWN_PAGE[CAPTION_START : CAPTION_START + 33]
    tags = chunk_figure_tags(caption, 1, CAPTION_START, CAPTION_START + 33, [FIGURE_1])
    assert tags == {"figure": "1"}
    # Same offsets on another page, or a figure without spans: no drawn-word tag.
    assert chunk_figure_tags(DRAWN_PAGE[block:end], 2, block, end, [FIGURE_1]) == {}
    plain = msgspec.structs.replace(FIGURE_1, text_spans=())
    assert chunk_figure_tags(DRAWN_PAGE[block:end], 1, block, end, [plain]) == {}


def test_attach_figures_keeps_only_spans_that_are_stored_picture_text() -> None:
    chunks: list[dict[str, object]] = [
        dict(paper_id="p", chunk_id=f"p:{i}", page_start=1, char_start=s, char_end=e, text=t)
        for i, (s, e, t) in enumerate(
            [(0, 27, "x"), (29, DRAWN_END + 4 + len(PICTURE_TEXT_END), "y")]
        )
    ]
    document: dict[str, object] = {"pages": [{"number": 1, "text": DRAWN_PAGE}], "chunks": chunks}
    stale = msgspec.structs.replace(FIGURE_1, text_spans=((DRAWN_START, DRAWN_END), (0, 5)))
    attach_figures(document, [stale])
    stored = document["figures"]
    assert isinstance(stored, list) and stored[0]["text_spans"] == ((DRAWN_START, DRAWN_END),)
    assert [c.get("figure_text") for c in chunks] == [None, True]


def drawn_row(text: str = "Encoder Retriever Decoder", section: str = "Results") -> EvidenceChunk:
    return EvidenceChunk(
        "p", "c5", text, 3, 100, 100 + len(text), section, figure="2", figure_text=True
    )


def test_figure_text_never_replaces_the_caption_chunk() -> None:
    scoped = [*figure_rows(), drawn_row()]
    _, c1, c2, c3, c4, c5 = scoped
    # The drawn words come after the caption and share its number; the caption is still linked.
    assert link_figures(plan("What does Figure 2 show?"), [c1, c2], scoped) == [c1, c3, c4, c2]
    # Drawn words at the top are not a caption, so they link nothing on their own.
    assert link_figures(plan("Which parts are drawn?"), [c5, c2], scoped) == [c5, c2]


def test_overviews_drop_words_drawn_inside_figures() -> None:
    sentence = "We propose an encoder that improved retrieval accuracy on Cedar."
    drawn = drawn_row(f"{PICTURE_TEXT_START}\n{sentence}\n{PICTURE_TEXT_END}", "Abstract")
    prose = EvidenceChunk("p", "c6", sentence, 1, 0, len(sentence), "Abstract")
    picked = retrieve("p", "Summarize the paper", [drawn, prose])
    assert [row.chunk_id for row in picked] == ["c6"]
    untagged = msgspec.structs.replace(drawn, figure="", figure_text=False)
    assert [r.chunk_id for r in retrieve("p", "Summarize the paper", [untagged, prose])] == ["c6"]
    question = msgspec.structs.replace(plan_answer("Summarize the paper"), overview=True)
    rows = read_for_answer(context("f"), "p", question, [*figure_rows(), drawn, prose])
    assert all(not row.figure_text for row in (*rows.rows, *rows.pool))
