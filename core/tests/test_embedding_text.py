"""S27: contextual embedding text and table row groups (embedding input only)."""

from __future__ import annotations

from arie_core.pipeline.embedding import (
    chunk_kind,
    embedding_text,
    embedding_units,
    vector_chunk_id,
    vector_revision,
)

PAPER: dict[str, object] = {
    "figures": [{"number": "3", "page": 2, "box": [0, 0, 1, 1], "caption": "Figure 3: Loss."}]
}


def test_prose_gets_only_its_section_first() -> None:
    chunk = {"text": "We train for ten epochs."}
    assert embedding_text(chunk, PAPER, "Method / Training") == (
        "Method / Training\nWe train for ten epochs."
    )
    assert embedding_text(chunk, PAPER, "") == "We train for ten epochs."
    assert chunk_kind(chunk) == "prose"


def test_figure_text_and_ocr_get_their_figure_caption() -> None:
    drawn = {"text": "epoch loss", "figure": "3", "figure_text": True}
    ocr = {"text": "0.5 1.0", "figure": "3", "figure_text": True, "ocr": True}
    assert embedding_text(drawn, PAPER, "Results") == "Results\nFigure 3: Loss.\nepoch loss"
    assert embedding_text(ocr, PAPER, "") == "Figure 3: Loss.\n0.5 1.0"
    assert (chunk_kind(drawn), chunk_kind(ocr)) == ("figure_text", "ocr")
    # A figure without a record keeps its label.
    lone = {"text": "x y", "figure": "9", "figure_text": True}
    assert embedding_text(lone, PAPER, "") == "Figure 9\nx y"
    assert chunk_kind({"text": "Figure 3: Loss.", "figure": "3"}) == "caption"


def test_table_caption_moves_first() -> None:
    table = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\nTable 4: Below."
    chunk = {"text": table}
    assert chunk_kind(chunk) == "table"
    assert embedding_text(chunk, PAPER, "Results") == (
        "Results\nTable 4: Below.\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    )


def _long_table(rows: int) -> str:
    lines = [f"| model-{i:02d} | {50 + i}.0 | baseline run number {i} |" for i in range(rows)]
    return "Table 7: Long results.\n\n| Model | F1 | Note |\n| --- | --- | --- |\n" + "\n".join(
        lines
    )


def _chunk(paper_id: str, chunk_id: str, text: str, start: int = 0) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "chunk_id": chunk_id,
        "page_start": 1,
        "char_start": start,
        "char_end": start + len(text),
        "text": text,
    }


def test_long_table_becomes_row_groups_with_caption_and_header_each() -> None:
    paper: dict[str, object] = {
        "chunks": [
            _chunk("p", "p:0", "## Results\nShort prose."),
            _chunk("p", "p:1", _long_table(100), 30),
            _chunk("other", "o:0", "Foreign."),
        ]
    }
    units = embedding_units("p", paper)
    groups = [unit for unit in units if unit.chunk_id == "p:1"]
    assert len(groups) > 2
    assert [unit.id for unit in groups] == [f"p:1#t{k}" for k in range(len(groups))]
    for unit in groups:
        assert unit.kind == "table"
        assert unit.text.startswith("Results\nTable 7: Long results.\n| Model | F1 | Note |\n")
        assert len(unit.text) <= 1500
    rows = [line for unit in groups for line in unit.text.splitlines() if "model-" in line]
    assert len(rows) == 100
    assert units[0].id == "p:0" and units[0].kind == "prose"
    assert all(unit.chunk_id != "o:0" for unit in units)


def test_short_table_stays_one_vector() -> None:
    paper: dict[str, object] = {"chunks": [_chunk("p", "p:0", _long_table(3))]}
    (unit,) = embedding_units("p", paper)
    assert unit.id == "p:0"


def test_vector_ids_map_back_to_chunk_ids() -> None:
    assert vector_chunk_id("p:1#t3", {"chunk_id": "p:1"}) == "p:1"
    assert vector_chunk_id("p:1#t3", {}) == "p:1"
    assert vector_chunk_id("2401.1:5", {}) == "2401.1:5"


def test_vector_revision_carries_the_embedding_format() -> None:
    assert vector_revision("abc", 12) == "abc:12:e2"
