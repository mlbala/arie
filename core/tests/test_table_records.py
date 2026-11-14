"""S27: table records and layouts from stored chunk text."""

from __future__ import annotations

from arie_core.pipeline.chunker import chunk_spans
from arie_core.pipeline.tables import chunk_table_tags, table_context, table_layout, table_records

TABLE = "| Model | F1 |\n| --- | --- |\n| BERT | 85.1 |\n| ELMo | 80.2 |"


def _chunks(page: str) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": f"p:{i}",
            "page_start": 3,
            "char_start": start,
            "char_end": end,
            "text": page[start:end],
        }
        for i, (start, end) in enumerate(chunk_spans(page))
    ]


def _check_offsets(page: str, chunks: list[dict[str, object]]) -> None:
    for record in table_records(chunks):
        assert [page[a:b] for a, b in record.rows] == ["| BERT | 85.1 |", "| ELMo | 80.2 |"]
        assert page[record.header_start : record.header_end] == "| Model | F1 |"
        if record.caption_start >= 0:
            assert page[record.caption_start : record.caption_end] == record.caption


def test_caption_above_joins_and_records_offsets_in_the_page_text() -> None:
    page = "Intro text here.\n\nTable 2: Scores on SQuAD.\n\n" + TABLE + "\n\nMore prose."
    chunks = _chunks(page)
    records = table_records(chunks)
    assert len(records) == 1
    record = records[0]
    assert (record.number, record.page, record.caption) == ("2", 3, "Table 2: Scores on SQuAD.")
    assert record.chunk_id == "p:1"
    _check_offsets(page, chunks)
    assert chunk_table_tags(str(chunks[1]["text"])) == {"table": "2"}
    assert chunk_table_tags(str(chunks[0]["text"])) == {}


def test_caption_below_is_found() -> None:
    page = "Before.\n\n" + TABLE + "\n\n**Table A1.** Ablations.\n\nAfter."
    chunks = _chunks(page)
    (record,) = table_records(chunks)
    assert (record.number, record.caption) == ("A1", "**Table A1.** Ablations.")
    _check_offsets(page, chunks)


def test_table_without_caption_has_no_number_and_no_tag() -> None:
    page = "Before.\n\n" + TABLE
    chunks = _chunks(page)
    (record,) = table_records(chunks)
    assert (record.number, record.caption, record.caption_start) == ("", "", -1)
    _check_offsets(page, chunks)
    assert chunk_table_tags(TABLE) == {}


def test_header_only_table_has_no_rows() -> None:
    layout = table_layout("Table 1: Empty.\n\n| A | B |\n| --- | --- |\n|  |  |")
    assert layout is not None
    assert layout.rows == ()
    assert layout.number == "1"


def test_no_table_and_ocr_chunks_give_no_record() -> None:
    assert table_layout("Plain prose with a | pipe? no.") is None
    chunk = {"chunk_id": "p:9", "page_start": 1, "char_start": 0, "text": TABLE, "ocr": True}
    assert table_records([chunk]) == []


def test_context_is_caption_and_header_only() -> None:
    text = "Table 2: Scores.\n" + TABLE
    layout = table_layout(text)
    assert layout is not None
    assert table_context(text, layout) == "Table 2: Scores.\n| Model | F1 |"
