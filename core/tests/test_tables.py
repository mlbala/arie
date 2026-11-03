"""Tables: captions stay with their table, long tables are citable, contents are skipped."""

from __future__ import annotations

from arie_core.harness.answer import table_excerpt
from arie_core.harness.reader import EvidenceChunk, eligible, is_table_of_contents
from arie_core.pipeline.chunker import chunk_spans

CAPTION = "**Table 1 Model configurations used in our experiments.** Layers and widths."
TABLE = "|Model|Layers|Width|\n|---|---|---|\n|70M|4|512|\n|100M|6|768|"
PAGE = f"Some prose before the table.\n\n{CAPTION}\n\n{TABLE}\n\nProse after the table."


def blocks(text: str) -> list[str]:
    return [text[start:end] for start, end in chunk_spans(text)]


def test_a_caption_above_a_table_joins_its_span_losslessly() -> None:
    spans = chunk_spans(PAGE)
    assert [PAGE[a:b] for a, b in spans] == [
        "Some prose before the table.",
        f"{CAPTION}\n\n{TABLE}",
        "Prose after the table.",
    ]


def test_a_caption_below_a_table_joins_it_but_plain_prose_does_not() -> None:
    below = f"{TABLE}\n\nTable 2: Results on the held-out set.\n\nThe model is fast."
    assert blocks(below) == [
        f"{TABLE}\n\nTable 2: Results on the held-out set.",
        "The model is fast.",
    ]
    prose = f"We describe the setup.\n\n{TABLE}"
    assert blocks(prose) == ["We describe the setup.", TABLE]


def test_a_long_table_offers_its_caption_header_and_first_rows_as_an_exact_quote() -> None:
    rows = "".join(f"|{size}M|{size // 10}|{size * 4}|\n" for size in range(100, 6000, 100))
    text = f"{CAPTION}\n\n|Model|Layers|Width|\n|---|---|---|\n{rows}"
    assert len(text.encode()) > 650
    excerpt = table_excerpt(text, limit=300)
    assert excerpt.startswith(CAPTION) and "|---|---|---|" in excerpt and "|100M|10|400|" in excerpt
    assert excerpt in text and len(excerpt.encode()) <= 300
    assert excerpt.endswith("|")  # ends on a whole row
    assert table_excerpt("No table here.") == ""
    assert table_excerpt("|A|B|\n|---|---|") == ""  # header only


def test_contents_listings_are_not_evidence() -> None:
    contents = (
        "|2.1|Different Ways to Loop . . . . . . . . . . . .|3|\n"
        "|2.2|Loop-Specific Specialization . . . . . . . . . .|4|\n"
        "|2.3|Training . . . . . . . . . . . . . . . . . . .|5|"
    )
    assert is_table_of_contents(contents) and not is_table_of_contents(TABLE)
    rows = [
        EvidenceChunk("p", "toc", contents, 2, 0, len(contents), "Contents"),
        EvidenceChunk("p", "t1", TABLE, 9, 0, len(TABLE), "Results"),
    ]
    assert [row.chunk_id for row in eligible("What are the model sizes?", rows)] == ["t1"]
