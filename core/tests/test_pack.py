"""The paper pack: markers map to chunks, noise is left out, bytes are stable."""

from __future__ import annotations

import random

from arie_core.harness.reader import EvidenceChunk
from arie_core.harness.session import estimate_tokens
from arie_core.pipeline.pack import (
    PaperPack,
    build_pack,
    escape_markers,
    marker_numbers,
    pack_fits,
    pack_revision,
)

PAPER = "2601.00001"


def chunk(number: int, text: str, page: int = 1, paper_id: str = PAPER) -> EvidenceChunk:
    start = number * 100
    return EvidenceChunk(paper_id, f"{paper_id}:{number}", text, page, start, start + len(text))


TOC = "1 Introduction . . . . . . . . 1\n2 Methods . . . . . . . . 3\n3 Results . . . . . . . . 5"
CHUNKS = [
    chunk(0, "## Introduction\nDeep models help. We study how they are used in practice."),
    chunk(1, "Our method is simple and it improves accuracy by four points."),
    chunk(2, "## Methods\nWe train for ten epochs with a learning rate of 0.1.", page=2),
    chunk(3, "$$ L = \\sum_i \\ell(x_i) $$", page=2),
    chunk(4, "Equal contribution. Listing order is random.", page=2),
    chunk(5, TOC, page=2),
    chunk(6, "## Results\nAccuracy rose to 91.2 percent. Compare [c1] with [c22].", page=3),
    chunk(7, "## Acknowledgements\nWe thank the reviewers for their help.", page=3),
    chunk(8, "## References\n[1] A. Author. A paper about models. 2020.", page=3),
]


def build(chunks: list[EvidenceChunk], paper_id: str = PAPER) -> PaperPack:
    return build_pack(paper_id, "A Study of Models", "We study models.", "hash:9", chunks)


def test_every_marker_maps_to_a_real_chunk_and_its_text() -> None:
    pack = build(CHUNKS)
    by_number = {int(c.chunk_id.rpartition(":")[2]): c for c in CHUNKS}
    assert pack.chunk_numbers == marker_numbers(pack.text) == (0, 1, 2, 3, 6)
    for number in pack.chunk_numbers:
        row = by_number[number]
        assert row.chunk_id == f"{PAPER}:{number}"
        body = escape_markers(row.text.strip())
        assert f"[c{number}] {body}" in pack.text
        # The offsets still address the page text; the pack only quotes the chunk.
        assert row.char_end - row.char_start == len(row.text)


def test_title_abstract_outline_and_sections_come_first_and_in_order() -> None:
    text = build(CHUNKS).text
    order = [
        text.index("Title: A Study of Models"),
        text.index("Abstract:\nWe study models."),
        text.index("Outline:\n- Introduction\n- Methods\n- Results"),
        text.index("Section: Introduction\n[c0]"),
        text.index("[c1]"),
        text.index("Section: Methods\n[c2]"),
        text.index("[c3]"),
        text.index("Section: Results\n[c6]"),
    ]
    assert order == sorted(order)
    assert text.count("Section: Introduction") == 1


def test_references_acknowledgements_credits_and_contents_are_absent() -> None:
    pack = build(CHUNKS)
    for number in (4, 5, 7, 8):
        assert number not in pack.chunk_numbers
    for noise in ("Equal contribution", ". . . . .", "thank the reviewers", "A. Author"):
        assert noise not in pack.text
    assert "References" not in pack.text
    assert "Acknowledgements" not in pack.text


def test_equations_stay_and_a_bare_heading_chunk_is_left_out() -> None:
    chunks = [*CHUNKS, chunk(9, "## Limitations", page=3), chunk(10, "We note one limit.", page=3)]
    pack = build(chunks)
    assert "$$ L = \\sum_i \\ell(x_i) $$" in pack.text
    assert 9 not in pack.chunk_numbers
    assert "Section: Limitations\n[c10]" in pack.text


def test_literal_markers_in_the_paper_are_escaped_and_never_counted() -> None:
    pack = build(CHUNKS)
    assert "Compare [\\c1] with [\\c22]." in pack.text
    assert marker_numbers(pack.text) == pack.chunk_numbers
    assert 22 not in marker_numbers(pack.text)
    hostile = build_pack(PAPER, "Study [c99]", "See [c98].", "r", CHUNKS)
    assert marker_numbers(hostile.text) == hostile.chunk_numbers
    assert "Title: Study [\\c99]" in hostile.text
    assert escape_markers("[c1][c20] and [c] and [cx1]") == "[\\c1][\\c20] and [c] and [cx1]"


def test_two_builds_give_identical_bytes_whatever_the_input_order() -> None:
    first = build(CHUNKS)
    shuffled = list(CHUNKS)
    random.Random(7).shuffle(shuffled)
    again = build(shuffled)
    assert again == first
    assert again.text.encode() == build(CHUNKS).text.encode()


def test_a_pack_holds_only_the_bound_papers_chunks() -> None:
    other = [
        chunk(0, "Another paper says models are cheap and we agree.", paper_id="2601.99999"),
        # A row that claims this paper but carries another paper's chunk id.
        EvidenceChunk(PAPER, "2601.99999:5", "Smuggled text that is a proper sentence.", 1, 0, 40),
        EvidenceChunk("2601.99999", f"{PAPER}:4", "Wrong paper but a borrowed id here.", 1, 0, 35),
    ]
    pack = build([*CHUNKS, *other])
    assert pack == build(CHUNKS)
    assert "Another paper" not in pack.text and "Smuggled" not in pack.text
    assert "borrowed" not in pack.text
    empty = build(other, paper_id="2601.11111")
    assert empty.chunk_numbers == ()


def test_token_estimate_and_revision() -> None:
    pack = build(CHUNKS)
    assert pack.token_estimate == estimate_tokens(len(pack.text.encode()))
    assert pack.revision == pack_revision("hash:9") == "hash:9:p1"


def test_pack_fits_uses_the_share_of_the_context_window() -> None:
    pack = PaperPack("t", (), 38_400, "r")
    assert pack_fits(pack, 64_000, 0.6)  # exactly 60%
    assert not pack_fits(PaperPack("t", (), 38_401, "r"), 64_000, 0.6)
    assert pack_fits(PaperPack("t", (), 100, "r"), 64_000, 0.6)
    assert not pack_fits(pack, 64_000, 0.5)
    assert pack_fits(pack, 38_400, 1.0)
