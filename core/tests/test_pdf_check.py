"""The structural PDF check (S18): PDFs are built here with pikepdf, never fetched."""

from __future__ import annotations

import asyncio
import pickle
import zlib
from collections.abc import Callable
from pathlib import Path

import pikepdf
import pytest
from arie_core.contracts import ParsedDoc
from arie_core.pipeline import IngestionPipeline, PdfRejected
from arie_core.registry.manager import PdfCheckConfig
from arie_worker.extraction import _extract  # pyright: ignore[reportPrivateUsage]
from arie_worker.pdf_check import check_pdf, inspect_pdf
from pikepdf import Array, Dictionary, Name, String

LIMITS = PdfCheckConfig()


def build(path: Path, edit: Callable[[pikepdf.Pdf], None] | None = None, **save: object) -> str:
    pdf = pikepdf.new()
    pdf.add_blank_page()
    if edit is not None:
        edit(pdf)
    # The save options (encryption, stream levels) differ per test, so they pass untyped.
    pdf.save(path, **save)  # pyright: ignore[reportArgumentType]
    return str(path)


def link(pdf: pikepdf.Pdf, action: Dictionary) -> None:
    annotation = Dictionary(
        Type=Name.Annot, Subtype=Name.Link, Rect=Array([0, 0, 10, 10]), A=action
    )
    pdf.pages[0].Annots = pdf.make_indirect(Array([annotation]))


def javascript_names(pdf: pikepdf.Pdf) -> None:
    script = pdf.make_indirect(Dictionary(S=Name.JavaScript, JS=String("app.alert(1)")))
    pdf.Root.Names = Dictionary(JavaScript=Dictionary(Names=Array([String("open"), script])))


def launch_link(pdf: pikepdf.Pdf) -> None:
    link(pdf, Dictionary(S=Name.Launch, F=String("cmd.exe")))


def embedded_file(pdf: pikepdf.Pdf) -> None:
    stream = pdf.make_stream(b"payload")
    spec = Dictionary(Type=Name.Filespec, F=String("a.txt"), EF=Dictionary(F=stream))
    pdf.Root.Names = Dictionary(EmbeddedFiles=Dictionary(Names=Array([String("a.txt"), spec])))


def xfa_form(pdf: pikepdf.Pdf) -> None:
    pdf.Root.AcroForm = Dictionary(Fields=Array([]), XFA=pdf.make_stream(b"<xdp/>"))


def submit_form_link(pdf: pikepdf.Pdf) -> None:
    link(pdf, Dictionary(S=Name.SubmitForm, F=String("https://example.org/collect")))


def goto_link(pdf: pikepdf.Pdf) -> None:
    link(pdf, Dictionary(S=Name.GoTo, D=Array([pdf.pages[0].obj, Name.Fit])))


def second_page(pdf: pikepdf.Pdf) -> None:
    pdf.add_blank_page()


def javascript_open_action(pdf: pikepdf.Pdf) -> None:
    pdf.Root.OpenAction = Dictionary(S=Name.JavaScript, JS=String("app.alert(1)"))


def hyperref(pdf: pikepdf.Pdf) -> None:
    # What hyperref writes into most arXiv PDFs: a destination OpenAction and URI links.
    pdf.Root.OpenAction = Array([pdf.pages[0].obj, Name.Fit])
    link(pdf, Dictionary(S=Name.URI, URI=String("https://arxiv.org/abs/2501.00001")))


def nested_filters(pdf: pikepdf.Pdf) -> None:
    data = b"x"
    for _ in range(6):
        data = zlib.compress(data)
    stream = pikepdf.Stream(pdf, data)
    stream.Filter = Array([Name.FlateDecode] * 6)
    pdf.Root.Extra = pdf.make_indirect(stream)


def test_a_plain_pdf_passes_and_is_measured(tmp_path: Path) -> None:
    path = build(tmp_path / "plain.pdf")
    check_pdf(path, LIMITS)
    facts = inspect_pdf(path)
    assert facts.pages == 1 and facts.objects >= 3 and facts.rules == ()


@pytest.mark.parametrize("edit", [hyperref, goto_link])
def test_latex_links_and_a_destination_open_action_pass(
    tmp_path: Path, edit: Callable[[pikepdf.Pdf], None]
) -> None:
    check_pdf(build(tmp_path / "links.pdf", edit), LIMITS)


def test_an_owner_password_alone_passes(tmp_path: Path) -> None:
    locked = pikepdf.Encryption(owner="owner", user="", allow=pikepdf.Permissions(extract=False))
    check_pdf(build(tmp_path / "owner.pdf", encryption=locked), LIMITS)


@pytest.mark.parametrize(
    ("edit", "rule"),
    [
        (javascript_names, "javascript"),
        (launch_link, "launch"),
        (embedded_file, "embedded_file"),
        (xfa_form, "xfa"),
        (submit_form_link, "submit_form"),
        (javascript_open_action, "javascript"),
    ],
)
def test_active_content_is_rejected_with_its_rule(
    tmp_path: Path, edit: Callable[[pikepdf.Pdf], None], rule: str
) -> None:
    path = build(tmp_path / "active.pdf", edit)
    with pytest.raises(PdfRejected, match=f"^quarantined: {rule}$"):
        check_pdf(path, LIMITS)
    assert rule in inspect_pdf(path).rules


def test_a_user_password_and_a_broken_file_are_rejected(tmp_path: Path) -> None:
    locked = pikepdf.Encryption(owner="owner", user="user")
    with pytest.raises(PdfRejected, match="^quarantined: encrypted$"):
        check_pdf(build(tmp_path / "user.pdf", encryption=locked), LIMITS)
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nnot a pdf")
    with pytest.raises(PdfRejected, match="^quarantined: unreadable$"):
        check_pdf(broken, LIMITS)


def test_size_limits_reject_objects_pages_and_filter_chains(tmp_path: Path) -> None:
    plain = build(tmp_path / "plain.pdf")
    with pytest.raises(PdfRejected, match="^quarantined: too_many_objects$"):
        check_pdf(plain, PdfCheckConfig(max_objects=2))
    two_pages = build(tmp_path / "two.pdf", second_page)
    with pytest.raises(PdfRejected, match="^quarantined: too_many_pages$"):
        check_pdf(two_pages, PdfCheckConfig(max_pages=1))
    # Saved as is: by default pikepdf would re-encode the chain as one filter.
    deep = build(
        tmp_path / "deep.pdf",
        nested_filters,
        compress_streams=False,
        stream_decode_level=pikepdf.StreamDecodeLevel.none,
    )
    assert inspect_pdf(deep).filter_depth == 6
    with pytest.raises(PdfRejected, match="^quarantined: filter_depth$"):
        check_pdf(deep, PdfCheckConfig(max_filter_depth=5))


def test_the_worker_checks_before_it_loads_the_parser(tmp_path: Path) -> None:
    path = build(tmp_path / "js.pdf", javascript_names)
    # No such parser exists: reaching the parser would raise a different error.
    with pytest.raises(PdfRejected, match="javascript"):
        _extract(".", "no-such-parser", "{}", 5.0, path, LIMITS)


def test_a_rejection_survives_the_trip_back_from_the_worker() -> None:
    error = pickle.loads(pickle.dumps(PdfRejected("quarantined: xfa")))
    assert isinstance(error, PdfRejected) and str(error) == "quarantined: xfa"
    assert not isinstance(error, RuntimeError)  # the sync's parse retry catches those


@pytest.mark.asyncio
async def test_parse_document_quarantines_a_rejected_pdf_at_once_and_deletes_it(
    tmp_path: Path,
) -> None:
    pipeline = IngestionPipeline(storage_root=tmp_path)
    path = build(tmp_path / "js.pdf", javascript_names)
    calls: list[str] = []

    async def checked_parse(file_path: str) -> ParsedDoc:
        calls.append(file_path)
        check_pdf(file_path, LIMITS)
        raise AssertionError("the parser must not run")

    pipeline._run_worker_parse = checked_parse  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(PdfRejected, match="^quarantined: javascript$"):
        await pipeline.parse_document("rejected", path)
    assert calls == [path]
    assert pipeline.is_quarantined("rejected")
    assert not await asyncio.to_thread(Path(path).exists)


@pytest.mark.asyncio
async def test_a_rejected_file_outside_the_storage_root_is_not_deleted(tmp_path: Path) -> None:
    pipeline = IngestionPipeline(storage_root=tmp_path / "papers")
    path = build(tmp_path / "elsewhere.pdf", javascript_names)

    async def rejected(file_path: str) -> ParsedDoc:
        raise PdfRejected("quarantined: javascript")

    pipeline._run_worker_parse = rejected  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(PdfRejected):
        await pipeline.parse_document("rejected", path)
    assert await asyncio.to_thread(Path(path).exists)
