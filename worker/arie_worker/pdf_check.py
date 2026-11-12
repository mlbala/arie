"""Structural PDF check, run in the PDF pool before the parser (SPEC 6.2, ADR 0013).

A stored PDF that needs a user password, does not open, carries active content or is
abnormally large is rejected with ``PdfRejected("quarantined: <rule>")``, so the parser
never sees it. Link actions LaTeX writes (``/URI``, ``/GoTo``, ``/GoToR``) and an
``/OpenAction`` that is a destination are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pikepdf
from arie_core.pipeline import PdfRejected
from arie_core.registry.manager import PdfCheckConfig

# Active-content rules in the order a rejection names them.
ACTIVE_RULES = (
    "javascript",
    "launch",
    "import_data",
    "submit_form",
    "goto_e",
    "embedded_file",
    "rich_media",
    "xfa",
)
_KEY_RULES = {
    "/JS": "javascript",
    "/JavaScript": "javascript",
    "/EmbeddedFiles": "embedded_file",
    "/EF": "embedded_file",
    "/XFA": "xfa",
}
_ACTION_RULES = {
    "/JavaScript": "javascript",
    "/Launch": "launch",
    "/ImportData": "import_data",
    "/SubmitForm": "submit_form",
    "/GoToE": "goto_e",
}


@dataclass(frozen=True, slots=True)
class PdfFacts:
    """What the check measured in one PDF; ``rules`` are the active-content rules hit."""

    objects: int
    pages: int
    filter_depth: int
    rules: tuple[str, ...]


def _open(path: str | Path) -> pikepdf.Pdf:
    try:
        return pikepdf.open(path)
    except pikepdf.PasswordError:
        raise PdfRejected("quarantined: encrypted") from None
    except Exception:
        raise PdfRejected("quarantined: unreadable") from None


def _scan(pdf: pikepdf.Pdf) -> tuple[int, set[str]]:
    """Walk every object, and every direct object inside one: filter depth and rules hit.

    Indirect references are not followed; each indirect object is visited once from the
    object list. A stack, not recursion, so deep nesting cannot overflow.
    """
    depth = 0
    rules: set[str] = set()
    for root in pdf.objects:
        stack: list[object] = [root]
        while stack:
            obj = stack.pop()
            if isinstance(obj, pikepdf.Array):
                stack.extend(item for item in obj if _direct_container(item))
                continue
            if not isinstance(obj, (pikepdf.Dictionary, pikepdf.Stream)):
                continue
            for key, value in obj.items():
                if key in _KEY_RULES:
                    rules.add(_KEY_RULES[key])
                if key == "/S" and isinstance(value, pikepdf.Name):
                    rule = _ACTION_RULES.get(str(value))
                    if rule:
                        rules.add(rule)
                elif key == "/Subtype" and value == pikepdf.Name.RichMedia:
                    rules.add("rich_media")
                elif key == "/Type" and value == pikepdf.Name.EmbeddedFile:
                    rules.add("embedded_file")
                elif key == "/Filter" and isinstance(obj, pikepdf.Stream):
                    depth = max(depth, len(value) if isinstance(value, pikepdf.Array) else 1)
                if _direct_container(value):
                    stack.append(value)
    return depth, rules


def _direct_container(value: object) -> bool:
    # pikepdf hands numbers and booleans back as Python values; only containers nest.
    return isinstance(value, (pikepdf.Dictionary, pikepdf.Array)) and not value.is_indirect


def _ordered(rules: set[str]) -> tuple[str, ...]:
    return tuple(rule for rule in ACTIVE_RULES if rule in rules)


def inspect_pdf(path: str | Path) -> PdfFacts:
    """Measure one PDF without limits (``make pdf-report``); open failures still raise."""
    with _open(path) as pdf:
        depth, rules = _scan(pdf)
        return PdfFacts(len(pdf.objects), len(pdf.pages), depth, _ordered(rules))


def check_pdf(path: str | Path, limits: PdfCheckConfig) -> None:
    """Raise ``PdfRejected`` naming the first rule this PDF fails; return if it passes."""
    with _open(path) as pdf:
        # The counts are cheap; check them before walking a huge file.
        if len(pdf.objects) > limits.max_objects:
            raise PdfRejected("quarantined: too_many_objects")
        if len(pdf.pages) > limits.max_pages:
            raise PdfRejected("quarantined: too_many_pages")
        depth, rules = _scan(pdf)
    if rules:
        raise PdfRejected(f"quarantined: {_ordered(rules)[0]}")
    if depth > limits.max_filter_depth:
        raise PdfRejected("quarantined: filter_depth")


__all__ = ["ACTIVE_RULES", "PdfFacts", "PdfRejected", "check_pdf", "inspect_pdf"]
