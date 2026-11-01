from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import yaml


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    mapping = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for key, item in mapping.items():
        result[str(key)] = item
    return result


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class PromptCacheBuilder:
    """Build the 3-zone prompt cache layout used by the grounded Q&A harness."""

    def __init__(self, *, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root) if project_root is not None else _project_root()

    def _load_prompt(self, name: str) -> dict[str, object]:
        path = self.project_root / "prompts" / name
        with path.open("r", encoding="utf-8") as handle:
            loaded_value: object = yaml.safe_load(handle) or {}
        mapping = _as_mapping(loaded_value)
        if not isinstance(loaded_value, dict):
            raise TypeError(f"Expected a mapping for prompt {name!r}")
        return mapping

    @staticmethod
    def _trim_to_tokens(value: str, *, max_tokens: int) -> str:
        words = value.split()
        if len(words) <= max_tokens:
            return value
        return " ".join(words[:max_tokens])

    def build_zone_a(self, *, playbook_name: str = "qa.v7.yaml") -> str:
        playbook = self._load_prompt(playbook_name)
        rules = playbook.get("rules", [])
        fragments: list[str] = []
        if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes)):
            for possible_rule in cast(Sequence[object], rules):
                if not isinstance(possible_rule, Mapping):
                    continue
                mapping = cast(Mapping[object, object], possible_rule)
                raw_value = mapping.get("text")
                if raw_value is None:
                    raw_value = mapping.get("description")
                text = str(raw_value or "")
                if text:
                    fragments.append(text)
        return "\n\n".join(fragments)

    def build_zone_b(
        self,
        *,
        paper_id: str,
        title: str,
        source: str,
        custom_metadata: Mapping[str, str] | None = None,
        outline: Sequence[str] | None = None,
        playbook_name: str = "qa.v7.yaml",
    ) -> str:
        playbook = self._load_prompt(playbook_name)
        zone_template = str(
            playbook.get(
                "zone_b_template", "Paper Card: {title}\nSource: {source}\nPaper ID: {paper_id}\n"
            )
        )
        metadata_map = {str(key): str(value) for key, value in (custom_metadata or {}).items()}
        sorted_metadata = [f"{key}: {metadata_map[key]}" for key in sorted(metadata_map)]
        ordered_outline = list(outline or [])
        metadata_block = "\n".join(sorted_metadata) if sorted_metadata else "metadata: {}"
        outline_block = (
            "\n".join(f"- {item}" for item in ordered_outline)
            if ordered_outline
            else "- outline unavailable"
        )
        return zone_template.format(
            paper_id=paper_id,
            title=title,
            source=source,
            metadata_block=metadata_block,
            outline_block=outline_block,
            custom_metadata=metadata_block,
            outline=outline_block,
        )

    def build_zone_c(
        self,
        *,
        query: str,
        evidence: Sequence[str],
        verified_pairs: Sequence[tuple[str, str]],
        skill_hint: str,
        bats_status: str,
        max_tokens: int = 4500,
    ) -> str:
        pair_entries: list[str] = []
        for question, answer in verified_pairs[-2:]:
            pair_entries.append(f"Q: {question}\nA: {answer}")
        evidence_entries: list[str] = [
            f'<paper_text chunk="{index}">{item}</paper_text>'
            for index, item in enumerate(evidence)
        ]
        block = "\n\n".join(
            [
                "Verified Q&A pairs:",
                *pair_entries,
                "Evidence:",
                *evidence_entries,
                f"Skill hint: {skill_hint}",
                f"BATS: {bats_status}",
                f"Current query: {query}",
            ]
        )
        return self._trim_to_tokens(block, max_tokens=max_tokens)

    def build_prompt_cache(
        self,
        *,
        query: str,
        paper_id: str,
        title: str,
        source: str,
        custom_metadata: Mapping[str, str] | None = None,
        outline: Sequence[str] | None = None,
        evidence: Sequence[str] | None = None,
        verified_pairs: Sequence[tuple[str, str]] | None = None,
        skill_hint: str = "ground claims with direct evidence and cite exact quotes.",
        bats_status: str = "BATS: calls=0/3 tokens=0/12000 wall=0.00s/25.0s",
    ) -> dict[str, str]:
        return {
            "zone_a": self.build_zone_a(),
            "zone_b": self.build_zone_b(
                paper_id=paper_id,
                title=title,
                source=source,
                custom_metadata=custom_metadata,
                outline=outline,
            ),
            "zone_c": self.build_zone_c(
                query=query,
                evidence=evidence or [],
                verified_pairs=verified_pairs or [],
                skill_hint=skill_hint,
                bats_status=bats_status,
            ),
        }


def build_zone_prompt(*args: object, **kwargs: object) -> dict[str, str]:
    return PromptCacheBuilder().build_prompt_cache(*args, **kwargs)


__all__ = ["PromptCacheBuilder", "build_zone_prompt"]
