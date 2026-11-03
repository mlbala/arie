"""Tier 3 entailment plugin: worker-only, fail-closed, and calibrated on real claims.

The real-model test runs only when the model is already in the Hugging Face cache (or
``ARIE_NLI_CACHE`` points at one), so the default suite never downloads ~280 MB.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from arie_core.harness.verifier import MIN_ENTAILMENT
from arie_core.registry.manager import RegistryManager

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cross-encoder/nli-deberta-v3-xsmall"


def entailment(**kwargs: object) -> Any:
    return RegistryManager(project_root=ROOT).build_plugin("entailment", "nli-onnx", **kwargs)


def test_entailment_plugin_is_discovered() -> None:
    assert "nli-onnx" in RegistryManager(project_root=ROOT).get_available_plugins("entailment")


@pytest.mark.asyncio
async def test_scoring_outside_a_worker_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARIE_ENTAILMENT_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="inside arie-worker"):
        await entailment().score(None, [("premise", "hypothesis")])


@pytest.mark.asyncio
async def test_a_model_without_an_entailment_label_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"id2label": {"0": "positive", "1": "negative"}}))
    plugin = entailment()
    monkeypatch.setenv("ARIE_ENTAILMENT_WORKER", "1")
    monkeypatch.setattr(plugin, "_download", lambda: {"config": config})
    with pytest.raises(ValueError, match="entailment"):
        await plugin.score(None, [("premise", "hypothesis")])


def cached_model() -> str | None:
    explicit = os.environ.get("ARIE_NLI_CACHE")
    if explicit:
        return explicit
    hub = pytest.importorskip("huggingface_hub")
    found = hub.try_to_load_from_cache(MODEL, "onnx/model.onnx")
    return None if not isinstance(found, str) else str(Path(found).parents[4])


@pytest.mark.asyncio
async def test_real_model_in_a_worker_rejects_polarity_flips_and_keeps_paraphrases() -> None:
    cache = cached_model()
    if cache is None:
        pytest.skip(f"{MODEL} is not cached; set ARIE_NLI_CACHE to run")
    from arie_worker.entailment import EntailmentPool

    pool = EntailmentPool(ROOT, "nli-onnx", {"model": MODEL, "cache_dir": cache})
    quote = "The model performs well on all benchmarks."
    transformer = (
        "The Transformer dispenses with recurrence and convolutions entirely, "
        "relying solely on attention."
    )
    try:
        scores = await pool.score(
            [
                (quote, "The model performs badly."),
                (transformer, "The Transformer uses only attention instead of recurrence."),
                (transformer, "The Transformer relies on recurrent layers."),
            ],
            kill_after=120.0,
        )
    finally:
        pool.close()
    flip, paraphrase, contradiction = scores
    assert flip < MIN_ENTAILMENT and contradiction < MIN_ENTAILMENT
    assert paraphrase >= 0.9
