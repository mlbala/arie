# S07: Streaming usage and cache stats in LLM plugins

**Needs:** S01 · **Size:** M

## Goal

Streaming calls report token usage, including cached input tokens, so explain mode can show its cache hit rate.

## Read first

- `core/arie_core/contracts/dtos.py` (`LLMDelta`, `TokenUsage` already has `cache_hit` and `cache_miss`)
- `stream` and `complete` in `plugins/llm-deepseek`, `plugins/llm-openai-compatible`, `plugins/llm-gemini`
- How each plugin calls its API (SDK or raw HTTP); this decides how to mock it in tests

## Change

- Add `usage: TokenUsage | None = None` to `LLMDelta`; the final delta carries it.
- Fill `cache_hit` and `cache_miss` in both `complete` and `stream`, using each API's own fields:
  - DeepSeek: `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`; streaming needs usage enabled in the request.
  - OpenAI-compatible: cached input tokens from the usage details of whichever API the plugin calls (chat completions and the Responses API name the field differently; check the plugin and the provider docs). Streaming needs usage enabled in the request.
  - Gemini: the cached token count in the response's usage metadata (check the field name for the SDK the plugin uses).
- If a provider reports nothing, leave the fields at 0 and say so in PROGRESS.

## Tests

- New `contract_tests/test_llm_plugins.py`: for each plugin, a mocked streaming response (no network) yields text deltas and a final delta whose usage has the cache fields.

## Done when

- `make check` passes. SPEC §4 notes that LLM streams end with usage.
