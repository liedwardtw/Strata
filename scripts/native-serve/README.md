# Native Strata serve + agent tool calling (RX 7900 XTX)

This folder wires the **native Strata serve** (`terminal/serve/server.py`) directly
to the optimized `strata-ik2` engine. That gives the full experience the lightweight
`serve_http.py` bridge did not: the OpenAI `/v1/chat/completions` and Anthropic
`/v1/messages` APIs, the `/api-monitor` page, the `/` web page, MCP, and — the
reason for this — **agent tool/function calling**.

## Why no engine change is needed

`server.py` talks to any engine over a small stdio protocol (`INFO`, `READY`,
`GEN`, `T`, `DONE`). It launches `<exe> --serve <args...>`. There is no
`engine_kind` allow-list, so `engine_kind=ik2` is accepted as an ordinary string.
`INFO`'s `batch_slots` / `slot_read` are honored if present; omit them for the
simple solo `GEN/T/DONE` path.

Tool calling is implemented **entirely in serve's Python layer**. The model emits
typed segments (`<|content_thinking|>`,
`<|content_invoke_tool_json|>{"name":...,"args":...}<|end_message|>`);
`serve/frontend.py` `InklingRewriter` / `InklingOutputParser` turn those into
`tool_calls`. `parser_for_template()` selects `InklingOutputParser` when the chat
template contains `<|content_thinking|>` / `<|content_invoke_tool_json|>` /
`effort_map` — so the **only switch that matters is the chat template** in the
tokenizer pack. The engine needs no tool-token handling (qwen36 does not have it
either).

## One-time setup

All machine-specific paths are environment variables (nothing is hardcoded).

```
$env:IK2_EXE          = 'C:\path\to\strata-ik2.exe'
$env:IK2_MODEL        = 'C:\path\to\Inkling-2-Mini-UD-Q4_K_XL.gguf'
$env:IK2_TOKENIZER    = 'C:\path\to\inkling-pack\tokenizer'
$env:ROCM_SDK         = 'C:\path\to\_rocm_sdk_devel'
$env:IK2_INT4_DIR     = 'C:\path\to\int4-bin'
$env:IK2_INT4_DENSE_DIR = 'C:\path\to\int4-dense-bin'
$env:STRATA_CONTROL_PY = 'C:\path\to\Strata\control\python\python.exe'
```

Build the tokenizer pack from the GGUF with the *Inkling-2 chat template* (the
one that documents tools). The stock HF template has llama.cpp-only
`{% generation %}` tags; strip them for stock Jinja2:

```
python tools/strata_tokenizer.py --gguf <gguf> --out ik2 --chat-template <hf-chat-template.jinja> --check
python scripts/native-serve/make_serve_template.py <hf-chat-template.jinja> ik2/tokenizer/chat_template.jinja
```

The `tokenizer/` directory must contain `vocab.json`, `merges.txt`,
`token_type.json`, `tokenizer.json` (`{"pre":"inkling2", ...}`) and
`chat_template.jinja`.

## Run / stop

```
start-native.bat        # native serve on http://127.0.0.1:8099/v1
stop-native.bat         # stops serve + engine (gpu_guard stays)
```

Or directly: `run-native-serve.ps1 -Ctx 32768 -Port 8099`.

## Verified (RX 7900 XTX, gfx1100)

- `/health`, `/v1/models`, `/`, `/api-monitor`, `/v1/status` — all OK.
- Plain chat: `content` = answer, `reasoning_content` = thinking (typed segments split).
- **Tool calling**: a request with `tools=[get_weather]` produced
  `tool_calls: [{function: {name: get_weather, arguments: {...}}}]`,
  finish_reason `tool_calls`.

Point the Strata Desktop custom provider at `http://127.0.0.1:8099/v1`
(model `inkling-2-mini`) to use the app with real tool calling.
