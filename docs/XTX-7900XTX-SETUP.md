# RX 7900 XTX — build & run from scratch

Everything needed to reproduce the single-stream 640K Inkling-2-Mini setup on Windows native HIP.

## Requirements

- GPU: AMD RX 7900 XTX (gfx1100), 24 GB. ROCm Windows HIP SDK (`hipcc`/clang targeting gfx1100).
- OS: Windows 11; Visual Studio build tools (for `mt.exe` manifest embedding); CMake + Ninja.
- RAM: 32 GB. Run only one model-loading process at a time; keep `gpu_guard` running.
- Model: `Inkling-2-Mini-UD-Q4_K_XL.gguf` (Unsloth quantization, ~17.6 GB).
- Python 3.12 with `numpy` + `scipy` and a `gguf-py` checkout for conversion tools; Python 3.12 with `jinja2` for the native serve layer.

## 1. Build

```powershell
.\scripts\build-windows-hip.ps1 `
  -EngineDir  <repo>\engine `
  -BuildDir   <repo>\build-hip-win `
  -RocmSdk    <path\to\_rocm_sdk_devel> `
  -Target     strata-ik2
```

Flags: `AMDGPU_TARGETS=gfx1100`, `GGML_HIP=ON`, `STRATA_ENABLE_CUDA=OFF`, `STRATA_ENABLE_HIP=ON`. At runtime put the ROCm SDK `bin` and `lib\llvm\bin` on `PATH` (launcher does this). Compile with the SDK clang/headers; run with the SDK HIP/hipBLAS/rocBLAS DLLs.

## 2. Convert INT4 sidecars

The engine reads the GGUF for routing/metadata and uses converted sidecars for the fast INT4 kernels:

```powershell
python tools\int4\nvfp4_to_int4.py   ...   # NVFP4 experts (layers 0-19) -> int4-bin
python tools\int4\q8_to_int4_l20.py  ...   # Q8_0 experts (layers 20-23) -> int4-bin
python tools\int4\q8_dense_to_int4.py ...  # dense projections + out_w -> int4-dense-bin
python tools\int4\validate_int4.py ...     # round-trip check
```

Set `IK2_INT4_DIR` / `IK2_INT4_DENSE_DIR` to those directories. **Wrong paths silently fall back to Q8 experts and collapse to ~3.9 tok/s** — check the `resident:` startup line.

## 3. Tokenizer pack

CLI mode consumes token-id files (`--tokens @file`, comma-separated) from the Inkling-2 tokenizer (`inkling2` / O200K BPE, `ignore_merges=true`). Serve mode generates tokens internally. For native serve, build a tokenizer directory with `vocab.json`, `merges.txt`, `token_type.json`, `tokenizer.json` (`pre:"inkling2"`), and `chat_template.jinja` (strip llama.cpp `{% generation %}` tags — see `scripts/native-serve/make_serve_template.py`).

## 4. Run (one click)

```bat
start.bat    :: starts native Strata serve on port 8099, prints OK, runs in background
stop.bat     :: stops serve + engine, leaves gpu_guard
```

Startup markers to expect:

```
fit(actual): experts 12.23 GiB + dense 1.10 + KV 5.36 + reserve 0.50 = 19.19 vs free 22.99 GiB -> FITS
resident: INT4 experts (l<20) + Q8_0 (l>=20) all in VRAM
conversation cache: 8192 MiB budget, 64 snapshots
INFO engine=ik2 ... kv=int4 ctx=655360 batch_slots=0
READY 655360 stop
```

GPU safety: refuse to start if another model process runs; never run two model-loading processes concurrently on a 24 GB WDDM card.

## 5. Endpoints

| What | URL |
| --- | --- |
| Web chat | `http://127.0.0.1:8099/` |
| Monitor | `http://127.0.0.1:8099/api-monitor` (or `/#monitor`) |
| OpenAI chat | `http://127.0.0.1:8099/v1/chat/completions` |
| Anthropic | `http://127.0.0.1:8099/v1/messages` |
| Health | `http://127.0.0.1:8099/health` |

```sh
curl http://127.0.0.1:8099/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"你好"}],"max_tokens":200}'
```

Portable launcher (all paths from params/env): `scripts/native-serve/run-native-serve.ps1`. Desktop App integration: `scripts/native-serve/README.md`.
