# OMEN

**Will it fit? How fast? What does it cost?**
OMEN is a GPU fit calculator for open AI models. Pick a model, a graphics card and how many people will use it, and OMEN tells you whether it fits in memory, roughly how fast it will run, and what each user costs per hour.

**Live demo: https://srijithlegend.github.io/OMEN/** (served by GitHub Pages from the `main` branch)

![OMEN calculator](docs/screenshot.png)

## What you get

- **A clear verdict**: fits or does not fit, with the headroom left over (and how many GPUs you would need if it does not fit).
- **Memory breakdown**: weights, per-user chat memory (KV cache) and overhead, drawn against your GPU's limit.
- **Quality loss**: how much worse each quantization is than fp16 (perplexity increase). Measured for GGUF formats (llama.cpp, Llama-2-7B), typical estimates for the rest.
- **Speed ceiling**: the best-case tokens per second for one user.
- **Cost per user-hour**: the GPU price divided by the users that fit.
- **Plenty of choices**: open models from Meta, Mistral, Qwen, Google, Microsoft, DeepSeek, OpenAI and others, filterable by use case (code, vision, finance, math and reasoning), many quantization formats, and a list of GPUs.
- **A plain-English Guide tab** that explains every term (VRAM, quantization, KV cache and more) for people new to running models.
- **Shareable links**: your setup is stored in the page URL.

## Try it

No install and no build step.

1. Download or clone this repository.
2. Open `index.html` in your browser.
3. Choose a model, quantization and GPU, set context and users, then press **Calculate**.

```bash
git clone https://github.com/SrijithLegend/OMEN.git
cd OMEN
# then open index.html in your browser
```

New to the terms? Open the **Guide** tab at the top of the page.

## How the numbers are worked out

| Result | Formula |
|---|---|
| Weights | parameters x bytes per parameter |
| KV cache per user | 2 x layers x KV heads x head dim x 2 bytes x context |
| Headroom | GPU memory - weights - overhead |
| Max users | headroom / KV cache per user (rounded down) |
| Speed ceiling | memory bandwidth / weight bytes read per token |
| Cost per user-hour | GPU price per hour / max users |

These are estimates, not benchmarks. Real speed depends on the software you use, batching and the hardware. Mixture-of-experts models use all parameters for memory but only the active ones for speed. GPU specs come from vendor datasheets and prices are assumptions, not quotes.

## How far off is the speed ceiling?

The speed ceiling is a best case, and real runs land below it. Against the five published llama.cpp runs for Llama-2-7B Q4_0 (single user, 128-token context, `measured.js`):

| GPU | Measured / predicted |
|---|---|
| RTX 4090 | about 70% |
| RTX 3090 | about 64% |
| RTX 5090 | about 61% |
| A100 80GB | about 35% |
| H100 80GB SXM | about 30% |

So the ceiling reads about 1.5x too high on consumer cards and about 3x too high on datacenter cards in this case. This is five rows from one model and one quantization on four different llama.cpp builds, so no correction factor is applied yet. The calculator shows this gap on the Measured vs predicted card whenever a measured run exists.

## Contribute a measured run

Real measurements beat estimates. If you have run a model, open the calculator for that combination and press **Submit your measured run**. It opens a pre-filled GitHub issue. Fill in the engine, tokens per second and a source link. The format is described in `runs.schema.json`, with an example in `run-template.json`.

## Contributing

- **Report a measured speed**: use **Submit your measured run** in the calculator (see above).
- **Report a measured memory use**: run the model, read the peak with `nvidia-smi --query-gpu=memory.used --format=csv`, and open an issue with the model, quantization, GPU, context length, users and the reading. The memory math (weights plus KV cache plus overhead) is not yet validated against real readings, so these are especially welcome.
- **Fix or add a model**: add its Hugging Face repo to `scripts/gen-models.js` and run `node scripts/gen-models.js`. Do not edit `models.js` by hand. Models are only included when their `config.json` and parameter count can be read.
- Run `node test.js` before opening a pull request.

## Project layout

| File | Purpose |
|---|---|
| `index.html` | The whole app (page, styles, logic) |
| `fit.js` | The fit calculation |
| `scripts/gen-models.js` | Rebuilds `models.js` from each model's Hugging Face `config.json` (`node scripts/gen-models.js`) |
| `models.js`, `quants.js`, `gpus.js`, `memtypes.js`, `usecases.js` | Data tables |
| `measured.js` | Community-measured runs |
| `test.js` | Checks the calculation: `node test.js` |

## Tests

```bash
node test.js
```

## Limits

OMEN assumes an fp16 KV cache and ignores multi-GPU interconnect losses. The multi-GPU counts shown are minimums. Treat results as a starting point, then measure.

## License

MIT, see `LICENSE`.

## Built from the product spec

OMEN is a static page, so everything that needs no server is built:
- **Fit and memory**: four-level fit status, memory breakdown with the biggest component named, context-length table, multi-GPU pooling.
- **Evidence labels**: measured vs estimated on every verdict, plus a Benchmarks tab that sets each measured run against the prediction.
- **Compare**: all GPUs, all quantizations, all context lengths, with quality loss per quantization and CSV export.
- **Workload and cost**: cost per 1M output tokens, reply time, monthly cost (per use and always on), copies needed to keep up.
- **Recommendations**: hard limits (budget, speed, quality) first, then one pick per priority. No single blended score.
- **Model import**: any public Hugging Face model, pinned to its commit. Missing fields are rejected, never guessed.
- **Saved setups**, configuration IDs and shareable links. Saved setups live in your browser only.
- **Result checker** for benchmark JSON, with submission through a prefilled GitHub issue.

## The API (backend/)

The page's **Live API** tab calls this server (default `http://localhost:8000`, remembered in the browser): memory and speed estimates, recommendations with confidence labels and an optional high-confidence-only filter, benchmarks and recorded prices. CORS is open by default; set `OMEN_CORS=https://your.site` to restrict it.

A FastAPI service built from the product spec, sharing the page's data (`node scripts/export-catalog.js` regenerates `backend/app/catalog.json`).

```bash
cd backend
pip install -r requirements.txt
python -m pytest                       # 16 tests
OMEN_ADMIN_KEY=choose-a-secret uvicorn app.main:app --reload   # docs at http://localhost:8000/docs
```

| Endpoint | What it does |
|---|---|
| `GET /api/v1/models`, `/models/{id}`, `/gpus`, `/gpus/{id}`, `/quantizations` | Catalog lookup |
| `POST /api/v1/estimate/memory` | Weights, KV cache, overhead, headroom, four-level fit status |
| `POST /api/v1/estimate/performance` | Decode ceiling, TTFT, TPOT, end to end, calibration. `null` where data is missing |
| `POST /api/v1/workloads/simulate`, `/cost/estimate` | Capacity, $ per 1M tokens, monthly cost, cloud vs self-hosted, break-even |
| `GET /api/v1/prices`, `POST /api/v1/prices` | Cloud price catalog (AWS, Google Cloud, Azure, Oracle, CoreWeave, Lambda, RunPod, Vast.ai, Paperspace, DigitalOcean, Together AI). Reviewer-entered, each with a source URL and date, append-only. `/cost/estimate` takes `provider` and `pricing_model` to use it |
| `POST /api/v1/configurations/compare` | Side by side; warns when workloads differ |
| `POST /api/v1/recommendations` | Hard limits first, then one pick per priority. Each pick carries a `confidence` (0.9 measured at your concurrency, 0.7 measured at another, 0.5 calibrated on other models on that GPU, 0.3 first-principles). **High confidence means 0.75 or above**, so only a measured setup qualifies today. `min_confidence: 0.75` drops everything else. The score is a fixed ladder, not a probability |
| `POST /api/v1/benchmarks/submit`, `GET /benchmarks`, `POST /benchmarks/{id}/verify` | Consent-gated submission, background validation, duplicate and physics checks, reviewer verification |
| `POST /api/v1/keys`, `/workspaces/{project}/analyses`, `/workspaces/{project}/changes` | Hashed API keys with a rate limit, private saved analyses, re-run and report material changes |
| `GET /metrics` | Prometheus metrics |

`backend/omen_bench.py` measures any OpenAI-compatible server (vLLM, llama.cpp server) and writes a result file. It submits only with `--consent`. Publishing and name attribution are both opt-in: the submitter name is hidden unless you also pass `--attribute`. (If you already have a database file from before this change, delete it: the schema gained a column.)

The database is SQLite by default. `OMEN_DB=postgresql+psycopg://...` selects PostgreSQL, and `docker compose up --build` in `backend/` starts the API, PostgreSQL and Prometheus. The PostgreSQL path and the Docker files are untested: the Docker daemon was not running when this was written. Only SQLite is tested.

## Not built, and why

- **React + TypeScript + Tailwind frontend.** By decision the existing look, colours and fonts are kept, so the page stays plain HTML/JS with no build step. The API is reached through the new **Live API** tab; the calculator tab still runs its own copy of the maths in the browser.
- **Supabase/OAuth sign-in, Razorpay/Stripe billing, and email or in-app alert delivery.** They need accounts with outside services. API keys and a pull-based `/changes` report cover the same ground without them.
- **Celery and Redis.** FastAPI background tasks do the benchmark validation. A queue is the upgrade once volume needs it.
- **Cloud price data and a benchmark dataset beyond the 5 published runs.** The price catalog exists but ships empty: someone has to read each provider's pricing page and enter the price with its source. Until then prices are assumptions, not quotes. Benchmark data has to come from contributors.
- **Prefill estimates for GPUs without a well-known compute figure**, and calibration beyond one model on one engine.
