# Litmus

A free, open-source evaluation and observability harness for LLM pipelines and agents.

Litmus answers the question every team shipping AI eventually asks: **"did my last change make the system better or worse?"** It runs versioned datasets against your pipeline, scores the outputs with deterministic checks and calibrated LLM judges, diffs runs against a baseline, and fails CI when quality regresses — then traces every score back to the exact model call that produced it.

## Why this exists

Most eval tooling is either a SaaS with a per-seat bill or a notebook nobody re-runs. Litmus is built to live in your repo and your CI:

- **Offline regression evals** on a curated, versioned golden dataset — the highest-signal eval loop.
- **Deterministic scorers** (exact match, contains, regex, JSON Schema) plus **LLM-as-judge** with pluggable providers.
- **Judge calibration**: measure your judge's agreement with human labels before you trust it.
- **CI quality gates**: `litmus run` exits non-zero when scores regress. Prompt changes become reviewable.
- **OpenTelemetry tracing** with GenAI semantic conventions, so scores link to traces.
- **Free to use, forever**: no paid APIs required. Judge inference runs on free tiers (Groq) or fully local models (Ollama). The dashboard and trace store are self-hosted.

## Quickstart

```bash
pip install -e .                        # core only
pip install -e ".[tracing]"             # + OpenTelemetry tracing (optional)
pytest                                    # 41 tests, no API keys needed

# Run the example eval (AstroDigest-style news scoring, stub system — no key needed)
PYTHONPATH=src python -m litmus.cli run \
  --dataset datasets/astrodigest_golden_seed_v1.jsonl \
  --system examples.score_and_summarize:score_stub \
  --scorer examples.score_and_summarize:scorers_basic \
  --name demo-run --report-md report.md
```

With a free Groq key you get the full run, including the LLM judge and the traced Groq system:

```bash
export GROQ_API_KEY=...   # free tier at groq.com
PYTHONPATH=src python -m litmus.cli run \
  --dataset datasets/astrodigest_golden_seed_v1.jsonl \
  --system examples.score_and_summarize:score_with_groq \
  --scorer examples.score_and_summarize:scorers_full \
  --name groq-run --trace \
  --gate-scorer newsworthy_match --fail-below 0.8
```

## Synthetic traffic (`litmus synth`)

Generate persona-based load against any async system, score the responses,
and gate on quality — every span labeled `litmus.synthetic: true` so
simulated traffic is trivially filterable and never confused with organic
traffic:

```bash
PYTHONPATH=src python -m litmus.cli synth \
  --personas examples/synth_personas.json \
  --provider mymod:MyProvider \          # async ModelProvider (writes the requests)
  --system mymod:my_system \             # async callable: request text -> response
  --n 50 --seed 0 --concurrency 8 --fail-below 0.7
```

Without `--scorer`, a reference-free LLM quality judge scores each response.
Reports include error rate, latency p50/p95, per-persona breakdowns, and the
standard scorer table.

## CI: evals on every PR

`.github/workflows/eval.yml` runs the full pipeline on every pull request and
push to main:

1. **test** job: pytest + ruff.
2. **eval** job: installs with no API keys, runs the astrodigest seed eval
   against the stub system (hermetic — deterministic scorers only, so it's
   free and stable), then `litmus diff` compares the candidate report against
   the committed `reports/baseline.json`. The diff lands in the job summary;
   any regression fails the check, and the candidate report is uploaded as an
   artifact for debugging.

Exit codes for `litmus diff`: `0` = no regression, `1` = regression detected,
`2` = bad inputs (e.g. missing report file).

**Refreshing the baseline** (when a quality change is intentional, on main):

```bash
PYTHONPATH=src python -m litmus.cli run \
  --dataset datasets/astrodigest_golden_seed_v1.jsonl \
  --system examples.score_and_summarize:score_stub \
  --scorer examples.score_and_summarize:scorers_basic \
  --name baseline --report-json reports/baseline.json
git add reports/baseline.json && git commit -m "Refresh eval baseline"
```

## Dashboard: scores over time

```bash
pip install 'litmus[dashboard]'
litmus dashboard --reports-dir reports
# open http://127.0.0.1:8000
```

The dashboard reads report JSON files from a directory (the filesystem is the
database — no database to operate) and renders:

- **Runs** — every eval run with per-scorer means, error counts, dataset versions.
- **Run detail** — per-case scores, latency, errors, and each `trace_id` linked
  into Jaeger for drill-down (with `--jaeger-url`).
- **Trends** — per-scorer mean and pass-rate over time (Chart.js).

`GET /api/runs` also exposes the run list as JSON for automation.

## Tracing pipeline: collector → Jaeger

`init_tracing` already spoke OTLP; the CLI now wires it up:

```bash
# terminal 1: local observability stack
cd deploy && docker compose up   # Jaeger UI at :16686, collector at :4317/:4318

# terminal 2: export spans instead of printing them
LITMUS_OTLP_ENDPOINT=http://localhost:4318 litmus run ... --report-json reports/x.json
# or: litmus run ... --otlp-endpoint http://localhost:4318
```

`--otlp-endpoint` (or the env var) overrides `--trace`. The same compose file
is what a server deploy reuses — see the roadmap.

## Second consumer: MCP demo agent

`examples/mcp_agent/` is a tiny real agent built on the official MCP SDK —
proof the harness generalizes beyond one pipeline:

- `server.py` — MCP server (stdio) with three deterministic tools:
  `calculator`, `weather_lookup`, `summarize`.
- `agent.py` — a minimal ReAct loop. The model emits fenced
  ` ```tool {"name": ..., "arguments": {...}} ` blocks; the agent executes
  them via the MCP client and feeds results back. Each tool call gets a
  litmus trace span.
- `tasks.jsonl` — 5 demo tasks with expected answers and expected tool
  sequences. `scorers.py` — a `ToolSequenceScorer` that grades whether the
  agent used the right tools (via a per-case tool-call registry, since
  scorers only see `(case, output)`).

```bash
pip install 'litmus[agent]'
litmus run --dataset examples/mcp_agent/tasks.jsonl \
  --system examples.mcp_agent.agent:agent_system \
  --scorer examples.mcp_agent.scorers:tool_scorer \
  --scorer litmus.scorers.exact:ExactMatchScorer
```

No API keys needed: `LITMUS_AGENT_PROVIDER` defaults to `stub` (scripted
responses for the demo tasks). Set it to `groq` (needs `GROQ_API_KEY`) or
`ollama` (needs a local Ollama server) to drive the same agent with a real
model.

## Tracing is optional and additive

The OTel SDK is an **optional** extra (`pip install ".[tracing]"`). Importing
`litmus` and running evals never requires it — every tracing helper is a
no-op until `init_tracing()` is called. Prompt/completion capture is **opt-in**
(`capture_content=True`) and recorded as span *events*, never attributes, so
payloads aren't indexed by your trace backend. The `litmus run` / `litmus
synth` CLIs only initialize tracing with `--trace`.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│  datasets/  │────▶│   runners    │────▶│    scorers     │
│ versioned   │     │ concurrency, │     │ exact / regex /│
│ .jsonl      │     │ timeouts,    │     │ jsonschema /   │
└─────────────┘     │ retries      │     │ llm-judge      │
                    └──────────────┘     └────────────────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐     ┌────────────────┐
                    │   tracing    │     │     report     │
                    │ OTel GenAI   │     │ md + json,     │
                    │ semconv      │     │ baseline diff  │
                    └──────────────┘     └────────────────┘
```

- `src/litmus/datasets.py` — `Case` / `Dataset` models, JSONL load/save.
- `src/litmus/runners.py` — async runner: bounded concurrency, per-case timeout, retries with backoff, per-case OTel spans.
- `src/litmus/scorers/` — `Score` protocol implementations; judges take any `ModelProvider` (`GroqProvider`, `OllamaProvider`).
- `src/litmus/calibration.py` — judge-vs-human agreement reporting.
- `src/litmus/report.py` — aggregate stats, markdown/JSON rendering, `compare_reports` for baseline diffs.
- `src/litmus/synth.py` — synthetic persona traffic: seeded generation, load run
  through the standard runner, scoring, and synth reports.
- `src/litmus/tracing.py` — OTel setup + GenAI-semconv span helpers. Optional
  extra; no-op safe; content capture opt-in.
- `src/litmus/cli.py` — `litmus run` with `--fail-below` / `--gate-scorer` CI gating.

## Honest scope notes

- The seed dataset (`datasets/astrodigest_golden_seed_v1.jsonl`) is a hand-built starter modeled on a real pipeline's shape. The path to a real golden set is one SQL export from the pipeline's database — documented in the roadmap, not faked here.
- Synthetic traffic (`litmus synth`) is **simulated load, not real user traffic**,
  and is labeled `litmus.synthetic: true` on every span and in every report.
  Offline evals are the primary workflow.

## Roadmap

- **M1** ✅ Core engine, CLI, scorers, calibration, tracing, first real eval run.
- **Synthetic traffic** ✅ Persona-based generation + load runner (`litmus synth`),
  reference-free quality judge, `litmus.synthetic` span labeling, CI gating.
- **M2** ✅ CI gating — GitHub Actions runs evals on every PR, `litmus diff`
  fails the check on regression vs a committed baseline.
- **M3** ✅ Score-over-time dashboard (`litmus dashboard`, FastAPI) + OTel
  collector pipeline (compose: Jaeger + collector, `--otlp-endpoint`).
- **M4** ✅ MCP demo agent — a tiny ReAct agent on the official MCP SDK
  (`examples/mcp_agent/`) evaluated by the same harness: `tools_used` +
  `exact_match` scorers, stub provider for keyless runs.
- **M5** — Docker Compose deployment to a Hetzner VPS alongside the dogfood pipeline.

## License

MIT — free for commercial and personal use.
