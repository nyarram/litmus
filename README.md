# litmus

Offline regression evals for agents and LLM pipelines. Run a frozen golden
dataset through the thing you ship, score every output with pluggable judges,
and gate deploys on the numbers — before a user ever sees a regression.

## Quickstart

```bash
pip install -e .
litmus run --suite datasets/examples/basic.jsonl \
           --target examples/echo_target.py:echo \
           --judges exact_match,containment \
           --threshold 0.7
```

Exit code is the CI gate: `0` on pass, `1` on fail. JSON and Markdown reports
land in `reports/`.

### LLM-as-judge

```bash
# vet the judge against human labels first
litmus calibrate --labeled datasets/examples/labeled.jsonl \
                 --client examples/stub_client.py:stub

# then use it in the gate
litmus run --suite datasets/examples/basic.jsonl \
           --target examples/echo_target.py:echo \
           --judges exact_match,llm_judge \
           --client examples/stub_client.py:stub \
           --threshold 0.7
```

The judge takes a **client callable**, not a provider SDK: any
`(prompt: str) -> str` function. Model choice, temperature, and auth live in
your client factory (see `examples/stub_client.py` for the pattern); the
judge owns only the versioned prompt template (`litmus/prompts/judge_v1.txt`),
verdict parsing (fences and chatter tolerated, one retry, then a flagged
`0.0` instead of a crash), and score normalization. `litmus calibrate`
reports MAE, bias, Pearson correlation, and a predicted-vs-human calibration
curve.

### Tracing

```bash
pip install "litmus[tracing]"
litmus run --suite datasets/examples/basic.jsonl \
           --target examples/echo_target.py:echo \
           --trace
```

Every case runs inside a `litmus.eval.case` span with `litmus.eval.target`
and `litmus.eval.judge` children, following GenAI semantic conventions
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`, …). Spans print
to stdout by default — no collector needed; pass `--trace-otlp-endpoint`
to ship them to an OTLP backend instead. Prompt/completion payloads are
never captured unless you opt in, so traces are safe to export. Tracing is
additive: omit `--trace` and the run is byte-identical.

## How it works

1. **Golden dataset** (`litmus/dataset.py`) — a JSONL file of frozen
   `input`/`reference` pairs with optional `tags` and `metadata`. Loaded with
   strict validation: every row needs `id`, `input`, `reference`; duplicate
   ids and malformed rows fail fast.
2. **Judges** (`litmus/judges.py`) — anything implementing
   `score(prediction, reference) -> Score`. Piece 1 ships two deterministic
   judges: `exact_match` and `containment`. `LLMJudge` is declared as an
   interface; the calibrated model-graded implementation is piece 2.
3. **Runner** (`litmus/runner.py`) — executes a target
   (`Callable[[str], str]`, usually a thin wrapper around your pipeline) over
   every case, aggregates per-judge mean scores, and produces an `EvalReport`
   with pass/fail against a threshold.

## Roadmap

- **Piece 1:** repo bootstrap + deterministic eval core + CLI gate.
- **Piece 2:** LLM-as-judge (versioned prompt templates,
  provider-agnostic client, robust verdict parsing) + `litmus calibrate`
  (MAE, bias, Pearson, calibration curve vs human labels).
- **Piece 3 (this):** OpenTelemetry tracing with GenAI semantic conventions
  (`litmus/tracing.py`, `--trace` on `litmus run`, console exporter by
  default, content capture opt-in).
- **Piece 2:** calibrated LLM-as-judge (prompt templates, calibration against
  human labels, agreement metrics).
- **Piece 3:** OpenTelemetry tracing for LLM/tool calls (GenAI semconv).
- **Piece 4:** synthetic persona-based traffic generator — an explicit
  stand-in for organic traffic, stated honestly here and in the code.
- **Piece 5:** score-over-time dashboard, plus dogfooding against real
  pipeline outputs.

## Design notes

- The runner is synchronous and single-process on purpose: determinism and
  debuggability first, throughput later.
- Judges never see `metadata` or `tags` — scoring inputs are only
  `(prediction, reference)`, so evals can't accidentally depend on labels.
- Thresholds are per-run, not per-dataset, so the same suite can gate a
  strict release and a lenient experiment.
