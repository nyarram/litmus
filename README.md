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

- **Piece 1 (this):** repo bootstrap + deterministic eval core + CLI gate.
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
