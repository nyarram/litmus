"""Deterministic stub LLM client for offline demos and tests.

Returns a model-style verdict without any network calls: score 1.0 when the
prediction section of the prompt matches the reference section, else 0.0.
It exists so the judge and calibration machinery can be exercised end to end
with zero API keys.

Production pattern — provider settings (model, temperature, auth) live in the
client factory as a closure, never in the judge:

    def make_client(model: str = "acme-large", temperature: float = 0.0):
        import acme
        client = acme.Client()
        def call(prompt: str) -> str:
            return client.complete(model=model, prompt=prompt,
                                   temperature=temperature).text
        return call
"""
import json


def _section(prompt: str, header: str) -> str:
    """Grab the text between a section header and the next blank line."""
    lines = prompt.splitlines()
    try:
        i = next(n for n, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        return ""
    body = []
    for line in lines[i + 1 :]:
        if not line.strip():
            break
        body.append(line)
    return "\n".join(body).strip()


def stub(prompt: str) -> str:
    reference = _section(prompt, "Reference answer:")
    prediction = _section(prompt, "Model prediction:")
    score = 1.0 if prediction and prediction == reference else 0.0
    return json.dumps(
        {"score": score, "rationale": "stub verdict (offline)", "flags": []}
    )
