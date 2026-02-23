"""Hypothesis generator."""


def default_hypotheses() -> list[dict]:
    return [
        {"name": "baseline_lightgbm", "kind": "model"},
        {"name": "target_encoding", "kind": "feature"},
    ]
