"""Leakage detection utilities."""


def detect_simple_leakage(columns: list[str], target: str) -> list[str]:
    return [c for c in columns if target in c.lower() and c != target]
