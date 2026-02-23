"""Ensemble architect."""


def blend_mean(predictions: list[list[float]]) -> list[float]:
    if not predictions:
        return []
    n = len(predictions)
    return [sum(vals) / n for vals in zip(*predictions)]
