from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any


@dataclass
class Experiment:
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    expected_gain: float = 0.005
    expected_runtime: float = 5.0
    overfit_risk: float = 0.2
    trials: int = 0
    success_count: int = 0

    def score(self) -> float:
        runtime = max(self.expected_runtime, 1e-6)
        risk_term = max(0.0, min(1.0, 1.0 - self.overfit_risk))
        return (self.expected_gain / runtime) * risk_term


class TEPE:
    """Time-Aware Experiment Policy Engine.

    Prioritizes experiments by expected value per runtime with overfit-risk penalty.
    Posterior success probability uses a Beta prior updated from experiment outcomes.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha_prior = alpha
        self.beta_prior = beta
        self._heap: list[tuple[float, int, str]] = []
        self._counter = 0
        self.experiments: dict[str, Experiment] = {}

    def _posterior_mean(self, exp: Experiment) -> float:
        a = self.alpha_prior + exp.success_count
        b = self.beta_prior + (exp.trials - exp.success_count)
        return a / (a + b)

    def _effective_score(self, exp: Experiment) -> float:
        base = exp.score()
        posterior = self._posterior_mean(exp)
        exploration = 1.0 / (1.0 + exp.trials)
        return base * (0.7 * posterior + 0.3 * exploration)

    def _push(self, exp: Experiment) -> None:
        self._counter += 1
        priority = -self._effective_score(exp)
        heapq.heappush(self._heap, (priority, self._counter, exp.name))

    def add_hypothesis(
        self,
        name: str,
        expected_gain: float,
        expected_runtime: float,
        overfit_risk: float,
        config: dict[str, Any] | None = None,
    ) -> None:
        exp = Experiment(
            name=name,
            config=config or {},
            expected_gain=expected_gain,
            expected_runtime=expected_runtime,
            overfit_risk=overfit_risk,
        )
        self.experiments[name] = exp
        self._push(exp)

    def get_next(self) -> Experiment | None:
        while self._heap:
            _, _, name = heapq.heappop(self._heap)
            exp = self.experiments.get(name)
            if exp is None:
                continue
            return exp
        return None

    def record_result(
        self,
        name: str,
        gain_observed: float,
        runtime_observed: float,
        overfit_flag: bool = False,
    ) -> None:
        if name not in self.experiments:
            raise KeyError(f"Unknown experiment: {name}")

        exp = self.experiments[name]
        exp.trials += 1

        success = gain_observed > 0
        if success:
            exp.success_count += 1

        # Exponential moving averages for stable adaptation
        lr = 0.35
        exp.expected_gain = (1 - lr) * exp.expected_gain + lr * max(gain_observed, 0.0)
        exp.expected_runtime = (1 - lr) * exp.expected_runtime + lr * max(runtime_observed, 1e-3)

        if overfit_flag:
            exp.overfit_risk = min(0.99, exp.overfit_risk + 0.1)
        else:
            exp.overfit_risk = max(0.01, exp.overfit_risk - 0.03 if success else exp.overfit_risk + 0.02)

        self._push(exp)

    def leaderboard(self, top_k: int = 10) -> list[dict[str, float | str | int]]:
        rows = []
        for exp in self.experiments.values():
            rows.append(
                {
                    "name": exp.name,
                    "score": self._effective_score(exp),
                    "expected_gain": exp.expected_gain,
                    "expected_runtime": exp.expected_runtime,
                    "overfit_risk": exp.overfit_risk,
                    "trials": exp.trials,
                }
            )
        rows.sort(key=lambda r: r["score"], reverse=True)
        return rows[:top_k]
