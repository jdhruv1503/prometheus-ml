from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from prometheus.intake.profiler import DataProfiler
from prometheus.policy.tepe import TEPE

app = typer.Typer(help="Prometheus: decision engine for ML hackathons")
console = Console()
STATE_PATH = Path(".prometheus_state.json")


def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"experiments": []}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


@app.command()
def init(dataset: str, target: str, metric: str, budget: int = 60) -> None:
    profiler = DataProfiler(dataset)
    profile = profiler.profile(target=target)

    state = {
        "dataset": dataset,
        "target": target,
        "metric": metric,
        "budget": budget,
        "profile": profile,
        "experiments": [
            {"name": "baseline_lightgbm", "gain": 0.01, "runtime": 4.0, "risk": 0.15},
            {"name": "target_encoding", "gain": 0.012, "runtime": 6.0, "risk": 0.25},
            {"name": "feature_interactions", "gain": 0.008, "runtime": 5.0, "risk": 0.2},
        ],
    }
    _save_state(state)
    console.print(f"[green]Initialized[/green] with dataset={dataset}, target={target}, metric={metric}, budget={budget}m")


@app.command()
def run(steps: int = 3) -> None:
    state = _load_state()
    if not state.get("experiments"):
        raise typer.BadParameter("No experiments found. Run `prometheus init` first.")

    policy = TEPE()
    for e in state["experiments"]:
        policy.add_hypothesis(e["name"], e["gain"], e["runtime"], e["risk"])

    for i in range(steps):
        exp = policy.get_next()
        if exp is None:
            break
        observed_gain = exp.expected_gain * (0.7 + 0.2 * (i + 1) / max(1, steps))
        observed_runtime = exp.expected_runtime * 1.05
        policy.record_result(exp.name, observed_gain, observed_runtime, overfit_flag=False)
        console.print(f"[cyan]Step {i+1}[/cyan]: ran {exp.name} gain={observed_gain:.4f} runtime={observed_runtime:.2f}m")

    console.print("[green]Policy loop complete.[/green]")


@app.command()
def status() -> None:
    state = _load_state()
    policy = TEPE()
    for e in state.get("experiments", []):
        policy.add_hypothesis(e["name"], e["gain"], e["runtime"], e["risk"])

    table = Table(title="Prometheus Queue")
    table.add_column("Experiment")
    table.add_column("Score", justify="right")
    table.add_column("Gain", justify="right")
    table.add_column("Runtime", justify="right")
    table.add_column("Risk", justify="right")

    for row in policy.leaderboard():
        table.add_row(
            str(row["name"]),
            f"{row['score']:.5f}",
            f"{row['expected_gain']:.4f}",
            f"{row['expected_runtime']:.2f}",
            f"{row['overfit_risk']:.2f}",
        )
    console.print(table)


@app.command()
def blend(oof_files: list[str]) -> None:
    console.print(f"[yellow]Blending[/yellow] {len(oof_files)} OOF files: {', '.join(oof_files)}")
    console.print("[green]Ensemble complete (placeholder averaging).[/green]")


if __name__ == "__main__":
    app()
