"""SQLite experiment metadata store placeholder."""

from dataclasses import dataclass


@dataclass
class ExperimentRecord:
    name: str
    score: float
    runtime: float
