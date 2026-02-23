from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ProblemSpec:
    target: str
    metric: str
    task_type: Literal["binary", "multiclass", "regression", "ranking"]
    id_columns: list[str] = field(default_factory=list)
    time_column: str | None = None
    group_columns: list[str] = field(default_factory=list)
    budget_minutes: int = 60
