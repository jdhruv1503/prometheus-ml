from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataProfiler:
    dataset_path: str | Path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    def _detect_temporal_columns(self, df: pd.DataFrame) -> list[str]:
        temporal_cols: list[str] = []
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                temporal_cols.append(col)
                continue
            if series.dtype == object:
                sample = series.dropna().astype(str).head(200)
                if sample.empty:
                    continue
                parsed = pd.to_datetime(sample, errors="coerce", utc=True)
                if parsed.notna().mean() > 0.8:
                    temporal_cols.append(col)
        return temporal_cols

    def profile(self, target: str | None = None) -> dict:
        df = self.load()
        n_rows, n_cols = df.shape

        dtypes = {c: str(t) for c, t in df.dtypes.items()}
        cardinality = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
        missingness = {c: float(df[c].isna().mean()) for c in df.columns}
        temporal_columns = self._detect_temporal_columns(df)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        summary: dict = {
            "shape": {"rows": int(n_rows), "columns": int(n_cols)},
            "dtypes": dtypes,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "cardinality": cardinality,
            "missingness": missingness,
            "temporal_columns": temporal_columns,
        }

        if target and target in df.columns:
            t = df[target]
            if pd.api.types.is_numeric_dtype(t):
                summary["target_distribution"] = {
                    "mean": float(t.mean()),
                    "std": float(t.std(ddof=0)) if len(t) > 1 else 0.0,
                    "min": float(t.min()),
                    "max": float(t.max()),
                }
            else:
                counts = t.value_counts(dropna=False, normalize=True)
                summary["target_distribution"] = {
                    str(k): float(v) for k, v in counts.to_dict().items()
                }

        return summary
