"""Append-only structured experiment registry."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ExperimentRecord:
    experiment_id: str
    dataset_version: str
    universe: str
    target: str
    features: list[str]
    model: str
    hyperparameters: dict[str, Any]
    cost_assumptions: dict[str, float]
    validation_scheme: str
    metrics: dict[str, float]
    status: str = "completed"


def save_record(record: ExperimentRecord, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["git_commit"] = _git_commit()
    with (output_dir / f"{record.experiment_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    flat = {**{k: v for k, v in payload.items() if k not in {"metrics", "hyperparameters"}},
            **{f"metric_{k}": v for k, v in record.metrics.items()}}
    csv_path = output_dir / "results.csv"
    old = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    result = pd.concat([old, pd.DataFrame([flat])], ignore_index=True)
    result.to_csv(csv_path, index=False)
    try:
        result.to_parquet(output_dir / "results.parquet", index=False)
    except ImportError:
        pass


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
