#!/usr/bin/env python3
"""Generate the committed synthetic portfolio demo (no network, no real data).

The output is a software demonstration fixture. Group-wise outcome
differences are inserted so the fairness UI is informative; they are not
estimates of real-world demographic or clinical relationships.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 400

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV_PATH = REPO_ROOT / "bias_audit_tool" / "sample_data" / "demo.csv"
DEMO_BENCHMARK_PATH = (
    REPO_ROOT / "bias_audit_tool" / "sample_data" / "demo_benchmark.json"
)

COLUMNS = [
    "feature_a",
    "feature_b",
    "feature_c",
    "age_band",
    "demo_group",
    "outcome",
]

# Equal shares are a demonstration benchmark, not a population prevalence.
SYNTHETIC_BENCHMARK = {"Group A": 0.5, "Group B": 0.5}


def build_demo_dataframe(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    """Return a deterministic synthetic binary-classification demo frame."""
    rng = np.random.default_rng(seed)

    demo_group = np.where(rng.random(n_rows) < 0.42, "Group A", "Group B")

    feature_a = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    feature_b = rng.normal(loc=0.2, scale=1.2, size=n_rows)
    feature_c = rng.choice(
        np.array(["low", "medium", "high"]),
        size=n_rows,
        p=np.array([0.30, 0.45, 0.25]),
    )
    age_band = rng.choice(
        np.array(["band_1", "band_2", "band_3"]),
        size=n_rows,
        p=np.array([0.34, 0.41, 0.25]),
    )

    feature_c_score = np.array(
        [{"low": -0.7, "medium": 0.0, "high": 0.8}[value] for value in feature_c]
    )
    # Intentional demo-only intercept shift so group-wise selection rates
    # differ after a classifier is trained. Not an empirical finding.
    group_shift = np.where(demo_group == "Group A", 0.85, -0.40)
    logit = (
        -0.15 + 1.35 * feature_a + 0.45 * feature_b + feature_c_score + group_shift
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    outcome = (rng.random(n_rows) < probability).astype(int)

    missing_mask = rng.random(n_rows) < 0.08
    feature_b = feature_b.astype(float)
    feature_b[missing_mask] = np.nan

    frame = pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "feature_c": feature_c,
            "age_band": age_band,
            "demo_group": demo_group,
            "outcome": outcome,
        }
    )
    return frame[COLUMNS]


def dataframe_to_csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, lineterminator="\n")


def write_demo_artifacts(
    csv_path: Path = DEMO_CSV_PATH,
    benchmark_path: Path = DEMO_BENCHMARK_PATH,
    seed: int = SEED,
    n_rows: int = N_ROWS,
) -> pd.DataFrame:
    df = build_demo_dataframe(seed=seed, n_rows=n_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(dataframe_to_csv_text(df), encoding="utf-8")
    benchmark_path.write_text(
        json.dumps(SYNTHETIC_BENCHMARK, indent=2) + "\n",
        encoding="utf-8",
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEMO_CSV_PATH,
        help="Destination for the synthetic demo CSV.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=DEMO_BENCHMARK_PATH,
        help="Destination for the synthetic demonstration benchmark JSON.",
    )
    args = parser.parse_args()
    write_demo_artifacts(csv_path=args.csv_path, benchmark_path=args.benchmark_path)


if __name__ == "__main__":
    main()
