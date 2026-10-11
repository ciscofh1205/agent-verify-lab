"""Run the pipeline with 5 seeds, log every run in MLflow, report mean ± SD.

Run:  PYTHONHASHSEED=0 python src/run_experiments.py
Out:  mlflow.db (all runs, kept out of Git) + results_summary.csv (in Git)
"""
import subprocess
import sys
from pathlib import Path

import mlflow
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.verify import AGENT, MIN_CONF, TRAIN_FRAC, main  # noqa: E402

SEEDS = [0, 1, 2, 3, 42]                       # 5 seeds: one seed is luck, five give a spread
METRICS = ["tvrd_b1", "tvrd_a", "trac_b1", "trac_a"]


def git_commit():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip() or "no-git"


def run():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("agent-verify-5seeds")
    commit, rows = git_commit(), []
    for seed in SEEDS:
        with mlflow.start_run(run_name=f"seed-{seed}"):
            r = main(seed)
            mlflow.log_params({**AGENT, "seed": seed, "git_commit": commit,
                               "data_sha256": r["data_sha256"], "split": f"temporal {TRAIN_FRAC:.0%} train",
                               "rule_template": "precedence", "min_conf": MIN_CONF})
            mlflow.log_metrics({k: r[k] for k in METRICS + ["n_rules"]})
            rows.append(r)

    table = pd.DataFrame(rows)[["seed"] + METRICS]
    table.to_csv("results_summary.csv", index=False)
    print(table.to_string(index=False))

    print("\nFinal metrics over 5 seeds (mean ± SD, sample SD ddof=1):")
    for m in METRICS:
        print(f"  {m:8s} {table[m].mean():.4f} ± {table[m].std(ddof=1):.4f}")
    reduction = 1 - table["tvrd_a"].mean() / table["tvrd_b1"].mean()
    print(f"  TVRD relative reduction (A vs B1): {reduction:.1%}")


if __name__ == "__main__":
    run()
