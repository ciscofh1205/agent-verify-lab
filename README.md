# Runtime Verification of LLM-Agent Actions · Reproducible Pipeline

**Research Methods & Scientific Integrity in AI · Session 5 · UNMSM Deep Tech Doctorate · Dr. Loveleen Gaur**
Henry Francisco Sánchez Parra

A small, fully reproducible miniature of my doctoral thesis: a **runtime verifier** that blocks the actions of an agent **before they execute**, using constraints **mined from past insurance-claims event logs**. It tests hypotheses H1 and H3 of my research protocol:

- **H1:** the verifier (arm **A**) lowers the hard-constraint violation rate compared with an agent without verification (arm **B1**).
- **H3:** it does not lower the rate of claims the agent resolves correctly on its own.

Anyone who clones this repository, installs `requirements.txt` and runs the two commands below gets **exactly** the numbers in this README.

## Quick start: the stranger test

```bash
git clone https://github.com/ciscofh1205/agent-verify-lab.git
cd agent-verify-lab
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python tests/test_no_leakage.py                         # 1. leakage checks
PYTHONHASHSEED=0 python src/verify.py --seed 42         # 2. one run, one seed
PYTHONHASHSEED=0 python src/run_experiments.py          # 3. five seeds, logged in MLflow
```

`PYTHONHASHSEED=0` goes **before** `python` on purpose: Python reads it only at start-up. On Windows PowerShell use `$env:PYTHONHASHSEED=0; python ...`.

To browse all runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

## Expected output

**Leakage test**
```
PASS: 560 train / 240 test claims, 17 rules mined from train only
```

**One seed** (`src/verify.py --seed 42`); the same line must appear every time:
```
{"seed": 42, "data_sha256": "4a0eace501cb8c05e0fab9de9723f3f75a9439f9c395dba8f8a1d224ef745be5", "n_train_cases": 560, "n_test_cases": 240, "n_rules": 17, "tvrd_b1": 0.3375, "tvrd_a": 0.00417, "trac_b1": 0.36667, "trac_a": 0.59583}
```

**Five seeds** (`src/run_experiments.py`, also saved in `results_summary.csv`):

| Seed | TVRD B1 | TVRD A | TRAC B1 | TRAC A |
|---|---|---|---|---|
| 0 | 0.31667 | 0.00417 | 0.38750 | 0.58333 |
| 1 | 0.25417 | 0.00000 | 0.40833 | 0.59167 |
| 2 | 0.33333 | 0.00417 | 0.38750 | 0.60000 |
| 3 | 0.27917 | 0.00833 | 0.39583 | 0.59167 |
| 42 | 0.33750 | 0.00417 | 0.36667 | 0.59583 |

## Final result (5 seeds, mean ± SD)

| Metric | B1: no verification | A: runtime verifier |
|---|---|---|
| **TVRD**: hard-constraint violation rate (lower is better) | **0.3042 ± 0.0362** | **0.0042 ± 0.0029** |
| **TRAC**: correct autonomous resolution rate (higher is better) | **0.3892 ± 0.0152** | **0.5925 ± 0.0062** |

TVRD relative reduction, A vs B1: **98.6%** (protocol target for H1: ≥ 50%). SD is the sample standard deviation (ddof = 1) across the 5 seeds. We report the mean ± SD, never the best seed.

TVRD for A is **not zero**, and this is expected. The expert rule "claims above 10,000 need an ExpertReview before approval" depends on the claim **amount**, and unconditional `precedence` rules cannot express it. This is a known coverage limit of mined rules, and the thesis architecture must address it.

⚠️ These numbers come from **synthetic data with known rules**. They show that the pipeline works and can be reproduced. They are **not** evidence for the thesis hypotheses.

## How the pipeline works

```
data/claims_log.csv ──► temporal split ──► RuntimeVerifier.fit(train) ──► evaluate on test: B1 vs A
   (SHA-256 checked)     70% old / 30% new     mines precedence(a, b) rules      5 seeds ──► MLflow ──► mean ± SD
```

| Step | File | What it does |
|---|---|---|
| Data | `src/make_synthetic_log.py` | Builds 800 synthetic claims (fixed seed 2026), each a sequence of activities: Register → CheckCoverage → Assess → [ExpertReview] → Approve → Pay → Close, or a Reject path. 3% of historical claims break a rule, as real logs do. |
| Pipeline | `src/verify.py` | Splits by time, mines rules from training claims, then replays each test claim with an agent. In B1 every action executes; in A the verifier checks each action first and blocks it if it breaks a mined rule. |
| Experiments | `src/run_experiments.py` | Runs 5 seeds, logs parameters and metrics of every run in MLflow, writes `results_summary.csv` and prints mean ± SD. |
| Checks | `tests/test_no_leakage.py` | Proves the split is clean and that the mined rules do not change when the test claims are scrambled. |

## Data leakage control

Mining the rules on the whole log and then testing the verifier on that same log is data leakage: it is the process-mining version of scaling before splitting. This pipeline prevents it in three ways.

1. **Split first, in time order.** Claims are sorted by opening time. The oldest 70% (560 claims) are used to learn rules, and the newest 30% (240 claims) only to test. Two `assert` statements stop the run if any claim falls on both sides, or if a training claim is newer than a test claim.
2. **Fit on train only.** `RuntimeVerifier.fit()` receives only the training claims, just as a scaler inside an sklearn `Pipeline` only sees training folds.
3. **The verifier is not the judge.** Violations are scored against a separate expert rule catalogue (`HARD_RULES`) that the verifier never sees, so the evaluation is not circular.

`tests/test_no_leakage.py` checks points 1 and 2 automatically.

## Data

**Synthetic proxy, not real insurer data.** Real claims logs are confidential under Peruvian Law 29733 and cannot be published. `src/make_synthetic_log.py` rebuilds the file on any machine.

| | |
|---|---|
| File | `data/claims_log.csv` (4,605 events, 800 claims, 2025) |
| SHA-256 | `4a0eace501cb8c05e0fab9de9723f3f75a9439f9c395dba8f8a1d224ef745be5` |
| Columns | `case_id`, `activity`, `amount`, `timestamp` |

Check it with `sha256sum data/claims_log.csv` (macOS: `shasum -a 256`). If the hash differs, you are not using the same data and the numbers will not match.

## Sources of randomness and how each is fixed

| Source | How it is fixed |
|---|---|
| Python `random` | `random.seed(seed)` |
| NumPy | `np.random.seed(seed)` plus an explicit `np.random.default_rng(seed)` per arm |
| Hash ordering | `PYTHONHASHSEED=0` set before Python starts |
| The agent | Stub policy logged in every run: `agent_id=stub-policy-v1`, `error_rate=0.15`, `prompt_sha256`. With a real LLM, also log the **dated** model version (never an alias) and the temperature. |
| The data | Fixed generator seed (2026) and the SHA-256 above |
| Library versions | Exact pins in `requirements.txt` |

B1 and A use the same seed, so the agent makes the same first mistakes in both arms. The only difference between them is the verifier.

## What MLflow records for every run

- **Parameters:** `seed`, `git_commit`, `data_sha256`, `agent_id`, `error_rate`, `prompt_sha256`, `split`, `rule_template`, `min_conf`.
- **Metrics:** `tvrd_b1`, `tvrd_a`, `trac_b1`, `trac_a`, `n_rules`.

The database `mlflow.db` is generated locally and kept out of Git; `results_summary.csv` is the versioned summary.

## The three R's

| | Who | With what | Guaranteed by this repository? |
|---|---|---|---|
| **Repeatability** | Me | My files | Yes: the same seed gives an identical line |
| **Reproducibility** | Anyone | My files | Yes: pinned versions, data hash, seeds and this README |
| **Replicability** | Another team | Their own data | No: it needs another insurer's logs. The thesis plans this as shadow-mode evaluation |

## Repository structure

```
agent-verify-lab/
├── README.md
├── requirements.txt
├── .gitignore                 # excludes mlflow.db, mlruns/, caches
├── data/
│   └── claims_log.csv         # synthetic event log (versioned)
├── src/
│   ├── __init__.py
│   ├── make_synthetic_log.py  # data generator (seed 2026)
│   ├── verify.py              # leakage-free pipeline: split -> fit -> evaluate
│   └── run_experiments.py     # 5 seeds -> MLflow -> mean ± SD
├── tests/
│   └── test_no_leakage.py
└── results_summary.csv        # per-seed results
```

## Environment

Tested with Python 3.13 on Linux, with the exact package versions in `requirements.txt` (numpy 2.5.3, pandas 3.0.5, mlflow 3.17.0).

## Next steps toward the thesis

1. Replace the synthetic log with a public BPI Challenge log, read with `pm4py`, keeping the same temporal split.
2. Replace the stub with a real LLM agent and log its dated model version, temperature and prompt hash.
3. Add the paired statistical tests of the protocol (McNemar, Wilcoxon) and more seeds.
