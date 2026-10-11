"""Build a synthetic insurance-claims event log: a public proxy for confidential data.

Why synthetic? Real claims logs are confidential (Peruvian Law 29733) and can never be
published. This script rebuilds the same public proxy on any machine.

Run:  python src/make_synthetic_log.py      ->  data/claims_log.csv
"""
import numpy as np
import pandas as pd

GEN_SEED = 2026   # Seed of the DATA, not of the experiment: it never changes.
N_CASES = 800     # 800 claims opened during 2025, one roughly every 11 hours.


def build_case(rng, i, start):
    """One claim = one ordered list of activities (a 'trace')."""
    case_id = f"C{i:04d}"
    opened = start + pd.Timedelta(hours=int(i * 10.9))
    amount = round(float(rng.lognormal(8.5, 0.9)), 2)      # claim amount in soles

    acts = ["Register", "CheckCoverage"]
    if rng.random() < 0.15:                                 # 15% not covered -> rejected
        acts += ["Reject"]
    else:
        acts += ["Assess"]
        if amount > 10000:                                  # business rule: big claims need an expert
            acts += ["ExpertReview"]
        acts += ["Approve", "Pay"] if rng.random() < 0.8 else ["Reject"]

    # 3% historical deviations (Pay before Approve), as in real logs. The miner must tolerate them.
    if "Pay" in acts and rng.random() < 0.03:
        k = acts.index("Pay")
        acts[k - 1], acts[k] = acts[k], acts[k - 1]
    acts += ["Close"]

    return [{"case_id": case_id, "activity": a, "amount": amount,
             "timestamp": (opened + pd.Timedelta(hours=2 * j)).isoformat()}
            for j, a in enumerate(acts)]


def main(path="data/claims_log.csv"):
    rng = np.random.default_rng(GEN_SEED)
    start = pd.Timestamp("2025-01-01")
    rows = [row for i in range(N_CASES) for row in build_case(rng, i, start)]
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    main()
