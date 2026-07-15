"""
check_regression.py — Gap 3: regression gate for the Novus Support Agent.

Project B previously had NO regression gate — baseline_scores.json existed but
nothing enforced it. This compares the latest eval_results.json against the saved
baseline and fails (exit 1) if any tracked metric drops by more than THRESHOLD.

Run in CI after any change to the agent / classifier / guardrails / prompts:
    python scripts/check_regression.py
    python scripts/check_regression.py --threshold 0.03

Metrics (all 0-1; faithfulness/correctness are stored normalized in the scorecard):
    classification_accuracy, routing_accuracy, faithfulness, correctness,
    refusal_pass_rate (Gap 3 — fraction of should-not-answer queries refused).

For routing, a DROP in correctness is bad; missed_escalation_rate is also tracked
inverted (a drop in catch-rate is a regression). refusal_pass_rate is skipped
automatically if a baseline predates the failure-class dataset.
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "eval_results.json"
BASELINE_PATH = Path(__file__).parent / "baseline_scores.json"

THRESHOLD = 0.05  # 5 percentage points — a meaningful signal, not noise

# Metrics where HIGHER is better; a drop beyond threshold is a regression.
METRICS = [
    "classification_accuracy",
    "routing_accuracy",
    "faithfulness",
    "correctness",
    "refusal_pass_rate",
]


def load_scorecard(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # eval_results.json wraps scorecard; baseline_scores.json has it under "scorecard" too
    if "scorecard" in data:
        return data["scorecard"]
    # compare-mode results store agent under "agent"
    if "agent" in data and "scorecard" in data["agent"]:
        return data["agent"]["scorecard"]
    return data


def check(baseline: dict, current: dict, threshold: float) -> tuple[bool, list[str]]:
    failures = []
    for metric in METRICS:
        base_val = baseline.get(metric)
        curr_val = current.get(metric)
        if base_val is None or curr_val is None:
            print(f"  ⚠️  Metric '{metric}' missing in one of the files — skipping.")
            continue
        drop = base_val - curr_val
        status = "✅" if drop <= threshold else "❌"
        print(
            f"  {status}  {metric:<24} baseline={base_val:.4f}  "
            f"current={curr_val:.4f}  delta={-drop:+.4f}"
        )
        if drop > threshold:
            failures.append(
                f"{metric}: dropped {drop:.4f} (>{threshold:.4f}). "
                f"Baseline={base_val:.4f}, Current={curr_val:.4f}"
            )
    return len(failures) == 0, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Novus Support Agent regression checker")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--baseline", type=str, default=str(BASELINE_PATH))
    parser.add_argument("--results", type=str, default=str(RESULTS_PATH))
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    results_path = Path(args.results)

    if not baseline_path.exists():
        print(f"No baseline at {baseline_path}. Run eval_harness.py --save-baseline first.")
        sys.exit(2)
    if not results_path.exists():
        print(f"No eval results at {results_path}. Run eval_harness.py first.")
        sys.exit(2)

    baseline = load_scorecard(baseline_path)
    current = load_scorecard(results_path)

    print(f"\nRegression check (threshold={args.threshold:.0%})")
    print(f"  Baseline : {baseline_path}")
    print(f"  Current  : {results_path}\n")

    passed, failures = check(baseline, current, args.threshold)

    print()
    if passed:
        print("✅  NO REGRESSION DETECTED — all metrics within threshold.")
        sys.exit(0)
    print("❌  REGRESSION DETECTED:")
    for f in failures:
        print(f"    • {f}")
    sys.exit(1)


if __name__ == "__main__":
    main()
