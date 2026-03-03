"""
Evaluates TIMER-Graph on hard negative scenarios.
Target: Demonstrate >20% improvement over semantic baseline.

Statistical additions (Phase 5D):
  - bootstrap_ci()        — 95% confidence interval on accuracy via bootstrap
  - calculate_mcnemar()   — McNemar's paired test (exact when b+c<25, asymptotic otherwise)
  - calculate_cohens_h()  — Effect size for two proportions
  - LaTeX export          — results/phase5/results_table.tex
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest, chi2
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from app.research.retrieval.scoring import TIMERScorer
from app.evaluation.metrics import compute_TRA, compute_recall_at_k
from typing import Dict, List, Any, Tuple


# ═══════════════════════════════════════════════════════════════
# Statistical helper functions (Phase 5D)
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(
    correct_array: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for accuracy.

    Parameters
    ----------
    correct_array : np.ndarray of 0/1 (or bool) values
        Binary correctness per query.
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    ci : float
        Confidence level, e.g. 0.95 for 95% CI.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
        CI bounds for accuracy.
    """
    arr = np.asarray(correct_array, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed=42)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1.0 - ci
    lower = float(np.percentile(boot_means, alpha / 2 * 100))
    upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))
    return (lower, upper)


def calculate_mcnemar(
    baseline_correct: np.ndarray,
    timer_correct: np.ndarray,
) -> float:
    """
    McNemar's paired significance test for two binary classifiers.

    Contingency table:
                        TIMER correct  TIMER wrong
        Baseline correct      a             b
        Baseline wrong        c             d

    Selection rule (SUPERVISOR-approved):
      b + c >= 25  → asymptotic chi2: (|b − c| − 1)^2 / (b + c)  with  df=1
      b + c <  25  → exact binomial:  P(X >= max(b,c)) where X ~ Binomial(b+c, 0.5)

    Parameters
    ----------
    baseline_correct : np.ndarray of bool/int
    timer_correct    : np.ndarray of bool/int

    Returns
    -------
    p_value : float
    """
    bc = np.asarray(baseline_correct, dtype=bool)
    tc = np.asarray(timer_correct, dtype=bool)

    # Discordant cells
    b = int(np.sum(bc & ~tc))   # baseline correct, timer wrong
    c = int(np.sum(~bc & tc))   # baseline wrong, timer correct

    discordant = b + c

    if discordant == 0:
        # Perfect agreement — p-value is undefined; return 1.0 (no evidence of difference)
        return 1.0

    if discordant < 25:
        # Exact binomial test — one-sided: probability that timer wins more than expected by chance
        result = binomtest(k=max(b, c), n=discordant, p=0.5, alternative="greater")
        return float(result.pvalue)
    else:
        # Asymptotic chi2 with continuity correction (Edwards)
        chi2_stat = (abs(b - c) - 1.0) ** 2 / discordant
        p_value = float(chi2.sf(chi2_stat, df=1))
        return p_value


def calculate_cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two independent proportions.

    Formula:  h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Convention (Cohen, 1988):
      |h| ≥ 0.8  → large effect
      |h| ≥ 0.5  → medium effect
      |h| ≥ 0.2  → small effect

    Parameters
    ----------
    p1, p2 : float  (proportions in [0, 1])

    Returns
    -------
    h : float  (absolute value)
    """
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = float(np.clip(p2, 0.0, 1.0))
    h = 2.0 * np.arcsin(np.sqrt(p1)) - 2.0 * np.arcsin(np.sqrt(p2))
    return abs(float(h))


def significance_stars(p_value: float) -> str:
    """Return publication-standard significance stars."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "ns"


def cohens_h_label(h: float) -> str:
    if h >= 0.8:
        return "large"
    elif h >= 0.5:
        return "medium"
    elif h >= 0.2:
        return "small"
    return "negligible"


# ═══════════════════════════════════════════════════════════════
# Evaluator class
# ═══════════════════════════════════════════════════════════════

class HardNegativeEvaluator:
    """
    Evaluates retrieval performance on hard negative scenarios.
    """

    def __init__(
        self,
        hard_negatives_path: str = "data/mocks/combined_hard_negatives.json",
        results_dir: str = "results/phase5",
    ):
        self.hard_negatives_path = Path(hard_negatives_path)
        if not self.hard_negatives_path.exists():
            raise FileNotFoundError(f"Dataset not found at {hard_negatives_path}")

        with open(hard_negatives_path, "r") as f:
            self.data = json.load(f)

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.scorer = TIMERScorer(lambda_decay=0.005)

    def simulate_semantic_baseline(self, query_data: Dict) -> Dict:
        """
        Simulates semantic-only retrieval (no temporal modulation).
        Rankings are determined purely by semantic_score.
        """
        notes = query_data["notes"]
        ranked = sorted(notes, key=lambda x: x["semantic_score"], reverse=True)

        return {
            "top_retrieval": ranked[0]["id"],
            "ranking": [n["id"] for n in ranked],
            "scores": {n["id"]: n["semantic_score"] for n in ranked},
        }

    def simulate_timer_retrieval(self, query_data: Dict) -> Dict:
        """
        Simulates TIMER retrieval with intent-modulated temporal scoring.
        """
        notes = query_data["notes"]
        query_text = query_data["text"]

        intent, confidence = self.scorer.classify_intent(query_text)
        beta = self.scorer.get_beta_intent(intent, confidence)

        timer_scores = {}
        for note in notes:
            score = self.scorer.score_node(
                semantic_score=note["semantic_score"],
                offset_days=note["offset_days"],
                beta=beta,
            )
            timer_scores[note["id"]] = score

        ranked = sorted(notes, key=lambda x: timer_scores[x["id"]], reverse=True)

        return {
            "top_retrieval": ranked[0]["id"],
            "ranking": [n["id"] for n in ranked],
            "scores": timer_scores,
            "intent": intent,
            "confidence": confidence,
        }

    def evaluate_scenario(self, scenario_name: str, queries: List[Dict]) -> pd.DataFrame:
        """
        Evaluates all queries in a scenario.
        """
        results = []

        for i, query in enumerate(queries):
            baseline_result = self.simulate_semantic_baseline(query)
            baseline_correct = baseline_result["top_retrieval"] == query["expected_retrieval"]

            timer_result = self.simulate_timer_retrieval(query)
            timer_correct = timer_result["top_retrieval"] == query["expected_retrieval"]

            if i == 0:
                print(f"  DEBUG Query[0] : {query['text']}")
                print(f"  DEBUG Intent   : {timer_result['intent']}")
                print(f"  DEBUG Scores   : {timer_result['scores']}")
                print(f"  DEBUG Expected : {query['expected_retrieval']} | Got: {timer_result['top_retrieval']}")

            results.append(
                {
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "intent": query["intent"],
                    "expected_note": query["expected_retrieval"],
                    "baseline_retrieval": baseline_result["top_retrieval"],
                    "baseline_correct": int(baseline_correct),
                    "timer_retrieval": timer_result["top_retrieval"],
                    "timer_correct": int(timer_correct),
                    "timer_intent_detected": timer_result["intent"],
                    "improvement": int(timer_correct and not baseline_correct),
                }
            )

        return pd.DataFrame(results)

    def run_full_evaluation(self) -> Dict:
        """
        Runs evaluation on all scenarios and generates report.
        """
        all_results = {}

        for scenario_name, queries in self.data["scenarios"].items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {scenario_name}")
            print(f"{'='*60}")

            df = self.evaluate_scenario(scenario_name, queries)
            all_results[scenario_name] = df

            baseline_acc = df["baseline_correct"].mean()
            timer_acc = df["timer_correct"].mean()
            improvement = timer_acc - baseline_acc

            print(f"\n📊 Results:")
            print(f"   Baseline Accuracy : {baseline_acc:.2%}")
            print(f"   TIMER Accuracy    : {timer_acc:.2%}")
            print(f"   Improvement       : {improvement:+.2%}")

            output_path = self.results_dir / f"{scenario_name}_results.csv"
            df.to_csv(output_path, index=False)
            print(f"   💾 Saved to: {output_path}")

        self.generate_summary_report(all_results)

        return all_results

    def generate_summary_report(self, all_results: Dict):
        """
        Generates publication-ready summary table with statistical tests.
        Exports CSV + LaTeX.
        """
        summary = []

        for scenario, df in all_results.items():
            baseline_arr = df["baseline_correct"].to_numpy()
            timer_arr = df["timer_correct"].to_numpy()

            baseline_acc = baseline_arr.mean()
            timer_acc = timer_arr.mean()
            delta = timer_acc - baseline_acc

            # ── Bootstrap CIs ─────────────────────────────────────────
            bl_lo, bl_hi = bootstrap_ci(baseline_arr)
            tm_lo, tm_hi = bootstrap_ci(timer_arr)

            # ── McNemar's test ────────────────────────────────────────
            p_val = calculate_mcnemar(baseline_arr, timer_arr)
            stars = significance_stars(p_val)

            # ── Cohen's h ─────────────────────────────────────────────
            h = calculate_cohens_h(timer_acc, baseline_acc)
            h_label = cohens_h_label(h)

            summary.append(
                {
                    "Scenario": scenario.replace("_", " ").title(),
                    "N": len(df),
                    "Baseline Acc": f"{baseline_acc:.2%}",
                    "Baseline 95% CI": f"[{bl_lo:.2%}, {bl_hi:.2%}]",
                    "TIMER Acc": f"{timer_acc:.2%}",
                    "TIMER 95% CI": f"[{tm_lo:.2%}, {tm_hi:.2%}]",
                    "Δ Acc": f"{delta:+.2%}",
                    "McNemar p": f"{p_val:.4f}",
                    "Sig": stars,
                    "Cohen's h": f"{h:.3f} ({h_label})",
                    "TIMER Wins": int(df["improvement"].sum()),
                }
            )

        # ── Overall row ───────────────────────────────────────────────
        total_n = sum(len(df) for df in all_results.values())

        if total_n > 0:
            all_baseline = np.concatenate(
                [df["baseline_correct"].to_numpy() for df in all_results.values()]
            )
            all_timer = np.concatenate(
                [df["timer_correct"].to_numpy() for df in all_results.values()]
            )

            overall_baseline = all_baseline.mean()
            overall_timer = all_timer.mean()
            overall_delta = overall_timer - overall_baseline

            bl_lo, bl_hi = bootstrap_ci(all_baseline)
            tm_lo, tm_hi = bootstrap_ci(all_timer)
            p_val = calculate_mcnemar(all_baseline, all_timer)
            stars = significance_stars(p_val)
            h = calculate_cohens_h(overall_timer, overall_baseline)
            h_label = cohens_h_label(h)

            summary.append(
                {
                    "Scenario": "**OVERALL**",
                    "N": total_n,
                    "Baseline Acc": f"{overall_baseline:.2%}",
                    "Baseline 95% CI": f"[{bl_lo:.2%}, {bl_hi:.2%}]",
                    "TIMER Acc": f"{overall_timer:.2%}",
                    "TIMER 95% CI": f"[{tm_lo:.2%}, {tm_hi:.2%}]",
                    "Δ Acc": f"{overall_delta:+.2%}",
                    "McNemar p": f"{p_val:.4f}",
                    "Sig": stars,
                    "Cohen's h": f"{h:.3f} ({h_label})",
                    "TIMER Wins": int(
                        sum(df["improvement"].sum() for df in all_results.values())
                    ),
                }
            )

        summary_df = pd.DataFrame(summary)

        # ── CSV ───────────────────────────────────────────────────────
        summary_path = self.results_dir / "summary_table.csv"
        summary_df.to_csv(summary_path, index=False)

        # ── LaTeX ─────────────────────────────────────────────────────
        tex_path = self.results_dir / "results_table.tex"
        # Select publication columns for LaTeX
        latex_cols = ["Scenario", "N", "Baseline Acc", "TIMER Acc", "Δ Acc",
                      "McNemar p", "Sig", "Cohen's h"]
        latex_df = summary_df[latex_cols].copy()
        latex_str = latex_df.to_latex(
            index=False,
            escape=True,
            caption=(
                "TIMER-Graph vs. Semantic Baseline on Hard Negative Evaluation Suite. "
                "Significance: * p<0.05, ** p<0.01, *** p<0.001. "
                "Cohen's h: small≥0.2, medium≥0.5, large≥0.8."
            ),
            label="tab:timer_hard_negatives",
        )
        tex_path.write_text(latex_str)

        # ── Print ─────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("📋 PHASE 5 SUMMARY REPORT (with statistical tests)")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print(f"\n💾 CSV    → {summary_path}")
        print(f"💾 LaTeX  → {tex_path}")

        # Success check
        if total_n > 0:
            if overall_timer - overall_baseline >= 0.20:
                print("\n✅ SUCCESS: TIMER achieves >20% improvement target!")
            else:
                print(
                    f"\n⚠️  TIMER improvement ({overall_timer - overall_baseline:.2%}) "
                    "below 20% target"
                )


if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(
        description="Run TIMER hard negative evaluation."
    )
    _parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="data/mocks/combined_hard_negatives.json",
        help="Path to the hard negatives JSON dataset.",
    )
    _parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results/phase5",
        help="Directory for output CSVs and LaTeX table.",
    )
    _args = _parser.parse_args()

    try:
        evaluator = HardNegativeEvaluator(
            hard_negatives_path=_args.dataset,
            results_dir=_args.results_dir,
        )
        results = evaluator.run_full_evaluation()
    except Exception as e:
        import traceback
        print(f"Evaluation Failed: {e}")
        traceback.print_exc()
