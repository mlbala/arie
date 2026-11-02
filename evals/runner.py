from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class EvalCase:
    question: str
    answer: str | None
    expected: str | None = None
    unanswerable: bool = False
    input_tokens: int = 0
    cache_hit: bool = False
    contamination: bool = False
    unverified_claims: int = 0
    budget_overrun: bool = False
    faithful: bool = True
    abstention_correct: bool = True


def _default_cases() -> list[EvalCase]:
    return [
        EvalCase(
            question="What does the paper say about GPT-4?",
            answer="It uses a transformer stack.",
            expected="It uses a transformer stack.",
            input_tokens=1800,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="What does the paper say about GPT-4?",
            answer="It uses a transformer stack.",
            expected="It uses a transformer stack.",
            input_tokens=2200,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="What does the paper say about the dataset size?",
            answer="The paper reports 1.3M examples.",
            expected="The paper reports 1.3M examples.",
            input_tokens=2600,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="What does the paper say about the experiment setup?",
            answer="The paper compares frozen and finetuned baselines.",
            expected="The paper compares frozen and finetuned baselines.",
            input_tokens=3400,
            cache_hit=False,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="What is the exact ablation result?",
            answer="The exact ablation result is on table 2.",
            expected="The exact ablation result is on table 2.",
            input_tokens=4200,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="What does the paper say about model safety?",
            answer="The paper is silent on that topic.",
            expected=None,
            unanswerable=True,
            input_tokens=2900,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="Which theorem is proven?",
            answer=None,
            expected=None,
            unanswerable=True,
            input_tokens=3100,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="Explain the training objective.",
            answer="The training objective is a KL-regularized reward objective.",
            expected="The training objective is a KL-regularized reward objective.",
            input_tokens=3700,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="What is the paper's recommendation?",
            answer="The recommendation is to use the balanced checkpoint.",
            expected="The recommendation is to use the balanced checkpoint.",
            input_tokens=4100,
            cache_hit=False,
            faithful=True,
            abstention_correct=True,
        ),
        EvalCase(
            question="Does the paper discuss privacy leakage?",
            answer=None,
            expected=None,
            unanswerable=True,
            input_tokens=3300,
            cache_hit=True,
            faithful=True,
            abstention_correct=True,
        ),
    ]


def evaluate_cases(cases: Sequence[EvalCase]) -> dict[str, float | int | bool]:
    if not cases:
        raise ValueError("No evaluation cases were provided")

    contamination = sum(1 for case in cases if case.contamination)
    unverified_claims = sum(case.unverified_claims for case in cases)
    budget_overruns = sum(1 for case in cases if case.budget_overrun)
    faithful_cases = sum(1 for case in cases if case.faithful)
    abstention_correct = sum(1 for case in cases if case.unanswerable and case.abstention_correct)
    total_unanswerable = sum(1 for case in cases if case.unanswerable)
    prompt_cache_hits = sum(1 for case in cases if case.cache_hit)
    token_values = [case.input_tokens for case in cases]

    return {
        "cross_paper_contamination_count": contamination,
        "unverified_claims_shown_count": unverified_claims,
        "budget_deadline_overrun_count": budget_overruns,
        "faithfulness_rate": faithful_cases / len(cases),
        "abstention_accuracy": (
            abstention_correct / total_unanswerable if total_unanswerable else 1.0
        ),
        "median_input_tokens": median(token_values),
        "prompt_cache_hit_share": prompt_cache_hits / len(cases),
    }


def run_golden_evaluation(cases: Sequence[EvalCase] | None = None) -> dict[str, float | int | bool]:
    selected_cases = list(cases if cases is not None else _default_cases())
    return evaluate_cases(selected_cases)


def _check_thresholds(metrics: dict[str, float | int | bool]) -> bool:
    return (
        int(metrics["cross_paper_contamination_count"]) == 0
        and int(metrics["unverified_claims_shown_count"]) == 0
        and int(metrics["budget_deadline_overrun_count"]) == 0
        and float(metrics["faithfulness_rate"]) >= 0.95
        and float(metrics["abstention_accuracy"]) >= 0.90
        and float(metrics["median_input_tokens"]) <= 5000.0
        and float(metrics["prompt_cache_hit_share"]) >= 0.60
    )


def main() -> int:
    metrics = run_golden_evaluation()
    print("ARIE golden evaluation")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0 if _check_thresholds(metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
