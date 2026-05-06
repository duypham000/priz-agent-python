"""DeepEval evaluation harness for agent output quality."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    _DEEPEVAL_AVAILABLE = False
    logger.warning("deepeval not installed — eval harness disabled")


def evaluate_answer_relevancy(
    input: str,
    actual_output: str,
    threshold: float = 0.7,
) -> dict:
    """Score answer relevancy. Returns {"score": float, "passed": bool, "reason": str}."""
    if not _DEEPEVAL_AVAILABLE:
        return {"score": 0.0, "passed": False, "reason": "deepeval not installed"}
    metric = AnswerRelevancyMetric(threshold=threshold)
    test_case = LLMTestCase(input=input, actual_output=actual_output)
    metric.measure(test_case)
    return {"score": metric.score, "passed": metric.success, "reason": metric.reason}


def evaluate_faithfulness(
    input: str,
    actual_output: str,
    retrieval_context: list[str],
    threshold: float = 0.7,
) -> dict:
    """Score faithfulness against retrieval context."""
    if not _DEEPEVAL_AVAILABLE:
        return {"score": 0.0, "passed": False, "reason": "deepeval not installed"}
    metric = FaithfulnessMetric(threshold=threshold)
    test_case = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )
    metric.measure(test_case)
    return {"score": metric.score, "passed": metric.success, "reason": metric.reason}


def run_golden_set_eval(
    golden_set: list[dict],
    threshold: float = 0.7,
) -> list[dict]:
    """
    Evaluate a list of test cases.
    Each item: {"input": str, "actual_output": str, "retrieval_context": list[str] (optional)}
    Returns list of result dicts with score, passed, reason, input.
    """
    results = []
    for item in golden_set:
        r = evaluate_answer_relevancy(
            input=item["input"],
            actual_output=item["actual_output"],
            threshold=threshold,
        )
        r["input"] = item["input"]
        results.append(r)
    return results
