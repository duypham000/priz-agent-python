"""DeepEval harness for Team 1: Documentation & Ops agents."""
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.llm.mock import MockAdapter

RELEVANCY_THRESHOLD = 0.7
FAITHFULNESS_THRESHOLD = 0.7


@pytest.fixture(scope="module")
def mock_adapter():
    return MockAdapter()


class TestSummarizerEval:
    def test_summarizer_answer_relevancy(self, docs_golden, mock_adapter):
        case = next(c for c in docs_golden if c["agent"] == "summarizer")
        model = mock_adapter.get_model()
        response = model.invoke(case["input"])
        actual_output = response.content if hasattr(response, "content") else str(response)

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case["expected_output"],
            retrieval_context=case["context"],
        )
        assert_test(test_case, [AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD)])

    def test_summarizer_faithfulness(self, docs_golden, mock_adapter):
        case = next(c for c in docs_golden if c["agent"] == "summarizer")
        model = mock_adapter.get_model()
        response = model.invoke(case["input"])
        actual_output = response.content if hasattr(response, "content") else str(response)

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case["expected_output"],
            retrieval_context=case["context"],
        )
        assert_test(test_case, [FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD)])


class TestTaskArchitectEval:
    def test_task_architect_answer_relevancy(self, docs_golden, mock_adapter):
        case = next(c for c in docs_golden if c["agent"] == "task_architect")
        model = mock_adapter.get_model()
        response = model.invoke(case["input"])
        actual_output = response.content if hasattr(response, "content") else str(response)

        test_case = LLMTestCase(
            input=case["input"],
            actual_output=actual_output,
            expected_output=case["expected_output"],
            retrieval_context=case["context"],
        )
        assert_test(test_case, [AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD)])
