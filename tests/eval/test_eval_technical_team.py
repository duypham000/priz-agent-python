"""DeepEval harness for Team 3: Technical Execution agents."""
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


class TestCodeArchitectEval:
    def test_code_architect_answer_relevancy(self, technical_golden, mock_adapter):
        case = next(c for c in technical_golden if c["agent"] == "code_architect")
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


class TestQualityGatekeeperEval:
    def test_quality_gatekeeper_answer_relevancy(self, technical_golden, mock_adapter):
        case = next(c for c in technical_golden if c["agent"] == "quality_gatekeeper")
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

    def test_quality_gatekeeper_faithfulness(self, technical_golden, mock_adapter):
        case = next(c for c in technical_golden if c["agent"] == "quality_gatekeeper")
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


class TestVisualInterpreterEval:
    def test_visual_interpreter_answer_relevancy(self, technical_golden, mock_adapter):
        case = next(c for c in technical_golden if c["agent"] == "visual_interpreter")
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
