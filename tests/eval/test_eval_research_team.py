"""DeepEval harness for Team 2: Research & Advisory agents."""
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


class TestMarketScoutEval:
    def test_market_scout_answer_relevancy(self, research_golden, mock_adapter):
        case = next(c for c in research_golden if c["agent"] == "market_scout")
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


class TestStrategicAdvisorEval:
    def test_strategic_advisor_answer_relevancy(self, research_golden, mock_adapter):
        case = next(c for c in research_golden if c["agent"] == "strategic_advisor")
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

    def test_strategic_advisor_faithfulness(self, research_golden, mock_adapter):
        case = next(c for c in research_golden if c["agent"] == "strategic_advisor")
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


class TestInternalBrainEval:
    def test_internal_brain_faithfulness(self, research_golden, mock_adapter):
        case = next(c for c in research_golden if c["agent"] == "internal_brain")
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
