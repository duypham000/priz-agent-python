"""DeepEval harness for Team 4: Knowledge & Learning agents."""
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


class TestKnowledgeScoutEval:
    def test_knowledge_scout_answer_relevancy(self, knowledge_golden, mock_adapter):
        case = next(c for c in knowledge_golden if c["agent"] == "knowledge_scout")
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

    def test_knowledge_scout_faithfulness(self, knowledge_golden, mock_adapter):
        case = next(c for c in knowledge_golden if c["agent"] == "knowledge_scout")
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


class TestExperienceArchivistEval:
    def test_experience_archivist_answer_relevancy(self, knowledge_golden, mock_adapter):
        case = next(c for c in knowledge_golden if c["agent"] == "experience_archivist")
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


class TestYamlArchitectEval:
    def test_yaml_architect_answer_relevancy(self, knowledge_golden, mock_adapter):
        case = next(c for c in knowledge_golden if c["agent"] == "yaml_architect")
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
