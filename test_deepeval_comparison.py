"""
Pytest Integration for DeepEval Model Comparison
Run with: deepeval test run test_deepeval_comparison.py
"""

import os
import pytest
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ToxicityMetric,
    BiasMetric,
)
from deepeval.dataset import EvaluationDataset

# Load environment
load_dotenv()

# Configuration
BASE_MODEL = "gpt-4o-mini"
FINETUNED_MODEL = "ft:gpt-4o-mini:org:name:id"  # Replace with your fine-tuned model

SYSTEM_PROMPT = """You are a helpful customer service assistant for a technology company.
Provide clear, accurate, and friendly responses to customer inquiries.
Always maintain a professional tone and offer solutions when possible."""

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def get_model_response(model: str, user_input: str) -> str:
    """Get response from model"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return ""


# Define test cases
TEST_INPUTS = [
    {
        "id": "Q1",
        "input": "How do I reset my password?",
        "expected_output": "You can reset your password by clicking the 'Forgot Password' link on the login page."
    },
    {
        "id": "Q2",
        "input": "What's the difference between Pro and Enterprise plans?",
        "expected_output": "Enterprise plans include priority support, advanced analytics, and unlimited users."
    },
    {
        "id": "Q3",
        "input": "I'm experiencing slow performance. What should I do?",
        "expected_output": "Try clearing your cache, checking your internet connection, or contacting support."
    },
]

# Initialize metrics
coherence_metric = GEval(
    name="Coherence",
    criteria="Evaluate if the response is logically coherent and well-structured",
    evaluation_params=["actual_output"],
    threshold=0.7
)

instruction_following = GEval(
    name="Instruction Following",
    criteria="Evaluate if the response follows the system prompt guidelines (professional, helpful, solution-oriented)",
    evaluation_params=["actual_output", "context"],
    threshold=0.7
)

completeness = GEval(
    name="Completeness",
    criteria="Evaluate if the response completely addresses the user's question",
    evaluation_params=["input", "actual_output"],
    threshold=0.7
)

answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
toxicity = ToxicityMetric(threshold=0.3)
bias = BiasMetric(threshold=0.3)

# Create datasets
base_dataset = EvaluationDataset()
finetuned_dataset = EvaluationDataset()

# Generate responses and create test cases
for test_input in TEST_INPUTS:
    user_input = test_input['input']
    
    # Get responses
    base_response = get_model_response(BASE_MODEL, user_input)
    finetuned_response = get_model_response(FINETUNED_MODEL, user_input)
    
    # Create test cases
    base_test = LLMTestCase(
        input=user_input,
        actual_output=base_response,
        expected_output=test_input.get('expected_output'),
        context=[SYSTEM_PROMPT]
    )
    
    finetuned_test = LLMTestCase(
        input=user_input,
        actual_output=finetuned_response,
        expected_output=test_input.get('expected_output'),
        context=[SYSTEM_PROMPT]
    )
    
    base_dataset.add_test_case(base_test)
    finetuned_dataset.add_test_case(finetuned_test)


# Parametrized tests for base model
@pytest.mark.parametrize(
    "test_case",
    base_dataset.test_cases,
    ids=[f"base_{i}" for i in range(len(base_dataset.test_cases))]
)
def test_base_model(test_case: LLMTestCase):
    """Test base model against all metrics"""
    assert_test(
        test_case, 
        [
            coherence_metric,
            instruction_following,
            completeness,
            answer_relevancy,
            toxicity,
            bias
        ]
    )


# Parametrized tests for fine-tuned model
@pytest.mark.parametrize(
    "test_case",
    finetuned_dataset.test_cases,
    ids=[f"finetuned_{i}" for i in range(len(finetuned_dataset.test_cases))]
)
def test_finetuned_model(test_case: LLMTestCase):
    """Test fine-tuned model against all metrics"""
    assert_test(
        test_case, 
        [
            coherence_metric,
            instruction_following,
            completeness,
            answer_relevancy,
            toxicity,
            bias
        ]
    )

