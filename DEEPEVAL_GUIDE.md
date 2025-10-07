# DeepEval Integration Guide

Complete guide for using [DeepEval](https://github.com/confident-ai/deepeval) to evaluate your base and fine-tuned GPT models with conversational data.

## 📚 Table of Contents

1. [What is DeepEval?](#what-is-deepeval)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Conversational Evaluation](#conversational-evaluation)
5. [Available Metrics](#available-metrics)
6. [Pytest Integration](#pytest-integration)
7. [Confident AI Dashboard](#confident-ai-dashboard)

---

## What is DeepEval?

**DeepEval** is a production-ready LLM evaluation framework with:
- ✅ **14+ built-in metrics** (GEval, Answer Relevancy, Faithfulness, etc.)
- ✅ **Conversational evaluation** for multi-turn dialogues
- ✅ **Confident AI integration** for visualization
- ✅ **Pytest integration** for CI/CD
- ✅ **11.5k+ GitHub stars** - battle-tested framework

### Why Use DeepEval?

- **Research-backed metrics**: Based on latest academic papers
- **Easy integration**: Works with OpenAI, Anthropic, any LLM
- **Production-ready**: Used by companies worldwide
- **Free & open source**: Apache 2.0 license

---

## Installation

### Option 1: DeepEval Only

```bash
pip install -r requirements_deepeval.txt
```

### Option 2: Full Framework (DeepEval + Custom Metrics)

```bash
pip install -r requirements.txt
pip install deepeval
```

### Verify Installation

```bash
deepeval --version
```

---

## Quick Start

### 1. Set Up Environment

```bash
# Copy and configure environment
cp env_template.txt .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-...
```

### 2. Configure Your Models

Edit the configuration in any script:

```python
BASE_MODEL = "gpt-4o-mini"
FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:org:name:id"

SYSTEM_PROMPT = """Your system prompt here"""
```

### 3. Run Evaluation

#### Method 1: Using Conversational Evaluator

```bash
python deepeval_conversational.py
```

#### Method 2: Using Pytest

```bash
deepeval test run test_deepeval_comparison.py
```

---

## Conversational Evaluation

### Single-Turn Evaluation

For simple question-answer pairs:

```python
from deepeval_conversational import ConversationalEvaluator

evaluator = ConversationalEvaluator(
    base_model="gpt-4o-mini",
    finetuned_model="ft:gpt-4o-mini:org:name:id",
    system_prompt="Your system prompt"
)

single_turn_cases = [
    {
        "id": "Q1",
        "input": "How do I reset my password?",
        "expected_output": "Click 'Forgot Password' on login page."
    }
]

results = evaluator.evaluate_single_turn(single_turn_cases)
report = evaluator.generate_comparison_report(results)
print(report)
```

### Multi-Turn Conversational Evaluation

For multi-turn conversations with context:

```python
multi_turn_conversations = [
    {
        "id": "CONV1",
        "turns": [
            {"user": "Hi, I need help with my order"},
            {"user": "It's order number 12345"},
            {"user": "I want to return it"}
        ]
    }
]

results = evaluator.evaluate_multi_turn(multi_turn_conversations)
report = evaluator.generate_comparison_report(results)
print(report)
```

### Loading from JSON

```python
import json

# Load test data
with open('conversational_test_data.json', 'r') as f:
    data = json.load(f)

# Single-turn
results_single = evaluator.evaluate_single_turn(data['single_turn_cases'])

# Multi-turn
results_multi = evaluator.evaluate_multi_turn(data['multi_turn_conversations'])
```

---

## Available Metrics

### 1. **GEval Metrics** (Custom Criteria)

Create custom evaluation criteria:

```python
from deepeval.metrics import GEval

coherence_metric = GEval(
    name="Coherence",
    criteria="Evaluate if the response is logically coherent and well-structured",
    evaluation_params=["actual_output"],
    threshold=0.7
)

instruction_following = GEval(
    name="Instruction Following",
    criteria="Evaluate if response follows system prompt guidelines",
    evaluation_params=["actual_output", "context"],
    threshold=0.7
)

completeness = GEval(
    name="Completeness",
    criteria="Evaluate if response completely addresses the question",
    evaluation_params=["input", "actual_output"],
    threshold=0.7
)
```

### 2. **Built-in Metrics**

#### Answer Relevancy
Measures how relevant the answer is to the question.

```python
from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
```

#### Faithfulness
Measures if response is faithful to the context (no hallucination).

```python
from deepeval.metrics import FaithfulnessMetric

faithfulness = FaithfulnessMetric(threshold=0.7)
```

#### Toxicity
Detects toxic/harmful content (lower is better).

```python
from deepeval.metrics import ToxicityMetric

toxicity = ToxicityMetric(threshold=0.3)  # Lower threshold = stricter
```

#### Bias
Detects biased content (lower is better).

```python
from deepeval.metrics import BiasMetric

bias = BiasMetric(threshold=0.3)
```

#### Contextual Relevancy
Measures if retrieval context is relevant.

```python
from deepeval.metrics import ContextualRelevancyMetric

contextual_relevancy = ContextualRelevancyMetric(threshold=0.7)
```

#### Hallucination
Detects hallucinated information.

```python
from deepeval.metrics import HallucinationMetric

hallucination = HallucinationMetric(threshold=0.3)
```

### 3. **Metric Comparison Table**

| Metric | What It Measures | Best For | Threshold |
|--------|------------------|----------|-----------|
| **GEval (Coherence)** | Logical flow & structure | All use cases | 0.7 |
| **GEval (Instruction Following)** | Adherence to system prompt | Brand voice, compliance | 0.7 |
| **GEval (Completeness)** | Thoroughness of answer | Customer support | 0.7 |
| **Answer Relevancy** | Question-answer alignment | Q&A systems | 0.7 |
| **Faithfulness** | No hallucination | RAG systems | 0.7 |
| **Toxicity** | Harmful content (lower=better) | Safety-critical apps | 0.3 |
| **Bias** | Biased content (lower=better) | Fair AI systems | 0.3 |

---

## Pytest Integration

### Why Use Pytest?

- ✅ **CI/CD integration**: Run tests automatically
- ✅ **Parallel execution**: Speed up evaluation
- ✅ **Clear reporting**: Pass/fail for each test
- ✅ **Confident AI sync**: Auto-upload results

### Basic Test Structure

```python
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

@pytest.mark.parametrize(
    "test_case",
    [
        LLMTestCase(
            input="How to reset password?",
            actual_output="Click forgot password link",
            expected_output="Use forgot password feature"
        )
    ]
)
def test_model(test_case: LLMTestCase):
    metric = AnswerRelevancyMetric(threshold=0.7)
    assert_test(test_case, [metric])
```

### Run Tests

```bash
# Run all tests
deepeval test run test_deepeval_comparison.py

# Run in parallel (4 workers)
deepeval test run test_deepeval_comparison.py -n 4

# Run specific test
deepeval test run test_deepeval_comparison.py::test_base_model

# Show detailed output
deepeval test run test_deepeval_comparison.py -v
```

### Test Output

```
✅ test_base_model[base_0] PASSED
✅ test_base_model[base_1] PASSED
✅ test_finetuned_model[finetuned_0] PASSED

=============== 3 passed in 12.5s ===============

View results: https://app.confident-ai.com/...
```

---

## Confident AI Dashboard

### What is Confident AI?

[Confident AI](https://confident-ai.com) is DeepEval's cloud platform for:
- 📊 **Visualizing results**: Interactive charts and graphs
- 📈 **Tracking experiments**: Compare model versions
- 🐛 **Debugging**: Trace evaluation results
- 👥 **Team collaboration**: Share results with stakeholders
- 📁 **Dataset management**: Store and version test cases

### Setup Confident AI

#### 1. Login

```bash
deepeval login
```

This will:
1. Open browser to Confident AI
2. Create account (or login)
3. Generate API key
4. Save to local config

#### 2. Run Evaluation

```bash
deepeval test run test_deepeval_comparison.py
```

Results automatically sync to dashboard!

#### 3. View Results

Click the link in terminal output:
```
View results: https://app.confident-ai.com/project/xyz/run/abc
```

### Dashboard Features

#### 📊 **Metric Visualization**
- Radar charts comparing metrics
- Bar charts for each test case
- Trend analysis over time

#### 🔍 **Detailed Inspection**
- View each test case
- See metric scores and reasoning
- Compare base vs fine-tuned responses

#### 📈 **Experiment Tracking**
- Track different model versions
- Compare prompt variations
- A/B test different approaches

#### 🎯 **Dataset Management**
- Upload test cases
- Version control datasets
- Share with team

---

## Example Workflows

### Workflow 1: Quick Evaluation

```bash
# 1. Configure models in deepeval_conversational.py
# 2. Run evaluation
python deepeval_conversational.py

# Output shows comparison report
```

### Workflow 2: CI/CD Integration

```bash
# .github/workflows/eval.yml
- name: Run LLM Evaluation
  run: deepeval test run test_deepeval_comparison.py
  
- name: Check if tests passed
  run: echo $?  # 0 = passed, 1 = failed
```

### Workflow 3: Dataset Evaluation

```python
import json
from deepeval_conversational import ConversationalEvaluator

# Load your production data
with open('conversational_test_data.json') as f:
    data = json.load(f)

evaluator = ConversationalEvaluator(...)

# Evaluate single-turn
results = evaluator.evaluate_single_turn(data['single_turn_cases'])

# Evaluate multi-turn
results = evaluator.evaluate_multi_turn(data['multi_turn_conversations'])

# Generate report
report = evaluator.generate_comparison_report(results)
report.to_csv('evaluation_report.csv')
```

### Workflow 4: Custom Metrics

```python
from deepeval.metrics import GEval

# Define business-specific criteria
brand_voice = GEval(
    name="Brand Voice Alignment",
    criteria="""Evaluate if response matches our brand voice:
    - Friendly but professional
    - Solution-oriented
    - Empathetic to customer needs
    - Uses simple language (no jargon)""",
    evaluation_params=["actual_output"],
    threshold=0.8
)

# Add to evaluation
evaluator.metrics.append(brand_voice)
```

---

## Comparison: DeepEval vs Custom Framework

| Feature | DeepEval | Custom Framework (model_evaluator.py) |
|---------|----------|--------------------------------------|
| **Built-in Metrics** | 14+ metrics | 11 metrics |
| **Conversational Support** | ✅ Native | ⚠️ Manual |
| **Dashboard** | ✅ Confident AI | ❌ Static visualizations |
| **Pytest Integration** | ✅ Built-in | ⚠️ Manual |
| **Statistical Tests** | ❌ Manual | ✅ Built-in (p-values) |
| **Custom Metrics** | ✅ GEval | ✅ Full customization |
| **Semantic Similarity** | ❌ Manual | ✅ BERT, Sentence-BERT |
| **CXO Visualizations** | ⚠️ Basic | ✅ Executive-ready charts |

### When to Use Each?

**Use DeepEval when:**
- ✅ You want quick setup with proven metrics
- ✅ You need conversational evaluation
- ✅ You want cloud dashboards (Confident AI)
- ✅ You need CI/CD integration

**Use Custom Framework when:**
- ✅ You need statistical validation (p-values)
- ✅ You want CXO-ready visualizations
- ✅ You need semantic similarity analysis
- ✅ You want full control over metrics

**Use Both when:**
- ✅ You want comprehensive evaluation
- ✅ You need both technical depth and business presentation
- ✅ You want to validate results across frameworks

---

## Troubleshooting

### Issue 1: API Key Not Found

```bash
# Error: OPENAI_API_KEY not found
# Solution:
echo 'OPENAI_API_KEY=sk-...' > .env
```

### Issue 2: Module Not Found

```bash
# Error: No module named 'deepeval'
# Solution:
pip install -r requirements_deepeval.txt
```

### Issue 3: Tests Not Running

```bash
# Error: No tests collected
# Solution: Use deepeval test run, not pytest
deepeval test run test_deepeval_comparison.py
```

### Issue 4: Confident AI Login Issues

```bash
# Clear cached credentials
rm -rf ~/.deepeval

# Login again
deepeval login
```

---

## Advanced Usage

### Custom Judge Model

```python
from deepeval.metrics import GEval

# Use different model for evaluation
metric = GEval(
    name="Quality",
    criteria="...",
    evaluation_params=["actual_output"],
    evaluation_model="gpt-4o",  # Use GPT-4 as judge
    threshold=0.7
)
```

### Async Evaluation

```python
from deepeval import evaluate_async

results = await evaluate_async(test_cases, metrics)
```

### Custom Test Case Fields

```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="...",
    actual_output="...",
    expected_output="...",
    context=["system prompt"],
    retrieval_context=["retrieved docs"],
    additional_metadata={
        "user_id": "123",
        "timestamp": "2024-01-01",
        "model_version": "v2.0"
    }
)
```

---

## Best Practices

### 1. **Start Small**
```python
# Test with 5-10 cases first
small_dataset = test_cases[:10]
results = evaluator.evaluate_single_turn(small_dataset)
```

### 2. **Use Appropriate Thresholds**
```python
# Customer support: Higher thresholds
metrics = [
    AnswerRelevancyMetric(threshold=0.8),
    ToxicityMetric(threshold=0.2)  # Stricter
]

# Creative content: Lower thresholds
metrics = [
    AnswerRelevancyMetric(threshold=0.6),
]
```

### 3. **Combine Metrics**
```python
# Use multiple metrics for robust evaluation
metrics = [
    GEval(name="Coherence", ...),
    GEval(name="Completeness", ...),
    AnswerRelevancyMetric(),
    ToxicityMetric(),
    BiasMetric()
]
```

### 4. **Version Your Tests**
```python
# Tag test cases with metadata
test_case = LLMTestCase(
    ...,
    additional_metadata={
        "test_version": "v1.0",
        "model_version": "ft-2024-01",
        "date": "2024-01-15"
    }
)
```

---

## Resources

- **DeepEval GitHub**: https://github.com/confident-ai/deepeval
- **Documentation**: https://docs.confident-ai.com
- **Confident AI Platform**: https://app.confident-ai.com
- **Community Discord**: Join via GitHub README

---

## Next Steps

1. ✅ Install DeepEval: `pip install -r requirements_deepeval.txt`
2. ✅ Configure models in scripts
3. ✅ Run your first evaluation: `python deepeval_conversational.py`
4. ✅ Try Pytest integration: `deepeval test run test_deepeval_comparison.py`
5. ✅ Login to Confident AI: `deepeval login`
6. ✅ View results in dashboard

**Happy Evaluating! 🚀**

