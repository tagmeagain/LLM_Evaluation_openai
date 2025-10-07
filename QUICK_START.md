# 🚀 Quick Start Guide - LLM Evaluation Framework

Get started evaluating your models in **under 5 minutes**!

## 📦 What You Have

This repository includes **two powerful evaluation approaches**:

1. **Custom Framework** - Statistical validation + CXO visualizations
2. **DeepEval** - Industry-standard metrics + Cloud dashboard

---

## ⚡ Quick Start: DeepEval (Recommended for Conversational Data)

### Step 1: Install Dependencies (30 seconds)

```bash
cd /Users/sahil/office-finetuning-openai
pip install -r requirements_deepeval.txt
```

### Step 2: Configure API Key (30 seconds)

```bash
# Create .env file
echo 'OPENAI_API_KEY=your_api_key_here' > .env
```

### Step 3: Edit Your Models (1 minute)

Open `run_deepeval.py` or any script and update:

```python
BASE_MODEL = "gpt-4o-mini"
FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:your-org:model-name:id"
SYSTEM_PROMPT = """Your system prompt here"""
```

### Step 4: Run Evaluation (2 minutes)

```bash
# Using CLI with pre-loaded test data
python run_deepeval.py \
  --base-model gpt-4o-mini \
  --finetuned-model ft:gpt-4o-mini:org:name:id \
  --system-prompt "Your system prompt" \
  --test-data conversational_test_data.json \
  --mode both
```

**OR use the Python script:**

```bash
python deepeval_conversational.py
```

### Step 5: View Results

Results are displayed in terminal and saved to CSV!

**Want cloud dashboards?**

```bash
deepeval login  # Login to Confident AI
# Re-run evaluation - results auto-sync to dashboard!
```

---

## 🎯 Quick Start: Custom Framework (For Statistical Analysis)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure

```bash
echo 'OPENAI_API_KEY=your_key' > .env
```

### Step 3: Edit Test Cases

Edit `example_evaluation.py`:

```python
BASE_MODEL = "gpt-4o-mini"
FINETUNED_MODEL = "ft:gpt-4o-mini:org:name:id"
SYSTEM_PROMPT = """Your system prompt"""

test_cases = [
    {"id": "Q1", "question": "Your question here"},
    # Add more...
]
```

### Step 4: Run

```bash
python example_evaluation.py
```

### Step 5: View CXO-Ready Reports

```
evaluation_results/
├── radar_comparison.html          # Open in browser
├── improvement_waterfall.html     # Open in browser
├── statistical_significance.png   # Use in presentations
├── aggregate_metrics.csv          # Use in spreadsheets
```

---

## 📊 Which Framework Should You Use?

### Use DeepEval If You:
- ✅ Have **conversational data** (multi-turn dialogues)
- ✅ Want **quick setup** (5 minutes)
- ✅ Need **cloud dashboards** (Confident AI)
- ✅ Want **CI/CD integration** (pytest)
- ✅ Need **built-in metrics** (14+ metrics ready to go)

### Use Custom Framework If You:
- ✅ Need **statistical validation** (p-values, t-tests)
- ✅ Want **CXO presentations** (executive-ready charts)
- ✅ Need **semantic analysis** (BERT, Sentence-BERT)
- ✅ Want **complete control** over metrics
- ✅ Need **publication-quality** visualizations

### Use BOTH If You:
- ✅ Want **comprehensive evaluation**
- ✅ Need **cross-validation** of results
- ✅ Present to **both technical and business** audiences

---

## 🎬 Example Workflows

### Workflow 1: Quick Conversational Evaluation

```bash
# 1. Edit your model names in deepeval_conversational.py
# 2. Run evaluation
python deepeval_conversational.py

# Done! View results in terminal
```

### Workflow 2: Pytest Integration for CI/CD

```bash
# 1. Edit test_deepeval_comparison.py with your models
# 2. Run tests
deepeval test run test_deepeval_comparison.py -n 4

# 3. View results
# Terminal shows pass/fail + link to dashboard
```

### Workflow 3: Custom Test Data

```bash
# 1. Create your_test_data.json
cat > my_tests.json << 'EOF'
{
  "single_turn_cases": [
    {
      "id": "Q1",
      "input": "Your question",
      "expected_output": "Expected answer"
    }
  ]
}
EOF

# 2. Run with your data
python run_deepeval.py \
  --test-data my_tests.json \
  --base-model gpt-4o-mini \
  --finetuned-model ft:... \
  --system-prompt "Your prompt"
```

### Workflow 4: Full Statistical Analysis

```bash
# 1. Edit example_evaluation.py
# 2. Run custom framework
python example_evaluation.py

# 3. Open visualizations
open evaluation_results/radar_comparison.html
open evaluation_results/statistical_significance.png
```

---

## 📝 Test Data Formats

### DeepEval Format (JSON)

```json
{
  "single_turn_cases": [
    {
      "id": "Q1",
      "input": "How do I reset my password?",
      "expected_output": "Click 'Forgot Password' on login page."
    }
  ],
  "multi_turn_conversations": [
    {
      "id": "CONV1",
      "turns": [
        {"user": "I need help with my order"},
        {"user": "Order number is 12345"},
        {"user": "I want to return it"}
      ]
    }
  ]
}
```

### Custom Framework Format (Python)

```python
test_cases = [
    {
        "id": "Q1",
        "question": "How do I reset my password?"
    },
    {
        "id": "Q2", 
        "question": "What are your business hours?"
    }
]
```

---

## 🔧 Configuration Checklist

- [ ] **API Key**: Set `OPENAI_API_KEY` in `.env`
- [ ] **Base Model**: Set to `gpt-4o-mini` or your base model
- [ ] **Fine-tuned Model**: Set to `ft:gpt-4o-mini:org:name:id`
- [ ] **System Prompt**: Your actual system prompt
- [ ] **Test Data**: Real questions from your use case

---

## 📈 Understanding Results

### DeepEval Metrics (0-1 scale, higher is better)

- **Coherence**: Logical flow (threshold: 0.7)
- **Instruction Following**: Adherence to system prompt (threshold: 0.7)
- **Completeness**: Thoroughness (threshold: 0.7)
- **Answer Relevancy**: Question-answer fit (threshold: 0.7)
- **Toxicity**: Harmful content (threshold: 0.3, **lower is better**)
- **Bias**: Biased content (threshold: 0.3, **lower is better**)

### Custom Framework Metrics

- **Instruction Adherence**: 0-10 (GPT-4 judged)
- **Semantic Similarity**: 0-1 (BERT embeddings)
- **P-value**: < 0.05 = statistically significant
- **Improvement %**: Positive = fine-tuned is better

---

## 🆘 Troubleshooting

### Error: "OPENAI_API_KEY not found"
```bash
echo 'OPENAI_API_KEY=sk-your-key' > .env
```

### Error: "Module not found"
```bash
pip install -r requirements_deepeval.txt
# OR
pip install -r requirements.txt
```

### Error: "deepeval command not found"
```bash
pip install deepeval
```

### DeepEval tests not running
```bash
# Use deepeval test run, not pytest
deepeval test run test_deepeval_comparison.py
```

### Confident AI login issues
```bash
rm -rf ~/.deepeval
deepeval login
```

---

## 📚 Full Documentation

- **DeepEval Guide**: [DEEPEVAL_GUIDE.md](DEEPEVAL_GUIDE.md)
- **Metrics Explanation**: [METRICS_GUIDE.md](METRICS_GUIDE.md)
- **Main README**: [README.md](README.md)

---

## 🎯 Next Steps

### For Quick Testing:
1. ✅ Run `python deepeval_conversational.py`
2. ✅ View results in terminal
3. ✅ Iterate on your models

### For Production:
1. ✅ Login to Confident AI: `deepeval login`
2. ✅ Run pytest: `deepeval test run test_deepeval_comparison.py`
3. ✅ Track experiments in dashboard

### For CXO Presentations:
1. ✅ Run `python example_evaluation.py`
2. ✅ Open `evaluation_results/` for charts
3. ✅ Use visualizations in slides

---

## 💡 Pro Tips

1. **Start small**: Test with 5-10 cases first
2. **Use both frameworks**: Cross-validate results
3. **Customize metrics**: Add your own with GEval
4. **Version your tests**: Track changes over time
5. **Automate**: Integrate into CI/CD

---

## 🚀 You're Ready!

Your evaluation framework is set up. Pick your approach and start evaluating! 

**Questions?** Check the full documentation or the example scripts.

**Happy Evaluating! 🎉**

