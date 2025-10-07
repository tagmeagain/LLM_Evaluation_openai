# Advanced LLM Model Evaluation Framework

A comprehensive evaluation framework for comparing base GPT models with fine-tuned versions using advanced, multi-dimensional metrics suitable for CXO presentations.

## 📊 Overview

This framework provides **10+ advanced metrics** across 4 key categories to scientifically evaluate and compare model performance:

### Metric Categories

#### 1. **Semantic Quality Metrics** 🎯
- **Semantic Similarity**: Measures how semantically similar responses are using Sentence-BERT embeddings (0-1 scale)
- **BERTScore F1**: Token-level semantic similarity between responses (0-1 scale)

#### 2. **Instruction Following Metrics** 📋
- **Instruction Adherence**: GPT-4 judged assessment of how well responses follow the system prompt (0-10 scale)
- **Response Relevance**: How relevant and on-topic the response is to the user's question (0-10 scale)

#### 3. **Quality Metrics** ⭐
- **Coherence Score**: Internal consistency, logical flow, and structure (0-10 scale)
- **Completeness Score**: How comprehensive and complete the answer is (0-10 scale)

#### 4. **Advanced Analytical Metrics** 🔬
- **Information Density**: Ratio of unique words to total words (measures information richness)
- **Specificity Score**: How specific vs generic the response is (0-10 scale)
- **Response Length**: Word count
- **Sentence Count**: Number of sentences
- **Average Sentence Length**: Words per sentence

## 🎨 Visualizations for CXO Presentations

The framework automatically generates 8+ professional visualizations:

1. **Radar Chart** - Multi-dimensional performance comparison
2. **Grouped Bar Chart** - Side-by-side metric comparison with error bars
3. **Waterfall Chart** - Performance improvement breakdown
4. **Statistical Significance Plot** - Shows which improvements are statistically valid
5. **Distribution Violin Plots** - Shows score distributions for key metrics
6. **Correlation Heatmap** - Reveals relationships between metrics
7. **Executive Summary Table** - Clean table of key metrics for slides

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this repository
cd office-finetuning-openai

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp env_template.txt .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_key_here
```

### 3. Prepare Your Test Data

Edit `example_evaluation.py` or load from `sample_test_cases.json`:

```python
# Your configuration
BASE_MODEL = "gpt-4o-mini"
FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:org:name:id"

SYSTEM_PROMPT = """Your system prompt here"""

test_cases = [
    {"id": "Q1", "question": "Your question 1"},
    {"id": "Q2", "question": "Your question 2"},
    # ... add more
]
```

### 4. Run Evaluation

```bash
python example_evaluation.py
```

## 📈 Understanding the Metrics

### Semantic Quality Metrics

**Semantic Similarity** (0-1 scale)
- Compares response embeddings using Sentence-BERT
- Higher values = responses are more semantically similar
- **Use Case**: Measure if fine-tuned model maintains similar semantic meaning
- **CXO Insight**: "Our fine-tuned model maintains 85% semantic alignment with base responses while improving quality"

**BERTScore F1** (0-1 scale)
- Token-level semantic similarity using contextual embeddings
- Balances precision and recall of semantic content
- **Use Case**: Detailed semantic comparison at word level
- **CXO Insight**: "Fine-tuned model achieves 0.92 BERTScore, indicating strong semantic preservation"

### Instruction Following Metrics

**Instruction Adherence** (0-10 scale)
- GPT-4 evaluates how well response follows system prompt
- Scored on:
  - 0-3: Poor adherence
  - 4-6: Moderate adherence
  - 7-8: Good adherence
  - 9-10: Excellent adherence
- **Use Case**: Ensure model follows business guidelines and tone
- **CXO Insight**: "Fine-tuned model shows 23% improvement in following company guidelines"

**Response Relevance** (0-10 scale)
- GPT-4 evaluates how relevant response is to user question
- Measures if response actually answers what was asked
- **Use Case**: Reduce off-topic or irrelevant responses
- **CXO Insight**: "Fine-tuned model delivers 15% more relevant responses to customer queries"

### Quality Metrics

**Coherence Score** (0-10 scale)
- GPT-4 evaluates logical flow and internal consistency
- Measures: structure, transitions, contradiction-free content
- **Use Case**: Ensure professional, well-structured responses
- **CXO Insight**: "Fine-tuned model produces 18% more coherent customer communications"

**Completeness Score** (0-10 scale)
- GPT-4 evaluates how comprehensive the answer is
- Checks if all aspects of question are addressed
- **Use Case**: Reduce follow-up questions and improve first-contact resolution
- **CXO Insight**: "Fine-tuned model provides 20% more complete answers, reducing support tickets"

### Advanced Analytical Metrics

**Information Density** (0-1 scale)
- Ratio of unique words to total words
- Higher = more information-rich, less repetitive
- **Use Case**: Measure response efficiency and value
- **CXO Insight**: "Fine-tuned model delivers 12% more information per word"

**Specificity Score** (0-10 scale)
- GPT-4 evaluates how specific vs generic responses are
- Measures concrete details, examples, and context-appropriateness
- **Use Case**: Ensure responses are actionable and detailed
- **CXO Insight**: "Fine-tuned model provides 25% more specific, actionable guidance"

### Statistical Validation

**P-value & Significance Testing**
- Paired t-test compares model performance
- P-value < 0.05 = statistically significant improvement
- **Use Case**: Prove improvements are real, not random chance
- **CXO Insight**: "8 out of 10 improvements are statistically significant (p<0.05)"

## 📁 Output Structure

After running evaluation:

```
evaluation_results/
├── base_model_detailed.csv           # Detailed scores for base model
├── finetuned_model_detailed.csv      # Detailed scores for fine-tuned model
├── aggregate_metrics.csv             # Summary statistics & comparison
├── results.json                      # All results in JSON format
├── radar_comparison.html             # Interactive radar chart
├── metric_comparison.html            # Interactive bar chart
├── improvement_waterfall.html        # Improvement breakdown
├── statistical_significance.png      # Significance visualization
├── distribution_comparison_*.html    # Distribution plots
├── correlation_heatmap_*.png         # Correlation matrices
└── executive_summary.html            # Key metrics table
```

## 🎯 Use Cases

### For CXO Presentations
- Use **radar_comparison.html** to show overall performance improvement
- Use **improvement_waterfall.html** to highlight specific metric gains
- Use **statistical_significance.png** to prove improvements are real
- Use **executive_summary.html** for slide tables

### For Technical Teams
- Use **correlation_heatmap** to understand metric relationships
- Use **distribution_comparison** to see score variability
- Use detailed CSV files for deep analysis

### For Business Stakeholders
- **Instruction Adherence**: "Model follows brand voice 30% better"
- **Response Relevance**: "15% reduction in off-topic responses"
- **Completeness**: "20% fewer follow-up questions needed"
- **Specificity**: "25% more actionable customer guidance"

## 🔧 Customization

### Add Custom Metrics

Edit `model_evaluator.py`:

```python
def custom_metric(self, response: str) -> float:
    # Your custom logic here
    return score

# Add to evaluate_response method
custom_score = self.custom_metric(response)
```

### Change Judge Model

```python
evaluator = AdvancedModelEvaluator(
    openai_api_key=API_KEY,
    judge_model="gpt-4o"  # Use different model for judging
)
```

### Modify Visualizations

Edit `visualizations.py` to customize colors, layouts, chart types.

## 📊 Sample Results Interpretation

### Example Output:

| Metric | Base Mean | Fine-tuned Mean | Improvement (%) | Significant |
|--------|-----------|-----------------|-----------------|-------------|
| Instruction Adherence | 7.2 | 8.9 | +23.6% | Yes |
| Response Relevance | 7.8 | 9.0 | +15.4% | Yes |
| Completeness Score | 7.0 | 8.4 | +20.0% | Yes |
| Specificity Score | 6.5 | 8.1 | +24.6% | Yes |

**CXO Message**: 
> "Our fine-tuned model shows statistically significant improvements across all key quality metrics, with an average 20% performance gain. This translates to better customer satisfaction, reduced support costs, and stronger brand consistency."

## 🤝 Contributing

Feel free to extend this framework:
- Add domain-specific metrics
- Create new visualizations
- Integrate with your ML pipeline
- Add support for other model providers

## 📝 License

MIT License - Feel free to use and modify for your needs.

## 🆘 Support

For issues or questions:
1. Check the example_evaluation.py for usage patterns
2. Review the docstrings in model_evaluator.py
3. Examine sample_test_cases.json for data format

## 🎓 Best Practices

1. **Test Set Size**: Use at least 20-30 diverse questions for reliable statistics
2. **System Prompt**: Keep system prompt consistent between models
3. **Temperature**: Use same temperature for fair comparison (default 0.7)
4. **Judge Model**: Use GPT-4 or better for quality judgments
5. **Multiple Runs**: Run evaluation 2-3 times and average results for stability

## 📚 References

- **BERTScore**: Zhang et al., 2020 - "BERTScore: Evaluating Text Generation with BERT"
- **Sentence-BERT**: Reimers & Gurevych, 2019 - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- **LLM-as-Judge**: Zheng et al., 2023 - "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"

---

**Built for data-driven model evaluation and CXO-ready insights** 📊✨

