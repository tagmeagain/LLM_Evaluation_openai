"""
Simple script to run evaluation with JSON test cases
"""

import os
import json
from dotenv import load_dotenv
from model_evaluator import AdvancedModelEvaluator
from visualizations import EvaluationVisualizer

def load_test_cases(filepath: str = "sample_test_cases.json"):
    """Load test cases from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your API key")
        print("See env_template.txt for format")
        return
    
    # Get user input for configuration
    print("="*60)
    print("Advanced Model Evaluation Framework")
    print("="*60)
    print()
    
    # Model configuration
    print("Enter your model details:")
    base_model = input("Base model name (e.g., gpt-4o-mini): ").strip() or "gpt-4o-mini"
    finetuned_model = input("Fine-tuned model name (e.g., ft:gpt-4o-mini:org:name:id): ").strip()
    
    if not finetuned_model:
        print("❌ Fine-tuned model name is required")
        return
    
    print("\nEnter your system prompt:")
    print("(Press Enter twice when done)")
    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0 and lines[-1] == "":
            break
        lines.append(line)
    system_prompt = "\n".join(lines[:-1]) if lines else ""
    
    if not system_prompt:
        print("❌ System prompt is required")
        return
    
    # Load test cases
    print("\nLoad test cases from:")
    test_file = input("File path (default: sample_test_cases.json): ").strip() or "sample_test_cases.json"
    
    try:
        test_cases = load_test_cases(test_file)
        print(f"✅ Loaded {len(test_cases)} test cases")
    except FileNotFoundError:
        print(f"❌ Error: File '{test_file}' not found")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{test_file}'")
        return
    
    # Judge model
    judge_model = input("\nJudge model (default: gpt-4o): ").strip() or "gpt-4o"
    
    # Confirm and run
    print("\n" + "="*60)
    print("Configuration Summary:")
    print("="*60)
    print(f"Base Model: {base_model}")
    print(f"Fine-tuned Model: {finetuned_model}")
    print(f"System Prompt: {system_prompt[:100]}...")
    print(f"Test Cases: {len(test_cases)} questions")
    print(f"Judge Model: {judge_model}")
    print("="*60)
    
    confirm = input("\nProceed with evaluation? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Evaluation cancelled")
        return
    
    # Initialize evaluator
    print("\n" + "="*60)
    print("Initializing Evaluator...")
    print("="*60)
    evaluator = AdvancedModelEvaluator(
        openai_api_key=OPENAI_API_KEY,
        judge_model=judge_model
    )
    
    # Run evaluation
    results = evaluator.batch_evaluate(
        test_cases=test_cases,
        base_model=base_model,
        finetuned_model=finetuned_model,
        system_prompt=system_prompt
    )
    
    # Calculate aggregate metrics
    print("\nCalculating aggregate metrics...")
    aggregate_df = evaluator.calculate_aggregate_metrics(results)
    
    # Display results
    print("\n" + "="*80)
    print("AGGREGATE METRICS - MODEL COMPARISON")
    print("="*80)
    print(aggregate_df.to_string(index=False))
    print("="*80)
    
    # Save results
    print("\nSaving results...")
    evaluator.save_results(results, aggregate_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = EvaluationVisualizer(output_dir="evaluation_results")
    visualizer.create_all_visualizations(results, aggregate_df)
    
    # Summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print("\nResults saved to: evaluation_results/")
    print("\nKey files for CXO presentation:")
    print("  📊 radar_comparison.html - Overall performance visualization")
    print("  📈 improvement_waterfall.html - Metric-by-metric improvements")
    print("  📉 statistical_significance.png - Statistically validated gains")
    print("  📋 executive_summary.html - Executive summary table")
    print("  📁 aggregate_metrics.csv - All metrics in spreadsheet format")
    print("\nOpen these files in your browser or include in your presentation!")
    print("="*80)

if __name__ == "__main__":
    main()

