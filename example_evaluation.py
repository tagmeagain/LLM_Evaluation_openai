"""
Example script showing how to use the Advanced Model Evaluator
"""

import os
from dotenv import load_dotenv
from model_evaluator import AdvancedModelEvaluator
from visualizations import EvaluationVisualizer

# Load environment variables
load_dotenv()

def main():
    # Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Model names
    BASE_MODEL = "gpt-4o-mini"  # Replace with your base model
    FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:your-org:model-name:id"  # Replace with your fine-tuned model
    
    # System prompt used for both models
    SYSTEM_PROMPT = """You are a helpful customer service assistant for a technology company. 
Provide clear, accurate, and friendly responses to customer inquiries. 
Always maintain a professional tone and offer solutions when possible."""
    
    # Test cases - your user questions
    test_cases = [
        {
            "id": "Q1",
            "question": "How do I reset my password?"
        },
        {
            "id": "Q2",
            "question": "What's the difference between the Pro and Enterprise plans?"
        },
        {
            "id": "Q3",
            "question": "I'm experiencing slow performance. What should I do?"
        },
        {
            "id": "Q4",
            "question": "Can I integrate your service with third-party tools?"
        },
        {
            "id": "Q5",
            "question": "What are your data security and privacy practices?"
        }
    ]
    
    # Initialize evaluator
    print("Initializing Advanced Model Evaluator...")
    evaluator = AdvancedModelEvaluator(
        openai_api_key=OPENAI_API_KEY,
        judge_model="gpt-4o"  # Model used to judge quality
    )
    
    # Run batch evaluation
    results = evaluator.batch_evaluate(
        test_cases=test_cases,
        base_model=BASE_MODEL,
        finetuned_model=FINETUNED_MODEL,
        system_prompt=SYSTEM_PROMPT
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
    evaluator.save_results(results, aggregate_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = EvaluationVisualizer(output_dir="evaluation_results")
    visualizer.create_all_visualizations(results, aggregate_df)
    
    print("\n✅ Evaluation complete!")
    print("\nNext steps:")
    print("1. Review the CSV files in 'evaluation_results/' directory")
    print("2. Open the HTML visualizations for interactive charts")
    print("3. Use the PNG charts in your CXO presentation")
    print("\nKey files for presentation:")
    print("  - radar_comparison.html (overall performance)")
    print("  - improvement_waterfall.html (improvement breakdown)")
    print("  - statistical_significance.png (significant improvements)")
    print("  - executive_summary.html (key metrics table)")


if __name__ == "__main__":
    main()

