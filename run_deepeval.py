"""
Simple runner for DeepEval conversational evaluation
Loads test data from JSON and runs evaluation
"""

import os
import json
import argparse
from dotenv import load_dotenv
from deepeval_conversational import ConversationalEvaluator

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description='Run DeepEval conversational evaluation')
    parser.add_argument(
        '--base-model',
        default='gpt-4o-mini',
        help='Base model identifier'
    )
    parser.add_argument(
        '--finetuned-model',
        required=True,
        help='Fine-tuned model identifier'
    )
    parser.add_argument(
        '--system-prompt',
        required=True,
        help='System prompt (or path to file containing it)'
    )
    parser.add_argument(
        '--test-data',
        default='conversational_test_data.json',
        help='Path to test data JSON file'
    )
    parser.add_argument(
        '--mode',
        choices=['single', 'multi', 'both'],
        default='both',
        help='Evaluation mode: single-turn, multi-turn, or both'
    )
    parser.add_argument(
        '--output',
        default='deepeval_results.csv',
        help='Output CSV file for results'
    )
    
    args = parser.parse_args()
    
    # Load system prompt
    if os.path.isfile(args.system_prompt):
        with open(args.system_prompt, 'r') as f:
            system_prompt = f.read().strip()
    else:
        system_prompt = args.system_prompt
    
    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    # Initialize evaluator
    print("\n" + "="*70)
    print("DeepEval Conversational Evaluation")
    print("="*70)
    print(f"Base Model: {args.base_model}")
    print(f"Fine-tuned Model: {args.finetuned_model}")
    print(f"System Prompt: {system_prompt[:100]}...")
    print(f"Test Data: {args.test_data}")
    print(f"Mode: {args.mode}")
    print("="*70 + "\n")
    
    evaluator = ConversationalEvaluator(
        base_model=args.base_model,
        finetuned_model=args.finetuned_model,
        system_prompt=system_prompt
    )
    
    all_reports = []
    
    # Run single-turn evaluation
    if args.mode in ['single', 'both'] and 'single_turn_cases' in test_data:
        print("\n" + "="*70)
        print("Running Single-Turn Evaluation")
        print("="*70)
        
        results_single = evaluator.evaluate_single_turn(
            test_data['single_turn_cases']
        )
        
        report_single = evaluator.generate_comparison_report(results_single)
        report_single['Evaluation Type'] = 'Single-Turn'
        all_reports.append(report_single)
        
        print("\n" + "="*70)
        print("SINGLE-TURN RESULTS")
        print("="*70)
        print(report_single.to_string(index=False))
    
    # Run multi-turn evaluation
    if args.mode in ['multi', 'both'] and 'multi_turn_conversations' in test_data:
        print("\n" + "="*70)
        print("Running Multi-Turn Evaluation")
        print("="*70)
        
        results_multi = evaluator.evaluate_multi_turn(
            test_data['multi_turn_conversations']
        )
        
        report_multi = evaluator.generate_comparison_report(results_multi)
        report_multi['Evaluation Type'] = 'Multi-Turn'
        all_reports.append(report_multi)
        
        print("\n" + "="*70)
        print("MULTI-TURN RESULTS")
        print("="*70)
        print(report_multi.to_string(index=False))
    
    # Save combined results
    if all_reports:
        import pandas as pd
        combined_report = pd.concat(all_reports, ignore_index=True)
        combined_report.to_csv(args.output, index=False)
        print(f"\n✅ Results saved to: {args.output}")
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print("\nTo view in Confident AI:")
    print("  1. Run: deepeval login")
    print("  2. Re-run evaluation")
    print("  3. Click the dashboard link")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

