"""
DeepEval Integration for Conversational LLM Evaluation
Supports multi-turn conversations with system prompts
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
)
from deepeval.dataset import EvaluationDataset, Golden

# Load environment
load_dotenv()


class ConversationalEvaluator:
    """
    Evaluate conversational LLM models using DeepEval
    Supports both single-turn and multi-turn conversations
    """
    
    def __init__(
        self, 
        base_model: str,
        finetuned_model: str,
        system_prompt: str,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize conversational evaluator
        
        Args:
            base_model: Base model identifier (e.g., 'gpt-4o-mini')
            finetuned_model: Fine-tuned model identifier
            system_prompt: System prompt used for conversations
            openai_api_key: OpenAI API key (defaults to env variable)
        """
        self.base_model = base_model
        self.finetuned_model = finetuned_model
        self.system_prompt = system_prompt
        
        api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        
        # Initialize metrics
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> List:
        """Initialize DeepEval metrics for conversational evaluation"""
        
        # GEval metrics for custom criteria
        coherence_metric = GEval(
            name="Coherence",
            criteria="Evaluate if the response is logically coherent and well-structured",
            evaluation_params=["actual_output"],
            threshold=0.7
        )
        
        instruction_following = GEval(
            name="Instruction Following",
            criteria="Evaluate if the response follows the system prompt instructions and guidelines",
            evaluation_params=["actual_output", "context"],
            threshold=0.7
        )
        
        completeness = GEval(
            name="Completeness",
            criteria="Evaluate if the response completely addresses the user's question",
            evaluation_params=["input", "actual_output"],
            threshold=0.7
        )
        
        # Built-in metrics
        answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
        toxicity = ToxicityMetric(threshold=0.3)  # Lower is better for toxicity
        bias = BiasMetric(threshold=0.3)  # Lower is better for bias
        
        return [
            coherence_metric,
            instruction_following,
            completeness,
            answer_relevancy,
            toxicity,
            bias
        ]
    
    def get_model_response(
        self, 
        model: str, 
        user_input: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Get response from model with conversation history
        
        Args:
            model: Model identifier
            user_input: Current user input
            conversation_history: Previous conversation turns
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting response from {model}: {e}")
            return ""
    
    def evaluate_single_turn(
        self,
        test_cases: List[Dict[str, str]],
        use_retrieval_context: bool = False
    ) -> Dict:
        """
        Evaluate single-turn conversations
        
        Args:
            test_cases: List of dicts with 'id', 'input', and optional 'expected_output'
            use_retrieval_context: Whether to include retrieval context
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*70}")
        print(f"DeepEval Single-Turn Conversational Evaluation")
        print(f"{'='*70}\n")
        
        base_dataset = EvaluationDataset()
        finetuned_dataset = EvaluationDataset()
        
        for i, test_case in enumerate(test_cases, 1):
            user_input = test_case['input']
            test_id = test_case.get('id', f'test_{i}')
            
            print(f"[{i}/{len(test_cases)}] Evaluating: {test_id}")
            
            # Get responses from both models
            base_response = self.get_model_response(self.base_model, user_input)
            finetuned_response = self.get_model_response(self.finetuned_model, user_input)
            
            # Create test cases with system prompt as context
            base_test = LLMTestCase(
                input=user_input,
                actual_output=base_response,
                expected_output=test_case.get('expected_output'),
                context=[self.system_prompt],  # System prompt as context
                retrieval_context=[self.system_prompt] if use_retrieval_context else None
            )
            
            finetuned_test = LLMTestCase(
                input=user_input,
                actual_output=finetuned_response,
                expected_output=test_case.get('expected_output'),
                context=[self.system_prompt],
                retrieval_context=[self.system_prompt] if use_retrieval_context else None
            )
            
            base_dataset.add_test_case(base_test)
            finetuned_dataset.add_test_case(finetuned_test)
        
        # Evaluate both datasets
        print(f"\n{'='*70}")
        print("Evaluating Base Model...")
        print(f"{'='*70}")
        base_results = evaluate(base_dataset, self.metrics)
        
        print(f"\n{'='*70}")
        print("Evaluating Fine-tuned Model...")
        print(f"{'='*70}")
        finetuned_results = evaluate(finetuned_dataset, self.metrics)
        
        return {
            'base': base_results,
            'finetuned': finetuned_results,
            'base_dataset': base_dataset,
            'finetuned_dataset': finetuned_dataset
        }
    
    def evaluate_multi_turn(
        self,
        conversations: List[Dict],
    ) -> Dict:
        """
        Evaluate multi-turn conversations
        
        Args:
            conversations: List of conversation dicts with 'id', 'turns' (list of turns)
                          Each turn has 'user' and optionally 'expected_assistant'
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*70}")
        print(f"DeepEval Multi-Turn Conversational Evaluation")
        print(f"{'='*70}\n")
        
        base_dataset = EvaluationDataset()
        finetuned_dataset = EvaluationDataset()
        
        for i, conversation in enumerate(conversations, 1):
            conv_id = conversation.get('id', f'conv_{i}')
            turns = conversation['turns']
            
            print(f"[{i}/{len(conversations)}] Evaluating conversation: {conv_id}")
            
            # Build conversation history
            base_history = []
            finetuned_history = []
            
            base_messages = []
            finetuned_messages = []
            
            for turn_idx, turn in enumerate(turns):
                user_msg = turn['user']
                
                # Get responses with conversation history
                base_response = self.get_model_response(
                    self.base_model, 
                    user_msg, 
                    base_history
                )
                finetuned_response = self.get_model_response(
                    self.finetuned_model, 
                    user_msg, 
                    finetuned_history
                )
                
                # Add to conversation history for next turn
                base_history.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": base_response}
                ])
                finetuned_history.extend([
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": finetuned_response}
                ])
                
                # Build messages for ConversationalTestCase
                base_messages.append([user_msg, base_response])
                finetuned_messages.append([user_msg, finetuned_response])
            
            # Create ConversationalTestCase for entire conversation
            base_conv_test = ConversationalTestCase(
                messages=base_messages,
                # You can add expected outputs for each turn if available
            )
            
            finetuned_conv_test = ConversationalTestCase(
                messages=finetuned_messages,
            )
            
            # For evaluation, we'll create individual test cases for each turn
            for turn_idx, (base_msg, finetuned_msg) in enumerate(
                zip(base_messages, finetuned_messages)
            ):
                base_test = LLMTestCase(
                    input=base_msg[0],
                    actual_output=base_msg[1],
                    context=[self.system_prompt],
                    additional_metadata={
                        'conversation_id': conv_id,
                        'turn': turn_idx + 1
                    }
                )
                
                finetuned_test = LLMTestCase(
                    input=finetuned_msg[0],
                    actual_output=finetuned_msg[1],
                    context=[self.system_prompt],
                    additional_metadata={
                        'conversation_id': conv_id,
                        'turn': turn_idx + 1
                    }
                )
                
                base_dataset.add_test_case(base_test)
                finetuned_dataset.add_test_case(finetuned_test)
        
        # Evaluate
        print(f"\n{'='*70}")
        print("Evaluating Base Model Conversations...")
        print(f"{'='*70}")
        base_results = evaluate(base_dataset, self.metrics)
        
        print(f"\n{'='*70}")
        print("Evaluating Fine-tuned Model Conversations...")
        print(f"{'='*70}")
        finetuned_results = evaluate(finetuned_dataset, self.metrics)
        
        return {
            'base': base_results,
            'finetuned': finetuned_results,
            'base_dataset': base_dataset,
            'finetuned_dataset': finetuned_dataset
        }
    
    def generate_comparison_report(self, results: Dict) -> pd.DataFrame:
        """
        Generate comparison report from evaluation results
        
        Args:
            results: Results dictionary from evaluation
            
        Returns:
            DataFrame with metric comparison
        """
        # Extract metric scores
        base_scores = {}
        finetuned_scores = {}
        
        # Get test results
        base_test_results = results['base_dataset'].test_cases
        finetuned_test_results = results['finetuned_dataset'].test_cases
        
        # Aggregate scores by metric
        for metric in self.metrics:
            metric_name = metric.name
            
            # Calculate average scores
            base_metric_scores = []
            finetuned_metric_scores = []
            
            for test_case in base_test_results:
                if hasattr(test_case, 'metrics_data'):
                    for m_data in test_case.metrics_data:
                        if m_data.name == metric_name:
                            base_metric_scores.append(m_data.score)
            
            for test_case in finetuned_test_results:
                if hasattr(test_case, 'metrics_data'):
                    for m_data in test_case.metrics_data:
                        if m_data.name == metric_name:
                            finetuned_metric_scores.append(m_data.score)
            
            base_scores[metric_name] = sum(base_metric_scores) / len(base_metric_scores) if base_metric_scores else 0
            finetuned_scores[metric_name] = sum(finetuned_metric_scores) / len(finetuned_metric_scores) if finetuned_metric_scores else 0
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Metric': list(base_scores.keys()),
            'Base Model': list(base_scores.values()),
            'Fine-tuned Model': list(finetuned_scores.values()),
        })
        
        comparison_df['Improvement (%)'] = (
            (comparison_df['Fine-tuned Model'] - comparison_df['Base Model']) / 
            comparison_df['Base Model'] * 100
        )
        
        comparison_df = comparison_df.round(3)
        
        return comparison_df


def main():
    """Example usage"""
    
    # Configuration
    BASE_MODEL = "gpt-4o-mini"
    FINETUNED_MODEL = "ft:gpt-4o-mini:org:name:id"  # Replace with your model
    
    SYSTEM_PROMPT = """You are a helpful customer service assistant for a technology company.
Provide clear, accurate, and friendly responses to customer inquiries.
Always maintain a professional tone and offer solutions when possible."""
    
    # Initialize evaluator
    evaluator = ConversationalEvaluator(
        base_model=BASE_MODEL,
        finetuned_model=FINETUNED_MODEL,
        system_prompt=SYSTEM_PROMPT
    )
    
    # Example 1: Single-turn evaluation
    print("\n" + "="*70)
    print("EXAMPLE 1: Single-Turn Evaluation")
    print("="*70)
    
    single_turn_cases = [
        {
            "id": "Q1",
            "input": "How do I reset my password?",
            "expected_output": "To reset your password, please visit our password reset page and follow the instructions."
        },
        {
            "id": "Q2",
            "input": "What are your business hours?",
            "expected_output": "We're available 24/7 for customer support."
        },
    ]
    
    results_single = evaluator.evaluate_single_turn(single_turn_cases)
    
    # Generate report
    report_single = evaluator.generate_comparison_report(results_single)
    print("\n" + "="*70)
    print("SINGLE-TURN COMPARISON REPORT")
    print("="*70)
    print(report_single.to_string(index=False))
    
    # Example 2: Multi-turn evaluation
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Multi-Turn Conversational Evaluation")
    print("="*70)
    
    multi_turn_conversations = [
        {
            "id": "CONV1",
            "turns": [
                {"user": "Hi, I need help with my order"},
                {"user": "It's order number 12345"},
                {"user": "I want to return it"},
            ]
        },
        {
            "id": "CONV2",
            "turns": [
                {"user": "What payment methods do you accept?"},
                {"user": "Do you accept PayPal?"},
                {"user": "Great, how do I add it to my account?"},
            ]
        },
    ]
    
    results_multi = evaluator.evaluate_multi_turn(multi_turn_conversations)
    
    # Generate report
    report_multi = evaluator.generate_comparison_report(results_multi)
    print("\n" + "="*70)
    print("MULTI-TURN COMPARISON REPORT")
    print("="*70)
    print(report_multi.to_string(index=False))
    
    print("\n✅ DeepEval Evaluation Complete!")
    print("\nTo view results in Confident AI dashboard, run:")
    print("  deepeval login")


if __name__ == "__main__":
    main()

