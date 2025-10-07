"""
Advanced Model Evaluation Framework
Compares base GPT-4.1-mini with fine-tuned model using multiple metrics
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# NLP and ML imports
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_rel, wilcoxon
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# OpenAI
from openai import OpenAI

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')


@dataclass
class EvaluationResult:
    """Store evaluation results for a single response"""
    question_id: str
    
    # Semantic Quality Metrics
    semantic_similarity: float  # How semantically similar responses are
    bert_score_f1: float  # BERTScore F1
    
    # Instruction Following Metrics
    instruction_adherence: float  # GPT-4 judged adherence to system prompt
    response_relevance: float  # How relevant response is to question
    
    # Quality Metrics
    coherence_score: float  # Internal consistency and flow
    completeness_score: float  # How complete the answer is
    
    # Quantitative Metrics
    response_length: int  # Token/word count
    sentence_count: int  # Number of sentences
    avg_sentence_length: float  # Average words per sentence
    
    # Advanced Metrics
    information_density: float  # Information per word
    specificity_score: float  # How specific vs generic
    
    # Model identifier
    model_name: str


class AdvancedModelEvaluator:
    """
    Comprehensive evaluation framework for comparing LLM models
    """
    
    def __init__(self, openai_api_key: str, judge_model: str = "gpt-4o"):
        """
        Initialize evaluator
        
        Args:
            openai_api_key: OpenAI API key
            judge_model: Model to use as judge for qualitative metrics
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.judge_model = judge_model
        
        # Load sentence transformer for semantic similarity
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        print("Evaluator initialized successfully!")
    
    def get_model_response(
        self, 
        model: str, 
        system_prompt: str, 
        user_question: str,
        temperature: float = 0.7
    ) -> str:
        """Get response from a model"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting response from {model}: {e}")
            return ""
    
    def calculate_semantic_similarity(
        self, 
        response1: str, 
        response2: str
    ) -> float:
        """
        Calculate semantic similarity between two responses using Sentence-BERT
        Range: 0-1 (higher is more similar)
        """
        embeddings = self.sentence_model.encode([response1, response2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity
    
    def calculate_bert_score(
        self, 
        candidate: str, 
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate BERTScore (semantic similarity at token level)
        Uses base model response as reference
        """
        P, R, F1 = bert_score(
            [candidate], 
            [reference], 
            lang="en", 
            verbose=False,
            device='cpu'
        )
        return {
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item()
        }
    
    def judge_instruction_adherence(
        self, 
        system_prompt: str, 
        user_question: str, 
        response: str
    ) -> float:
        """
        Use GPT-4 to judge how well the response follows the system prompt
        Returns score 0-10
        """
        judge_prompt = f"""You are an expert evaluator. Assess how well the following response adheres to the given system prompt and answers the user question.

System Prompt: {system_prompt}

User Question: {user_question}

Response to Evaluate: {response}

Evaluate on a scale of 0-10 where:
- 0-3: Poor adherence, misses key instructions
- 4-6: Moderate adherence, follows some instructions
- 7-8: Good adherence, follows most instructions
- 9-10: Excellent adherence, perfectly follows all instructions

Respond with ONLY a number between 0 and 10, no explanation."""

        try:
            result = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.3
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0), 10)  # Clamp to 0-10
        except Exception as e:
            print(f"Error in judge_instruction_adherence: {e}")
            return 5.0
    
    def judge_response_relevance(
        self, 
        user_question: str, 
        response: str
    ) -> float:
        """
        Use GPT-4 to judge how relevant the response is to the question
        Returns score 0-10
        """
        judge_prompt = f"""You are an expert evaluator. Assess how relevant and on-topic the following response is to the user's question.

User Question: {user_question}

Response to Evaluate: {response}

Evaluate on a scale of 0-10 where:
- 0-3: Irrelevant or off-topic
- 4-6: Somewhat relevant but missing key points
- 7-8: Relevant and addresses the question
- 9-10: Highly relevant, directly and completely addresses the question

Respond with ONLY a number between 0 and 10, no explanation."""

        try:
            result = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.3
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0), 10)
        except Exception as e:
            print(f"Error in judge_response_relevance: {e}")
            return 5.0
    
    def judge_coherence(self, response: str) -> float:
        """
        Use GPT-4 to judge the internal coherence and flow of the response
        Returns score 0-10
        """
        judge_prompt = f"""You are an expert evaluator. Assess the coherence, logical flow, and internal consistency of the following response.

Response to Evaluate: {response}

Evaluate on a scale of 0-10 where:
- 0-3: Incoherent, contradictory, or poorly structured
- 4-6: Somewhat coherent but with logical gaps
- 7-8: Coherent with good logical flow
- 9-10: Highly coherent, excellent structure and flow

Respond with ONLY a number between 0 and 10, no explanation."""

        try:
            result = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.3
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0), 10)
        except Exception as e:
            print(f"Error in judge_coherence: {e}")
            return 5.0
    
    def judge_completeness(
        self, 
        user_question: str, 
        response: str
    ) -> float:
        """
        Use GPT-4 to judge how complete and comprehensive the response is
        Returns score 0-10
        """
        judge_prompt = f"""You are an expert evaluator. Assess how complete and comprehensive the following response is to the user's question.

User Question: {user_question}

Response to Evaluate: {response}

Evaluate on a scale of 0-10 where:
- 0-3: Incomplete, missing critical information
- 4-6: Partially complete, addresses some aspects
- 7-8: Complete, covers all main aspects
- 9-10: Highly comprehensive, thorough and detailed

Respond with ONLY a number between 0 and 10, no explanation."""

        try:
            result = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.3
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0), 10)
        except Exception as e:
            print(f"Error in judge_completeness: {e}")
            return 5.0
    
    def calculate_information_density(self, response: str) -> float:
        """
        Calculate information density (unique words / total words)
        Higher values indicate more information-rich content
        """
        words = word_tokenize(response.lower())
        if len(words) == 0:
            return 0.0
        unique_words = len(set(words))
        return unique_words / len(words)
    
    def judge_specificity(self, response: str) -> float:
        """
        Use GPT-4 to judge how specific vs generic the response is
        Returns score 0-10
        """
        judge_prompt = f"""You are an expert evaluator. Assess how specific and concrete the following response is, versus being generic and vague.

Response to Evaluate: {response}

Evaluate on a scale of 0-10 where:
- 0-3: Very generic, vague, could apply to many contexts
- 4-6: Somewhat specific but still fairly general
- 7-8: Specific with concrete details and examples
- 9-10: Highly specific, detailed, and context-appropriate

Respond with ONLY a number between 0 and 10, no explanation."""

        try:
            result = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.3
            )
            score = float(result.choices[0].message.content.strip())
            return min(max(score, 0), 10)
        except Exception as e:
            print(f"Error in judge_specificity: {e}")
            return 5.0
    
    def calculate_quantitative_metrics(self, response: str) -> Dict[str, Any]:
        """Calculate basic quantitative metrics"""
        sentences = sent_tokenize(response)
        words = word_tokenize(response)
        
        return {
            'response_length': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def evaluate_response(
        self,
        question_id: str,
        system_prompt: str,
        user_question: str,
        response: str,
        reference_response: str,
        model_name: str
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation of a single response
        
        Args:
            question_id: Identifier for the question
            system_prompt: The system prompt used
            user_question: The user's question
            response: The model's response to evaluate
            reference_response: Reference response (usually from base model)
            model_name: Name of the model being evaluated
        """
        print(f"  Evaluating {model_name} for question {question_id}...")
        
        # Semantic similarity metrics
        semantic_sim = self.calculate_semantic_similarity(response, reference_response)
        bert_scores = self.calculate_bert_score(response, reference_response)
        
        # Instruction following metrics
        instruction_adherence = self.judge_instruction_adherence(
            system_prompt, user_question, response
        )
        response_relevance = self.judge_response_relevance(user_question, response)
        
        # Quality metrics
        coherence = self.judge_coherence(response)
        completeness = self.judge_completeness(user_question, response)
        
        # Quantitative metrics
        quant_metrics = self.calculate_quantitative_metrics(response)
        
        # Advanced metrics
        info_density = self.calculate_information_density(response)
        specificity = self.judge_specificity(response)
        
        return EvaluationResult(
            question_id=question_id,
            semantic_similarity=semantic_sim,
            bert_score_f1=bert_scores['f1'],
            instruction_adherence=instruction_adherence,
            response_relevance=response_relevance,
            coherence_score=coherence,
            completeness_score=completeness,
            response_length=quant_metrics['response_length'],
            sentence_count=quant_metrics['sentence_count'],
            avg_sentence_length=quant_metrics['avg_sentence_length'],
            information_density=info_density,
            specificity_score=specificity,
            model_name=model_name
        )
    
    def batch_evaluate(
        self,
        test_cases: List[Dict[str, str]],
        base_model: str,
        finetuned_model: str,
        system_prompt: str
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate both models on a batch of test cases
        
        Args:
            test_cases: List of dicts with 'id' and 'question' keys
            base_model: Name of base model (e.g., 'gpt-4o-mini')
            finetuned_model: Name of fine-tuned model
            system_prompt: The system prompt to use
            
        Returns:
            Dict with 'base' and 'finetuned' keys containing evaluation results
        """
        results = {
            'base': [],
            'finetuned': []
        }
        
        print(f"\n{'='*60}")
        print(f"Starting batch evaluation of {len(test_cases)} test cases")
        print(f"{'='*60}\n")
        
        for i, test_case in enumerate(test_cases, 1):
            question_id = test_case.get('id', f'Q{i}')
            user_question = test_case['question']
            
            print(f"\n[{i}/{len(test_cases)}] Processing: {question_id}")
            print(f"Question: {user_question[:100]}...")
            
            # Get responses from both models
            print(f"  Getting response from base model ({base_model})...")
            base_response = self.get_model_response(
                base_model, system_prompt, user_question
            )
            
            print(f"  Getting response from fine-tuned model ({finetuned_model})...")
            finetuned_response = self.get_model_response(
                finetuned_model, system_prompt, user_question
            )
            
            if not base_response or not finetuned_response:
                print(f"  Skipping {question_id} due to empty response")
                continue
            
            # Evaluate base model (using itself as reference)
            base_eval = self.evaluate_response(
                question_id=question_id,
                system_prompt=system_prompt,
                user_question=user_question,
                response=base_response,
                reference_response=base_response,
                model_name=base_model
            )
            results['base'].append(base_eval)
            
            # Evaluate fine-tuned model (using base as reference for semantic similarity)
            finetuned_eval = self.evaluate_response(
                question_id=question_id,
                system_prompt=system_prompt,
                user_question=user_question,
                response=finetuned_response,
                reference_response=base_response,
                model_name=finetuned_model
            )
            results['finetuned'].append(finetuned_eval)
        
        print(f"\n{'='*60}")
        print(f"Batch evaluation complete!")
        print(f"{'='*60}\n")
        
        return results
    
    def calculate_aggregate_metrics(
        self, 
        results: Dict[str, List[EvaluationResult]]
    ) -> pd.DataFrame:
        """
        Calculate aggregate statistics for all metrics
        
        Returns:
            DataFrame with mean, std, and statistical comparison
        """
        # Convert to DataFrames
        base_df = pd.DataFrame([asdict(r) for r in results['base']])
        finetuned_df = pd.DataFrame([asdict(r) for r in results['finetuned']])
        
        # Metrics to aggregate
        metrics = [
            'semantic_similarity', 'bert_score_f1', 'instruction_adherence',
            'response_relevance', 'coherence_score', 'completeness_score',
            'response_length', 'sentence_count', 'avg_sentence_length',
            'information_density', 'specificity_score'
        ]
        
        aggregate_data = []
        
        for metric in metrics:
            base_values = base_df[metric].values
            finetuned_values = finetuned_df[metric].values
            
            # Calculate statistics
            base_mean = np.mean(base_values)
            base_std = np.std(base_values)
            finetuned_mean = np.mean(finetuned_values)
            finetuned_std = np.std(finetuned_values)
            
            # Calculate improvement
            improvement = ((finetuned_mean - base_mean) / base_mean * 100) if base_mean != 0 else 0
            
            # Statistical significance test (paired t-test)
            if len(base_values) > 1:
                try:
                    t_stat, p_value = ttest_rel(finetuned_values, base_values)
                    significant = "Yes" if p_value < 0.05 else "No"
                except:
                    p_value = 1.0
                    significant = "N/A"
            else:
                p_value = 1.0
                significant = "N/A"
            
            aggregate_data.append({
                'Metric': metric,
                'Base Mean': base_mean,
                'Base Std': base_std,
                'Fine-tuned Mean': finetuned_mean,
                'Fine-tuned Std': finetuned_std,
                'Improvement (%)': improvement,
                'P-value': p_value,
                'Significant': significant
            })
        
        return pd.DataFrame(aggregate_data)
    
    def save_results(
        self, 
        results: Dict[str, List[EvaluationResult]], 
        aggregate_df: pd.DataFrame,
        output_dir: str = "evaluation_results"
    ):
        """Save evaluation results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        base_df = pd.DataFrame([asdict(r) for r in results['base']])
        finetuned_df = pd.DataFrame([asdict(r) for r in results['finetuned']])
        
        base_df.to_csv(f"{output_dir}/base_model_detailed.csv", index=False)
        finetuned_df.to_csv(f"{output_dir}/finetuned_model_detailed.csv", index=False)
        
        # Save aggregate metrics
        aggregate_df.to_csv(f"{output_dir}/aggregate_metrics.csv", index=False)
        
        # Save as JSON for easy reading
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump({
                'base': [asdict(r) for r in results['base']],
                'finetuned': [asdict(r) for r in results['finetuned']]
            }, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/")
        print(f"  - base_model_detailed.csv")
        print(f"  - finetuned_model_detailed.csv")
        print(f"  - aggregate_metrics.csv")
        print(f"  - results.json")


if __name__ == "__main__":
    print("Advanced Model Evaluator - Use this as a module")
    print("See example_evaluation.py for usage examples")

