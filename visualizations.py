"""
Visualization and Reporting Module for Model Evaluation
Creates CXO-ready charts and reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import Dict, List
from dataclasses import asdict
from model_evaluator import EvaluationResult

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class EvaluationVisualizer:
    """Create professional visualizations for CXO presentations"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'base': '#3498db',      # Blue
            'finetuned': '#2ecc71', # Green
            'accent': '#e74c3c'     # Red for highlights
        }
    
    def create_comparison_radar_chart(
        self, 
        aggregate_df: pd.DataFrame,
        filename: str = "radar_comparison.html"
    ):
        """
        Create interactive radar chart comparing models across all metrics
        """
        # Select key metrics for radar chart (normalized to 0-10 scale)
        metrics_for_radar = [
            'instruction_adherence', 'response_relevance', 
            'coherence_score', 'completeness_score', 'specificity_score'
        ]
        
        # Filter for these metrics
        df_radar = aggregate_df[aggregate_df['Metric'].isin(metrics_for_radar)].copy()
        
        # Create figure
        fig = go.Figure()
        
        # Add base model trace
        fig.add_trace(go.Scatterpolar(
            r=df_radar['Base Mean'].tolist() + [df_radar['Base Mean'].tolist()[0]],
            theta=df_radar['Metric'].tolist() + [df_radar['Metric'].tolist()[0]],
            fill='toself',
            name='Base Model',
            line_color=self.colors['base'],
            fillcolor=self.colors['base'],
            opacity=0.6
        ))
        
        # Add fine-tuned model trace
        fig.add_trace(go.Scatterpolar(
            r=df_radar['Fine-tuned Mean'].tolist() + [df_radar['Fine-tuned Mean'].tolist()[0]],
            theta=df_radar['Metric'].tolist() + [df_radar['Metric'].tolist()[0]],
            fill='toself',
            name='Fine-tuned Model',
            line_color=self.colors['finetuned'],
            fillcolor=self.colors['finetuned'],
            opacity=0.6
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True,
            title="Model Performance Comparison - Key Quality Metrics",
            font=dict(size=14)
        )
        
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved radar chart to {filepath}")
        
        return fig
    
    def create_metric_comparison_bars(
        self, 
        aggregate_df: pd.DataFrame,
        filename: str = "metric_comparison.html"
    ):
        """
        Create grouped bar chart comparing all metrics
        """
        # Prepare data
        metrics = aggregate_df['Metric'].tolist()
        base_means = aggregate_df['Base Mean'].tolist()
        finetuned_means = aggregate_df['Fine-tuned Mean'].tolist()
        
        fig = go.Figure()
        
        # Add bars for base model
        fig.add_trace(go.Bar(
            name='Base Model',
            x=metrics,
            y=base_means,
            marker_color=self.colors['base'],
            error_y=dict(
                type='data',
                array=aggregate_df['Base Std'].tolist(),
                visible=True
            )
        ))
        
        # Add bars for fine-tuned model
        fig.add_trace(go.Bar(
            name='Fine-tuned Model',
            x=metrics,
            y=finetuned_means,
            marker_color=self.colors['finetuned'],
            error_y=dict(
                type='data',
                array=aggregate_df['Fine-tuned Std'].tolist(),
                visible=True
            )
        ))
        
        fig.update_layout(
            title="Comprehensive Metric Comparison",
            xaxis_title="Metrics",
            yaxis_title="Score",
            barmode='group',
            xaxis_tickangle=-45,
            height=600,
            font=dict(size=12)
        )
        
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved bar chart to {filepath}")
        
        return fig
    
    def create_improvement_waterfall(
        self, 
        aggregate_df: pd.DataFrame,
        filename: str = "improvement_waterfall.html"
    ):
        """
        Create waterfall chart showing improvements
        """
        # Sort by improvement
        df_sorted = aggregate_df.sort_values('Improvement (%)', ascending=False)
        
        fig = go.Figure(go.Waterfall(
            name="Improvement",
            orientation="v",
            x=df_sorted['Metric'].tolist(),
            textposition="outside",
            text=[f"{x:+.1f}%" for x in df_sorted['Improvement (%)'].tolist()],
            y=df_sorted['Improvement (%)'].tolist(),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": self.colors['accent']}},
            increasing={"marker": {"color": self.colors['finetuned']}},
            totals={"marker": {"color": "rgb(128, 128, 128)"}}
        ))
        
        fig.update_layout(
            title="Performance Improvement by Metric (%)",
            xaxis_title="Metrics",
            yaxis_title="Improvement (%)",
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )
        
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved waterfall chart to {filepath}")
        
        return fig
    
    def create_statistical_significance_plot(
        self, 
        aggregate_df: pd.DataFrame,
        filename: str = "statistical_significance.png"
    ):
        """
        Create plot showing which improvements are statistically significant
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        df_plot = aggregate_df.copy()
        df_plot['Significant_num'] = df_plot['Significant'].apply(
            lambda x: 1 if x == 'Yes' else 0
        )
        df_plot = df_plot.sort_values('Improvement (%)', ascending=True)
        
        # Create horizontal bar chart
        colors = [self.colors['finetuned'] if sig == 1 else self.colors['base'] 
                  for sig in df_plot['Significant_num']]
        
        bars = ax.barh(df_plot['Metric'], df_plot['Improvement (%)'], color=colors)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        # Labels and title
        ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_title('Performance Improvement with Statistical Significance', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for i, (bar, val, sig) in enumerate(zip(bars, 
                                                  df_plot['Improvement (%)'], 
                                                  df_plot['Significant'])):
            label = f"{val:+.1f}%"
            if sig == 'Yes':
                label += " *"
            x_pos = val + (1 if val > 0 else -1)
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                   ha='left' if val > 0 else 'right', va='center', fontsize=9)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.colors['finetuned'], label='Statistically Significant (p<0.05)'),
            Patch(facecolor=self.colors['base'], label='Not Significant')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved significance plot to {filepath}")
        plt.close()
    
    def create_distribution_comparison(
        self, 
        results: Dict[str, List[EvaluationResult]],
        metric: str = 'instruction_adherence',
        filename: str = "distribution_comparison.html"
    ):
        """
        Create violin/box plot comparing distributions of a specific metric
        """
        # Prepare data
        base_values = [getattr(r, metric) for r in results['base']]
        finetuned_values = [getattr(r, metric) for r in results['finetuned']]
        
        fig = go.Figure()
        
        # Add violin plots
        fig.add_trace(go.Violin(
            y=base_values,
            name='Base Model',
            box_visible=True,
            meanline_visible=True,
            fillcolor=self.colors['base'],
            opacity=0.6,
            x0='Base Model'
        ))
        
        fig.add_trace(go.Violin(
            y=finetuned_values,
            name='Fine-tuned Model',
            box_visible=True,
            meanline_visible=True,
            fillcolor=self.colors['finetuned'],
            opacity=0.6,
            x0='Fine-tuned Model'
        ))
        
        fig.update_layout(
            title=f"Distribution Comparison - {metric.replace('_', ' ').title()}",
            yaxis_title="Score",
            showlegend=True,
            height=500
        )
        
        filepath = os.path.join(self.output_dir, filename.replace('.html', f'_{metric}.html'))
        fig.write_html(filepath)
        print(f"Saved distribution plot to {filepath}")
        
        return fig
    
    def create_correlation_heatmap(
        self, 
        results: Dict[str, List[EvaluationResult]],
        model_type: str = 'finetuned',
        filename: str = "correlation_heatmap.png"
    ):
        """
        Create correlation heatmap between different metrics
        """
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results[model_type]])
        
        # Select numeric columns
        numeric_cols = [
            'semantic_similarity', 'bert_score_f1', 'instruction_adherence',
            'response_relevance', 'coherence_score', 'completeness_score',
            'information_density', 'specificity_score'
        ]
        
        # Calculate correlation
        corr = df[numeric_cols].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr, 
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )
        
        ax.set_title(f'Metric Correlation Matrix - {model_type.title()} Model', 
                     fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename.replace('.png', f'_{model_type}.png'))
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved correlation heatmap to {filepath}")
        plt.close()
    
    def create_executive_summary_table(
        self, 
        aggregate_df: pd.DataFrame,
        filename: str = "executive_summary.html"
    ):
        """
        Create executive summary table with key metrics
        """
        # Select top metrics
        key_metrics = [
            'instruction_adherence', 'response_relevance', 
            'completeness_score', 'specificity_score'
        ]
        
        df_summary = aggregate_df[aggregate_df['Metric'].isin(key_metrics)].copy()
        df_summary = df_summary[[
            'Metric', 'Base Mean', 'Fine-tuned Mean', 
            'Improvement (%)', 'Significant'
        ]]
        
        # Format for display
        df_summary['Metric'] = df_summary['Metric'].str.replace('_', ' ').str.title()
        df_summary['Base Mean'] = df_summary['Base Mean'].round(2)
        df_summary['Fine-tuned Mean'] = df_summary['Fine-tuned Mean'].round(2)
        df_summary['Improvement (%)'] = df_summary['Improvement (%)'].round(1)
        
        # Create plotly table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_summary.columns),
                fill_color=self.colors['base'],
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[df_summary[col] for col in df_summary.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=11)
            )
        )])
        
        fig.update_layout(
            title="Executive Summary - Key Performance Metrics",
            height=400
        )
        
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved executive summary to {filepath}")
        
        return fig
    
    def create_all_visualizations(
        self, 
        results: Dict[str, List[EvaluationResult]], 
        aggregate_df: pd.DataFrame
    ):
        """
        Create all visualizations at once
        """
        print("\n" + "="*60)
        print("Creating Visualizations for CXO Presentation")
        print("="*60 + "\n")
        
        # 1. Radar chart
        print("1. Creating radar comparison chart...")
        self.create_comparison_radar_chart(aggregate_df)
        
        # 2. Bar chart
        print("2. Creating metric comparison bars...")
        self.create_metric_comparison_bars(aggregate_df)
        
        # 3. Waterfall chart
        print("3. Creating improvement waterfall...")
        self.create_improvement_waterfall(aggregate_df)
        
        # 4. Statistical significance
        print("4. Creating statistical significance plot...")
        self.create_statistical_significance_plot(aggregate_df)
        
        # 5. Distribution comparisons for key metrics
        key_metrics = ['instruction_adherence', 'response_relevance', 'completeness_score']
        for metric in key_metrics:
            print(f"5. Creating distribution comparison for {metric}...")
            self.create_distribution_comparison(results, metric)
        
        # 6. Correlation heatmaps
        print("6. Creating correlation heatmaps...")
        self.create_correlation_heatmap(results, 'base')
        self.create_correlation_heatmap(results, 'finetuned')
        
        # 7. Executive summary
        print("7. Creating executive summary table...")
        self.create_executive_summary_table(aggregate_df)
        
        print("\n" + "="*60)
        print("All visualizations created successfully!")
        print(f"Check the '{self.output_dir}' directory")
        print("="*60 + "\n")


if __name__ == "__main__":
    print("Visualization module - Import and use with evaluation results")

