import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import matplotlib.gridspec as gridspec


class CorrelationVisualizer:
    """
    Correlation coefficient statistical analysis visualizer

    Analyzes correlation data by scene and task, provides multiple visualization methods
    """

    def __init__(self, data_dict: Dict[str, Dict[str, List[List[float]]]],
                 sig_level: float = 0.05):
        """
        Initialize visualizer

        Args:
        ----------
        data_dict : Dict[str, Dict[str, List[List[float]]]]
            Nested dict with scene, task, and correlation data
            Format: {'scene_name': {'task_name': [[corr, p_value], ...]}}
        sig_level : float, default=0.05
            Significance level for determining significant correlations
        """
        self.data_dict = data_dict
        self.sig_level = sig_level
        self.df = self._prepare_dataframe()
        self.scenes = list(data_dict.keys())
        self.tasks = ['global', 'share', 'io', 'total']

    def _prepare_dataframe(self) -> pd.DataFrame:
        """Convert raw dict data to DataFrame format"""
        records = []

        for scene, task_dict in self.data_dict.items():
            for task, corr_pairs in task_dict.items():
                for corr, p_val in corr_pairs:
                    records.append({
                        'scene': scene,
                        'task': task,
                        'correlation': corr,
                        'p_value': p_val,
                        'is_sig': abs(p_val) < self.sig_level,
                        'abs_corr': abs(corr)
                    })

        return pd.DataFrame(records)

    def get_summary_stats(self) -> pd.DataFrame:
        """Get basic descriptive statistics"""
        summary = []

        for scene in self.scenes:
            for task in self.tasks:
                data = self.df[(self.df['scene'] == scene) & (self.df['task'] == task)]

                if len(data) > 0:
                    corr_vals = data['correlation'].values

                    stats = {
                        'scene': scene,
                        'task': task,
                        'n_samples': len(corr_vals),
                        'mean': np.mean(corr_vals),
                        'median': np.median(corr_vals),
                        'std': np.std(corr_vals),
                        'min': np.min(corr_vals),
                        'max': np.max(corr_vals),
                        'sig_ratio': np.mean(data['is_sig']) * 100
                    }

                    summary.append(stats)

        return pd.DataFrame(summary)

    def plot_violin(self, figsize=(15, 10), title="Correlation Distribution - Violin Plot"):
        """Plot violin plot for correlation distribution"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for idx, task in enumerate(self.tasks[:4]):
            ax = axes[idx]
            task_data = self.df[self.df['task'] == task]

            if len(task_data) > 0:
                plot_data = []
                labels = []

                for scene in self.scenes:
                    scene_data = task_data[task_data['scene'] == scene]['correlation']
                    if len(scene_data) > 0:
                        plot_data.append(scene_data.values)
                        labels.append(scene)

                if plot_data:
                    violin_parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)

                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(plt.cm.tab20c(idx / 4))
                        pc.set_alpha(0.7)

                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels, rotation=45, ha='right')
                    ax.set_ylabel('Correlation')
                    ax.set_title(f'Task: {task}')
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_histograms(self, figsize=(16, 12), title="Correlation Distribution - Histograms"):
        """Plot histogram grid by scene and task"""
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(len(self.scenes), len(self.tasks), hspace=0.3, wspace=0.3)

        for i, scene in enumerate(self.scenes):
            for j, task in enumerate(self.tasks):
                ax = plt.subplot(gs[i, j])

                data = self.df[(self.df['scene'] == scene) & (self.df['task'] == task)]

                if len(data) > 0:
                    corr_vals = data['correlation'].values

                    ax.hist(corr_vals, bins=15, alpha=0.7,
                            color=plt.cm.tab20c(j / 4),
                            edgecolor='black', density=True)

                    mean_val = np.mean(corr_vals)
                    ax.axvline(mean_val, color='green', linestyle='--', linewidth=2)
                    ax.axvline(0, color='red', linestyle='-', alpha=0.5)

                    sig_ratio = data['is_sig'].mean() * 100

                    ax.set_title(f'{scene} - {task}\nSig Ratio: {sig_ratio:.1f}%', fontsize=10)
                    ax.set_xlabel('Correlation')
                    ax.set_ylabel('Density')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([-1.1, 1.1])

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_sig_heatmap(self, figsize=(12, 8), title="Significant Sample Ratio Heatmap"):
        """Plot heatmap of significant sample ratios"""
        pivot = self.df.pivot_table(
            values='is_sig',
            index='scene',
            columns='task',
            aggfunc='mean'
        ) * 100

        pivot = pivot[self.tasks]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=1, linecolor='white', ax=ax,
                    cbar_kws={'label': 'Significant Ratio (%)'})

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Task')
        ax.set_ylabel('Scene')

        return fig

    def plot_comparison(self, figsize=(14, 10), title="Correlation Comparison Analysis"):
        """Plot comprehensive comparison analysis"""
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        # Boxplot by task
        ax1 = plt.subplot(gs[0, 0])
        sns.boxplot(data=self.df, x='task', y='correlation', ax=ax1)
        ax1.set_title('Correlation by Task')
        ax1.set_xlabel('Task')
        ax1.set_ylabel('Correlation')
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)

        # Boxplot by scene
        ax2 = plt.subplot(gs[0, 1])
        scene_order = sorted(self.scenes)
        sns.boxplot(data=self.df, x='scene', y='correlation', order=scene_order, ax=ax2)
        ax2.set_title('Correlation by Scene')
        ax2.set_xlabel('Scene')
        ax2.set_ylabel('Correlation')
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # Significant ratio bar chart
        ax3 = plt.subplot(gs[0, 2])
        sig_counts = self.df.groupby(['scene', 'task'])['is_sig'].mean().unstack() * 100
        sig_counts.plot(kind='bar', ax=ax3, alpha=0.8)
        ax3.set_title('Significant Ratio (%)')
        ax3.set_xlabel('Scene')
        ax3.set_ylabel('Significant Ratio (%)')
        ax3.legend(title='Task', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3, axis='y')

        # Density plots by task
        ax4 = plt.subplot(gs[1, :])
        for task in self.tasks:
            task_data = self.df[self.df['task'] == task]['correlation']
            if len(task_data) > 0:
                sns.kdeplot(task_data, label=task, ax=ax4, linewidth=2)

        ax4.set_title('Correlation Density by Task')
        ax4.set_xlabel('Correlation')
        ax4.set_ylabel('Density')
        ax4.legend(title='Task')
        ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([-1, 1])

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def generate_report(self, output_path: Optional[str] = None):
        """Generate comprehensive analysis report"""
        fig = plt.figure(figsize=(18, 24))
        gs = gridspec.GridSpec(4, 2, hspace=0.4, wspace=0.3)

        # Violin plot
        ax1 = plt.subplot(gs[0, 0])
        task_data = []
        task_labels = []

        for task in self.tasks:
            corr_vals = self.df[self.df['task'] == task]['correlation'].values
            if len(corr_vals) > 0:
                task_data.append(corr_vals)
                task_labels.append(task)

        violin_parts = ax1.violinplot(task_data, showmeans=True)
        ax1.set_xticks(range(1, len(task_labels) + 1))
        ax1.set_xticklabels(task_labels)
        ax1.set_ylabel('Correlation')
        ax1.set_title('Correlation by Task - Violin Plot')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # Significant ratio heatmap
        ax2 = plt.subplot(gs[0, 1])
        pivot = self.df.pivot_table(
            values='is_sig',
            index='scene',
            columns='task',
            aggfunc='mean'
        ) * 100

        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Significant Ratio Heatmap')

        # Boxplot by scene and task
        ax3 = plt.subplot(gs[1, :])
        scene_order = sorted(self.scenes)
        sns.boxplot(data=self.df, x='scene', y='correlation', hue='task',
                    order=scene_order, ax=ax3)
        ax3.set_title('Correlation by Scene and Task')
        ax3.set_xlabel('Scene')
        ax3.set_ylabel('Correlation')
        ax3.legend(title='Task', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)

        # Overall histogram
        ax4 = plt.subplot(gs[2, 0])
        all_corr = self.df['correlation'].values
        ax4.hist(all_corr, bins=30, alpha=0.7, color='steelblue',
                 edgecolor='black', density=True)

        ax4.set_title('Overall Correlation Distribution')
        ax4.set_xlabel('Correlation')
        ax4.set_ylabel('Density')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5)

        # Significant vs non-significant comparison
        ax5 = plt.subplot(gs[2, 1])
        sig_data = self.df[self.df['is_sig'] == True]['correlation']
        non_sig_data = self.df[self.df['is_sig'] == False]['correlation']

        bp = ax5.boxplot([sig_data.values, non_sig_data.values],
                         labels=['Significant', 'Non-Significant'],
                         patch_artist=True)

        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax5.set_title('Significant vs Non-Significant Comparison')
        ax5.set_ylabel('Correlation')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        # Summary table
        ax6 = plt.subplot(gs[3, :])
        ax6.axis('tight')
        ax6.axis('off')

        stats = self.get_summary_stats()
        overall = {
            'Total Samples': len(self.df),
            'Significant Samples': self.df['is_sig'].sum(),
            'Significant Ratio (%)': self.df['is_sig'].mean() * 100,
            'Mean Correlation': self.df['correlation'].mean(),
            'Correlation Std': self.df['correlation'].std(),
            'Min Correlation': self.df['correlation'].min(),
            'Max Correlation': self.df['correlation'].max(),
            'Median Correlation': self.df['correlation'].median()
        }

        table_data = [[k, f'{v:.3f}' if isinstance(v, float) else v]
                      for k, v in overall.items()]

        table = ax6.table(cellText=table_data,
                          colLabels=['Metric', 'Value'],
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.4, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        ax6.set_title('Overall Statistics', fontsize=12, fontweight='bold', y=0.95)

        plt.suptitle('Correlation Analysis Report', fontsize=20, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.show()

        return {
            'summary_stats': stats,
            'overall_stats': overall,
            'pivot_table': pivot
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    with open('mse.json', 'r') as f:
        sample_data = json.load(f)

    # Create visualizer
    visualizer = CorrelationVisualizer(sample_data, sig_level=0.05)

    # Get summary statistics
    summary = visualizer.get_summary_stats()
    print("Summary Statistics:")
    print(summary.to_string())

    # Generate visualizations
    visualizer.plot_violin()
    visualizer.plot_histograms()
    visualizer.plot_sig_heatmap()
    visualizer.plot_comparison()

    # Generate comprehensive report
    report = visualizer.generate_report('report.png')