import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class CorrelationAnalyzer:
    def __init__(self, json_path):
        """Initialize analyzer and load JSON data"""
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self):
        """Convert JSON data to DataFrame format"""
        records = []
        for scene_name, scene_data in self.data.items():
            for corr_type in ['p_global', 'p_share', 'p_io']:
                if corr_type in scene_data:
                    for i, (corr, p_value) in enumerate(scene_data[corr_type]):
                        records.append({
                            'scene': scene_name,
                            'correlation_type': corr_type,
                            'correlation': corr,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'correlation_strength': self._classify_correlation_strength(corr)
                        })
        return pd.DataFrame(records)

    def _classify_correlation_strength(self, corr):
        """Classify correlation strength"""
        abs_corr = corr
        if corr >= 0.8:
            return 'Very Strong'
        elif corr >= 0.6:
            return 'Strong'
        elif corr >= 0.4:
            return 'Moderate'
        elif corr >= 0.2:
            return 'Weak'
        elif corr>=0:
            return 'Very Weak'
        else:
            return 'Negative'

    def positive_correlation_analysis(self):
        """Focus on positive correlations only"""
        positive_df = self.df[self.df['correlation'] > 0]

        if len(positive_df) == 0:
            print("No positive correlations found.")
            return None

        print("=" * 60)
        print("           POSITIVE CORRELATION ANALYSIS")
        print("=" * 60)

        # Basic statistics
        total_samples = len(self.df)
        positive_samples = len(positive_df)
        positive_ratio = positive_samples / total_samples * 100

        print(f"Total samples: {total_samples}")
        print(f"Positive correlation samples: {positive_samples} ({positive_ratio:.1f}%)")
        print(f"Average positive correlation: {positive_df['correlation'].mean():.3f}")
        print(f"Std of positive correlations: {positive_df['correlation'].std():.3f}")

        # Significance in positive correlations
        significant_positive = positive_df['significant'].sum()
        significant_ratio = significant_positive / positive_samples * 100
        print(
            f"Significant positive correlations: {significant_positive}/{positive_samples} ({significant_ratio:.1f}%)")

        # By scene analysis
        print("\nPositive correlations by scene:")
        scene_stats = positive_df.groupby('scene').agg({
            'correlation': ['count', 'mean', 'std'],
            'significant': 'mean'
        }).round(3)
        print(scene_stats)

        # By type analysis
        print("\nPositive correlations by type:")
        type_stats = positive_df.groupby('correlation_type').agg({
            'correlation': ['count', 'mean', 'std'],
            'significant': 'mean'
        }).round(3)
        print(type_stats)

        return positive_df

    def comprehensive_analysis(self):
        """Perform comprehensive statistical analysis"""
        print("=" * 60)
        print("           CORRELATION DATA ANALYSIS REPORT")
        print("=" * 60)

        # Overall statistics
        self._overall_statistics()
        print()

        # Scene-based statistics
        self._scene_based_analysis()
        print()

        # Type-based statistics
        self._type_based_analysis()
        print()

        # Significance analysis
        self._significance_analysis()
        print()

        # Positive correlation analysis
        self.positive_correlation_analysis()

    def _overall_statistics(self):
        """Overall statistical analysis"""
        print(" Overall Statistics:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Number of scenes: {self.df['scene'].nunique()}")
        print(f"  Average correlation: {self.df['correlation'].mean():.3f} ± {self.df['correlation'].std():.3f}")
        print(f"  Median correlation: {self.df['correlation'].median():.3f}")
        print(f"  Range: [{self.df['correlation'].min():.3f}, {self.df['correlation'].max():.3f}]")

        # Correlation strength distribution
        strength_counts = self.df['correlation_strength'].value_counts()
        print("  Correlation strength distribution:")
        for strength, count in strength_counts.items():
            percentage = count / len(self.df) * 100
            print(f"    {strength}: {count} ({percentage:.1f}%)")

    def _scene_based_analysis(self):
        """Scene-based statistical analysis"""
        print(" Scene-based Statistics:")
        scene_stats = self.df.groupby('scene').agg({
            'correlation': ['count', 'mean', 'std', 'min', 'max'],
            'significant': 'mean'
        }).round(3)

        # Rename columns
        scene_stats.columns = ['sample_count', 'mean', 'std', 'min', 'max', 'significant_ratio']
        print(scene_stats)

        # Find best and worst scenes
        best_scene = self.df.groupby('scene')['correlation'].mean().idxmax()
        worst_scene = self.df.groupby('scene')['correlation'].mean().idxmin()
        print(
            f"  Best performing scene: {best_scene} (mean r={self.df.groupby('scene')['correlation'].mean().max():.3f})")
        print(
            f"  Worst performing scene: {worst_scene} (mean r={self.df.groupby('scene')['correlation'].mean().min():.3f})")

    def _type_based_analysis(self):
        """Correlation type-based statistical analysis"""
        print(" Type-based Statistics:")
        type_stats = self.df.groupby('correlation_type').agg({
            'correlation': ['count', 'mean', 'std', 'min', 'max'],
            'significant': 'mean'
        }).round(3)

        type_stats.columns = ['sample_count', 'mean', 'std', 'min', 'max', 'significant_ratio']
        print(type_stats)

        # Test differences between types
        types = self.df['correlation_type'].unique()
        if len(types) > 1:
            print("  Type difference test (ANOVA):")
            anova_data = [self.df[self.df['correlation_type'] == t]['correlation'] for t in types]
            f_stat, p_value = stats.f_oneway(*anova_data)
            print(f"  F={f_stat:.3f}, p={p_value:.3f}")

    def _significance_analysis(self):
        """Significance analysis"""
        print(" Significance Analysis:")
        total_samples = len(self.df)
        significant_samples = self.df['significant'].sum()
        significance_rate = significant_samples / total_samples * 100

        print(f"  Significant samples: {significant_samples}/{total_samples} ({significance_rate:.1f}%)")
        print(
            f"  Non-significant samples: {total_samples - significant_samples}/{total_samples} ({(100 - significance_rate):.1f}%)")

        # Significance by scene
        scene_significance = self.df.groupby('scene')['significant'].mean().sort_values(ascending=False)
        print("  Significance ratio by scene (top 5):")
        for scene, rate in scene_significance.head(5).items():
            print(f"    {scene}: {rate:.1%}")

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 10))

        plt.suptitle('8x8 patch layout', fontsize=16, fontweight='bold')
        plt.subplot(2, 3, 1)
        self._plot_overall_distribution()

        plt.subplot(2, 3, 2)
        self._plot_scene_boxplot()

        plt.subplot(2, 3, 3)
        self._plot_type_boxplot()

        plt.subplot(2, 3, 4)
        self._plot_correlation_strength()

        plt.subplot(2, 3, 5)
        self._plot_positive_correlation_analysis()

        plt.subplot(2, 3, 6)
        self._plot_significance_pie()

        plt.tight_layout()
        plt.show()

    def _plot_overall_distribution(self):
        """Plot overall distribution histogram"""
        plt.hist(self.df['correlation'], bins=30, alpha=0.7, edgecolor='black', color='skyblue')
        plt.axvline(self.df['correlation'].mean(), color='red', linestyle='--',
                    label=f'Mean: {self.df["correlation"].mean():.3f}')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.title('Overall Correlation Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_scene_boxplot(self):
        """Plot scene-based boxplot"""
        # Select scenes with sufficient samples
        scene_counts = self.df['scene'].value_counts()
        top_scenes = scene_counts[scene_counts >= 5].index.tolist()
        filtered_df = self.df[self.df['scene'].isin(top_scenes)]

        if len(top_scenes) > 0:
            scene_means = filtered_df.groupby('scene')['correlation'].mean().sort_values()
            ordered_scenes = scene_means.index.tolist()

            box_data = [filtered_df[filtered_df['scene'] == scene]['correlation'] for scene in ordered_scenes]
            plt.boxplot(box_data, labels=ordered_scenes, vert=False)
            plt.xlabel('Correlation Coefficient')
            plt.title('Correlation Distribution by Scene')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
            plt.title('Scene Distribution (Insufficient Data)')

    def _plot_type_boxplot(self):
        """Plot type-based boxplot"""
        type_data = [self.df[self.df['correlation_type'] == t]['correlation']
                     for t in self.df['correlation_type'].unique()]

        plt.boxplot(type_data, labels=self.df['correlation_type'].unique())
        plt.ylabel('Correlation Coefficient')
        plt.title('Correlation Comparison by Type')
        plt.grid(True, alpha=0.3)

        # Add mean markers
        for i, corr_type in enumerate(self.df['correlation_type'].unique(), 1):
            mean_val = self.df[self.df['correlation_type'] == corr_type]['correlation'].mean()
            plt.text(i, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')

    def _plot_significance_pie(self):
        """Plot significance ratio pie chart"""
        positive_df = self.df[self.df['correlation'] > 0]
        significant_count = positive_df['significant'].sum()
        non_significant_count = len(positive_df) - significant_count

        sizes = [significant_count, non_significant_count]
        labels = [f'Significant\n{significant_count}', f'Non-significant\n{non_significant_count}']
        colors = ['lightcoral', 'lightblue']

        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Statistical Significance Ratio for Positive samples')

    def _plot_correlation_strength(self):
        """Plot correlation strength distribution"""
        strength_counts = self.df['correlation_strength'].value_counts()

        # Order by strength
        strength_order = ['Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak', 'Negative']
        strength_counts = strength_counts.reindex(strength_order, fill_value=0)

        colors = ['darkred', 'red', 'orange', 'lightblue', 'lightgray', 'gray']
        bars = plt.bar(strength_counts.index, strength_counts.values, color=colors, alpha=0.7)

        plt.xlabel('Correlation Strength')
        plt.ylabel('Sample Count')
        plt.title('Correlation Strength Distribution')
        plt.xticks(rotation=45)

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')
        plt.grid(True, alpha=0.3)

    def _plot_positive_correlation_analysis(self):
        """Visualize positive correlation analysis"""
        positive_df = self.df[self.df['correlation'] > 0]

        if len(positive_df) == 0:
            plt.text(0.5, 0.5, 'No Positive Correlations', ha='center', va='center')
            plt.title('Positive Correlation Analysis')
            return

        # Plot positive correlation distribution
        plt.hist(positive_df['correlation'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.axvline(positive_df['correlation'].mean(), color='red', linestyle='--',
                    label=f'Mean: {positive_df["correlation"].mean():.3f}')
        plt.xlabel('Positive Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.title('Positive Correlation Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add text box with statistics
        positive_ratio = len(positive_df) / len(self.df) * 100
        significant_positive = positive_df['significant'].sum()
        textstr = f'Positive samples: {len(positive_df)}\nPositive ratio: {positive_ratio:.1f}%\nSignificant: {significant_positive}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)


# Usage example
if __name__ == "__main__":
    # Initialize analyzer (replace with your JSON file path)
    correlation = 'spearman'
    dens_error_type = 'mse'
    recon_error_type = 'mse'
    mask_layout = (8, 8)
    analyzer = CorrelationAnalyzer(f'{correlation}_dens-{dens_error_type}-vs-recon-{recon_error_type}_correlation_patch{mask_layout[0]}{mask_layout[1]}.json')

    # Perform comprehensive analysis (includes positive correlation analysis)
    analyzer.comprehensive_analysis()

    # Create visualizations
    analyzer.create_visualizations()