import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class CorrelationDataProcessor:
    """处理相关系数数据格式，转换为Seaborn友好的格式"""

    @staticmethod
    def parse_correlation_dict(data_dict: Dict) -> pd.DataFrame:
        """
        解析复杂的嵌套字典结构：
        {
            'scene_name1': {
                'global': [(pearson1, p1), (pearson2, p2), ...],
                'share': [(pearson1, p1), ...],
                'in_out': [(pearson1, p1), ...]
            },
            'scene_name2': {...}
        }
        """
        records = []

        for scene_name, channels in data_dict.items():
            for channel_name, samples in channels.items():
                for i, (pearson, p_value) in enumerate(samples):
                    records.append({
                        'scene': scene_name,
                        'channel': channel_name,
                        'pearson': pearson,
                        'p_value': p_value,
                        'sample_id': f"{scene_name}_{channel_name}_{i}",
                        'is_significant': p_value < 0.05,
                        'significance': 'significant' if p_value < 0.05 else 'non-significant'
                    })

        return pd.DataFrame(records)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class CorrelationVisualizer:
    """简化的可视化类"""

    def __init__(self):
        # 设置基本样式，Seaborn会自动处理颜色
        sns.set_style("whitegrid")
        # Set2是一个很好的分类调色板，Seaborn会自动为不同类别分配颜色
        self.palette = "Set2"

    def load_data(self, data_dict):
        """直接转换数据为DataFrame，Seaborn可以直接使用"""
        records = []

        for scene_name, channels in data_dict.items():
            for channel_name, samples in channels.items():
                for i, (pearson, p_value) in enumerate(samples):
                    records.append({
                        'scene': scene_name,
                        'channel': channel_name,
                        'pearson': pearson,
                        'p_value': p_value,
                        'is_sig': p_value < 0.05  # 添加显著性标记
                    })

        self.df = pd.DataFrame(records)
        print(f"加载了 {len(self.df)} 条数据")
        print(f"场景: {self.df['scene'].unique().tolist()}")
        print(f"通道: {self.df['channel'].unique().tolist()}")
        return self.df

    def plot_violin_by_scene(self, figsize=(12, 6)):
        """按场景绘制小提琴图 - 简化的Seaborn用法"""
        if not hasattr(self, 'df'):
            print("请先使用load_data()加载数据")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Seaborn的核心用法：直接传入DataFrame和列名
        # x: 场景列，y: 数值列，hue: 分组列
        sns.violinplot(
            data=self.df,  # DataFrame数据
            x='scene',  # x轴是场景
            y='pearson',  # y轴是相关系数
            hue='channel',  # 用颜色区分通道
            palette=self.palette,  # 调色板
            inner='quartile',  # 内部显示四分位数
            ax=ax
        )

        # 添加标题和标签
        ax.set_title('Correlation Coefficients by Scene', fontsize=14)
        ax.set_xlabel('Scene')
        ax.set_ylabel('Spearman Correlation')

        # 添加0线
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 自动调整布局
        plt.tight_layout()

        return fig, ax

    def plot_significant_violin_by_scene(self, figsize=(12, 6)):
        """只绘制显著样本（p < 0.05）的小提琴图"""
        if not hasattr(self, 'df'):
            print("请先使用load_data()加载数据")
            return

        # 筛选显著样本
        sig_df = self.df[self.df['is_sig']].copy()

        if len(sig_df) == 0:
            print("警告：没有显著样本（p < 0.05）")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制显著样本的小提琴图
        sns.violinplot(
            data=sig_df,
            x='scene',
            y='pearson',
            hue='channel',
            palette=self.palette,
            inner='quartile',
            ax=ax
        )

        # 添加标题和标签
        ax.set_title('Significant Correlation Coefficients by Scene (p < 0.05)', fontsize=14)
        ax.set_xlabel('Scene')
        ax.set_ylabel('Spearman Correlation')

        # 添加0线
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # 自动调整布局
        plt.tight_layout()

        return fig, ax

    def plot_significance_pie_charts(self, figsize=(15, 5)):
        """绘制三个饼图，分别显示每个通道的显著样本占比"""
        if not hasattr(self, 'df'):
            print("请先使用load_data()加载数据")
            return

        # 创建子图
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        # 获取所有通道
        channels = self.df['channel'].unique()

        # 确保有三个通道
        # if len(channels) != 3:
        #     print(f"警告：数据包含 {len(channels)} 个通道，但预期是3个")
        #     # 只绘制实际存在的通道
        #     fig.delaxes(axes[2])
        #     if len(channels) < 2:
        #         fig.delaxes(axes[1])

        for idx, channel in enumerate(channels):  # 最多处理3个通道
            # if idx >= 3:  # 如果超过3个通道，跳出循环
            #     break

            # 筛选当前通道的数据
            channel_data = self.df[self.df['channel'] == channel]

            # 计算显著和非显著样本数量
            sig_count = channel_data['is_sig'].sum()
            nonsig_count = len(channel_data) - sig_count

            # 数据
            sizes = [sig_count, nonsig_count]
            labels = ['Significant', 'Non-significant']

            # 颜色设置
            colors = ['#ff6b6b', '#c8d6e5']  # 红色表示显著，浅蓝色表示非显著

            # 绘制饼图
            axes[idx].pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.05, 0)  # 突出显著部分
            )

            # 设置标题
            axes[idx].set_title(f'Channel: {channel}\nTotal: {len(channel_data)} samples', fontsize=12)

            # 添加图例
            axes[idx].legend(loc='upper right')

        # 主标题
        fig.suptitle('Significance Distribution by Channel (p < 0.05)', fontsize=16, fontweight='bold')

        # 自动调整布局
        plt.tight_layout()

        return fig, axes

# class CorrelationVisualizer:
#     """基础可视化类"""
#
#     def __init__(self, data_dict: Dict = None):
#         self.df = None
#         if data_dict:
#             self.load_data(data_dict)
#
#         # 设置Seaborn样式
#         sns.set_style("whitegrid")
#         # 使用Seaborn默认调色板 - 它会自动分配颜色
#         self.palette = "Set2"  # 这是一个不错的分类调色板
#
#     def load_data(self, data_dict: Dict):
#         """加载数据"""
#         self.df = CorrelationDataProcessor.parse_correlation_dict(data_dict)
#         print(f"数据加载完成，共 {len(self.df)} 条记录")
#         print(f"场景: {self.df['scene'].unique().tolist()}")
#         print(f"通道: {self.df['channel'].unique().tolist()}")
#
#     def plot_violin_by_scene(self, figsize: Tuple = (12, 6)):
#         """
#         按场景绘制小提琴图，所有场景并排显示
#         每个场景内显示三个通道的分布
#         """
#         if self.df is None:
#             print("请先加载数据")
#             return
#
#         fig, ax = plt.subplots(figsize=figsize)
#
#         # 使用Seaborn绘制小提琴图
#         # x轴: scene (场景)
#         # y轴: pearson (相关系数)
#         # hue: channel (通道)，Seaborn会自动用不同颜色区分
#         # split: True 会让小提琴图分成两半显示，更节省空间
#         # inner: 显示内部统计量
#
#         sns.violinplot(
#             data=self.df,
#             x='scene',
#             y='pearson',
#             hue='channel',
#             palette=self.palette,
#             split=True,  # 将小提琴分成两半，不同通道并排
#             inner='quartile',  # 显示四分位线
#             ax=ax
#         )
#
#         # 添加标题和标签
#         ax.set_title('Correlation Coefficients by Scene and Channel', fontsize=14, fontweight='bold')
#         ax.set_xlabel('Scene', fontsize=12)
#         ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12)
#
#         # 添加水平参考线
#         ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
#
#         # 调整图例位置，避免遮挡图形
#         ax.legend(title='Channel', bbox_to_anchor=(1.05, 1), loc='upper left')
#
#         # 自动调整布局
#         plt.tight_layout()
#
#         return fig, ax


# 创建示例数据（模拟你的数据结构）
def create_sample_data():
    """创建示例数据用于测试"""
    return {
        'Scene_A': {
            'global': [(0.85, 0.001), (0.78, 0.003), (0.92, 0.0005)],
            'share': [(0.45, 0.032), (0.32, 0.045), (0.51, 0.028)],
            'in_out': [(-0.23, 0.078), (-0.31, 0.024), (-0.18, 0.091)]
        },
        'Scene_B': {
            'global': [(0.72, 0.002), (0.68, 0.004), (0.81, 0.001)],
            'share': [(0.38, 0.021), (0.41, 0.018), (0.35, 0.025)],
            'in_out': [(-0.28, 0.035), (-0.22, 0.042), (-0.31, 0.019)]
        },
        'Scene_C': {
            'global': [(0.88, 0.0003), (0.79, 0.002), (0.85, 0.001)],
            'share': [(0.52, 0.015), (0.48, 0.022), (0.55, 0.011)],
            'in_out': [(-0.19, 0.067), (-0.25, 0.038), (-0.21, 0.049)]
        }
    }


# 使用示例
if __name__ == "__main__":
    # 1. 创建示例数据
    # with open('spearman_dens-mse-vs-recon-mse_correlation_patch88.json', 'r') as f:
    #     sample_data = json.load(f)
    with open('mae.json', 'r') as f:
        sample_data = json.load(f)
    # sample_data =
    data = {}
    for scene in sample_data.keys():
        main_scene = scene.split('/')[0]
        data[main_scene] = sample_data[scene]
    # 2. 创建可视化器并加载数据
    viz = CorrelationVisualizer()
    viz.load_data(data)

    # 3. 绘制小提琴图
    fig, ax = viz.plot_violin_by_scene(figsize=(14, 7))
    viz.plot_significance_pie_charts()
    viz.plot_significant_violin_by_scene()

    # 4. 保存图表（可选）
    fig.savefig('violin_by_scene.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'violin_by_scene.png'")

    # 5. 显示图表
    plt.show()