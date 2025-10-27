# -*- coding: utf-8 -*-
"""
文件名: 04_feature_analysis.py
本脚本是数据处理流程的第四步，核心任务是根据分析结果进行特征选择，包括：
1.  根据预定义的列表，手动移除指定的特征。
2.  对特征选择后的数据，重新进行一次完整的EDA，以验证特征效果。
3.  保存一份最终可用于模型训练的、干净的数据集。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. 全局配置区 ---

# --- 【输入路径配置】 ---
INPUT_DIR = 'results'
BASE_DATA_FILE = '03_merged_data.xlsx' 

# --- 【输出路径与文件配置】 ---
RESULTS_DIR = 'results'
FINAL_DATA_FILE = '04_data_selected.xlsx'
EDA_PLOT_DIR = '04_eda_plots'
CORR_HEATMAP_FILE = '04_correlation_heatmap.png'
CORR_MATRIX_FILE = '04_correlation_matrix.csv'
VIF_SCORES_FILE = '04_vif_scores.csv'

# --- 【列定义】 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]
CHEMISTRY_COLUMNS = [
    "碳含量", "硅含量", "锰含量", "磷含量", "硫含量", "铝含量",
    "钛含量", "铌含量", "氧含量", "氮含量"
]

# --- 【核心配置：特征删除与EDA控制】 ---

# 1. EDA 子开关：是否生成分布直方图
ENABLE_HISTOGRAM_PLOTS = True

# 2. EDA 子开关：是否生成散点图
ENABLE_SCATTER_PLOTS = True

# 3. 根据03代码生成的直方图和vif的分析结果，在此处手动配置要删除的特征列表
FEATURES_TO_REMOVE = [
    '升温起始温度差', '降温结束温度差', '冷点_升温起始温度', '冷点_升温结束温度',
    '热点_升温幅度', '热点_升温起始温度', '热点_升温结束温度', '热点_平均升温速率',
    '冷点_平均升温速率', '峰值温度差', '保温起始温度差', '保温结束温度差',
    '热点_保温起始温度', '冷点_保温起始温度', '热点_保温结束温度', '冷点_保温结束温度',
    '冷点_保温温度标准差', '冷点_降温起始温度', '热点_降温速率标准差', '热点_降温结束温度',
    '热点_保温平均温度', '冷点_保温平均温度', '热点_降温起始温度', '冷点_降温结束温度',
    '冷点_降温幅度', '降温时长', '平均降温速率差', '降温时长占比', '升温时长占比','热点_最大瞬时降温速率',
    '保温时长占比', '冷点_保温阶段温降', '热点_保温阶段温降', '降温起始温度差','工艺结束时刻',
    '升温结束温度差', '总工艺时长','升温开始时刻','保温开始时刻','降温开始时刻','热点_峰值温度时刻','冷点_峰值温度时刻','峰值时刻差'
]

# --- 【绘图设置】 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# --- 2. 辅助与核心分析函数 ---

def sanitize_filename(filename: str) -> str:
    """移除文件名中的非法字符。"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def analyze_final_correlations(df: pd.DataFrame, output_dir: str):
    """计算特征选择后的相关性矩阵和热力图。"""
    print("--- 步骤 3.1/4: 重新进行相关性分析 ---")
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    corr_matrix.to_csv(os.path.join(output_dir, CORR_MATRIX_FILE), encoding='utf-8-sig')
    
    plt.figure(figsize=(30, 24))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 7})
    plt.title('最终特征相关性热力图 (特征选择后)', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, CORR_HEATMAP_FILE), dpi=150)
    plt.close()
    print(f"    [成功] 特征选择后的相关性热力图和矩阵已保存。")

def analyze_final_vif(df: pd.DataFrame, output_dir: str):
    """计算特征选择后的VIF分数。"""
    print("--- 步骤 3.2/4: 重新进行VIF分析 ---")
    features_to_check = [col for col in df.columns if col not in [ID_COLUMN] + PERFORMANCE_METRICS]
    features_df = df[features_to_check].select_dtypes(include=['number']).dropna()
    
    if features_df.shape[1] < 2:
        print("    [警告] 数值型输入特征不足，无法计算VIF。")
        return

    print(f"    将在 {features_df.shape[1]} 个最终输入特征上计算VIF...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features_df.columns
    vif_data["VIF"] = [variance_inflation_factor(features_df.values, i) for i in range(features_df.shape[1])]
    vif_data_sorted = vif_data.sort_values('VIF', ascending=False)
    
    print("\n    --- VIF 分数报告 (特征选择后) ---")
    print(vif_data_sorted.to_string(index=False))
    
    vif_data_sorted.to_csv(os.path.join(output_dir, VIF_SCORES_FILE), index=False, encoding='utf-8-sig')
    print(f"\n    [成功] 特征选择后的VIF分数已保存。")

def perform_final_eda_plotting(df: pd.DataFrame, plot_dir: str):
    """生成特征选择后的分布直方图和散点图。"""
    print("--- 步骤 3.3/4: 生成最终EDA图表 ---")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # 生成分布直方图
    if ENABLE_HISTOGRAM_PLOTS:
        print("    正在生成分布直方图...")
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'特征分布 - {col}')
            plt.savefig(os.path.join(plot_dir, f'hist_selected_{sanitize_filename(col)}.png'))
            plt.close()
        print("    [成功] 分布直方图已生成。")
    else:
        print("    [已跳过] 子开关 ENABLE_HISTOGRAM_PLOTS=False，跳过分布直方图生成。")

    # 生成散点图
    if ENABLE_SCATTER_PLOTS:
        input_features = [col for col in numeric_cols if col not in [ID_COLUMN] + PERFORMANCE_METRICS]
        print("    正在生成输入特征与性能指标的散点图...")
        for feature in input_features:
            for metric in PERFORMANCE_METRICS:
                if metric in df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=df, x=feature, y=metric, alpha=0.6)
                    plt.title(f'关系探索 - {feature} vs {metric}')
                    plt.savefig(os.path.join(plot_dir, f'scatter_selected_{sanitize_filename(feature)}_vs_{sanitize_filename(metric)}.png'))
                    plt.close()
        print("    [成功] 散点图已生成。")
    else:
        print("    [已跳过] 子开关 ENABLE_SCATTER_PLOTS=False，跳过散点图生成。")
    
    print(f"    所有最终EDA图表已保存至: {plot_dir}")


# --- 3. 主程序入口 ---
def main():
    """主程序，执行特征删除和最终的EDA。"""
    print("=" * 60)
    print("--- 启动脚本: 04_feature_analysis.py ---")
    print("=" * 60)

    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    
    input_file_path = os.path.join(INPUT_DIR, BASE_DATA_FILE)
    print(f"--- 步骤 0/4: 加载数据文件: {input_file_path} ---")
    if not os.path.exists(input_file_path):
        print(f"    [错误] 数据文件不存在。请先运行 '03_merge_clean_and_eda.py'。")
        return
    df = pd.read_excel(input_file_path)
    print(f"    加载成功，数据集包含 {len(df)} 行和 {len(df.columns)} 列。")

    print("\n--- 步骤 1/4: 执行手动特征删除 ---")
    existing_features_to_remove = [col for col in FEATURES_TO_REMOVE if col in df.columns]
    df_selected = df.drop(columns=existing_features_to_remove)
    print(f"    [成功] 移除了 {len(existing_features_to_remove)} 个指定的特征。")
    print(f"    当前数据集剩余 {len(df_selected.columns)} 个特征。")
    
    print("\n--- 步骤 2/4: 保存特征选择后的数据集 ---")
    output_final_data_path = os.path.join(RESULTS_DIR, FINAL_DATA_FILE)
    df_selected.to_excel(output_final_data_path, index=False)
    print(f"    [成功] 特征选择后的数据已保存至: {output_final_data_path}")

    print("\n--- 步骤 3/4: 对特征选择后的数据进行最终的EDA ---")
    final_plot_dir = os.path.join(RESULTS_DIR, EDA_PLOT_DIR)
    analyze_final_correlations(df_selected, RESULTS_DIR)
    analyze_final_vif(df_selected, RESULTS_DIR)
    perform_final_eda_plotting(df_selected, final_plot_dir)

    print("\n" + "=" * 60)
    print("--- 步骤4全部任务完成 ---")
    print(f"最终数据集包含 {len(df_selected)} 行和 {len(df_selected.columns)} 列。")
    print("=" * 60)

if __name__ == '__main__':
    main()