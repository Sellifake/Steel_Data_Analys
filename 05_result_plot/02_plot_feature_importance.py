# -*- coding: utf-8 -*-
"""
文件名: 02_plot_feature_importance.py
功能: 
    1. 读取所有模型的特征重要性CSV文件
    2. 统计前5个重要特征的名字及出现次数
    3. 为每个性能指标绘制特征重要性对比图
    4. 保存到'results/特征重要性'文件夹
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
INPUT_DIR = '特征重要性'
OUTPUT_DIR = 'results/特征重要性'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]
MODEL_NAMES = {
    'random_frost': '随机森林',
    'XGBoost_Tuned': 'XGBoost',
    'LightGBM_Tuned': 'LightGBM', 
    '01_GPR_Tuned': '高斯过程回归',
    '02_SVR_Tuned': '支持向量回归',
    '01a_TabNet': 'TabNet'
}

def load_feature_importance_data():
    """加载所有模型的特征重要性数据"""
    all_data = {}
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # 解析文件名获取模型名和性能指标
        parts = filename.replace('.csv', '').split('_')
        
        # 确定模型名
        if 'random_frost' in filename:
            model_name = 'random_frost'
            metric_part = filename.replace('random_frost_', '').replace('.csv', '')
        elif 'XGBoost_Tuned' in filename:
            model_name = 'XGBoost_Tuned'
            metric_part = filename.replace('XGBoost_Tuned_', '').replace('.csv', '')
        elif 'LightGBM_Tuned' in filename:
            model_name = 'LightGBM_Tuned'
            metric_part = filename.replace('LightGBM_Tuned_', '').replace('.csv', '')
        elif '01_GPR_Tuned' in filename:
            model_name = '01_GPR_Tuned'
            metric_part = filename.replace('01_GPR_Tuned_', '').replace('.csv', '')
        elif '02_SVR_Tuned' in filename:
            model_name = '02_SVR_Tuned'
            metric_part = filename.replace('02_SVR_Tuned_', '').replace('.csv', '')
        elif '01a_TabNet' in filename:
            model_name = '01a_TabNet'
            metric_part = filename.replace('01a_TabNet_', '').replace('.csv', '')
        else:
            continue
        
        # 确定性能指标
        if '抗拉强度' in metric_part:
            metric = '抗拉强度'
        elif '屈服Rp0.2值' in metric_part:
            metric = '屈服Rp0.2值*'
        elif '断后伸长率' in metric_part:
            metric = '断后伸长率'
        else:
            continue
        
        # 读取数据
        try:
            df = pd.read_csv(file_path)
            print(f"处理文件: {filename}")
            print(f"列名: {df.columns.tolist()}")
            
            # 处理不同的列名格式
            importance_col = None
            if 'Importance' in df.columns:
                importance_col = 'Importance'
            elif 'Importance (TabNet)' in df.columns:
                importance_col = 'Importance (TabNet)'
            elif 'Importance (Permutation)' in df.columns:
                importance_col = 'Importance (Permutation)'
            
            if importance_col and 'Feature' in df.columns:
                # 按重要性排序，取前5个
                df_sorted = df.sort_values(importance_col, ascending=False).head(5)
                
                key = f"{model_name}_{metric}"
                all_data[key] = {
                    'model': model_name,
                    'metric': metric,
                    'top_features': df_sorted['Feature'].tolist(),
                    'importances': df_sorted[importance_col].tolist()
                }
                print(f"成功加载: {model_name} - {metric}, 前5特征: {df_sorted['Feature'].tolist()}")
            else:
                print(f"文件 {filename} 缺少必要的列")
                
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            continue
    
    return all_data

def analyze_top_features(all_data):
    """分析前5个重要特征的统计信息"""
    feature_counts = {}
    
    print(f"\n=== 开始分析特征统计 ===")
    print(f"总共加载了 {len(all_data)} 个模型-指标组合")
    
    for metric in PERFORMANCE_METRICS:
        feature_counts[metric] = Counter()
        metric_models = []
        all_features_list = []  # 存储所有特征，包括重复的
        
        # 收集该性能指标下所有模型的前5个特征
        for key, data in all_data.items():
            if data['metric'] == metric:
                metric_models.append(data['model'])
                print(f"  {metric}: 找到模型 {data['model']}, 前5特征: {data['top_features']}")
                for feature in data['top_features']:
                    feature_counts[metric][feature] += 1
                    all_features_list.append(feature)  # 添加到总列表中
        
        print(f"  {metric}: 共找到 {len(metric_models)} 个模型: {metric_models}")
        print(f"  {metric}: 总特征出现次数: {sum(feature_counts[metric].values())}")
        print(f"  {metric}: 所有特征列表(包括重复): {all_features_list}")
        print(f"  {metric}: 所有特征列表长度: {len(all_features_list)} (应该是30)")
    
    return feature_counts

# 删除单个模型特征重要性对比图功能

def plot_feature_frequency_analysis(feature_counts, output_dir):
    """绘制特征出现频率分析图"""
    
    for metric, counts in feature_counts.items():
        if not counts:
            continue
            
        # 获取出现次数最多的特征
        top_features = counts.most_common()  
        
        if not top_features:
            continue
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features, frequencies = zip(*top_features)
        
        # 绘制水平柱状图
        bars = ax.barh(range(len(features)), frequencies, 
                      color=plt.cm.plasma(np.linspace(0, 1, len(features))), 
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 设置标签
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=12)
        ax.set_xlabel('出现次数', fontsize=14)
        ax.set_title(f'{metric} - 前5重要特征出现频率分析', fontsize=16, fontweight='bold')
        
        # 添加数值标签
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{freq}次', ha='left', va='center', fontsize=11, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # 保存图片
        output_filename = f'02_feature_frequency_analysis_{metric.replace("*", "").replace("Rp0.2值", "Rp02")}.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存频率分析图: {output_path}")
        
        plt.show()

# 删除热力图功能

def main():
    """主函数"""
    print("=" * 60)
    print("--- 启动脚本: 02_plot_feature_importance.py ---")
    print("--- 目的: 绘制所有模型的特征重要性对比图 ---")
    print("=" * 60)
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: '{OUTPUT_DIR}'")
    
    # 加载所有特征重要性数据
    print("正在加载所有模型的特征重要性数据...")
    all_data = load_feature_importance_data()
    
    if not all_data:
        print("未找到任何特征重要性数据，程序退出")
        return
    
    print(f"成功加载数据，共 {len(all_data)} 个模型-指标组合")
    print("加载的数据详情:")
    for key, data in all_data.items():
        print(f"  {key}: {data['model']} - {data['metric']}")
    
    # 分析前5个重要特征的统计信息
    print("正在分析前5个重要特征的统计信息...")
    feature_counts = analyze_top_features(all_data)
    
    # 打印统计结果
    for metric, counts in feature_counts.items():
        print(f"\n{metric} 前5重要特征出现次数统计:")
        total_features = sum(counts.values())
        unique_features = len(counts)
        print(f"  总特征出现次数: {total_features} (应该是30)")
        print(f"  唯一特征数量: {unique_features}")
        print(f"  所有特征出现次数:")
        for feature, count in counts.most_common():
            print(f"    {feature}: {count}次")
        
        # 验证：显示每个模型的前5特征
        print(f"  各模型前5特征详情:")
        for key, data in all_data.items():
            if data['metric'] == metric:
                print(f"    {data['model']}: {data['top_features']}")
    
    # 绘制特征出现频率分析图
    print(f"\n正在绘制特征出现频率分析图...")
    plot_feature_frequency_analysis(feature_counts, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("--- 所有特征重要性图表绘制完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()
