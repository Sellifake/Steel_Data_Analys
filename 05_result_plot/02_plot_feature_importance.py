# -*- coding: utf-8 -*-
"""
文件名: 02_plot_feature_importance.py
功能: 
    1. 读取所有模型的特征重要性CSV文件
    2. 统计前5个重要特征的出现次数
    3. 统计后5个重要特征的出现次数
    4. 保存统计结果到CSV文件并生成可视化图表
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

def load_feature_importance_data():
    """加载所有模型的特征重要性数据"""
    all_data = {}
    
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(INPUT_DIR, '*.csv'))
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # 解析文件名获取模型名和性能指标
        model_name = None
        metric = None
        
        if filename.startswith('01a_TabNet_'):
            model_name = '01a_TabNet'
            metric_part = filename.replace('01a_TabNet_', '').replace('_feature_importance.csv', '')
        elif filename.startswith('02_SVR_Tuned_'):
            model_name = '02_SVR_Tuned'
            metric_part = filename.replace('02_SVR_Tuned_', '').replace('_feature_importance.csv', '')
        elif filename.startswith('rf_'):
            model_name = 'RandomForest'
            metric_part = filename.replace('rf_', '').replace('_feature_importance.csv', '')
        elif filename.startswith('xgboost_'):
            model_name = 'XGBoost_Tuned'
            metric_part = filename.replace('xgboost_', '').replace('_feature_importance.csv', '')
        elif filename.startswith('lightgbm_'):
            model_name = 'LightGBM_Tuned'
            metric_part = filename.replace('lightgbm_', '').replace('_feature_importance.csv', '')
        else:
            print(f"无法识别文件名格式: {filename}")
            continue
        
        # 确定性能指标
        if '抗拉强度' in metric_part:
            metric = '抗拉强度'
        elif '屈服Rp0.2值' in metric_part:
            metric = '屈服Rp0.2值*'
        elif '断后伸长率' in metric_part:
            metric = '断后伸长率'
        else:
            print(f"无法识别性能指标: {metric_part}")
            continue
        
        # 读取数据
        try:
            df = pd.read_csv(file_path)
            print(f"处理文件: {filename}")
            
            # 处理不同的列名格式
            importance_col = None
            if 'Importance' in df.columns:
                importance_col = 'Importance'
            elif 'Importance (TabNet)' in df.columns:
                importance_col = 'Importance (TabNet)'
            elif 'Importance (Permutation)' in df.columns:
                importance_col = 'Importance (Permutation)'
            
            if importance_col and 'Feature' in df.columns:
                # 按重要性排序
                df_sorted = df.sort_values(importance_col, ascending=False)
                
                # 获取前5个和后5个特征
                top5_features = df_sorted.head(5)['Feature'].tolist()
                bottom5_features = df_sorted.tail(5)['Feature'].tolist()
                
                key = f"{model_name}_{metric}"
                all_data[key] = {
                    'model': model_name,
                    'metric': metric,
                    'top5_features': top5_features,
                    'bottom5_features': bottom5_features,
                    'all_features': df_sorted['Feature'].tolist()
                }
                print(f"成功加载: {model_name} - {metric}")
                print(f"  前5特征: {top5_features}")
                print(f"  后5特征: {bottom5_features}")
            else:
                print(f"文件 {filename} 缺少必要的列")
                
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            continue
    
    return all_data

def analyze_feature_counts(all_data):
    """分析前5个和后5个重要特征的统计信息"""
    top5_counts = {}
    bottom5_counts = {}
    
    print(f"\n=== 开始分析特征统计 ===")
    print(f"总共加载了 {len(all_data)} 个模型-指标组合")
    
    for metric in PERFORMANCE_METRICS:
        top5_counts[metric] = Counter()
        bottom5_counts[metric] = Counter()
        metric_models = []
        
        # 收集该性能指标下所有模型的前5个和后5个特征
        for key, data in all_data.items():
            if data['metric'] == metric:
                metric_models.append(data['model'])
                print(f"  {metric}: 找到模型 {data['model']}")
                print(f"    前5特征: {data['top5_features']}")
                print(f"    后5特征: {data['bottom5_features']}")
                
                # 统计前5个特征
                for feature in data['top5_features']:
                    top5_counts[metric][feature] += 1
                
                # 统计后5个特征
                for feature in data['bottom5_features']:
                    bottom5_counts[metric][feature] += 1
        
        print(f"  {metric}: 共找到 {len(metric_models)} 个模型: {metric_models}")
        print(f"  {metric}: 前5特征总出现次数: {sum(top5_counts[metric].values())}")
        print(f"  {metric}: 后5特征总出现次数: {sum(bottom5_counts[metric].values())}")
    
    return top5_counts, bottom5_counts

def plot_feature_frequency_analysis(top5_counts, bottom5_counts, output_dir):
    """绘制特征出现频率分析图 - 前5和后5特征左右排列"""
    
    for metric in PERFORMANCE_METRICS:
        # 检查是否有数据
        if not top5_counts[metric] and not bottom5_counts[metric]:
            continue
            
        # 创建子图，左右排列
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f'{metric} - 特征重要性频率分析', fontsize=18, fontweight='bold')
        
        # 绘制前5重要特征（左侧）
        if top5_counts[metric]:
            top_features = top5_counts[metric].most_common()
            if top_features:
                features, frequencies = zip(*top_features)
                
                # 绘制水平柱状图
                bars = ax1.barh(range(len(features)), frequencies, 
                              color=plt.cm.plasma(np.linspace(0, 1, len(features))), 
                              alpha=0.8, edgecolor='black', linewidth=0.5, height=0.6)
                
                # 设置标签
                ax1.set_yticks(range(len(features)))
                ax1.set_yticklabels(features, fontsize=11)
                ax1.set_xlabel('出现次数', fontsize=12, fontweight='bold')
                ax1.set_title('前5重要特征', fontsize=14, fontweight='bold')
                
                # 添加数值标签
                for i, (bar, freq) in enumerate(zip(bars, frequencies)):
                    width = bar.get_width()
                    ax1.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                           f'{freq}次', ha='left', va='center', fontsize=10, fontweight='bold')
                
                # 设置网格和范围
                ax1.grid(True, alpha=0.3, axis='x')
                ax1.set_axisbelow(True)
                ax1.set_xlim(0, max(frequencies) + 0.5)
        
        # 绘制后5重要特征（右侧）
        if bottom5_counts[metric]:
            bottom_features = bottom5_counts[metric].most_common()
            if bottom_features:
                features, frequencies = zip(*bottom_features)
                
                # 绘制水平柱状图
                bars = ax2.barh(range(len(features)), frequencies, 
                              color=plt.cm.viridis(np.linspace(0, 1, len(features))), 
                              alpha=0.8, edgecolor='black', linewidth=0.5, height=0.6)
                
                # 设置标签
                ax2.set_yticks(range(len(features)))
                ax2.set_yticklabels(features, fontsize=11)
                ax2.set_xlabel('出现次数', fontsize=12, fontweight='bold')
                ax2.set_title('后5重要特征', fontsize=14, fontweight='bold')
                
                # 添加数值标签
                for i, (bar, freq) in enumerate(zip(bars, frequencies)):
                    width = bar.get_width()
                    ax2.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                           f'{freq}次', ha='left', va='center', fontsize=10, fontweight='bold')
                
                # 设置网格和范围
                ax2.grid(True, alpha=0.3, axis='x')
                ax2.set_axisbelow(True)
                ax2.set_xlim(0, max(frequencies) + 0.5)
        
        plt.tight_layout()
        
        # 保存图片
        output_filename = f'feature_frequency_analysis_{metric.replace("*", "").replace("Rp0.2值", "Rp02")}.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"已保存特征频率分析图: {output_path}")
        
        plt.show()

def save_feature_counts_to_csv(top5_counts, bottom5_counts, output_dir):
    """保存特征统计结果到CSV文件"""
    
    for metric in PERFORMANCE_METRICS:
        # 保存前5特征统计
        top5_data = []
        for feature, count in top5_counts[metric].most_common():
            top5_data.append({
                '特征名称': feature,
                '出现次数': count,
                '性能指标': metric,
                '特征类型': '前5重要'
            })
        
        top5_df = pd.DataFrame(top5_data)
        top5_filename = f'top5_feature_counts_{metric.replace("*", "").replace("Rp0.2值", "Rp02")}.csv'
        top5_path = os.path.join(output_dir, top5_filename)
        top5_df.to_csv(top5_path, index=False, encoding='utf-8-sig')
        print(f"已保存前5特征统计: {top5_path}")
        
        # 保存后5特征统计
        bottom5_data = []
        for feature, count in bottom5_counts[metric].most_common():
            bottom5_data.append({
                '特征名称': feature,
                '出现次数': count,
                '性能指标': metric,
                '特征类型': '后5重要'
            })
        
        bottom5_df = pd.DataFrame(bottom5_data)
        bottom5_filename = f'bottom5_feature_counts_{metric.replace("*", "").replace("Rp0.2值", "Rp02")}.csv'
        bottom5_path = os.path.join(output_dir, bottom5_filename)
        bottom5_df.to_csv(bottom5_path, index=False, encoding='utf-8-sig')
        print(f"已保存后5特征统计: {bottom5_path}")
    
    # 保存综合统计
    all_data = []
    for metric in PERFORMANCE_METRICS:
        for feature, count in top5_counts[metric].most_common():
            all_data.append({
                '特征名称': feature,
                '出现次数': count,
                '性能指标': metric,
                '特征类型': '前5重要'
            })
        for feature, count in bottom5_counts[metric].most_common():
            all_data.append({
                '特征名称': feature,
                '出现次数': count,
                '性能指标': metric,
                '特征类型': '后5重要'
            })
    
    all_df = pd.DataFrame(all_data)
    all_path = os.path.join(output_dir, 'all_feature_counts_summary.csv')
    all_df.to_csv(all_path, index=False, encoding='utf-8-sig')
    print(f"已保存综合统计: {all_path}")

def main():
    """主函数"""
    print("=" * 60)
    print("--- 启动脚本: 02_plot_feature_importance.py ---")
    print("--- 目的: 统计所有模型的特征重要性出现次数 ---")
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
    
    # 分析前5个和后5个重要特征的统计信息
    print("正在分析特征统计信息...")
    top5_counts, bottom5_counts = analyze_feature_counts(all_data)
    
    # 打印统计结果
    for metric in PERFORMANCE_METRICS:
        print(f"\n{metric} 特征出现次数统计:")
        print(f"  前5重要特征:")
        for feature, count in top5_counts[metric].most_common():
            print(f"    {feature}: {count}次")
        
        print(f"  后5重要特征:")
        for feature, count in bottom5_counts[metric].most_common():
            print(f"    {feature}: {count}次")
    
    # 保存统计结果到CSV文件
    print(f"\n正在保存统计结果到CSV文件...")
    save_feature_counts_to_csv(top5_counts, bottom5_counts, OUTPUT_DIR)
    
    # 绘制特征出现频率分析图
    print(f"\n正在绘制特征出现频率分析图...")
    plot_feature_frequency_analysis(top5_counts, bottom5_counts, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("--- 所有特征重要性统计和图表绘制完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()
