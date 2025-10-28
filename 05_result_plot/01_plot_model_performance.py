# -*- coding: utf-8 -*-
"""
文件名: 01_plot_model_performance.py
功能: 
    1. 读取所有模型的性能指标CSV文件
    2. 为三个性能指标分别绘制对比图，显示6个模型的性能
    3. 生成柱状图展示R²、MAE、RMSE指标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
INPUT_DIR = '回归指标'
OUTPUT_DIR = 'results/回归指标分析'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]
MODEL_NAMES = {
    'RandomForest': '随机森林',
    'XGBoost_Tuned': 'XGBoost',
    'LightGBM_Tuned': 'LightGBM', 
    '02_SVR_Tuned': '支持向量回归',
    '01a_TabNet': 'TabNet'
}

def load_all_model_data():
    """加载所有模型的性能数据"""
    all_data = []
    
    # 1. 随机森林数据（没有模型列）
    rf_file = os.path.join(INPUT_DIR, 'random_forest_performance_summary.csv')
    if os.path.exists(rf_file):
        rf_data = pd.read_csv(rf_file)
        rf_data['模型'] = 'RandomForest'
        all_data.append(rf_data)
        print(f"已加载随机森林数据: {len(rf_data)} 条记录")
    
    # 2. XGBoost数据（没有模型列）
    xgb_file = os.path.join(INPUT_DIR, 'xgboost_tuned_performance_summary.csv')
    if os.path.exists(xgb_file):
        xgb_data = pd.read_csv(xgb_file)
        xgb_data['模型'] = 'XGBoost_Tuned'
        all_data.append(xgb_data)
        print(f"已加载XGBoost数据: {len(xgb_data)} 条记录")
    
    # 3. LightGBM数据（没有模型列）
    lgb_file = os.path.join(INPUT_DIR, 'lightgbm_tuned_performance_summary.csv')
    if os.path.exists(lgb_file):
        lgb_data = pd.read_csv(lgb_file)
        lgb_data['模型'] = 'LightGBM_Tuned'
        all_data.append(lgb_data)
        print(f"已加载LightGBM数据: {len(lgb_data)} 条记录")
    
    # 4. SVR数据（有模型列）
    svr_file = os.path.join(INPUT_DIR, 'svr_tuned_performance_summary.csv')
    if os.path.exists(svr_file):
        svr_data = pd.read_csv(svr_file)
        all_data.append(svr_data)
        print(f"已加载SVR数据: {len(svr_data)} 条记录")
    
    # 5. TabNet数据（现在有完整的三个指标）
    tabnet_file = os.path.join(INPUT_DIR, 'TabNet_performance_summary.csv')
    if os.path.exists(tabnet_file):
        tabnet_data = pd.read_csv(tabnet_file)
        tabnet_data['模型'] = '01a_TabNet'
        all_data.append(tabnet_data)
        print(f"已加载TabNet数据: {len(tabnet_data)} 条记录")
    
    # 合并所有数据
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data
    else:
        print("未找到任何性能数据文件")
        return None

def plot_performance_comparison(data, metric, output_dir):
    """为指定指标绘制性能对比图"""
    
    # 筛选数据
    metric_data = data[data['性能指标'] == metric].copy()
    
    if metric_data.empty:
        print(f"未找到 {metric} 的数据")
        return
    
    # 重命名模型为中文
    metric_data['模型中文名'] = metric_data['模型'].map(MODEL_NAMES)
    
    # 创建子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{metric} - 模型性能对比', fontsize=16, fontweight='bold')
    
    # 定义指标和对应的轴
    metrics = [
        ('R2 Score (平均值)', 'R方分数', axes[0]),
        ('MAE (平均值)', '平均绝对误差 (MAE)', axes[1]),
        ('RMSE (平均值)', '均方根误差 (RMSE)', axes[2])
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (col_name, ylabel, ax) in enumerate(metrics):
        # 过滤掉NaN值
        plot_data = metric_data.dropna(subset=[col_name])
        
        if plot_data.empty:
            ax.text(0.5, 0.5, f'{ylabel}\n数据不可用', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(ylabel)
            continue
        
        # 绘制柱状图
        bars = ax.bar(range(len(plot_data)), plot_data[col_name], 
                     color=colors[:len(plot_data)], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 设置标签
        ax.set_xticks(range(len(plot_data)))
        ax.set_xticklabels(plot_data['模型中文名'], rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=14, fontweight='bold')
        
        # 添加数值标签
        for j, (bar, value) in enumerate(zip(bars, plot_data[col_name])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 设置网格
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存图片
    output_filename = f'01_model_performance_comparison_{metric.replace("*", "").replace("Rp0.2值", "Rp02")}.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存图片: {output_path}")
    
    plt.show()

def plot_comprehensive_comparison(data, output_dir):
    """绘制综合性能对比图"""
    
    # 创建综合对比图
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('所有模型性能指标综合对比', fontsize=18, fontweight='bold')
    
    metrics_info = [
        ('R2 Score (平均值)', 'R方分数', '越高越好'),
        ('MAE (平均值)', '平均绝对误差 (MAE)', '越低越好'),
        ('RMSE (平均值)', '均方根误差 (RMSE)', '越低越好')
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, metric in enumerate(PERFORMANCE_METRICS):
        metric_data = data[data['性能指标'] == metric].copy()
        metric_data['模型中文名'] = metric_data['模型'].map(MODEL_NAMES)
        
        for j, (col_name, ylabel, direction) in enumerate(metrics_info):
            ax = axes[i, j]
            
            # 过滤掉NaN值
            plot_data = metric_data.dropna(subset=[col_name])
            
            if plot_data.empty:
                ax.text(0.5, 0.5, f'{metric}\n{ylabel}\n数据不可用', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{metric} - {ylabel}', fontsize=12, fontweight='bold')
                continue
            
            # 绘制柱状图
            bars = ax.bar(range(len(plot_data)), plot_data[col_name], 
                         color=colors[:len(plot_data)], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # 设置标签
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels(plot_data['模型中文名'], rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(f'{metric} - {ylabel}', fontsize=12, fontweight='bold')
            
            # 添加数值标签
            for k, (bar, value) in enumerate(zip(bars, plot_data[col_name])):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # 设置网格
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # 保存综合对比图
    output_filename = '01_comprehensive_model_performance_comparison.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存综合对比图: {output_path}")
    
    plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("--- 启动脚本: 01_plot_model_performance.py ---")
    print("--- 目的: 绘制所有模型的性能对比图 ---")
    print("=" * 60)
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: '{OUTPUT_DIR}'")
    
    # 加载所有模型数据
    print("正在加载所有模型的性能数据...")
    all_data = load_all_model_data()
    
    if all_data is None or all_data.empty:
        print("未找到任何性能数据，程序退出")
        return
    
    print(f"成功加载数据，共 {len(all_data)} 条记录")
    print("数据概览:")
    print(all_data[['模型', '性能指标', 'R2 Score (平均值)']].head(10))
    
    # 为每个性能指标绘制单独的对比图
    for metric in PERFORMANCE_METRICS:
        print(f"\n正在绘制 {metric} 的性能对比图...")
        plot_performance_comparison(all_data, metric, OUTPUT_DIR)
    
    # 绘制综合对比图
    print(f"\n正在绘制综合性能对比图...")
    plot_comprehensive_comparison(all_data, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("--- 所有图表绘制完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()
