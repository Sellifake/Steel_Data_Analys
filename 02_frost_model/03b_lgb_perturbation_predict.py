# -*- coding: utf-8 -*-
"""
文件名: 03b_lgb_perturbation_predict.py
功能: 
    1. 加载 03a 脚本训练好的 LightGBM (LGBM) 模型。
    2. 加载 05 脚本生成的扰动数据集 (perturbed datasets)。
    3. 使用 LGBM 模型对扰动数据集进行批量预测。
    4. 保存详细的原始预测结果 (ICE - 个体条件期望)。
    5. 对预测结果按“扰动值”进行分组统计 (均值, 中位数, 分位数等)。
    6. 保存统计汇总结果 (PDP - 部分依赖图数据)。
    7. 基于统计汇总结果，绘制并保存 PDP 效应图 (带 95% 置信区间)。

脚本放置位置: 
    - 假设此脚本位于: .../steel/02_frost_model/
"""

import pandas as pd
import numpy as np
import os
import re
import joblib
import lightgbm as lgb # 导入lgb，确保joblib反序列化时能找到库
from tqdm import tqdm # 用于显示处理进度
import matplotlib.pyplot as plt # 导入绘图库

# --- 1. 配置区 ---

# --- 路径配置 ---
# 假设此脚本位于 .../steel/02_frost_model/
# 模型输入/输出路径: .../steel/02_frost_model/results/03_lightgbm_results/
MODEL_IO_DIR = 'results/03_lightgbm_results'
SAVED_MODEL_DIR = os.path.join(MODEL_IO_DIR, 'saved_models')

# 扰动数据输入路径: .../steel/01_read_data/results/
PERTURBED_DATA_DIR = '../01_read_data/results'

# --- 列定义 ---
# ID列 (与 05 脚本中定义的一致)
ID_COLUMN = '罩退钢卷号'
# 性能指标 (与 03a 脚本中定义的一致)
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]
# 扰动值列 (与 05 脚本中定义的一致)
PERTURBATION_COLUMN = '扰动值'

# --- 扰动特征配置 ---
# 定义我们期望处理的扰动特征 (与 05 脚本中一致)
PERTURBED_FEATURES_LIST = ['热点_峰值温度', '冷点_峰值温度', '保温时长']

# --- 【新增修改点 1】: 定义性能指标的基线 (使用您提供的最新值) ---
PERFORMANCE_BASELINES = {
    "抗拉强度": {"min": 260, "max": 370},
    "屈服Rp0.2值*": {"min": 120, "max": 240},  # 确保Key与PERFORMANCE_METRICS中的名称一致
    "断后伸长率": {"min": 34, "max": float('inf')} # 'inf' 表示正无穷，不绘制最大值基线
}


# --- 2. 辅助函数 ---

def sanitize_filename(filename):
    """
    清洗文件名中的非法字符，确保文件名符合系统要求
    与01a_random_forest_modeling.py中的函数保持一致
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def load_models(model_dir, metrics):
    """
    加载所有训练好的LightGBM模型
    返回模型字典，键为性能指标名称
    """
    print("--- 正在加载 LightGBM 模型... ---")
    models = {}
    
    if not os.path.exists(model_dir):
        print(f"    [严重错误] 模型目录未找到: {model_dir}")
        print("    请检查脚本是否放置在 '02_frost_model' 文件夹中。")
        return None

    for metric in metrics:
        safe_metric_name = sanitize_filename(metric)
        # (仿照 01b) 特殊处理 "屈服Rp0.2值*" (因为 sanitize_filename 会替换 *)
        if metric == "屈服Rp0.2值*":
            safe_metric_name = "屈服Rp0.2值_"
        
        model_filename = f'{safe_metric_name}_model.joblib'
        model_path = os.path.join(model_dir, model_filename)
        
        try:
            models[metric] = joblib.load(model_path)
            print(f"    [成功] 已加载模型: {model_filename}")
        except FileNotFoundError:
            print(f"    [严重错误] 模型文件未找到: {model_path}")
            # 更新错误提示
            print("    请确保已运行 03a_lightgbm_modeling.py 脚本并成功保存了模型。")
            return None
    print(f"--- 共加载 {len(models)} 个模型 ---")
    return models

def get_feature_columns(df):
    """
    从加载的扰动数据集中自动识别所有特征列
    排除ID列、扰动值列和性能指标列
    """
    # 基础排除列
    exclude_cols = [ID_COLUMN, PERTURBATION_COLUMN]
    # 理论上性能指标列不应该在 05 生成的文件中，但为保险起见加入
    exclude_cols.extend(PERFORMANCE_METRICS)
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def calculate_summary_stats(df, group_col, pred_cols):
    """
    计算分组统计汇总
    包括均值、中位数、95%置信区间和标准差
    """
    print("    正在计算统计汇总 (均值, 中位数, P2.5, P97.5, 标准差)...")
    
    # --- 【修改点 2】: 修改为 95% 置信区间 ---
    p025 = lambda x: x.quantile(0.025)
    p025.__name__ = 'P025'
    
    p975 = lambda x: x.quantile(0.975)
    p975.__name__ = 'P975'
    
    agg_funcs = ['mean', 'median', p025, p975, 'std']
    # --- 修改结束 ---
    
    # 对所有预测列执行分组和聚合
    summary = df.groupby(group_col)[pred_cols].agg(agg_funcs)
    
    # 调整列名，使其更易读 (例如: 'pred_抗拉强度_mean')
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    return summary.reset_index()

def setup_plot_style():
    """
    设置matplotlib绘图风格，支持中文显示
    配置字体和负号显示
    """
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("    [绘图] 中文字体 'SimHei' 设置成功。")
    except:
        print("    [绘图警告] 未找到 'SimHei' 字体，中文可能显示为方框。")


# --- 3. 主执行函数 ---

def main():
    """
    主执行函数
    使用LightGBM模型对扰动数据进行预测分析和可视化
    """
    print("=" * 60)
    print("--- 启动脚本: 03b_lgb_perturbation_predict.py ---")
    print("--- 目的: 使用 LGBM 模型对扰动数据进行预测、统计与可视化 ---")
    print("=" * 60)

    # 确保输出目录存在
    if not os.path.exists(MODEL_IO_DIR):
        os.makedirs(MODEL_IO_DIR)
        print(f"已创建输出文件夹: '{MODEL_IO_DIR}'")

    # --- 0. 设置绘图风格 ---
    setup_plot_style()

    # --- 1. 加载模型 ---
    models = load_models(SAVED_MODEL_DIR, PERFORMANCE_METRICS)
    if models is None:
        print("模型加载失败，脚本终止。")
        return

    # --- 2. 循环处理每一种扰动数据 ---
    for feature_name in PERTURBED_FEATURES_LIST:
        
        safe_feature_name = sanitize_filename(feature_name)
        data_filename = f'05_perturbed_dataset_{safe_feature_name}.csv'
        data_path = os.path.join(PERTURBED_DATA_DIR, data_filename)
        
        print(f"\n{'='*20} 正在处理: {feature_name} (文件: {data_filename}) {'='*20}")

        # 2a. 加载扰动数据
        try:
            df_perturbed = pd.read_csv(data_path)
            print(f"    [成功] 加载扰动数据，形状: {df_perturbed.shape}")
        except FileNotFoundError:
            print(f"    [严重错误] 扰动数据文件未找到: {data_path}")
            print("    请确保已运行 05_generate_perturbed_data.py 脚本。")
            continue
        except Exception as e:
            print(f"    [错误] 读取CSV文件时出错: {e}")
            continue
            
        # 自动识别特征列
        x_columns = get_feature_columns(df_perturbed)
        if len(x_columns) == 0:
            print("    [严重错误] 未能在数据中识别出任何特征列。")
            continue
        print(f"    已识别 {len(x_columns)} 个特征用于预测。")
        
        # 准备一个DataFrame来存储所有预测结果
        df_predictions_ice = df_perturbed[[ID_COLUMN, PERTURBATION_COLUMN]].copy()
        
        pred_cols_list = [] # 用于存储预测列的名称

        # 2b. 执行批量预测任务
        print("    正在执行批量预测 (LightGBM)...")
        for metric, model in tqdm(models.items(), desc="    预测性能"):
            
            # (仿照 01b) 构造预测列名 (例如: 'pred_抗拉强度')
            if metric == "屈服Rp0.2值*":
                pred_col_name = "pred_屈服Rp0.2值_"
            else:
                pred_col_name = f'pred_{sanitize_filename(metric)}'
            
            pred_cols_list.append(pred_col_name)

            # 执行预测 (LightGBM的 predict 接口与 sklearn 一致)
            predictions = model.predict(df_perturbed[x_columns])
            
            # 存储预测结果
            df_predictions_ice[pred_col_name] = predictions

        # 2c. 保存原始预测结果(ICE - 个体条件期望)
        # 更改输出文件名
        output_ice_filename = f'03b_lgb_prediction_results_{safe_feature_name}.csv'
        output_ice_path = os.path.join(MODEL_IO_DIR, output_ice_filename)
        print(f"    正在保存原始预测结果 (ICE)...")
        df_predictions_ice.to_csv(output_ice_path, index=False, encoding='utf-8-sig')
        print(f"    [成功] 原始预测已保存至: {output_ice_path}")
        
        # 2d. 计算统计汇总(PDP - 部分依赖图数据)
        df_summary_pdp = calculate_summary_stats(
            df_predictions_ice, 
            PERTURBATION_COLUMN, 
            pred_cols_list
        )
        
        # 2e. 保存统计汇总结果
        # 更改输出文件名
        output_pdp_filename = f'03b_lgb_summary_stats_{safe_feature_name}.csv'
        output_pdp_path = os.path.join(MODEL_IO_DIR, output_pdp_filename)
        print(f"    正在保存统计汇总结果 (PDP)...")
        df_summary_pdp.to_csv(output_pdp_path, index=False, encoding='utf-8-sig')
        print(f"    [成功] 统计汇总已保存至: {output_pdp_path}")

        # 2f. 生成可视化图表(PDP效应图)
        print(f"    正在为 '{feature_name}' 生成可视化图表...")
        
        # 循环为 3 个性能指标分别绘图
        for metric in PERFORMANCE_METRICS:
            
            # (仿照 01b) 构造列名
            if metric == "屈服Rp0.2值*":
                base_col_name = "pred_屈服Rp0.2值_"
            else:
                base_col_name = f'pred_{sanitize_filename(metric)}'
            
            # --- 【修改点 3】: 访问 P025 和 P975 列 ---
            mean_col = f'{base_col_name}_mean'
            p025_col = f'{base_col_name}_P025' # 从 P05 修改
            p975_col = f'{base_col_name}_P975' # 从 P95 修改
            
            # 检查列是否存在
            if not all(col in df_summary_pdp.columns for col in [mean_col, p025_col, p975_col]):
                print(f"    [绘图警告] 缺少 {metric} 的统计列，跳过绘图。")
                continue

            # 开始绘图
            plt.figure(figsize=(10, 6))
            
            x_values = df_summary_pdp[PERTURBATION_COLUMN]
            mean_values = df_summary_pdp[mean_col]
            p025_values = df_summary_pdp[p025_col] # 从 P05 修改
            p975_values = df_summary_pdp[p975_col] # 从 P95 修改

            # 绘制均值线
            plt.plot(x_values, mean_values, 'b-', label='预测均值 (Mean)')
            
            # --- 【修改点 4】: 绘制 95% 置信区间 (P2.5 到 P97.5) ---
            plt.fill_between(
                x_values, 
                p025_values, # 从 P05 修改
                p975_values, # 从 P95 修改
                color='blue', 
                alpha=0.1, 
                label='95% 置信区间 (P2.5-P97.5)' # 标签从 90% 修改
            )
            
            # --- 【修改点 5】: 添加性能指标基线 ---
            baselines = PERFORMANCE_BASELINES.get(metric)
            custom_yticks = set() # 用于存储基线值以添加到Y轴
            
            if baselines:
                min_val = baselines.get('min')
                max_val = baselines.get('max')
                
                # 绘制最小值基线
                if min_val is not None and min_val != float('-inf'):
                    plt.axhline(y=min_val, color='red', linestyle='--', linewidth=1.2, 
                                label=f'最小值基线 ({min_val})')
                    custom_yticks.add(min_val) # 添加到Y轴刻度
                    
                # 绘制最大值基线 (排除无穷大)
                if max_val is not None and max_val != float('inf'):
                    plt.axhline(y=max_val, color='red', linestyle='--', linewidth=1.2, 
                                label=f'最大值基线 ({max_val})')
                    custom_yticks.add(max_val) # 添加到Y轴刻度
            
            # 设置图表标题和标签
            plot_title = f'{feature_name} 扰动 对 {metric} 的影响 (PDP - LightGBM)'
            plt.title(plot_title, fontsize=16)
            plt.xlabel(f'{feature_name} 扰动值 (delta)', fontsize=12)
            plt.ylabel(f'预测的 {metric}', fontsize=12)
            
            # --- 【修改点 6】: 确保基线值显示在Y轴上 ---
            if custom_yticks:
                # 获取matplotlib自动生成的Y轴刻度
                current_ticks = list(plt.yticks()[0])
                # 合并自动刻度和我们的基线刻度，去重并排序
                new_ticks = sorted(list(set(current_ticks) | custom_yticks))
                # 设置新的Y轴刻度
                plt.yticks(new_ticks)

            # --- 【修改点 7】: 调整图例位置 ---
            plt.legend(loc='upper right')
            
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # (仿照 01b) 保存图表
            safe_metric_name = sanitize_filename(metric)
            if metric == "屈服Rp0.2值*":
                safe_metric_name = "屈服Rp0.2值_"
                
            # 更改输出文件名
            plot_filename = f'03b_pdp_plot_{safe_feature_name}_vs_{safe_metric_name}.png'
            plot_path = os.path.join(MODEL_IO_DIR, plot_filename)
            
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            print(f"    [成功] 图表已保存至: {plot_path}")


    print("\n" + "=" * 60)
    print("--- LightGBM扰动预测与可视化任务全部完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()