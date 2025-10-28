# -*- coding: utf-8 -*-
"""
文件名: 01b_tabnet_perturbation_predict.py
(注意: 此脚本应放置在 04_neural_network 文件夹中)

功能: 
    1. 加载 01a 脚本训练好的 TabNet Pipeline (Scaler) 和模型权重 (.zip)。
    2. 加载 05 (01_read_data) 脚本生成的扰动数据集。
    3. [新增] 清洗扰动数据，移除所有特征列均为空(NaN)或0的行，并用0填充剩余NaN。
    4. 使用 TabNet Pipeline 对*清洗后*的数据集进行批量预测 (np.float32)。
    5. 保存详细的原始预测结果 (ICE)。
    6. 对预测结果按“扰动值”进行分组统计 (PDP)。
    7. [修改] 绘制并保存 PDP 效应图 (带 95% 置信区间和性能基线)。
"""

import pandas as pd
import numpy as np
import os
import re
import joblib
from tqdm import tqdm 
import matplotlib.pyplot as plt

# --- [关键] 导入 Pipeline 依赖 ---
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pytorch_tabnet.tab_model import TabNetRegressor

# --- 【新增修改点 1】: GPU配置 (确保预测也在GPU上运行) ---
import torch
GPU_DEVICE_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_ID)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. 配置区 ---

# --- 路径配置 ---
MODEL_IO_DIR = 'results/01_tabnet_results'
SAVED_MODEL_DIR = os.path.join(MODEL_IO_DIR, 'saved_models')
PERTURBED_DATA_DIR = '../01_read_data/results'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]
PERTURBATION_COLUMN = '扰动值'
PERTURBED_FEATURES_LIST = ['热点_峰值温度', '冷点_峰值温度', '保温时长']

# --- 模型配置 ---
MODEL_NAME_PREFIX = '01a_TabNet' 

# --- 【新增修改点 2】: 定义性能指标的基线 (使用您提供的最新值) ---
PERFORMANCE_BASELINES = {
    "抗拉强度": {"min": 260, "max": 410},
    "屈服Rp0.2值*": {"min": 120, "max": 240},
    "断后伸长率": {"min": 34, "max": float('inf')} # 'inf' 表示正无穷
}

# --- 2. 辅助函数 ---

def sanitize_filename(filename):
    """(复制自 01a) 清洗文件名中的非法字符。"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def load_models(model_dir, metrics, model_name_prefix):
    """
    [修改] 加载 TabNet 模型
    TabNet 模型需要加载 Pipeline (用于 Scaler) 和 .zip 权重文件
    """
    print(f"--- 正在加载 {model_name_prefix} (Pipeline + Weights) 模型... ---")
    models = {}
    
    if not os.path.exists(model_dir):
        print(f"    [严重错误] 模型目录未找到: {model_dir}")
        return None

    for metric in metrics:
        safe_metric_name = sanitize_filename(metric)
        if metric == "屈服Rp0.2值*":
            safe_metric_name = "屈服Rp0.2值_"
        
        # 1. 加载 Pipeline (包含 Scaler 和 *未训练* 的 TabNet 结构)
        pipeline_filename = f'{model_name_prefix}_{safe_metric_name}_model.joblib'
        pipeline_path = os.path.join(model_dir, pipeline_filename)
        
        # 2. 加载模型权重 (.zip 文件)
        # (01a 中保存的路径是 model_path，不带.zip)
        weights_filename = f'{model_name_prefix}_{safe_metric_name}_model.zip'
        weights_path = os.path.join(model_dir, weights_filename)
        
        try:
            # 加载 Pipeline
            pipeline = joblib.load(pipeline_path)
            
            # --- 【新增修改点 3】: 确保模型在 GPU/CPU 上加载 ---
            # (joblib 无法正确序列化 PyTorch 模型，所以我们重新加载)
            loaded_model = TabNetRegressor(device_name=device) # 指定设备
            
            # 从 .zip 文件加载权重
            # TabNet load_model 会自动将模型移动到 device_name 指定的设备
            loaded_model.load_model(weights_path)
            
            # 将加载了权重的模型 "注入" 回 Pipeline
            pipeline.named_steps['model'] = loaded_model
            
            models[metric] = pipeline
            print(f"    [成功] 已加载模型 (将在 {device} 上运行): {pipeline_filename} + {weights_filename}")
            
        except FileNotFoundError as e:
            print(f"    [严重错误] 模型文件未找到: {e.filename}")
            print("    请确保已运行 01a_tabnet_modeling.py 脚本并成功保存了 .joblib 和 .zip 文件。")
            return None
        except Exception as e:
            print(f"    [严重错误] 加载模型时出错: {e}")
            return None
            
    print(f"--- 共加载 {len(models)} 个模型 ---")
    return models

def get_feature_columns(df):
    """(复制自 01b) 自动识别所有特征列 (X)。"""
    exclude_cols = [ID_COLUMN, PERTURBATION_COLUMN] + PERFORMANCE_METRICS
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def calculate_summary_stats(df, group_col, pred_cols):
    """(复制自 01b, 并修改为 95% CI) 计算分组统计汇总"""
    print("    正在计算统计汇总 (均值, 中位数, P2.5, P97.5, 标准差)...")
    
    # --- 【新增修改点 4】: 修改为 95% 置信区间 ---
    p025 = lambda x: x.quantile(0.025); p025.__name__ = 'P025'
    p975 = lambda x: x.quantile(0.975); p975.__name__ = 'P975'
    agg_funcs = ['mean', 'median', p025, p975, 'std']
    
    summary = df.groupby(group_col)[pred_cols].agg(agg_funcs)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()

def setup_plot_style():
    """(复制自 01b) 设置 matplotlib 绘图风格"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("    [绘图] 中文字体 'SimHei' 设置成功。")
    except:
        print("    [绘图警告] 未找到 'SimHei' 字体。")


# --- 3. 主执行函数 ---

def main():
    """主执行函数"""
    print("=" * 60)
    print("--- 启动脚本: 01b_tabnet_perturbation_predict.py ---")
    print(f"--- 目的: 使用 {MODEL_NAME_PREFIX} 模型对扰动数据进行预测 (Device: {device}) ---")
    print("=" * 60)

    if not os.path.exists(MODEL_IO_DIR):
        os.makedirs(MODEL_IO_DIR)
        print(f"已创建输出文件夹: '{MODEL_IO_DIR}'")

    # --- 0. 设置绘图风格 ---
    setup_plot_style()

    # --- 1. 加载模型 ---
    models = load_models(SAVED_MODEL_DIR, PERFORMANCE_METRICS, MODEL_NAME_PREFIX)
    if models is None:
        print("模型加载失败，脚本终止。")
        return

    # --- 2. 循环处理每一种扰动数据 ---
    for feature_name in PERTURBED_FEATURES_LIST:
        
        safe_feature_name = sanitize_filename(feature_name)
        data_filename = f'05_perturbed_dataset_{safe_feature_name}.csv'
        data_path = os.path.join(PERTURBED_DATA_DIR, data_filename)
        
        print(f"\n{'='*20} 正在处理: {feature_name} (TabNet) {'='*20}")

        # 2a. 加载扰动数据
        try:
            df_perturbed = pd.read_csv(data_path)
            print(f"    [成功] 加载扰动数据，原始形状: {df_perturbed.shape}")
        except FileNotFoundError:
            print(f"    [严重错误] 扰动数据文件未找到: {data_path}")
            continue
        except Exception as e:
            print(f"    [错误] 读取CSV文件时出错: {e}")
            continue
            
        # --- [SVR特定] 数据清洗：移除空行 + 0填充 (保留此逻辑) ---
        x_columns = get_feature_columns(df_perturbed)
        if len(x_columns) == 0:
            print("    [严重错误] 未能在数据中识别出任何特征列。")
            continue
        
        n_rows_before = len(df_perturbed)
        
        # 1. 移除所有特征列都为 NaN 的行 (CSV中的空行)
        df_perturbed = df_perturbed.dropna(subset=x_columns, how='all')
        
        # 2. 用 0 填充剩余的 NaN (匹配训练数据 0 填充的假设)
        df_perturbed = df_perturbed.fillna(0)
        
        # 3. 移除所有特征列都为 0 的行
        if not df_perturbed.empty:
            numeric_x_columns = df_perturbed[x_columns].select_dtypes(include=np.number).columns
            is_all_zero = (df_perturbed[numeric_x_columns] == 0).all(axis=1)
            df_perturbed = df_perturbed[~is_all_zero]
        
        n_rows_after = len(df_perturbed)
        
        if n_rows_before > n_rows_after:
            print(f"    [数据清洗] 已移除 {n_rows_before - n_rows_after} 行 (因空或全0)。")
            print(f"    [数据清洗] 清洗后形状: {df_perturbed.shape}")
        
        if df_perturbed.empty:
            print("    [警告] 清洗后没有剩余数据，跳过此扰动特征。")
            continue
        # --- [SVR特定] 清洗结束 ---
            
        print(f"    已识别 {len(x_columns)} 个特征用于预测。")
        
        df_predictions_ice = df_perturbed[[ID_COLUMN, PERTURBATION_COLUMN]].copy()
        pred_cols_list = [] 

        # 2b. (任务一) 批量预测
        print("    正在执行批量预测 (TabNet Pipeline)...")
        for metric, model in tqdm(models.items(), desc="    预测性能"):
            
            if metric == "屈服Rp0.2值*":
                pred_col_name = "pred_屈服Rp0.2值_"
            else:
                pred_col_name = f'pred_{sanitize_filename(metric)}'
            
            pred_cols_list.append(pred_col_name)

            # [新增] 转换为 np.float32
            X_predict = df_perturbed[x_columns].values.astype(np.float32)
            
            # model 是 Pipeline, 它会自动 .transform() 和 .predict()
            predictions = model.predict(X_predict)
            
            # TabNet predict 返回 (n, 1) 2D数组, 转为 1D
            df_predictions_ice[pred_col_name] = predictions.flatten()

        # 2c. 保存原始预测结果 (ICE)
        output_ice_filename = f'01b_tabnet_prediction_results_{safe_feature_name}.csv'
        output_ice_path = os.path.join(MODEL_IO_DIR, output_ice_filename)
        print(f"    正在保存原始预测结果 (ICE)...")
        df_predictions_ice.to_csv(output_ice_path, index=False, encoding='utf-8-sig')
        print(f"    [成功] 原始预测已保存至: {output_ice_path}")
        
        # 2d. (任务二) 统计汇总 (PDP)
        df_summary_pdp = calculate_summary_stats(
            df_predictions_ice, 
            PERTURBATION_COLUMN, 
            pred_cols_list
        )
        
        # [关键修复] 确保横坐标正确：将扰动值列转为数值并按其排序
        df_summary_pdp = df_summary_pdp.copy()
        df_summary_pdp[PERTURBATION_COLUMN] = pd.to_numeric(
            df_summary_pdp[PERTURBATION_COLUMN], errors='coerce'
        )
        df_summary_pdp = df_summary_pdp.dropna(subset=[PERTURBATION_COLUMN])
        df_summary_pdp = df_summary_pdp.sort_values(PERTURBATION_COLUMN).reset_index(drop=True)
        
        # 2e. 保存统计汇总结果 (PDP)
        output_pdp_filename = f'01b_tabnet_summary_stats_{safe_feature_name}.csv'
        output_pdp_path = os.path.join(MODEL_IO_DIR, output_pdp_filename)
        print(f"    正在保存统计汇总结果 (PDP)...")
        df_summary_pdp.to_csv(output_pdp_path, index=False, encoding='utf-8-sig')
        print(f"    [成功] 统计汇总已保存至: {output_pdp_path}")

        # 2f. (任务三) 可视化统计结果 (PDP)
        print(f"    正在为 '{feature_name}' 生成可视化图表...")
        
        for metric in PERFORMANCE_METRICS:
            
            if metric == "屈服Rp0.2值*":
                base_col_name = "pred_屈服Rp0.2值_"
            else:
                base_col_name = f'pred_{sanitize_filename(metric)}'
            
            # --- 【新增修改点 5】: 访问 P025 和 P975 列 ---
            mean_col = f'{base_col_name}_mean'
            p025_col = f'{base_col_name}_P025' # 从 P05 修改
            p975_col = f'{base_col_name}_P975' # 从 P95 修改
            
            if not all(col in df_summary_pdp.columns for col in [mean_col, p025_col, p975_col]):
                print(f"    [绘图警告] 缺少 {metric} 的统计列，跳过绘图。")
                continue

            plt.figure(figsize=(10, 6))
            
            x_values = df_summary_pdp[PERTURBATION_COLUMN]
            p025_values = df_summary_pdp[p025_col] # 从 P05 修改
            p975_values = df_summary_pdp[p975_col] # 从 P95 修改
            
            # 绘制均值线
            plt.plot(x_values, df_summary_pdp[mean_col], 'b-', label='预测均值 (Mean)')
            
            # --- 【新增修改点 6】: 绘制 95% 置信区间 (P2.5 到 P97.5) ---
            plt.fill_between(
                x_values, 
                p025_values, # 从 P05 修改
                p975_values, # 从 P95 修改
                color='blue', 
                alpha=0.1, 
                label='95% 置信区间 (P2.5-P97.5)' # 标签从 90% 修改
            )
            # 保证置信区间完整显示
            plt.xlim(x_values.min(), x_values.max())
            
            # --- 【新增修改点 7】: 添加性能指标基线 ---
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
            
            plot_title = f'{feature_name} 扰动 对 {metric} 的影响 (PDP - TabNet)'
            plt.title(plot_title, fontsize=16)
            plt.xlabel(f'{feature_name} 扰动值 (delta)', fontsize=12)
            plt.ylabel(f'预测的 {metric}', fontsize=12)
            
            # --- 【新增修改点 8】: 确保基线值显示在Y轴上 ---
            if custom_yticks:
                # 获取matplotlib自动生成的Y轴刻度
                current_ticks = list(plt.yticks()[0])
                # 合并自动刻度和我们的基线刻度，去重并排序
                new_ticks = sorted(list(set(current_ticks) | custom_yticks))
                # 设置新的Y轴刻度
                plt.yticks(new_ticks)
            
            # --- 【新增修改点 9】: 调整图例位置 ---
            plt.legend(loc='upper right')
            
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            safe_metric_name = sanitize_filename(metric)
            if metric == "屈服Rp0.2值*":
                safe_metric_name = "屈服Rp0.2值_"
                
            plot_filename = f'01b_pdp_plot_{safe_feature_name}_vs_{safe_metric_name}.png'
            plot_path = os.path.join(MODEL_IO_DIR, plot_filename)
            
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            print(f"    [成功] 图表已保存至: {plot_path}")


    print("\n" + "=" * 60)
    print("--- 步骤 01b (TabNet扰动预测) 全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()