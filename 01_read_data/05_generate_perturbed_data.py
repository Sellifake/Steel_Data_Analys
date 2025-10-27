# -*- coding: utf-8 -*-
"""
文件名: 05_generate_perturbed_data.py
功能: 
    1. 加载 01_read_data/results 文件夹中的原始数据集 (04_data_selected.xlsx)。
    2. 按照 PERTURBATION_CONFIG 中定义的规则，生成新的“扰动数据集”。
    3. 将每个特征扰动生成的“虚拟数据集”分别保存到独立的CSV文件中，
       以便后续的各个模型（RF, XGBoost等）可以加载它们进行预测。

数据生成规则 (ICE - 个体条件期望):
    - 基础: 以原始数据中的每一条记录 (共N条) 作为基准。
    - 扰动: 选择一个特征 (如 '热点_峰值温度')。
    - 范围: 针对该特征，生成一个扰动值数组 (如 M=[-10, -9, ..., +10])。
    - 生成: 
        - 对N条记录中的每一条，都应用M个扰动值。
        - 新特征值 = 原始特征值 + 扰动值
        - 保持所有其他特征不变。
    - 结果: 每个被扰动的特征都会生成一个 N * M 行的新数据集。

"""

import pandas as pd
import numpy as np
import os
import re

# --- 1. 配置区 ---

# --- 路径配置 ---
INPUT_DIR = 'results'
OUTPUT_DIR = 'results' 
# 使用01_read_data生成的最终清理数据
DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
# 定义这些列是为了在加载数据时将它们从特征 (X) 中排除
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 扰动配置 ---
#
# 详细定义要扰动的特征及其生成规则
# 规则: 
# 1. 键 (key) 是数据文件中必须存在的特征列名。
# 2. 值 (value) 是一个 numpy 数组，代表要施加的“变化量” (delta)。
#
PERTURBATION_CONFIG = {
    
    # '热点_峰值温度': 
    # 规则: 在原始值的基础上，从-10°C 变化到 +10°C，每次变化的步长为 1°C。
    # 对应的“扰动值”(delta) 数组为: [-10, -9, ..., 0, ..., 9, 10]
    '热点_峰值温度': np.arange(-10, 10 + 1, 1),
    
    # '冷点_峰值温度': 
    # 规则: 同上，在原始值的基础上，从-10°C 变化到 +10°C，步长为 1°C。
    '冷点_峰值温度': np.arange(-10, 10 + 1, 1),
    
    # '保温时长': 
    # 规则: 在原始值的基础上，从-3600秒 (-1小时) 变化到 +3600秒 (+1小时)。
    # 每次变化的步长为 60秒 (1分钟)。
    # 对应的“扰动值”(delta) 数组为: [-3600, -3540, ..., 0, ..., 3540, 3600]
    '保温时长': np.arange(-3600, 3600 + 60, 60)
}

# --- 2. 辅助函数 ---

def sanitize_filename(filename):
    """
    清洗文件名中的非法字符。
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def create_perturbed_dataset(original_x_data, feature_to_perturb, perturbations):
    """
    高效地创建扰动数据集 (向量化操作)。
    
    详细规则:
    假设 original_x_data 有 N 行 (例如 2760)。
    假设 perturbations 有 M 个值 (例如 21 个 [-10, ..., +10])。
    
    1. 重复原始数据 (np.repeat): 
       将 [Row1, Row2, ...] 变为 [Row1, Row1(共M次), Row2, Row2(共M次), ...]
       总行数变为 N * M。
       
    2. 创建平铺扰动值 (np.tile): 
       将 [p1, p2, ..., pM] 变为 [p1, ..., pM, p1, ..., pM, ... (共N次)]
       总行数变为 N * M。
       
    3. 应用扰动:
       获取第1步中重复后的数据 (df_repeated)，
       取出其中 'feature_to_perturb' 列的原始值 (base_values)，
       与第2步中的扰动值 (perturb_tiled) 相加。
       
       最终: df_repeated[feature_to_perturb] = base_values + perturb_tiled
       
    参数:
    - original_x_data (pd.DataFrame): 原始的特征数据 (N条)
    - feature_to_perturb (str): 要扰动的列名
    - perturbations (np.array): 扰动值的数组 (M个)
    
    返回:
    - (pd.DataFrame): 包含 N * M 条记录的扰动数据集 (即所有17个特征的新值)
    - (np.array): N * M 条记录对应的扰动值 (delta)
    """
    
    n_original = len(original_x_data)
    n_perturb = len(perturbations)
    
    # 1. 重复原始数据 (N * M 行)
    df_repeated = pd.DataFrame(
        np.repeat(original_x_data.values, n_perturb, axis=0),
        columns=original_x_data.columns
    )
    
    # 2. 创建平铺的扰动值 (N * M 行)
    perturb_tiled = np.tile(perturbations, n_original)
    
    # 3. 应用扰动 (核心规则)
    base_values = df_repeated[feature_to_perturb].values
    
    # 新特征值 = 原始特征值 + 扰动值
    df_repeated[feature_to_perturb] = base_values + perturb_tiled
    
    return df_repeated, perturb_tiled


# --- 3. 主执行函数 ---

def main():
    """主执行函数"""
    print("=" * 60)
    print("--- 启动脚本: 05_generate_perturbed_data.py ---")
    print("--- 目的: 仅生成用于敏感性分析的扰动数据集 ---")
    print("=" * 60)

    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: '{OUTPUT_DIR}'")

    # --- 1. 加载原始数据 ---
    data_path = os.path.join(INPUT_DIR, DATA_FILE)
    try:
        df_original = pd.read_excel(data_path)
        print(f"成功加载原始数据: {data_path}")
        print(f"原始数据形状: {df_original.shape}")
    except FileNotFoundError:
        print(f"[严重错误] 原始数据文件未找到 '{data_path}'。")
        print(f"    请确保 '{DATA_FILE}' 文件位于 '{INPUT_DIR}' 文件夹中。")
        return
    except Exception as e:
        print(f"[错误] 读取Excel文件时出错: {e}")
        print("    请确保已安装 'openpyxl' 库: pip install openpyxl")
        return
        
    # 提取特征列 (X) 和 ID列
    # 确保特征列的顺序与训练时一致
    x_columns = [col for col in df_original.columns if col not in [ID_COLUMN] + PERFORMANCE_METRICS]
    original_x_data = df_original[x_columns]
    original_id_data = df_original[ID_COLUMN]
    
    print(f"已识别 {len(x_columns)} 个特征。")

    # --- 2. 循环处理每一种扰动配置 ---
    for feature_name, perturbation_values in PERTURBATION_CONFIG.items():
        
        if feature_name not in original_x_data.columns:
            print(f"\n[警告] 特征 '{feature_name}' 在数据文件中未找到，跳过此扰动。")
            continue
            
        print(f"\n{'='*20} 正在为特征: '{feature_name}' 生成数据 {'='*20}")
        
        n_original = len(original_x_data)
        n_perturb = len(perturbation_values)
        total_samples = n_original * n_perturb
        
        print(f"    将为 {n_original} 条原始数据生成 {n_perturb} 个扰动点")
        print(f"    总计生成新样本数: {total_samples}")

        # 生成大型扰动数据集 (包含所有特征)
        print("    正在生成扰动特征矩阵...")
        df_perturbed_features, perturb_tiled = create_perturbed_dataset(
            original_x_data, 
            feature_name, 
            perturbation_values
        )
        
        # 准备ID和扰动值列 (用于合并)
        print("    正在准备ID和扰动值 (delta) 列...")
        # 重复ID，使其与 N * M 行匹配
        id_repeated = np.repeat(original_id_data.values, n_perturb)
        
        # 组合成最终的输出DataFrame
        # 目标: [ID, 扰动值, 特征1, 特征2, ..., 特征17]
        
        # 复制特征数据
        results_df = df_perturbed_features.copy()
        
        # 将ID和扰动值插入到最前面
        # 插入 '罩退钢卷号' (例如 '钢卷A', '钢卷A', ..., '钢卷B', ...)
        results_df.insert(0, ID_COLUMN, id_repeated)
        # 插入 '扰动值' (例如 -10, -9, ..., 10, -10, ...)
        results_df.insert(1, '扰动值', perturb_tiled)

        # 缺失值检测与清洗
        # 规则: 若一行（除ID列）全为0或空，则删除；随后将剩余NaN填充为0
        vals = results_df.drop(columns=[ID_COLUMN]).replace(r'^\s*$', np.nan, regex=True)
        numeric_vals = vals.apply(pd.to_numeric, errors='coerce')
        mask_all_zero_or_nan = numeric_vals.fillna(0).eq(0).all(axis=1)
        num_dropped = int(mask_all_zero_or_nan.sum())
        if num_dropped > 0:
            print(f"    检测到 {num_dropped} 行全为0或空，已删除。")
        results_df = results_df.loc[np.logical_not(mask_all_zero_or_nan)].reset_index(drop=True)
        results_df = results_df.fillna(0)

        # 保存结果
        safe_feature_name = sanitize_filename(feature_name)
        
        output_filename = f'05_perturbed_dataset_{safe_feature_name}.csv'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"    正在保存扰动数据集...")
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"    [成功] 扰动数据集已保存至: {output_path}")

    print("\n" + "=" * 60)
    print("--- 步骤5 (扰动数据生成) 全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()