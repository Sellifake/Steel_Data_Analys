# -*- coding: utf-8 -*-
"""
文件名：02_read_data.py
本脚本的核心功能是根据特征定义计算出特征值，其主要流程如下：
1.  从预处理阶段生成的Parquet文件 ('all_coils_data.parquet') 中读取所有钢卷的时间序列数据。
2.  对每个钢卷，根据其温度曲线（控制、热点、冷点）的变化趋势，划分出升温、保温、降温三个工艺阶段。
3.  基于划分的阶段，计算一系列详细的工艺特征，例如各阶段的时长、温升/温降幅度、平均速率、峰值温度等。
4.  对计算出的特征集进行去重，移除那些工艺过程完全相同的重复记录。
5.  将最终的特征数据保存到一个Excel文件 ('results/extract_data.xlsx') 中。
6.  将去重后所有有效的钢卷号保存到一个总的文本文件 ('steel_coil_ids.txt')，并进一步将其分割成多个小文件，
    存放在指定文件夹 ('钢卷编号') 中，便于分批处理。
"""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

# --- 1. 全局配置区 ---

# --- 【路径配置】 ---
PREPROCESSED_INPUT_FILE = 'all_coils_data.parquet'
OUTPUT_FILENAME = 'results/02_extract_data.xlsx'
COIL_IDS_FILENAME = 'steel_coil_ids.txt'
SPLIT_COIL_IDS_FOLDER = '钢卷编号'

# --- 【参数配置】 ---
# 每个分割后的钢卷号文件的包含数量
COIL_IDS_CHUNK_SIZE = 400


# --- 2. 核心功能函数 ---

def get_feature_name_list() -> List[str]:
    """
    此列表定义了最终输出Excel文件中的列顺序。
    """
    return [
        '罩退钢卷号', '升温开始时刻', '保温开始时刻', '降温开始时刻', '工艺结束时刻',
        '总工艺时长', '热点_峰值温度', '热点_峰值温度时刻', '冷点_峰值温度', '冷点_峰值温度时刻',
        
        # 升温阶段
        '升温时长', '升温时长占比',
        '热点_升温起始温度', '热点_升温结束温度', '热点_升温幅度', '热点_平均升温速率', '热点_升温速率标准差', '热点_最大瞬时升温速率',
        '冷点_升温起始温度', '冷点_升温结束温度', '冷点_升温幅度', '冷点_平均升温速率', '冷点_升温速率标准差', '冷点_最大瞬时升温速率',
        
        # 保温阶段
        '保温时长', '保温时长占比',
        '热点_保温起始温度', '热点_保温结束温度', '热点_保温阶段温降', '热点_保温平均温度', '热点_保温温度标准差',
        '冷点_保温起始温度', '冷点_保温结束温度', '冷点_保温阶段温降', '冷点_保温平均温度', '冷点_保温温度标准差',
        
        # 降温阶段
        '降温时长', '降温时长占比',
        '热点_降温起始温度', '热点_降温结束温度', '热点_降温幅度', '热点_平均降温速率', '热点_降温速率标准差', '热点_最大瞬时降温速率', 
        '冷点_降温起始温度', '冷点_降温结束温度', '冷点_降温幅度', '冷点_平均降温速率', '冷点_降温速率标准差', '冷点_最大瞬时降温速率',
        
        # 差异特征
        '峰值温度差', '峰值时刻差', '升温起始温度差', '升温结束温度差', '平均升温速率差',
        '保温起始温度差', '保温结束温度差', '保温平均温度差', '降温起始温度差', '降温结束温度差', '平均降温速率差'
    ]


def calculate_new_features(df: pd.DataFrame) -> Dict[str, Any]:
    """
    根据单个钢卷的时间序列数据，计算所有预定义的工艺特征。
    采用三阶段划分法（升温、保温、降温）进行计算。

    Args:
        df (pd.DataFrame): 单个钢卷的数据，需包含时间及各温度曲线。

    Returns:
        Dict[str, Any]: 一个包含所有计算出的特征及其值的字典。
    """
    feature_names = get_feature_name_list()
    features = {name: None for name in feature_names if name != '罩退钢卷号'}
    
    guide_curve = '控制温度实际温度'
    hot_target_curve = '控制温度(热点温度)'
    cold_curve = '压紧温度(冷点温度)'
    time_col = '时间(s)'
    cover_temp_curve = '加热罩实际温度'

    try:
        has_hot_target_curve = hot_target_curve in df.columns and not df[hot_target_curve].isnull().all()
        has_cold_curve = cold_curve in df.columns and not df[cold_curve].isnull().all()

        cover_temp_series = df[cover_temp_curve]
        first_hot_indices = cover_temp_series.index[cover_temp_series > 0]
        if first_hot_indices.empty: return features
        t_start_idx = first_hot_indices[0]
        
        process_df = df.loc[t_start_idx:].reset_index(drop=True)
        
        t_soak_start_idx_local = process_df[guide_curve].idxmax()
        t_soak_start_idx = process_df.index[t_soak_start_idx_local] + t_start_idx

        if pd.isna(t_soak_start_idx): return features

        soaking_cooling_df = df.loc[t_soak_start_idx:]
        temp_diff = soaking_cooling_df[guide_curve].diff()
        first_drop_indices = temp_diff.index[temp_diff < 0]
        t_cool_start_idx = first_drop_indices[0] if not first_drop_indices.empty else None
        
        t_end_idx = df.index[-1]

        features['升温开始时刻'] = df.loc[t_start_idx, time_col]
        features['保温开始时刻'] = df.loc[t_soak_start_idx, time_col]
        features['工艺结束时刻'] = df.loc[t_end_idx, time_col]
        if t_cool_start_idx is not None:
             features['降温开始时刻'] = df.loc[t_cool_start_idx, time_col]
        
        t_start, t_soak_start, t_cool_start, t_end = (features['升温开始时刻'], features['保温开始时刻'], features['降温开始时刻'], features['工艺结束时刻'])
        
        features['总工艺时长'] = t_end - t_start if t_end is not None and t_start is not None else None
        
        if has_hot_target_curve:
            hot_peak_idx_local = process_df[hot_target_curve].idxmax()
            hot_peak_idx = process_df.index[hot_peak_idx_local] + t_start_idx
            features['热点_峰值温度'] = df.loc[hot_peak_idx, hot_target_curve]
            features['热点_峰值温度时刻'] = df.loc[hot_peak_idx, time_col]
        if has_cold_curve:
            cold_peak_idx_local = process_df[cold_curve].idxmax()
            cold_peak_idx = process_df.index[cold_peak_idx_local] + t_start_idx
            features['冷点_峰值温度'] = df.loc[cold_peak_idx, cold_curve]
            features['冷点_峰值温度时刻'] = df.loc[cold_peak_idx, time_col]

        curves_to_process = []
        if has_hot_target_curve: curves_to_process.append((hot_target_curve, '热点'))
        if has_cold_curve: curves_to_process.append((cold_curve, '冷点'))

        if t_soak_start is not None and t_start is not None and t_soak_start >= t_start:
            stage_df = df.loc[t_start_idx:t_soak_start_idx]
            duration = t_soak_start - t_start
            features['升温时长'] = duration
            if features['总工艺时长'] and features['总工艺时长'] > 0: features['升温时长占比'] = duration / features['总工艺时长']
            
            for curve, prefix in curves_to_process:
                start_temp, end_temp = stage_df.loc[t_start_idx, curve], stage_df.loc[t_soak_start_idx, curve]
                temp_change = end_temp - start_temp
                features[f'{prefix}_升温起始温度'], features[f'{prefix}_升温结束温度'], features[f'{prefix}_升温幅度'] = start_temp, end_temp, temp_change
                if duration > 0: features[f'{prefix}_平均升温速率'] = temp_change / duration
                
                if not stage_df.empty and len(stage_df) > 1:
                    rates = stage_df[curve].diff() / stage_df[time_col].diff()
                    features[f'{prefix}_升温速率标准差'], features[f'{prefix}_最大瞬时升温速率'] = rates.std(), rates.max()

        if t_cool_start is not None and t_soak_start is not None and t_cool_start > t_soak_start:
            stage_df = df.loc[t_soak_start_idx:t_cool_start_idx]
            duration = t_cool_start - t_soak_start
            features['保温时长'] = duration
            if features['总工艺时长'] and features['总工艺时长'] > 0: features['保温时长占比'] = duration / features['总工艺时长']
                
            for curve, prefix in curves_to_process:
                start_temp, end_temp = stage_df.loc[t_soak_start_idx, curve], stage_df.loc[t_cool_start_idx, curve]
                features[f'{prefix}_保温起始温度'], features[f'{prefix}_保温结束温度'] = start_temp, end_temp
                features[f'{prefix}_保温阶段温降'], features[f'{prefix}_保温平均温度'], features[f'{prefix}_保温温度标准差'] = start_temp - end_temp, stage_df[curve].mean(), stage_df[curve].std()
        
        if t_cool_start is not None and t_end is not None and t_end > t_cool_start:
            stage_df = df.loc[t_cool_start_idx:t_end_idx]
            duration = t_end - t_cool_start
            features['降温时长'] = duration
            if features['总工艺时长'] and features['总工艺时长'] > 0: features['降温时长占比'] = duration / features['总工艺时长']

            for curve, prefix in curves_to_process:
                start_temp, end_temp = stage_df.loc[t_cool_start_idx, curve], stage_df.loc[t_end_idx, curve]
                temp_change = start_temp - end_temp
                features[f'{prefix}_降温起始温度'], features[f'{prefix}_降温结束温度'], features[f'{prefix}_降温幅度'] = start_temp, end_temp, temp_change
                if duration > 0: features[f'{prefix}_平均降温速率'] = temp_change / duration
                
                if not stage_df.empty and len(stage_df) > 1:
                    rates = (stage_df[curve].diff() / stage_df[time_col].diff()).abs()
                    features[f'{prefix}_降温速率标准差'], features[f'{prefix}_最大瞬时降温速率'] = rates.std(), rates.max()

        if has_hot_target_curve and has_cold_curve:
            def calculate_diff(key1, key2):
                return features.get(key1) - features.get(key2) if features.get(key1) is not None and features.get(key2) is not None else None
            
            features['峰值温度差'] = calculate_diff('热点_峰值温度', '冷点_峰值温度')
            features['峰值时刻差'] = calculate_diff('热点_峰值温度时刻', '冷点_峰值温度时刻')
            features['升温起始温度差'] = calculate_diff('热点_升温起始温度', '冷点_升温起始温度')
            features['升温结束温度差'] = calculate_diff('热点_升温结束温度', '冷点_升温结束温度')
            features['平均升温速率差'] = calculate_diff('热点_平均升温速率', '冷点_平均升温速率')
            features['保温起始温度差'] = calculate_diff('热点_保温起始温度', '冷点_保温起始温度')
            features['保温结束温度差'] = calculate_diff('热点_保温结束温度', '冷点_保温结束温度')
            features['保温平均温度差'] = calculate_diff('热点_保温平均温度', '冷点_保温平均温度')
            features['降温起始温度差'] = calculate_diff('热点_降温起始温度', '冷点_降温起始温度')
            features['降温结束温度差'] = calculate_diff('热点_降温结束温度', '冷点_降温结束温度')
            features['平均降温速率差'] = calculate_diff('热点_平均降温速率', '冷点_平均降温速率')

    except Exception:
        return {name: None for name in feature_names if name != '罩退钢卷号'}
        
    return features


def split_coil_ids_file(main_filename: str, folder: str, chunk_size: int):
    """
    读取主钢卷号文件，并将其分割成多个小文件，方便分批处理。
    """
    print(f"\n--- 步骤 4/4: 开始分割钢卷号文件 ---")
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"    已创建文件夹: '{folder}'")
        with open(main_filename, 'r', encoding='utf-8') as f:
            coil_ids = [cid.strip() for cid in f.read().split(',') if cid.strip()]
        
        total_ids = len(coil_ids)
        print(f"    共有 {total_ids} 个有效钢卷号待分割。")

        num_files = 0
        for i in range(0, total_ids, chunk_size):
            chunk = coil_ids[i:i + chunk_size]
            file_index = (i // chunk_size) + 1
            output_path = os.path.join(folder, f'steel_coil_ids_{file_index}.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(','.join(chunk))
            num_files += 1
        
        print(f"    [成功] 已将钢卷号分割成 {num_files} 个文件，保存在 '{folder}' 文件夹中。")

    except FileNotFoundError:
        print(f"    [警告] 主钢卷号文件 '{main_filename}' 未找到，跳过分割步骤。")
    except Exception as e:
        print(f"    [错误] 分割钢卷号文件时发生错误: {e}")


def main():
    """主程序：从Parquet读取数据，计算特征，去重，并保存结果。"""
    print("=" * 60)
    print("--- 启动脚本: 01_read_data.py ---")
    print("=" * 60)

    print(f"--- 步骤 1/4: 从 {PREPROCESSED_INPUT_FILE} 快速读取数据 ---")
    if not os.path.exists(PREPROCESSED_INPUT_FILE):
        print(f"    [错误] 预处理文件 '{PREPROCESSED_INPUT_FILE}' 不存在。请先运行01_preprocess_data.py")
        return
    
    all_data_df = pd.read_parquet(PREPROCESSED_INPUT_FILE)
    grouped = all_data_df.groupby('罩退钢卷号')
    print(f"    [成功] 读取完毕，共找到 {len(grouped)} 个钢卷的数据。")
    
    print(f"\n--- 步骤 2/4: 开始为所有钢卷计算工艺特征 ---")
    results_list = []
    for coil_id, df_single_coil in tqdm(grouped, desc="计算特征进度"):
        calculated_features = calculate_new_features(df_single_coil.reset_index(drop=True))
        result = {'罩退钢卷号': coil_id, **calculated_features}
        results_list.append(result)
    
    print("\n--- 步骤 3/4: 开始进行数据后处理与保存 ---")
    if not results_list:
        print("    [错误] 未能计算出任何有效特征，无法生成结果文件。")
        return

    results_df = pd.DataFrame(results_list)
    
    feature_columns = [col for col in results_df.columns if col != '罩退钢卷号']
    original_count = len(results_df)
    results_df.dropna(subset=feature_columns, how='all', inplace=True)
    results_df.drop_duplicates(subset=feature_columns, keep='first', inplace=True)
    deduplicated_count = len(results_df)
    print(f"    数据去重完成: 原始记录 {original_count} 条，去重后保留 {deduplicated_count} 条唯一工艺记录。")
    
    column_order = get_feature_name_list()
    results_df = results_df.loc[:, ~results_df.columns.duplicated()]
    
    for col in column_order:
        if col not in results_df.columns:
            results_df[col] = None
    results_df = results_df[column_order]
    
    results_df.to_excel(OUTPUT_FILENAME, index=False, engine='openpyxl')
    print(f"    [成功] 新特征数据已保存到: {OUTPUT_FILENAME}")
    
    valid_coil_ids = results_df['罩退钢卷号'].astype(str).dropna().unique().tolist()
    with open(COIL_IDS_FILENAME, 'w', encoding='utf-8') as f:
        f.write(",".join(valid_coil_ids))
    print(f"    [成功] 总钢卷号列表已保存到: {COIL_IDS_FILENAME}")
    
    split_coil_ids_file(COIL_IDS_FILENAME, SPLIT_COIL_IDS_FOLDER, COIL_IDS_CHUNK_SIZE)
    
    print("\n" + "=" * 60)
    print("--- 全部任务顺利完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()