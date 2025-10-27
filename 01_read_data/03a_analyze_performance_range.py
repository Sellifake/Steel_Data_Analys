# -*- coding: utf-8 -*-
"""
文件名: 03a_analyze_performance_range.py
本脚本的核心功能是分析性能数据中三个核心指标（抗拉强度、屈服强度、断后伸长率）
的“平均”允许范围。

主要流程:
1.  复用 03 脚本的逻辑，高效加载 performance 文件夹中的所有性能数据。
2.  仅筛选出分析所需的列（钢卷号、6个性能上下限指标）。
3.  强制将性能指标转换为数值类型，并清除无效数据（如缺失值）。
4.  计算每个钢卷（每一行）的三个性能指标的实际区间长度（最大值 - 最小值）。
5.  筛选出有效的区间（区间长度 > 0）。
6.  计算所有有效区间的平均最小值、平均最大值和平均区间长度。
7.  打印（Print）这三个指标的平均区间 [平均最小值, 平均最大值]。
"""

import pandas as pd
import os
import glob
import re

# --- 1. 全局配置区 ---

# --- 【输入路径配置】 ---
PERFORMANCE_DATA_FOLDER = 'performance'

# --- 【列定义】 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_SECTION_TITLE = '钢卷性能数据'

# 核心：定义此次分析必需的性能上下限列
TARGET_COLUMNS = [
    '抗拉强度最大值', '抗拉强度最小值',
    '屈服强度最大值', '屈服强度最小值',
    '断后伸长率最大值', '断后伸长率最小值'
]

# 定义三个指标的分析对
METRIC_PAIRS = {
    "抗拉强度": ('抗拉强度最小值', '抗拉强度最大值'),
    "屈服强度": ('屈服强度最小值', '屈服强度最大值'),
    "断后伸长率": ('断后伸长率最小值', '断后伸长率最大值')
}


# --- 2. 核心分析函数 ---

def load_performance_data_for_range_analysis(folder_path: str) -> pd.DataFrame | None:
    """
    高效加载、合并并准备性能数据，仅用于区间分析。
    1. 高效读取所有Excel文件。
    2. **仅筛选出必需的列** (ID_COLUMN 和 TARGET_COLUMNS)。
    3. 根据钢卷号去重，确保唯一性。
    (此函数基于 03_merge_clean_and_eda.py 中的函数修改)
    """
    print(f"--- 步骤 1/4: 扫描文件夹 '{folder_path}' 并准备性能数据 ---")
    if not os.path.isdir(folder_path):
        print(f"    [错误] 文件夹 '{folder_path}' 不存在。")
        return None

    excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))
    if not excel_files:
        print(f"    [错误] 在 '{folder_path}' 中未找到 .xlsx 文件。")
        return None
    
    all_dfs = []
    print(f"    发现 {len(excel_files)} 个文件，开始读取...")
    for file in excel_files:
        try:
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                header_row_index = -1
                for idx, row in df_raw.iterrows():
                    # 寻找包含特定标题的行
                    if row.astype(str).str.contains(PERFORMANCE_SECTION_TITLE, na=False).any():
                        header_row_index = idx + 1
                        break
                if header_row_index != -1 and header_row_index < len(df_raw):
                    new_header = df_raw.iloc[header_row_index]
                    df_data = df_raw.iloc[header_row_index + 1:].copy()
                    df_data.columns = new_header
                    all_dfs.append(df_data)
                    # 假设每个文件只找一个匹配的sheet
                    break
        except Exception as e:
            print(f"       [警告] 读取文件 {os.path.basename(file)} 失败: {e}")
    
    if not all_dfs:
        print("    [错误] 未能从任何文件中加载数据。")
        return None
    
    combined_perf_df = pd.concat(all_dfs, ignore_index=True)
    print(f"    读取完成，共合并 {len(combined_perf_df)} 行原始性能记录。")

    # 仅保留此次分析所需的列
    required_cols = [ID_COLUMN] + TARGET_COLUMNS
    existing_required_cols = [col for col in required_cols if col in combined_perf_df.columns]
    
    missing_cols = set(required_cols) - set(existing_required_cols)
    if missing_cols:
        print(f"    [警告] 性能数据中缺少以下目标列，将跳过: {missing_cols}")

    if ID_COLUMN not in existing_required_cols:
        print(f"    [严重错误] 性能数据中未找到关键的ID列: '{ID_COLUMN}'")
        return None
        
    perf_df_subset = combined_perf_df[existing_required_cols].copy()
    print(f"    已从性能数据中筛选出 {len(perf_df_subset.columns)} 个必需列。")

    initial_rows = len(perf_df_subset)
    # 确保ID列存在且不为空
    perf_df_subset.dropna(subset=[ID_COLUMN], inplace=True)
    # 按ID去重，保留第一个出现的记录
    perf_df_subset.drop_duplicates(subset=[ID_COLUMN], keep='first', inplace=True)
    print(f"    对性能数据去重: 从 {initial_rows} 行减少到 {len(perf_df_subset)} 行唯一钢卷记录。")
    
    return perf_df_subset

# --- 3. 主程序入口 ---

def main():
    """主程序，执行性能区间分析。"""
    print("=" * 60)
    print("--- 启动脚本: 03a_analyze_performance_range.py ---")
    print("--- 目标: 查找三个性能指标的平均区间 ---")
    print("=" * 60)

    # --- 步骤 1: 加载数据 ---
    df = load_performance_data_for_range_analysis(PERFORMANCE_DATA_FOLDER)
    if df is None:
        print("\n--- 任务终止：无法加载或准备性能数据。 ---")
        return

    # --- 步骤 2: 数据类型转换与清洗 ---
    print(f"\n--- 步骤 2/4: 转换数据类型并清洗无效数据 ---")
    
    # 确定实际存在于DataFrame中的目标列
    cols_to_convert = [col for col in TARGET_COLUMNS if col in df.columns]
    
    if not cols_to_convert:
        print("    [错误] 数据中不包含任何一个目标性能列。无法分析。")
        return

    # 强制转换为数值类型，无法转换的变为NaN
    df[cols_to_convert] = df[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    # 移除任何目标列为NaN的行
    initial_rows = len(df)
    df.dropna(subset=cols_to_convert, inplace=True)
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"    [清洗] 移除了 {dropped_rows} 行，因为它们在目标列中包含缺失值。")
    
    print(f"    数据准备完毕，剩余 {len(df)} 行有效记录用于分析。")

    # --- 步骤 3: 计算每个指标的平均区间与均值 ---
    print(f"\n--- 步骤 3/4: 计算每个指标的平均区间与均值 ---")
    
    analysis_results = {}

    for metric_name, (min_col, max_col) in METRIC_PAIRS.items():
        if min_col in df.columns and max_col in df.columns:
            # 计算区间长度
            range_col_name = f"{metric_name}_区间长度"
            df[range_col_name] = df[max_col] - df[min_col]
            
            # 筛选有效区间（长度必须大于0）
            df_valid = df[df[range_col_name] > 0].copy()
            
            if df_valid.empty:
                print(f"    [警告] 指标 '{metric_name}': 未找到任何有效区间 (最大值 > 最小值)。")
                analysis_results[metric_name] = "无有效区间"
            else:
                
                # 核心逻辑: 计算所有有效数据的均值
                mean_min = df_valid[min_col].mean()
                mean_max = df_valid[max_col].mean()
                mean_range = df_valid[range_col_name].mean()
                
                # 记录结果
                result = {
                    "mean_min": mean_min,
                    "mean_max": mean_max,
                    "mean_range": mean_range,
                    "valid_count": len(df_valid)
                }
                analysis_results[metric_name] = result
                print(f"    [分析] 指标 '{metric_name}': 基于 {len(df_valid)} 条有效记录计算均值完成。")
                
        else:
            print(f"    [跳过] 指标 '{metric_name}': 缺少 '{min_col}' 或 '{max_col}' 列。")

    # --- 步骤 4: 输出最终结果 ---
    print("\n" + "=" * 60)
    print("--- 最终分析结果：性能指标平均区间 ---")
    
    for metric_name, result in analysis_results.items():
        if isinstance(result, dict):
            # 打印均值结果
            print(f"\n  {metric_name} (基于 {result['valid_count']} 条有效数据):")
            print(f"    平均区间: [ {result['mean_min']:.4f}, {result['mean_max']:.4f} ]")
            print(f"    平均区间长度: {result['mean_range']:.4f}")
        else:
            # 打印警告信息（如"无有效区间"）
            print(f"\n  {metric_name}: {result}")
            
    print("\n" + "=" * 60)
    print("--- 步骤3a全部任务完成 ---")
    print("=" * 60)


if __name__ == '__main__':
    main()