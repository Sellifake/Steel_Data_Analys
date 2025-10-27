# -*- coding: utf-8 -*-
"""
文件名：01_preprocess_data.py
本脚本用于数据预处理，其主要功能是：
1. 遍历指定文件夹（'data_origin'）下的所有原始Excel文件。
2. 从每个文件的每个工作表（Sheet）中提取钢卷的时间序列数据。
3. 对提取的数据进行清洗，包括：统一列名、处理重复列、转换数据类型、去除无效数据等。
4. 基于钢卷号进行去重，确保每个钢卷的数据只被处理一次。
5. 将所有清洗干净的钢卷数据合并，并保存为一个高效的Parquet格式文件（'all_coils_data.parquet'），
   以供后续进行快速的数据分析和建模。
"""
import os
import pandas as pd
from typing import Dict, Any, Optional
from tqdm import tqdm

# --- 全局配置 ---
# 存放原始Excel文件的文件夹路径
ORIGIN_DATA_PATH = 'data_origin'
# 预处理后输出的Parquet文件名
PREPROCESSED_OUTPUT_FILE = 'all_coils_data.parquet'
# 数据表中必须包含的列，如果缺少则该钢卷数据被视为无效
MIN_REQUIRED_COLS = ['时间(s)', '控制温度实际温度', '加热罩实际温度']
# 所有需要转换为数值类型的列
ALL_POSSIBLE_NUMERIC_COLS = ['时间(s)', '控制温度实际温度', '压紧温度(冷点温度)', '加热罩实际温度', '控制温度(热点温度)']


def find_value_by_label(df: pd.DataFrame, label: str) -> Optional[Any]:
    """
    在DataFrame的前几行中查找指定的标签文本，并返回其右侧相邻单元格的值。
    用于从文件头信息中提取钢卷号等单个信息。

    Args:
        df (pd.DataFrame): 待搜索的DataFrame，通常是Excel文件的头部区域。
        label (str): 要查找的标签字符串，例如 "钢卷号"。

    Returns:
        Optional[Any]: 找到的标签右侧单元格的值，如果未找到则返回 None。
    """
    for row_idx in range(df.shape[0]):
        for col_idx in range(df.shape[1] - 1):
            cell_value = df.iloc[row_idx, col_idx]
            if isinstance(cell_value, str) and label in cell_value:
                return df.iloc[row_idx, col_idx + 1]
    return None


def find_header_row(df: pd.DataFrame, anchor: str) -> Optional[int]:
    """
    在DataFrame中根据锚点文本（数据表的第一列列名）定位数据表的表头所在的行索引。

    Args:
        df (pd.DataFrame): 待搜索的DataFrame。
        anchor (str): 表头第一列的文本，例如 "TIMECOUNTER"。

    Returns:
        Optional[int]: 表头所在的行索引，如果未找到则返回 None。
    """
    for row_idx in range(df.shape[0]):
        # strip() 用于去除可能的空格
        if str(df.iloc[row_idx, 0]).strip() == anchor:
            return row_idx
    return None


def extract_sheet_data_integrated(file_path: str, sheet_name: str) -> Optional[Dict[str, Any]]:
    """
    从单个Excel工作表中提取、清洗并格式化钢卷数据。

    Args:
        file_path (str): Excel文件的完整路径。
        sheet_name (str): 要处理的工作表名称。

    Returns:
        Optional[Dict[str, Any]]: 如果处理成功，返回一个包含钢卷号和对应数据DataFrame的字典。
                                  如果工作表无效或处理失败，则返回 None。
    """
    try:
        # 1. 扫描文件头部，获取元信息（如钢卷号）和表头位置
        scan_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=40, engine='openpyxl')
        
        # 查找钢卷号
        steel_coil_id = find_value_by_label(scan_df, "钢卷号")
        if steel_coil_id is None:
            # 如果没有钢卷号，说明不是有效数据页，直接跳过
            return None
        
        # 定位数据表头
        header_row_index = find_header_row(scan_df, "TIMECOUNTER")
        if header_row_index is None:
            # 如果找不到表头，也视为无效数据页
            return None
            
        # 2. 读取主要数据区域
        data_df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index, engine='openpyxl')
        
        # 3. 数据清洗流程
        if not data_df.empty:
            # 将表头的下一行作为实际的列名
            data_df.columns = data_df.iloc[0].astype(str).str.strip().tolist()
            data_df = data_df.iloc[1:].reset_index(drop=True)
            # 统一括号格式，便于后续处理
            data_df.columns = [col.replace('（', '(').replace('）', ')') for col in data_df.columns]
        
        # 处理重复的列名，只保留第一个
        if data_df.columns.has_duplicates:
            data_df = data_df.loc[:, ~data_df.columns.duplicated(keep='first')]
        # 删除所有值都为空的列
        data_df.dropna(axis=1, how='all', inplace=True)
        
        # 检查是否包含所有必需的列
        if not all(col in data_df.columns for col in MIN_REQUIRED_COLS):
            return None 

        # 将指定列转换为数值类型，无法转换的值会变为NaN
        cols_to_numeric = [col for col in ALL_POSSIBLE_NUMERIC_COLS if col in data_df.columns]
        for col in cols_to_numeric:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
                
        # 删除在关键列上存在NaN值的行
        data_df.dropna(subset=MIN_REQUIRED_COLS, inplace=True)
        # 按时间排序
        data_df = data_df.sort_values(by='时间(s)').reset_index(drop=True)
        
        # 如果清洗后数据为空，则判定为无效
        if data_df.empty:
            return None

        # 4. 为数据添加钢卷号标识，用于后续合并
        data_df['罩退钢卷号'] = steel_coil_id
        
        return {"steel_coil_id": steel_coil_id, "data": data_df}
    
    except Exception as e:
        # 处理异常情况
        print(f"    [警告] 处理文件 '{os.path.basename(file_path)}' 的工作表 '{sheet_name}' 时出错: {e}")
        return None


def main():
    """
    主函数，执行整个数据预处理流程：
    1. 查找并遍历所有原始Excel文件。
    2. 对每个文件中的每个工作表进行数据提取和清洗。
    3. 合并所有有效钢卷的数据。
    4. 将最终结果保存为Parquet文件。
    """
    print("--- (1/3) 开始扫描并读取原始数据文件 ---")
    if not os.path.exists(ORIGIN_DATA_PATH):
        print(f"    [错误] 原始数据文件夹 '{ORIGIN_DATA_PATH}' 不存在，请检查路径配置。")
        return
        
    excel_files = [f for f in os.listdir(ORIGIN_DATA_PATH) if f.endswith('.xlsx') and not f.startswith('~$')]
    if not excel_files:
        print(f"    [错误] 在文件夹 '{ORIGIN_DATA_PATH}' 中没有找到任何 .xlsx 文件。")
        return
    
    print(f"    发现 {len(excel_files)} 个Excel文件，开始进行预处理...")
    
    all_coil_dfs = []
    processed_ids = set()  # 使用集合来存储已处理的钢卷号，用于高效去重

    # 使用tqdm创建进度条
    pbar = tqdm(excel_files, desc="预处理Excel文件进度")
    for filename in pbar:
        file_path = os.path.join(ORIGIN_DATA_PATH, filename)
        try:
            # 一次性读取Excel文件，获取所有sheet名称，效率更高
            xls = pd.ExcelFile(file_path, engine='openpyxl')
            for sheet_name in xls.sheet_names:
                extracted_data = extract_sheet_data_integrated(file_path, sheet_name)
                # 只有在数据有效且钢卷号未被处理过的情况下，才添加
                if extracted_data and extracted_data["steel_coil_id"] not in processed_ids:
                    all_coil_dfs.append(extracted_data["data"])
                    processed_ids.add(extracted_data["steel_coil_id"])
        except Exception as e:
            print(f"    [严重错误] 读取或处理文件 {filename} 时发生意外: {e}\n")
    
    print(f"\n--- (2/3) 原始数据读取与清洗完毕 ---")
    print(f"    成功提取了 {len(all_coil_dfs)} 个不重复的有效钢卷数据。\n")

    if not all_coil_dfs:
        print("    [警告] 未能提取到任何有效数据，程序终止。")
        return
        
    print(f"--- (3/3) 开始合并数据并保存为Parquet文件 ---")
    try:
        # 将所有DataFrame合并成一个
        combined_df = pd.concat(all_coil_dfs, ignore_index=True)
        
        # 保存为Parquet格式，读取速度快，占用空间小
        # 需要安装 pyarrow: pip install pyarrow
        combined_df.to_parquet(PREPROCESSED_OUTPUT_FILE, index=False)
        print(f"    成功！所有数据已合并并保存至: {PREPROCESSED_OUTPUT_FILE}")
        print(f"    - 总行数: {len(combined_df)}")
        print(f"    - 总列数: {len(combined_df.columns)}")
        print(f"    - 涉及钢卷数: {combined_df['罩退钢卷号'].nunique()}")
        
    except Exception as e:
        print(f"    [严重错误] 保存Parquet文件时失败: {e}")
        print("    请确保已安装 'pyarrow' 库 (pip install pyarrow)。")

    print("\n--- 所有预处理任务完成 ---")


if __name__ == '__main__':
    main()
