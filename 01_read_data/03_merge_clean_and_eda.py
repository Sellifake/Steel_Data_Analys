# -*- coding: utf-8 -*-
"""
文件名:03_merge_clean_and_eda.py
本脚本的核心功能是将化学成分和性能指标合并到数据集中，其主要流程如下：
1.  加载由02代码生成的工艺特征数据 (extract_data.xlsx) 作为基准。
2.  从 performance 文件夹中高效读取所有性能与化学数据。
3.  从性能数据中，仅筛选出必需的列（钢卷号、指定的性能指标、化学成分以及新增的工艺特征）。
4.  以工艺特征数据为基准，进行左合并（left join），确保数据集的行数和基准一致。
5.  强制类型转换，确保所有分析列都为数值类型。
6.  进行详细、步骤清晰的基础数据清洗（缺失值的判断，删除缺失值所在列）。
7.  严格按照“工艺特征->化学成分->性能指标”的顺序排列最终列。
8. 保存一份合并和初步清洗后的完整数据集，作为后续步骤的输入。
9. 对这份完整数据集进行全面的探索性分析（相关性、VIF、数据分布直方图、散点图），为特征选择提供依据。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 1. 全局配置区 ---

# --- 【输入路径配置】 ---
BASE_FEATURE_FILE = 'results/extract_data.xlsx'
PERFORMANCE_DATA_FOLDER = 'performance'

# --- 【输出路径与文件配置】 ---
RESULTS_DIR = 'results'
MERGED_DATA_FILE = '03_merged_data.xlsx'
EDA_PLOT_DIR = '03_eda_plots'
CORR_HEATMAP_FILE = '03_correlation_heatmap.png'
CORR_MATRIX_FILE = '03_correlation_matrix.csv'
VIF_SCORES_FILE = '03_vif_scores.csv'

# --- 【列定义】 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_SECTION_TITLE = '钢卷性能数据'
# 核心：定义最终需要的性能和化学列
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]
CHEMISTRY_COLUMNS = [
    "碳含量", "硅含量", "锰含量", "磷含量", "硫含量", "铝含量",
    "钛含量", "铌含量", "氧含量", "氮含量"
]

# 需要从性能文件中提取的额外工艺特征
ADDITIONAL_PROCESS_FEATURES = [
    '退火卷宽度', '冷轧入口厚度', '冷轧生产实际厚度', '热轧在炉时间',
    '热轧精轧入口平均温度', '热轧精轧出口平均温度', '热轧卷取平均温度'
]

# --- 【绘图设置】 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 全局EDA绘图控制开关
# True: 执行步骤6.3，生成所有直方图和散点图 (耗时)
# False: 跳过步骤6.3，不生成图表 (仍会计算VIF和相关性CSV)
RUN_EDA_PLOTTING = True


# --- 2. 辅助与核心分析函数 ---

def sanitize_filename(filename: str) -> str:
    """移除文件名中的非法字符。"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def load_and_prepare_performance_data(folder_path: str) -> pd.DataFrame | None:
    """
    高效加载、合并并准备性能与化学数据。
    1. 高效读取所有Excel文件。
    2. **仅筛选出必需的列** (包含新增工艺特征)。
    3. 根据钢卷号去重，确保唯一性。
    """
    print(f"--- 步骤 1.2/6: 扫描文件夹 '{folder_path}' 并准备性能/化学数据 ---")
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
                    if row.astype(str).str.contains(PERFORMANCE_SECTION_TITLE, na=False).any():
                        header_row_index = idx + 1
                        break
                if header_row_index != -1 and header_row_index < len(df_raw):
                    new_header = df_raw.iloc[header_row_index]
                    df_data = df_raw.iloc[header_row_index + 1:].copy()
                    df_data.columns = new_header
                    all_dfs.append(df_data)
                    break
        except Exception as e:
            print(f"       [警告] 读取文件 {os.path.basename(file)} 失败: {e}")
    
    if not all_dfs:
        print("    [错误] 未能从任何文件中加载数据。")
        return None
    
    combined_perf_df = pd.concat(all_dfs, ignore_index=True)
    print(f"    读取完成，共合并 {len(combined_perf_df)} 行原始性能/化学记录。")

    # 将 ADDITIONAL_PROCESS_FEATURES 添加到必需列中
    required_cols = [ID_COLUMN] + PERFORMANCE_METRICS + CHEMISTRY_COLUMNS + ADDITIONAL_PROCESS_FEATURES
    existing_required_cols = [col for col in required_cols if col in combined_perf_df.columns]
    
    if ID_COLUMN not in existing_required_cols:
        print(f"    [严重错误] 性能数据中未找到关键的ID列: '{ID_COLUMN}'")
        return None
        
    perf_df_subset = combined_perf_df[existing_required_cols].copy()
    print(f"    已从性能数据中筛选出 {len(perf_df_subset.columns)} 个必需列。")

    initial_rows = len(perf_df_subset)
    perf_df_subset.dropna(subset=[ID_COLUMN], inplace=True)
    perf_df_subset.drop_duplicates(subset=[ID_COLUMN], keep='first', inplace=True)
    print(f"    对性能数据去重: 从 {initial_rows} 行减少到 {len(perf_df_subset)} 行唯一钢卷记录。")
    
    return perf_df_subset

def analyze_correlations(df: pd.DataFrame, output_dir: str):
    """
    计算**所有数值列**（工艺、化学、性能）的相关性矩阵并绘制热力图。
    """
    print("--- 步骤 6.1/6: 进行相关性分析 ---")
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr()
    
    corr_matrix.to_csv(os.path.join(output_dir, CORR_MATRIX_FILE), encoding='utf-8-sig')
    print(f"    [成功] 包含所有数值特征的相关性矩阵已保存。")
    
    plt.figure(figsize=(30, 24))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 7})
    plt.title('特征相关性热力图 (包含工艺、化学与性能)', fontsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, CORR_HEATMAP_FILE), dpi=150)
    plt.close()
    print(f"    [成功] 包含所有数值特征的相关性热力图已保存。")

def analyze_vif(df: pd.DataFrame, output_dir: str):
    """
    计算VIF，**输入特征仅包含工艺特征和化学成分**。
    """
    print("--- 步骤 6.2/6: 进行VIF分析 ---")
    features_to_check = [col for col in df.columns if col not in [ID_COLUMN] + PERFORMANCE_METRICS]
    features_df = df[features_to_check].select_dtypes(include=['number']).dropna()
    
    if features_df.shape[1] < 2:
        print("    [警告] 数值型输入特征不足，无法计算VIF。")
        return

    print(f"    将在 {features_df.shape[1]} 个输入特征 (工艺+化学) 上计算VIF...")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features_df.columns
    vif_data["VIF"] = [variance_inflation_factor(features_df.values, i) for i in range(features_df.shape[1])]
    vif_data_sorted = vif_data.sort_values('VIF', ascending=False)
    
    print("\n    --- VIF 分数报告 (仅输入特征: 工艺+化学) ---")
    print(vif_data_sorted.to_string(index=False))
    
    vif_data_sorted.to_csv(os.path.join(output_dir, VIF_SCORES_FILE), index=False, encoding='utf-8-sig')
    print(f"\n    [成功] VIF分数已保存。")

def perform_eda_plotting(df: pd.DataFrame, plot_dir: str):
    """生成分布直方图和散点图。"""
    print("--- 步骤 6.3/6: 生成EDA图表 ---")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    print("    正在生成分布直方图...")
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'特征分布 - {col}')
        plt.savefig(os.path.join(plot_dir, f'hist_{sanitize_filename(col)}.png'))
        plt.close()

    input_features = [col for col in numeric_cols if col not in [ID_COLUMN] + PERFORMANCE_METRICS]
    print("    正在生成输入特征与性能指标的散点图...")
    for feature in input_features:
        for metric in PERFORMANCE_METRICS:
            if metric in df.columns:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df, x=feature, y=metric, alpha=0.6)
                plt.title(f'关系探索 - {feature} vs {metric}')
                plt.savefig(os.path.join(plot_dir, f'scatter_{sanitize_filename(feature)}_vs_{sanitize_filename(metric)}.png'))
                plt.close()
    print(f"    [成功] 所有EDA图表已保存至: {plot_dir}")


# --- 3. 主程序入口 ---

def main():
    """主程序，执行数据合并、初步清洗和全面的探索性分析。"""
    print("=" * 60)
    print("--- 启动脚本: 03_merge_clean_and_eda.py，进行化学含量和性能指标的合并 ---")
    print("=" * 60)

    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    # --- 步骤 1: 加载与合并数据 ---
    print(f"--- 步骤 1.1/6: 加载特征主文件: {BASE_FEATURE_FILE} ---")
    if not os.path.exists(BASE_FEATURE_FILE):
        print(f"    [错误] 特征文件不存在。请先运行 '02_read_data.py'。")
        return
    df_feature = pd.read_excel(BASE_FEATURE_FILE)
    # 存储来自 02_read_data.py 的原始工艺特征列名
    process_feature_cols = df_feature.columns.tolist() 
    print(f"    加载成功，基准数据集包含 {len(df_feature)} 条工艺记录。")

    df_performance_prepared = load_and_prepare_performance_data(PERFORMANCE_DATA_FOLDER)
    if df_performance_prepared is None:
        print("\n--- 任务终止：无法加载或准备性能数据。 ---")
        return
    
    print("\n--- 步骤 1.3/6: 以工艺特征数据为基准，进行左合并 ---")
    df_merged = pd.merge(df_feature, df_performance_prepared, on=ID_COLUMN, how='left')
    print(f"    合并完成。当前数据集共 {len(df_merged)} 行。")

    # --- 步骤 2: 【关键修正】统一数据类型 ---
    print("\n--- 步骤 2/6: 统一所有特征和指标为数值类型 ---")
    # 确定需要转换为数值的列（除ID列外的所有列）
    cols_to_convert = [col for col in df_merged.columns if col != ID_COLUMN]
    
    print(f"    将对 {len(cols_to_convert)} 个列尝试强制转换为数值类型...")
    # 使用 .apply() 对选定的所有列执行 pd.to_numeric
    # errors='coerce' 会将无法转换的值（如文本）替换为NaN
    df_merged[cols_to_convert] = df_merged[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    
    print("    [成功] 类型转换完成。任何非数值数据已被转换为缺失值 (NaN)。")


    # --- 步骤 3: 基础数据清洗 ---
    print("\n--- 步骤 3/6: 进行基础数据清洗与缺失值处理 ---")
    current_rows = len(df_merged)
    print(f"    清洗前总行数: {current_rows}")

    if '冷点_峰值温度' in df_merged.columns:
        df_merged = df_merged[df_merged['冷点_峰值温度'] != 0].copy()
        rows_after = len(df_merged)
        if current_rows > rows_after:
            print(f"    [操作] 原因: '冷点_峰值温度'为0。移除了 {current_rows - rows_after} 行。")
            current_rows = rows_after

    chem_cols_in_df = [col for col in CHEMISTRY_COLUMNS if col in df_merged.columns]
    if chem_cols_in_df:
        is_all_zero = (df_merged[chem_cols_in_df] == 0).all(axis=1)
        is_all_na = df_merged[chem_cols_in_df].isnull().all(axis=1)
        df_merged = df_merged[~(is_all_zero | is_all_na)]
        rows_after = len(df_merged)
        if current_rows > rows_after:
            print(f"    [操作] 原因: 化学成分数据全部为0或全部缺失。移除了 {current_rows - rows_after} 行。")
            current_rows = rows_after
    
    perf_cols_in_df = [col for col in PERFORMANCE_METRICS if col in df_merged.columns]
    df_merged.dropna(subset=perf_cols_in_df, inplace=True)
    rows_after = len(df_merged)
    if current_rows > rows_after:
        print(f"    [操作] 原因: 目标变量(性能指标)缺失。移除了 {current_rows - rows_after} 行。")
        current_rows = rows_after
    
    print(f"\n    清洗完成，最终剩余 {len(df_merged)} 行有效数据。")

    # 步骤 3.5: 对指定的新增特征进行中位数填充
    print("\n--- 步骤 3.5/6: 对新增工艺特征执行中位数填充 ---")
    for col in ADDITIONAL_PROCESS_FEATURES:
        if col in df_merged.columns:
            # 1. 检查是否存在缺失值
            missing_count = df_merged[col].isnull().sum()
            if missing_count > 0:
                # 2. 计算中位数
                median_val = df_merged[col].median()
                # 3. 填充
                df_merged[col].fillna(median_val, inplace=True)
                print(f"    [填充] 特征 '{col}': {missing_count} 个缺失值已用中位数 ({median_val:.4f}) 填充。")
            else:
                print(f"    [跳过] 特征 '{col}': 无缺失值。")
        else:
            print(f"    [警告] 特征 '{col}': 在数据集中不存在，跳过填充。")

    # 步骤 3.6: 计算新特征 '压下率'
    print("\n--- 步骤 3.6/6: 计算新特征 '压下率' ---")
    if '冷轧入口厚度' in df_merged.columns and '冷轧生产实际厚度' in df_merged.columns:
        # 确保分母（冷轧入口厚度）不为0
        denominator = df_merged['冷轧入口厚度']
        numerator = df_merged['冷轧入口厚度'] - df_merged['冷轧生产实际厚度']
        
        # 使用 np.where 避免除零错误
        df_merged['压下率'] = np.where(denominator == 0, 0, numerator / denominator)
        print("    [成功] 已计算新特征 '压下率'。")
    else:
        print("    [警告] 计算'压下率'失败：缺少'冷轧入口厚度'或'冷轧生产实际厚度'列。")


    # 步骤 4: 重新排列列顺序
    print("\n--- 步骤 4/6: 按照“工艺-化学-性能”顺序重新排列数据列 ---")
    
    # 1. 原始工艺特征 (来自 02_read_data.py, 排除ID)
    process_cols_final = [col for col in process_feature_cols if col in df_merged.columns and col != ID_COLUMN]
    
    # 2. 新增工艺特征 (来自 performance 文件夹)
    new_process_cols_final = [col for col in ADDITIONAL_PROCESS_FEATURES if col in df_merged.columns]
    
    # 3. 添加 '压下率' (如果存在)
    if '压下率' in df_merged.columns and '压下率' not in new_process_cols_final:
        new_process_cols_final.append('压下率')
    
    # 4. 化学成分
    chem_cols_final = [col for col in CHEMISTRY_COLUMNS if col in df_merged.columns]
    
    # 5. 性能指标
    perf_cols_final = [col for col in PERFORMANCE_METRICS if col in df_merged.columns]

    # 6. 组合所有期望的列
    # 顺序：ID -> (原始工艺) -> (新增工艺+压下率) -> (化学) -> (性能)
    expected_order = [ID_COLUMN] + process_cols_final + new_process_cols_final + chem_cols_final + perf_cols_final
    
    # 7. 去重并保留顺序 (使用 dict.fromkeys)
    ordered_cols = list(dict.fromkeys(expected_order))
    
    # 8. 找出遗漏的列 (存在于数据框但未被定义的列)
    defined_set = set(ordered_cols)
    all_cols_set = set(df_merged.columns)
    other_cols = [col for col in df_merged.columns if col not in defined_set]
    
    if other_cols:
        print(f"    [警告] 发现未明确排序的列: {other_cols}。已将它们放置在化学成分之前。")
        # 将 "other_cols" 插入到化学成分之前
        final_column_order = [ID_COLUMN] + process_cols_final + new_process_cols_final + other_cols + chem_cols_final + perf_cols_final
        # 再次去重
        ordered_cols = list(dict.fromkeys(final_column_order))
    
    # 9. 确保最终列表中的所有列都实际存在于 df_merged 中
    final_columns = [col for col in ordered_cols if col in df_merged.columns]
        
    df_final = df_merged[final_columns]
    print(f"    列顺序重排完成。最终数据集包含 {len(df_final.columns)} 列。")


    # --- 步骤 5: 保存合并后的数据 ---
    print("\n--- 步骤 5/6: 保存合并和初步清洗后的数据 ---")
    output_merged_path = os.path.join(RESULTS_DIR, MERGED_DATA_FILE)
    df_final.to_excel(output_merged_path, index=False)
    print(f"    [成功] 数据已保存至: {output_merged_path}")

    # --- 步骤 6: 全面探索性分析 ---
    print("\n--- 步骤 6/6: 对清洗后的完整数据进行全面EDA ---")
    analyze_correlations(df_final, RESULTS_DIR)
    analyze_vif(df_final, RESULTS_DIR)
    
    # 根据全局开关决定是否执行绘图
    if RUN_EDA_PLOTTING:
        perform_eda_plotting(df_final, os.path.join(RESULTS_DIR, EDA_PLOT_DIR))
    else:
        print("--- 步骤 6.3/6: 已跳过生成EDA图表 (RUN_EDA_PLOTTING = False) ---")

    print("\n" + "=" * 60)
    print("--- 步骤3全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()