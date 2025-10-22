# -*- coding: utf-8 -*-
"""
文件名: 03a_lightgbm_modeling.py
功能: 
    1. 加载01_read_data文件夹中经过特征筛选后的最终数据集。
    2. 对每个性能指标，使用设置了随机种子的K-折交叉验证来评估 LightGBM 模型的性能。
    3. 在完整数据集上训练最终模型，并进行保存，确保模型可复现。
    4. 生成并保存特征重要性 (图+CSV)。
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb # 导入 LightGBM
from sklearn.model_selection import KFold, cross_validate

# --- 1. 配置区 ---

# --- 输入/输出文件配置 ---
INPUT_DIR = '../01_read_data/results'
OUTPUT_DIR = 'results/03_lightgbm_results' # 更改为LightGBM的特定输出目录
# 使用01_read_data生成的最终清理数据
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 模型配置 ---
# 设置全局随机种子以确保实验可复现
RANDOM_STATE = 42
K_FOLDS = 5

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


def sanitize_filename(filename):
    """(复制自 01a) 清洗文件名中的非法字符。"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)



def main():
    """主执行函数"""
    print("=" * 60)
    print("--- 启动脚本: 03a_lightgbm_modeling.py ---")
    print("=" * 60)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: '{OUTPUT_DIR}'")

    # --- 1. 加载和预处理数据 ---
    input_file_path = os.path.join(INPUT_DIR, CLEANED_DATA_FILE)
    try:
        df = pd.read_excel(input_file_path)
        print(f"成功加载数据: {input_file_path}")
    except FileNotFoundError:
        print(f"[严重错误] 清洗后的文件未找到 '{input_file_path}'。请先运行01_read_data文件夹中的脚本。")
        return

    X_all = df.drop(columns=[ID_COLUMN] + PERFORMANCE_METRICS, errors='ignore').select_dtypes(include=np.number)
    Y_all = df[PERFORMANCE_METRICS]
    print(f"数据准备完成，共包含 {len(X_all.columns)} 个输入特征。")

    # --- 2. 循环为每个性能指标建模 ---
    cv_performance_summary = []

    for metric in Y_all.columns:
        print(f"\n\n{'='*20} 正在为性能指标: '{metric}' 进行建模 (LightGBM) {'='*20}")
        safe_metric_name = sanitize_filename(metric)

        # 准备当前目标的数据集 (数据完整性检测，移除包含NaN的行)
        metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna()
        if metric_data.empty:
            print(f"    [警告] 移除NaN后，没有剩余数据可用于评估 '{metric}'，跳过此指标。")
            continue
        
        X_train = metric_data[X_all.columns]
        y_train = metric_data[metric]
        
        # LightGBM 不需要 base_score 修复

        # --- 2a. K-折交叉验证 ---
        print(f"\n--- (1/3) 正在执行 {K_FOLDS}-折交叉验证 ---")
        
        # 实例化 LightGBM 回归模型
        # 为确保可复现性，除了 random_state，还需设置 boosting_type='gbdt' (默认)
        # 增加 'verbose=-1' 关闭LGBM的啰嗦输出
        model = lgb.LGBMRegressor(
            n_estimators=100, 
            random_state=RANDOM_STATE, 
            n_jobs=-1,
            verbose=-1 # 关闭训练过程中的输出
        )
        
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        scores = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring_metrics)
        
        # 打印每一折的详细结果
        print(f"    --- 每一折的详细结果 ---")
        for fold in range(K_FOLDS):
            r2_fold = scores['test_r2'][fold]
            mae_fold = -scores['test_neg_mean_absolute_error'][fold]
            rmse_fold = np.sqrt(-scores['test_neg_mean_squared_error'][fold])
            print(f"    第{fold+1}折: R2={r2_fold:.4f}, MAE={mae_fold:.4f}, RMSE={rmse_fold:.4f}")
        
        mean_r2 = np.mean(scores['test_r2'])
        mean_mae = -np.mean(scores['test_neg_mean_absolute_error'])
        mean_rmse = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))
        
        cv_performance_summary.append({
            '性能指标': metric, 'R2 Score (平均值)': f"{mean_r2:.4f}",
            'MAE (平均值)': f"{mean_mae:.4f}", 'RMSE (平均值)': f"{mean_rmse:.4f}"
        })
        print(f"    交叉验证完成 (平均值): R2={mean_r2:.4f}, MAE={mean_mae:.4f}, RMSE={mean_rmse:.4f}")

        # --- 2b. 训练最终模型并保存 ---
        print("\n--- (2/3) 正在训练最终模型并保存 ---")
        
        # 实例化最终的 LightGBM 模型
        final_model = lgb.LGBMRegressor(
            n_estimators=100, 
            random_state=RANDOM_STATE, 
            n_jobs=-1,
            verbose=-1
        )
        final_model.fit(X_train, y_train)

        model_save_dir = os.path.join(OUTPUT_DIR, 'saved_models')
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
        model_path = os.path.join(model_save_dir, f'{safe_metric_name}_model.joblib')
        
        # LightGBM 模型同样可以使用 joblib 保存
        joblib.dump(final_model, model_path)
        print(f"    模型文件已保存至: {model_path}")

        # --- 2c. 模型解释与分析 (特征重要性) ---
        print("\n--- (3/3) 正在进行模型解释 (特征重要性) ---")
        
        # 特征重要性 (LightGBM 同样支持 .feature_importances_)
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': final_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        csv_path = os.path.join(OUTPUT_DIR, f'{safe_metric_name}_feature_importance.csv')
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        plt.figure(figsize=(12, 8))
        importance_df.head(20).sort_values(by='Importance', ascending=True).plot(kind='barh', x='Feature', y='Importance', legend=False, figsize=(12, 8))
        plt.title(f'"{metric}" 的Top 20重要特征 (LightGBM)', fontsize=16)
        plt.xlabel('特征重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f'{safe_metric_name}_feature_importance.png')
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"    - 特征重要性分析完成 (CSV和图表已保存)。")


    # --- 3. 汇总并保存交叉验证结果 ---
    if cv_performance_summary:
        summary_df = pd.DataFrame(cv_performance_summary)
        print(f"\n\n--- LightGBM 交叉验证性能汇总表 ({K_FOLDS}-Fold CV, random_state={RANDOM_STATE}) ---")
        print(summary_df.to_string(index=False))
        
        # 更改汇总文件名
        summary_path = os.path.join(OUTPUT_DIR, 'lightgbm_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n交叉验证性能汇总表已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print("--- 步骤3a (LightGBM) 全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()