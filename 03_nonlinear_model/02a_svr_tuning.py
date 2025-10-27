# -*- coding: utf-8 -*-
"""
文件名: 02a_svr_tuning.py
(注意: 此脚本应放置在 03_nonlinear_model 文件夹中)

功能: 
    1. 加载最终数据集。
    2. [新增] 引入 RandomizedSearchCV 对 SVR(RBF核) 的超参数 C 和 gamma 进行调优。
    3. [保持] 使用 Pipeline 捆绑 StandardScaler 和 SVR。
    4. 对每个性能指标，使用 K-折交叉验证评估*调优后*的最佳 SVR 模型的性能。
    5. 训练并保存最终的*最佳* Pipeline (Scaler + SVR)。
    6. [保持] 使用 "排列重要性" (Permutation Importance) 作为SVR的特征重要性衡量标准。
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
# 导入 loguniform 来定义SVR的参数搜索空间
from scipy.stats import loguniform

# --- 1. 配置区 ---

# --- 输入/输出文件配置 ---
# 假设此脚本位于 .../steel/03_nonlinear_model/
INPUT_DIR = '../01_read_data/results'
# 专属的输出目录: .../steel/03_nonlinear_model/results/02_svr_tuned_results/
OUTPUT_DIR = 'results/02_svr_tuned_results' 
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 模型配置 ---
RANDOM_STATE = 42
K_FOLDS = 5
MODEL_NAME_PREFIX = '02_SVR_Tuned' # 用于文件命名的模型前缀

# --- 调优配置 ---
# 随机搜索的迭代次数
N_ITER_SEARCH = 50 
# 并行度设置 (仿照 04a_boosting_tuning.py)
N_JOBS_PARALLEL = 10 

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


# --- 2. SVR 调优参数空间定义 ---

# SVR(RBF)的性能由 C 和 gamma 决定，它们都在对数尺度上
svr_param_dist = {
    # 'svr__C': C 是惩罚系数。在 0.1 到 1000 之间搜索
    'svr__C': loguniform(0.1, 1000),
    # 'svr__gamma': RBF核宽度。在 0.0001 到 0.1 之间搜索
    'svr__gamma': loguniform(0.0001, 0.1)
}


# --- 3. 辅助函数 ---

def sanitize_filename(filename):
    """
    清洗文件名中的非法字符，确保文件名符合系统要求
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def plot_permutation_importance(model, X_train, y_train, feature_names, metric_name, model_name_prefix, output_dir):
    """
    计算并可视化排列重要性
    通过随机打乱特征值来评估特征对模型性能的影响
    """
    print("    - 正在计算排列重要性 (Permutation Importance)...")
    print("    (这可能需要几秒钟，因为它需要多次重新预测...)")
    
    try:
        result = permutation_importance(
            model, 
            X_train, 
            y_train, 
            n_repeats=10, 
            random_state=RANDOM_STATE, 
            n_jobs=N_JOBS_PARALLEL # 使用并行
        )
        
        importance = result.importances_mean
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance (Permutation)': importance
        }).sort_values(by='Importance (Permutation)', ascending=False)
        
        # 保存CSV
        csv_filename = f'{model_name_prefix}_{metric_name}_feature_importance.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 绘制图表
        plt.figure(figsize=(12, 8))
        importance_df.head(20).sort_values(by='Importance (Permutation)', ascending=True).plot(
            kind='barh', x='Feature', y='Importance (Permutation)', 
            legend=False, figsize=(12, 8)
        )
        plt.title(f'"{metric_name}" Top 20 特征 ({model_name_prefix} - Permutation Importance)', fontsize=16)
        plt.xlabel('特征重要性 (R2分数下降)', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        
        plot_filename = f'{metric_name}_feature_importance.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"    - 排列重要性分析完成 (CSV和图表已保存)。")
        
    except Exception as e:
        print(f"    [错误] 排列重要性分析失败: {e}")


# --- 4. 主执行函数 ---

def main():
    """
    主执行函数
    执行SVR超参数调优和建模的完整流程
    """
    print("=" * 60)
    print("--- 启动脚本: 02a_svr_tuning.py ---")
    print(f"--- 目的: SVR(RBF核)超参数调优 (N_Iter={N_ITER_SEARCH}) ---")
    print(f"--- 输出目录: {OUTPUT_DIR} ---")
    print("=" * 60)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出文件夹: '{OUTPUT_DIR}'")
    
    model_save_dir = os.path.join(OUTPUT_DIR, 'saved_models')
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)


    # --- 1. 加载数据 ---
    input_file_path = os.path.join(INPUT_DIR, CLEANED_DATA_FILE)
    try:
        df = pd.read_excel(input_file_path)
        print(f"成功加载数据: {input_file_path}")
    except FileNotFoundError:
        print(f"[严重错误] 清洗后的文件未找到 '{input_file_path}'。")
        return
    
    X_all = df.drop(columns=[ID_COLUMN] + PERFORMANCE_METRICS, errors='ignore').select_dtypes(include=np.number)
    Y_all = df[PERFORMANCE_METRICS]
    feature_names = X_all.columns
    n_features = len(feature_names)
    print(f"数据准备完成，共包含 {n_features} 个输入特征。")
    
    
    # --- 2. 循环执行模型训练 ---
    all_cv_performance_summary = []

    for metric in Y_all.columns:
        print(f"\n\n{'='*20} 正在为性能指标: '{metric}' 进行SVR超参数调优 {'='*20}")
        safe_metric_name = sanitize_filename(metric)

        # 准备数据
        metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna()
        if metric_data.empty:
            print(f"    [警告] 移除NaN后，没有剩余数据可用于评估 '{metric}'，跳过。")
            continue
        
        X_train = metric_data[X_all.columns]
        y_train = metric_data[metric]
        
        # --- 2a. 定义处理流程Pipeline作为调优基础 ---
        scaler = StandardScaler()
        svr = SVR(kernel='rbf') 
        pipeline = Pipeline([
            ('scaler', scaler),
            ('svr', svr)
        ])

        # --- 2b. 执行RandomizedSearchCV超参数调优 ---
        print(f"\n--- (1/3) 正在执行 {N_ITER_SEARCH} 次迭代的 {K_FOLDS}-折交叉验证调优 ---")
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=svr_param_dist, # 使用定义的SVR参数空间
            n_iter=N_ITER_SEARCH,
            cv=kfold,
            scoring='r2', # 调优目标
            n_jobs=N_JOBS_PARALLEL, # 限制并行
            random_state=RANDOM_STATE,
            verbose=1 # 显示调优进度
        )
        
        search.fit(X_train, y_train)
        
        print(f"    调优完成。最佳R2 (Tuning): {search.best_score_:.4f}")
        print(f"    最佳参数: {search.best_params_}")

        # --- 2c. 使用最佳参数重新评估所有性能指标 ---
        print(f"    正在使用最佳参数重新运行交叉验证以获取所有指标...")
        
        # search.best_estimator_ 是 *已经* 在全部数据上 refit 过的最终模型
        final_model = search.best_estimator_ 
        
        scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        
        # 我们重新运行 cross_validate 来获取 K-Fold 的(MAE, RMSE)指标
        scores = cross_validate(final_model, X_train, y_train, cv=kfold, 
                                scoring=scoring_metrics, n_jobs=N_JOBS_PARALLEL)
        
        mean_r2 = np.mean(scores['test_r2'])
        mean_mae = -np.mean(scores['test_neg_mean_absolute_error'])
        mean_rmse = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))
        
        all_cv_performance_summary.append({
            '模型': MODEL_NAME_PREFIX,
            '性能指标': metric, 
            'R2 Score (平均值)': f"{mean_r2:.4f}",
            'MAE (平均值)': f"{mean_mae:.4f}", 
            'RMSE (平均值)': f"{mean_rmse:.4f}",
            '最佳参数': str(search.best_params_)
        })
        print(f"    CV完成 (平均值): R2={mean_r2:.4f}, MAE={mean_mae:.4f}, RMSE={mean_rmse:.4f}")

        # --- 2d. 训练最终模型并保存到磁盘 ---
        print("\n--- (3/3) 正在保存最终模型 ---")
        
        # final_model (search.best_estimator_) 已经被 RandomizedSearchCV 在全部数据上训练过了
        # 无需
        # 
        # 
        # 重新 .fit()
        
        model_filename = f'{MODEL_NAME_PREFIX}_{safe_metric_name}_model.joblib'
        model_path = os.path.join(model_save_dir, model_filename)
        
        joblib.dump(final_model, model_path)
        print(f"    Pipeline (Scaler+SVR_Tuned) 已保存至: {model_path}")

        # --- 2e. 进行模型解释和特征重要性分析 ---
        print("\n--- (4/4) 正在进行模型解释 (特征重要性) ---")
        
        plot_permutation_importance(
            final_model, 
            X_train, 
            y_train,
            feature_names, 
            safe_metric_name, 
            MODEL_NAME_PREFIX, 
            OUTPUT_DIR
        )

    # --- 4. 汇总并保存所有模型的交叉验证结果 ---
    if all_cv_performance_summary:
        summary_df = pd.DataFrame(all_cv_performance_summary)
        print(f"\n\n--- SVR (Tuned) 交叉验证性能汇总表 ({K_FOLDS}-Fold CV) ---")
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join(OUTPUT_DIR, 'svr_tuned_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n交叉验证性能汇总表已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print("--- SVR超参数调优任务全部完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()