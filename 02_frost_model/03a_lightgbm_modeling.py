# -*- coding: utf-8 -*-
"""
文件名: 03a_lightgbm_modeling.py
功能: 
    1. 加载01_read_data文件夹中经过特征筛选后的最终数据集。
    2. 对每个性能指标，使用 RandomizedSearchCV (随机搜索) 进行 K-折交叉验证和超参数调优。
    3. 使用*调优后*的最佳模型，重新运行 K-折交叉验证以获取 R2, MAE, RMSE。
    4. 在完整数据集上训练*调优后*的最终模型，并进行保存。
    5. 生成并保存调优后模型的特征重要性 (图+CSV)。
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb # 导入 LightGBM
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
# RandomizedSearchCV 依赖 scipy.stats
from scipy.stats import randint, uniform

# --- 1. 配置区 ---

# --- 输入/输出文件配置 ---
INPUT_DIR = '../01_read_data/results'
OUTPUT_DIR = 'results/03_lightgbm_results' 
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 模型与调优配置 ---
# 设置全局随机种子以确保实验可复现
RANDOM_STATE = 42
K_FOLDS = 5
# 随机搜索的迭代次数 (n_iter)
N_ITER_SEARCH = 200 
# 并行度设置 (使用5个CPU核心)
N_JOBS_PARALLEL = 5  

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 调优参数空间定义 (LightGBM) ---
# 搜索空间已适当扩展
lgbm_param_dist = {
    'n_estimators': randint(100, 1200),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 12),
    'num_leaves': randint(20, 60),             # 关键参数：叶子节点数
    'subsample': uniform(0.6, 0.4),            # (0.6-1.0)
    'colsample_bytree': uniform(0.6, 0.4),     # (0.6-1.0)
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}


def sanitize_filename(filename):
    """
    清洗文件名中的非法字符，确保文件名符合系统要求
    与01a_random_forest_modeling.py中的函数保持一致
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)



def main():
    """
    主执行函数
    执行LightGBM超参数调优和建模的完整流程
    """
    print("=" * 60)
    print("--- 启动脚本: 03a_lightgbm_modeling.py (集成超参数调优) ---")
    print(f"--- 调优迭代次数: {N_ITER_SEARCH}, 并行度: {N_JOBS_PARALLEL} ---")
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
        print(f"\n\n{'='*20} 正在为性能指标: '{metric}' 进行LightGBM超参数调优 {'='*20}")
        safe_metric_name = sanitize_filename(metric)

        # 准备当前目标的数据集
        metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna()
        if metric_data.empty:
            print(f"    [警告] 移除NaN后，没有剩余数据可用于评估 '{metric}'，跳过此指标。")
            continue
        
        X_train = metric_data[X_all.columns]
        y_train = metric_data[metric]
        
        # --- 2a. 执行RandomizedSearchCV超参数调优 ---
        print(f"\n--- (1/4) 正在执行 {N_ITER_SEARCH} 次迭代的 {K_FOLDS}-折交叉验证调优 ---")
        
        # 实例化 *基础* LightGBM 回归模型 (用于调优)
        base_model = lgb.LGBMRegressor(
            random_state=RANDOM_STATE, 
            n_jobs=N_JOBS_PARALLEL, # 使用设定的并行度
            verbose=-1 # 关闭LGBM的啰嗦输出
        )
        
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        
        # 使用 R2 作为调优的主要评分标准
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=lgbm_param_dist,
            n_iter=N_ITER_SEARCH,
            cv=kfold,
            scoring='r2', # 调优目标
            n_jobs=N_JOBS_PARALLEL, # 限制并行搜索
            random_state=RANDOM_STATE,
            verbose=1 # 显示调优进度
        )
        
        search.fit(X_train, y_train)
            
        print(f"    调优完成。最佳R2 (Tuning): {search.best_score_:.4f}")
        print(f"    最佳参数: {search.best_params_}")
        
        # --- 2b. 使用最佳参数重新评估所有性能指标 ---
        print(f"\n--- (2/4) 正在使用最佳参数重新运行交叉验证以获取所有指标 ---")
        final_model = search.best_estimator_ # 这就是调优后的最佳模型
        
        scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
        scores = cross_validate(final_model, X_train, y_train, cv=kfold, 
                                scoring=scoring_metrics, n_jobs=N_JOBS_PARALLEL)
        
        # 打印每一折的详细结果
        print(f"    --- 每一折的详细结果 (使用最佳参数) ---")
        for fold in range(K_FOLDS):
            r2_fold = scores['test_r2'][fold]
            mae_fold = -scores['test_neg_mean_absolute_error'][fold]
            rmse_fold = np.sqrt(-scores['test_neg_mean_squared_error'][fold])
            print(f"    第{fold+1}折: R2={r2_fold:.4f}, MAE={mae_fold:.4f}, RMSE={rmse_fold:.4f}")

        mean_r2 = np.mean(scores['test_r2'])
        mean_mae = -np.mean(scores['test_neg_mean_absolute_error'])
        mean_rmse = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))
        
        cv_performance_summary.append({
            '性能指标': metric, 
            'R2 Score (平均值)': f"{mean_r2:.4f}",
            'MAE (平均值)': f"{mean_mae:.4f}", 
            'RMSE (平均值)': f"{mean_rmse:.4f}",
            '最佳参数': str(search.best_params_) # 记录最佳参数
        })
        print(f"    交叉验证完成 (平均值): R2={mean_r2:.4f}, MAE={mean_mae:.4f}, RMSE={mean_rmse:.4f}")

        # --- 2c. 训练最终模型并保存到磁盘 ---
        print("\n--- (3/4) 正在训练最终模型并保存 ---")
        
        # 遵循 04_boosting_tuning.py 的逻辑，在全部数据上重新fit
        final_model.fit(X_train, y_train)

        model_save_dir = os.path.join(OUTPUT_DIR, 'saved_models')
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
        model_path = os.path.join(model_save_dir, f'{safe_metric_name}_model.joblib')
        
        joblib.dump(final_model, model_path)
        print(f"    模型文件已保存至: {model_path}")

        # --- 2d. 进行模型解释和特征重要性分析 ---
        print("\n--- (4/4) 正在进行模型解释 (特征重要性) ---")
        
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': final_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        csv_path = os.path.join(OUTPUT_DIR, f'{safe_metric_name}_feature_importance.csv')
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        plt.figure(figsize=(12, 8))
        importance_df.head(20).sort_values(by='Importance', ascending=True).plot(kind='barh', x='Feature', y='Importance', legend=False, figsize=(12, 8))
        plt.title(f'"{metric}" 的Top 20重要特征 (LightGBM - Tuned)', fontsize=16)
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
        print(f"\n\n--- LightGBM 调优后交叉验证性能汇总表 ({K_FOLDS}-Fold CV, random_state={RANDOM_STATE}) ---")
        # 按照R2排序
        summary_df_sorted = summary_df.sort_values(by='R2 Score (平均值)', ascending=False)
        print(summary_df_sorted.to_string(index=False))
        
        summary_path = os.path.join(OUTPUT_DIR, 'lightgbm_tuned_performance_summary.csv')
        summary_df_sorted.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n交叉验证性能汇总表已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print("--- LightGBM超参数调优任务全部完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()