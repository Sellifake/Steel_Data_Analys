# -*- coding: utf-8 -*-
"""
文件名: 04_boosting_tuning.py
功能: 
    1. 加载最终数据集。
    2. 定义 XGBoost 和 LightGBM 的超参数搜索空间。
    3. 对每个性能指标，使用 RandomizedSearchCV (随机搜索) 进行 K-折交叉验证和超参数调优。
    4. 在完整数据集上训练*调优后*的最终模型，并将其保存到专属的 'results/04_boosting_tuned_results' 文件夹中。
    5. 生成并保存调优后模型的特征重要性 (图+CSV)。
    6. 将两个模型的调优后性能汇总到一个CSV文件中。
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
# RandomizedSearchCV 依赖 scipy.stats
from scipy.stats import randint, uniform

# --- 1. 配置区 ---

# --- 输入/输出文件配置 ---
INPUT_DIR = '../01_read_data/results'
# 专属的输出目录，不覆盖任何现有结果
OUTPUT_DIR_BASE = 'results/04_boosting_tuned_results' 
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 模型与调优配置 ---
# 设置全局随机种子以确保实验可复现
RANDOM_STATE = 42
K_FOLDS = 5
# 随机搜索的迭代次数 (n_iter)。增加此值可提高找到最佳参数的几率，但会增加时间。
N_ITER_SEARCH = 200 
# 并行度设置：降低CPU占用 (根据你的最新代码设置)
N_JOBS_PARALLEL = 5  # 使用5个CPU核心

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


# --- 2. 辅助函数 ---

def sanitize_filename(filename):
    """清洗文件名中的非法字符。"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

# [已移除] visualize_decision_tree_xgb 函数
# [已移除] visualize_decision_tree_lgb 函数


# --- 3. 调优参数空间定义 ---

# XGBoost 的参数搜索空间
xgb_param_dist = {
    'n_estimators': randint(100, 1000),       # 树的数量
    'learning_rate': uniform(0.01, 0.2),      # 学习率
    'max_depth': randint(3, 10),              # 树的最大深度
    'subsample': uniform(0.7, 0.3),           # 训练样本采样率 (0.7-1.0)
    'colsample_bytree': uniform(0.7, 0.3),    # 特征采样率 (0.7-1.0)
    'gamma': uniform(0, 0.5),                 # 节点分裂的最小损失降低
    'reg_alpha': uniform(0, 1),               # L1 正则化
    'reg_lambda': uniform(0, 1)               # L2 正则化
}

# LightGBM 的参数搜索空间
lgbm_param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 10),
    'num_leaves': randint(20, 50),             # 关键参数：叶子节点数
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}


# --- 4. 主执行函数 ---

def main():
    """主执行函数"""
    print("=" * 60)
    print("--- 启动脚本: 04a_boosting_tuning.py ---")
    print(f"--- 目的: 对XGBoost和LightGBM进行 {N_ITER_SEARCH} 次迭代的随机搜索调优 ---")
    print(f"--- 输出目录: {OUTPUT_DIR_BASE} ---")
    print("=" * 60)
    
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE)
        print(f"已创建输出文件夹: '{OUTPUT_DIR_BASE}'")
    
    model_save_dir = os.path.join(OUTPUT_DIR_BASE, 'saved_models')
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
    print(f"数据准备完成，共包含 {len(X_all.columns)} 个输入特征。")
    
    # --- 2. 定义要调优的模型配置 ---
    MODELS_TO_TUNE = [
        {
            'name': 'XGBoost_Tuned', # 增加后缀以区分
            'base_model': xgb.XGBRegressor(
                random_state=RANDOM_STATE, 
                n_jobs=N_JOBS_PARALLEL, # 使用你设定的并行度
                objective='reg:squarederror'
            ),
            'param_dist': xgb_param_dist,
        },
        {
            'name': 'LightGBM_Tuned', # 增加后缀以区分
            'base_model': lgb.LGBMRegressor(
                random_state=RANDOM_STATE, 
                n_jobs=N_JOBS_PARALLEL, # 使用你设定的并行度
                verbose=-1 # 关闭LGBM的啰嗦输出
            ),
            'param_dist': lgbm_param_dist,
        }
    ]

    # --- 3. 循环执行模型调优 ---
    
    # 用于存储所有模型和所有指标的汇总结果
    all_cv_performance_summary = []

    for config in MODELS_TO_TUNE:
        
        model_name = config['name']
        
        print(f"\n\n{'='*25} 正在开始 {model_name} 的超参数调优 {'='*25}")

        # 循环为每个性能指标建模
        for metric in Y_all.columns:
            print(f"\n\n{'-'*20} 正在为指标: '{metric}' (模型: {model_name}) 进行调优 {'-'*20}")
            safe_metric_name = sanitize_filename(metric)

            # 准备数据
            metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna()
            if metric_data.empty:
                print(f"    [警告] 移除NaN后，没有剩余数据可用于评估 '{metric}'，跳过。")
                continue
            
            X_train = metric_data[X_all.columns]
            y_train = metric_data[metric]
            
            # --- 2a. 执行 RandomizedSearchCV ---
            print(f"\n--- (1/3) 正在执行 {N_ITER_SEARCH} 次迭代的 {K_FOLDS}-折交叉验证调优 ---")
            
            base_estimator = config['base_model']
            
            # 针对 XGBoost，应用 base_score 修复
            if 'XGBoost' in model_name:
                base_score_value = y_train.mean()
                base_estimator.set_params(base_score=base_score_value)
            
            kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
            
            # 使用 R2 作为调优的主要评分标准
            search = RandomizedSearchCV(
                estimator=base_estimator,
                param_distributions=config['param_dist'],
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
            
            # --- 2b. 使用最佳模型重新评估所有指标 (R2, MAE, RMSE) ---
            print(f"    正在使用最佳参数重新运行交叉验证以获取所有指标...")
            final_model = search.best_estimator_ # 这就是调优后的最佳模型
            
            scoring_metrics = ['r2', 'neg_mean_absolute_error', 'neg_mean_squared_error']
            # 注意: cross_validate 也受 n_jobs 影响
            scores = cross_validate(final_model, X_train, y_train, cv=kfold, 
                                    scoring=scoring_metrics, n_jobs=N_JOBS_PARALLEL)
            
            mean_r2 = np.mean(scores['test_r2'])
            mean_mae = -np.mean(scores['test_neg_mean_absolute_error'])
            mean_rmse = np.sqrt(-np.mean(scores['test_neg_mean_squared_error']))
            
            # 存储到总的汇总表
            all_cv_performance_summary.append({
                '模型': model_name,
                '性能指标': metric, 
                'R2 Score (平均值)': f"{mean_r2:.4f}",
                'MAE (平均值)': f"{mean_mae:.4f}", 
                'RMSE (平均值)': f"{mean_rmse:.4f}",
                '最佳参数': str(search.best_params_) # 记录最佳参数
            })
            print(f"    CV完成 (平均值): R2={mean_r2:.4f}, MAE={mean_mae:.4f}, RMSE={mean_rmse:.4f}")

            # --- 2c. 训练最终模型并保存 (使用最佳参数) ---
            print("\n--- (2/3) 正在训练最终模型并保存 ---")
            final_model.fit(X_train, y_train) # 在全部数据上重新训练

            # 使用 {model_name}_{metric_name}_model.joblib 命名
            model_filename = f'{model_name}_{safe_metric_name}_model.joblib'
            model_path = os.path.join(model_save_dir, model_filename)
            
            joblib.dump(final_model, model_path)
            print(f"    模型文件已保存至: {model_path}")

            # --- 2d. 模型解释 (特征重要性) ---
            print("\n--- (3/3) 正在进行模型解释 (特征重要性) ---")
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': final_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            # 保存到 04 目录
            csv_filename = f'{model_name}_{safe_metric_name}_feature_importance.csv'
            csv_path = os.path.join(OUTPUT_DIR_BASE, csv_filename)
            importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            plt.figure(figsize=(12, 8))
            importance_df.head(20).sort_values(by='Importance', ascending=True).plot(kind='barh', x='Feature', y='Importance', legend=False, figsize=(12, 8))
            plt.title(f'"{metric}" Top 20 特征 ({model_name})', fontsize=16)
            plt.tight_layout()
            
            plot_filename = f'{model_name}_{safe_metric_name}_feature_importance.png'
            plot_path = os.path.join(OUTPUT_DIR_BASE, plot_filename)
            plt.savefig(plot_path, dpi=120)
            plt.close()
            print(f"    - 特征重要性分析完成。")
            

    # --- 4. 汇总并保存所有模型的交叉验证结果 ---
    if all_cv_performance_summary:
        summary_df = pd.DataFrame(all_cv_performance_summary)
        print(f"\n\n--- 调优后 Boosting 交叉验证性能汇总表 ({K_FOLDS}-Fold CV) ---")
        # 按照模型和指标排序，方便查看
        summary_df = summary_df.sort_values(by=['模型', '性能指标'])
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join(OUTPUT_DIR_BASE, 'boosting_tuned_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n交叉验证性能汇总表已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print("--- 步骤4a (Boosting调优) 全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()