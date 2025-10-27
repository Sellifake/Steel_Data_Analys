# -*- coding: utf-8 -*-
"""
文件名: 01_random_forest_modeling.py
功能: 
    1. 加载01_read_data文件夹中经过特征筛选后的最终数据集。
    2. 对每个性能指标，使用设置了随机种子的K-折交叉验证来评估随机森林模型的性能。
    3. 在完整数据集上训练最终模型，并进行保存，确保模型可复现。
    4. 生成并保存特征重要性 (图+CSV)。
    5. 生成并保存SHAP分析图 (概要图+依赖图)，优化了依赖图的保存结构。
    6. 为每个模型导出一棵决策树的结构图，可视化其分裂节点。
"""

import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold, cross_validate

# --- 1. 配置区 ---

# --- 输入/输出文件配置 ---
INPUT_DIR = '../01_read_data/results'
OUTPUT_DIR = 'results/01_random_forest_results' 
# 使用01_read_data生成的最终清理数据
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 模型配置 ---
# 设置全局随机种子以确保实验可复现
RANDOM_STATE = 42
K_FOLDS = 5

# --- SHAP分析开关 ---
# 是否生成SHAP概要图
ENABLE_SHAP_SUMMARY_PLOT = True
# 是否生成SHAP依赖图
ENABLE_SHAP_DEPENDENCE_PLOTS = True

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


def sanitize_filename(filename):
    """
    清洗文件名中的非法字符，确保文件名符合系统要求
    """
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def visualize_decision_tree(model, feature_names, metric_name, output_dir):
    """
    从随机森林中提取第一棵树，并将其结构可视化
    用于理解模型决策过程和特征分裂规则
    """
    tree_viz_dir = os.path.join(output_dir, 'decision_tree_visualizations')
    if not os.path.exists(tree_viz_dir):
        os.makedirs(tree_viz_dir)

    # 提取森林中的第一棵决策树作为样本
    tree_to_plot = model.estimators_[0]
    
    plt.figure(figsize=(25, 15))
    plot_tree(tree_to_plot,
              feature_names=feature_names,
              filled=True,
              rounded=True,
              max_depth=3, # 限制深度以便观察
              fontsize=10)

    plt.title(f"决策树可视化 (样本) - 预测目标: {metric_name}\n (仅显示前3层)", fontsize=16)
    
    safe_metric_name = sanitize_filename(metric_name)
    plot_path = os.path.join(tree_viz_dir, f'{safe_metric_name}_decision_tree.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"    - 决策树结构图已保存至: {plot_path}")


def main():
    """
    主执行函数
    执行随机森林建模的完整流程
    """
    print("=" * 60)
    print("--- 启动脚本: 01a_random_forest_modeling.py ---")
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
        print(f"\n\n{'='*20} 正在为性能指标: '{metric}' 进行随机森林建模 {'='*20}")
        safe_metric_name = sanitize_filename(metric)

        # 准备当前目标的数据集，移除包含NaN的行以确保数据完整性
        metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna()
        if metric_data.empty:
            print(f"    [警告] 移除NaN后，没有剩余数据可用于评估 '{metric}'，跳过此指标。")
            continue
        
        X_train = metric_data[X_all.columns]
        y_train = metric_data[metric]

        # --- 2a. 执行K-折交叉验证评估模型性能 ---
        print(f"\n--- (1/4) 正在执行 {K_FOLDS}-折交叉验证 ---")
        model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
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

        # --- 2b. 训练最终模型并保存到磁盘 ---
        print("\n--- (2/4) 正在训练最终模型并保存 ---")
        final_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        final_model.fit(X_train, y_train)

        model_save_dir = os.path.join(OUTPUT_DIR, 'saved_models')
        if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
        model_path = os.path.join(model_save_dir, f'{safe_metric_name}_model.joblib')
        joblib.dump(final_model, model_path)
        print(f"    模型文件已保存至: {model_path}")

        # --- 2c. 进行模型解释与分析，包括特征重要性分析 ---
        print("\n--- (3/4) 正在进行模型解释与分析 ---")
        
        # 特征重要性
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': final_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        csv_path = os.path.join(OUTPUT_DIR, f'{safe_metric_name}_feature_importance.csv')
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        plt.figure(figsize=(12, 8))
        importance_df.head(20).sort_values(by='Importance', ascending=True).plot(kind='barh', x='Feature', y='Importance', legend=False, figsize=(12, 8))
        plt.title(f'"{metric}" 的Top 20重要特征', fontsize=16)
        plt.xlabel('特征重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, f'{safe_metric_name}_feature_importance.png')
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"    - 特征重要性分析完成 (CSV和图表已保存)。")

        # SHAP 分析
        if ENABLE_SHAP_SUMMARY_PLOT or ENABLE_SHAP_DEPENDENCE_PLOTS:
            print("    - 正在进行SHAP分析...")
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer.shap_values(X_train)
        else:
            print("    - [已跳过] 所有SHAP分析开关已关闭，跳过SHAP分析。")
            shap_values = None
        
        # SHAP概要图
        if ENABLE_SHAP_SUMMARY_PLOT and shap_values is not None:
            plt.figure()
            shap.summary_plot(shap_values, X_train, show=False, max_display=20)
            plt.title(f'SHAP概要图 - {metric}', fontsize=16)
            plt.tight_layout()
            summary_plot_path = os.path.join(OUTPUT_DIR, f'{safe_metric_name}_shap_summary.png')
            plt.savefig(summary_plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"    - SHAP概要图已保存。")
        else:
            print("    - [已跳过] 子开关 ENABLE_SHAP_SUMMARY_PLOT=False，跳过SHAP概要图生成。")

        # SHAP依赖图
        if ENABLE_SHAP_DEPENDENCE_PLOTS and shap_values is not None:
            shap_dep_folder = os.path.join(OUTPUT_DIR, 'shap_dependence_plots')
            if not os.path.exists(shap_dep_folder): os.makedirs(shap_dep_folder)
            print(f"    - 正在为 {len(X_train.columns)} 个特征生成SHAP依赖图...")
            for feature in X_train.columns:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, X_train, interaction_index=None, show=False, alpha=0.4)
                ax = plt.gca()
                feature_idx = X_train.columns.get_loc(feature)
                sns.regplot(x=X_train[feature].values, y=shap_values[:, feature_idx], scatter=False, lowess=True, 
                            line_kws={'color': 'red', 'linestyle': '--', 'linewidth': 2}, ax=ax)
                ax.set_title(f'SHAP依赖图: {feature} 对 {metric}', fontsize=16)
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylabel(f'SHAP value for {feature}')
                plt.tight_layout()
                
                # 文件名中加入指标名称以区分
                dep_plot_path = os.path.join(shap_dep_folder, f'{safe_metric_name}_shap_dep_{sanitize_filename(feature)}.png')
                plt.savefig(dep_plot_path, dpi=120)
                plt.close()
            print(f"    - 所有SHAP依赖图已保存至 '{shap_dep_folder}'。")
        else:
            print("    - [已跳过] 子开关 ENABLE_SHAP_DEPENDENCE_PLOTS=False，跳过SHAP依赖图生成。")
        
        # --- 2d. 可视化决策树分裂节点结构 ---
        print("\n--- (4/4) 正在可视化决策树的分裂节点 ---")
        visualize_decision_tree(final_model, X_train.columns, metric, OUTPUT_DIR)

    # --- 3. 汇总并保存交叉验证结果 ---
    if cv_performance_summary:
        summary_df = pd.DataFrame(cv_performance_summary)
        print(f"\n\n--- 交叉验证性能汇总表 ({K_FOLDS}-Fold CV, random_state={RANDOM_STATE}) ---")
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join(OUTPUT_DIR, 'random_forest_performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n交叉验证性能汇总表已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print("--- 随机森林建模任务全部完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()