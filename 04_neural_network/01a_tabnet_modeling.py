# -*- coding: utf-8 -*-
"""
文件名: 01a_tabnet_modeling.py
(注意: 此脚本应放置在 04_neural_network 文件夹中)

功能: 
    1. 加载最终数据集 (假设已用0填充NaN)。
    2. [新增] 引入 StandardScaler 进行数据标准化。
    3. [新增] 使用 Pipeline 将 Scaler 和 TabNetRegressor 捆绑。
    4. [新增] TabNet 需要 np.float32 类型的数据。
    5. 对每个性能指标，使用K-折交叉验证评估 TabNet 模型的性能。
    6. [新增] 随机搜索多组超参数, 以R2均值选择最佳组合, 再用全量数据重训并保存。
    7. 提取 TabNet 内置的 'feature_importances_' 作为特征重要性。
"""

# --- 0. 并行控制 (需在数值库加载前设置) ---
import os
N_JOBS_PARALLEL = 8  # 限制CPU核心数
os.environ["OMP_NUM_THREADS"] = str(N_JOBS_PARALLEL)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS_PARALLEL)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS_PARALLEL)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS_PARALLEL)

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# 评价指标
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# PyTorch 线程控制
import torch
torch.set_num_threads(N_JOBS_PARALLEL)
# 导入 TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

# --- 1. 配置区 ---

# --- 输入/输出文件配置 ---
# 假设此脚本位于 .../steel/04_neural_network/
INPUT_DIR = '../01_read_data/results'
# 专属的输出目录: .../steel/04_neural_network/results/01_tabnet_results/
OUTPUT_DIR = 'results/01_tabnet_results' 
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 模型配置 ---
RANDOM_STATE = 42
K_FOLDS = 5
MODEL_NAME_PREFIX = '01a_TabNet' 

# TabNet 训练参数 (早停+充足训练上限，交给早停裁剪)
TABNET_EPOCHS = 300         # 较高上限配合早停
TABNET_PATIENCE = 30        # 早停耐心
TABNET_BATCH_SIZE = 1024    # 批次大小

# 随机搜索控制
N_ITER_SEARCH = 40          # 随机采样超参数组合次数 (可按机器时间调整)

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


# --- 2. 辅助函数 ---

def sanitize_filename(filename):
    """清洗文件名中的非法字符。"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def plot_tabnet_importance(model, feature_names, metric_name, model_name_prefix, output_dir):
    """
    提取 TabNet 的 feature_importances_ 并可视化。
    """
    print("    - 正在提取 TabNet 内置特征重要性...")
    try:
        # 在 Pipeline 中, model 位于 'model' 步骤
        importance = model.named_steps['model'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance (TabNet)': importance
        }).sort_values(by='Importance (TabNet)', ascending=False)
        
        # 保存CSV (保持你的最新命名风格: 不加前缀, 只用指标名)
        csv_filename = f'{metric_name}_feature_importance.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 绘制图表
        plt.figure(figsize=(12, 8))
        importance_df.head(20).sort_values(by='Importance (TabNet)', ascending=True).plot(
            kind='barh', x='Feature', y='Importance (TabNet)', 
            legend=False, figsize=(12, 8)
        )
        plt.title(f'"{metric_name}" Top 20 特征 ({model_name_prefix})', fontsize=16)
        plt.xlabel('TabNet 特征重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        
        plot_filename = f'{metric_name}_feature_importance.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=120)
        plt.close()
        print(f"    - TabNet 特征重要性分析完成 (CSV和图表已保存)。")
    except Exception as e:
        print(f"    [错误] TabNet 特征重要性分析失败: {e}")

def sample_hparams(rng: np.random.RandomState):
    """
    随机采样一组 TabNet 超参 (采用对数均匀+离散候选, 兼顾稳定性与表达能力)。
    """
    n_d = int(rng.choice([16, 24, 32, 40]))
    n_a = n_d  # 常见做法: n_d 与 n_a 相同
    n_steps = int(rng.choice([3, 4, 5, 6]))
    gamma = float(rng.uniform(1.0, 2.0))  # 特征复用多样性
    lambda_sparse = float(10 ** rng.uniform(-5, -2))  # 1e-5 ~ 1e-2 (对数均匀)
    learning_rate = float(10 ** rng.uniform(-3.2, -2.2))  # ~[6.3e-4, 6.3e-3]
    virtual_batch_size = int(rng.choice([128, 256, 512])) # Ghost BN 的统计更稳定
    return dict(
        n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
        lambda_sparse=lambda_sparse, learning_rate=learning_rate,
        virtual_batch_size=virtual_batch_size
    )


# --- 3. 主执行函数 ---

def main():
    """主执行函数"""
    print("=" * 60)
    print("--- 启动脚本: 01a_tabnet_modeling.py ---")
    print(f"--- 目的: TabNet 随机超参搜索(K折)→以R2选优→全量重训→重要性 ---")
    print(f"--- (Epochs={TABNET_EPOCHS}, Patience={TABNET_PATIENCE}, Iter={N_ITER_SEARCH}, CPU={N_JOBS_PARALLEL}) ---")
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
    
    # 假设数据已用0填充，但为保险起见
    df = df.fillna(0)
    
    X_all = df.drop(columns=[ID_COLUMN] + PERFORMANCE_METRICS, errors='ignore').select_dtypes(include=np.number)
    Y_all = df[PERFORMANCE_METRICS]
    feature_names = X_all.columns
    n_features = len(feature_names)
    print(f"数据准备完成，共包含 {n_features} 个输入特征。")
    
    # --- 2. 循环执行模型训练 ---
    all_cv_performance_summary = []
    rng = np.random.RandomState(RANDOM_STATE)

    for metric in Y_all.columns:
        print(f"\n\n{'='*20} 正在为性能指标: '{metric}' 进行建模 (TabNet) {'='*20}")
        safe_metric_name = sanitize_filename(metric)

        # 准备数据 (TabNet 需要 1D y)
        metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna(subset=[metric])
        if metric_data.empty:
            print(f"    [警告] 移除 Y 为 NaN 的行后，没有剩余数据可用于评估 '{metric}'，跳过。")
            continue
        
        # 转换为 np.float32
        X = metric_data[X_all.columns].values.astype(np.float32)
        y = metric_data[metric].values.astype(np.float32)

        # --- 2a. (1/3) 随机超参搜索 + K折交叉验证 ---
        print(f"\n--- (1/3) 正在执行随机超参搜索 (共 {K_FOLDS}-折, 迭代 {N_ITER_SEARCH} 次) ---")
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        best_mean_r2 = -np.inf
        best_params = None

        # 可选: 记录每次搜索结果到CSV，便于观察R²随超参波动
        search_csv_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_{safe_metric_name}_hparam_search.csv')
        if os.path.exists(search_csv_path):
            os.remove(search_csv_path)

        for it in range(1, N_ITER_SEARCH + 1):
            params = sample_hparams(rng)
            cv_scores = {'r2': [], 'mae': [], 'mse': []}

            for fold, (train_index, test_index) in enumerate(kfold.split(X)):
                # 1) 标准化 (每折仅用训练集拟合)
                scaler_fold = StandardScaler()
                X_tr = scaler_fold.fit_transform(X[train_index])
                X_te = scaler_fold.transform(X[test_index])
                y_tr, y_te = y[train_index], y[test_index]

                # 2) 拟合 TabNet (注意 y 需为 2D)
                model_fold = TabNetRegressor(
                    seed=RANDOM_STATE,
                    verbose=0,
                    n_d=params["n_d"], n_a=params["n_a"],
                    n_steps=params["n_steps"],
                    gamma=params["gamma"],
                    lambda_sparse=params["lambda_sparse"],
                    optimizer_params=dict(lr=params["learning_rate"]),
                    # 注意: virtual_batch_size 不能放在构造器里
                )
                model_fold.fit(
                    X_tr, y_tr.reshape(-1, 1),
                    max_epochs=TABNET_EPOCHS,
                    patience=TABNET_PATIENCE,
                    batch_size=TABNET_BATCH_SIZE,
                    # 放到 fit() 中
                    virtual_batch_size=params["virtual_batch_size"]
                )

                # 3) 验证
                y_pred = model_fold.predict(X_te).flatten()  # 预测为2D, 需展平
                cv_scores['r2'].append(r2_score(y_te, y_pred))
                cv_scores['mae'].append(mean_absolute_error(y_te, y_pred))
                cv_scores['mse'].append(mean_squared_error(y_te, y_pred))

            mean_r2 = float(np.mean(cv_scores['r2']))
            mean_mae = float(np.mean(cv_scores['mae']))
            mean_rmse = float(np.sqrt(np.mean(cv_scores['mse'])))

            # 记录此次搜索的结果
            row = pd.DataFrame([{
                '指标': metric, 'iter': it,
                'R2(mean)': round(mean_r2, 6),
                'MAE(mean)': round(mean_mae, 6),
                'RMSE(mean)': round(mean_rmse, 6),
                **params
            }])
            row.to_csv(search_csv_path, mode='a', header=not os.path.exists(search_csv_path),
                       index=False, encoding='utf-8-sig')

            print(f"    - iter={it:02d} | R2={mean_r2:.4f}, MAE={mean_mae:.4f}, RMSE={mean_rmse:.4f} | params={params}")

            # 以 R² 均值最大为优
            if mean_r2 > best_mean_r2:
                best_mean_r2 = mean_r2
                best_params = params

        print(f"\n    [最优] '{metric}' 的最优CV均值: R2={best_mean_r2:.4f}")
        print(f"           最优超参: {best_params}")

        # --- 2b. (2/3) 训练最终模型并保存 (使用最优超参) ---
        print("\n--- (2/3) 正在训练最终模型并保存 ---")
        print(f"    (TabNet 正在训练 {TABNET_EPOCHS} epochs, patience={TABNET_PATIENCE}...)")
        
        # 我们必须保存一个 Pipeline (Scaler + Model)
        final_scaler = StandardScaler()
        X_scaled = final_scaler.fit_transform(X)
        
        final_model = TabNetRegressor(
            seed=RANDOM_STATE,
            verbose=0,
            n_d=best_params["n_d"], n_a=best_params["n_a"],
            n_steps=best_params["n_steps"],
            gamma=best_params["gamma"],
            lambda_sparse=best_params["lambda_sparse"],
            optimizer_params=dict(lr=best_params["learning_rate"]),
            # 不在构造器里放 virtual_batch_size
        )
        final_model.fit(
            X_scaled, y.reshape(-1, 1),
            max_epochs=TABNET_EPOCHS,
            patience=TABNET_PATIENCE,
            batch_size=TABNET_BATCH_SIZE,
            virtual_batch_size=best_params["virtual_batch_size"]  # 放到 fit() 中
        )

        final_pipeline = Pipeline([
            ('scaler', final_scaler),
            ('model', final_model)
        ])
        
        # 保存整个 Pipeline (保持你的命名风格)
        model_filename = f'{MODEL_NAME_PREFIX}_{safe_metric_name}_model.joblib'
        model_path = os.path.join(model_save_dir, model_filename)
        joblib.dump(final_pipeline, model_path)
        print(f"    Pipeline (Scaler+TabNet) 已保存至: {model_path}")
        
        # 同步保存 TabNet 原生 .zip (保持你的命名风格)
        tabnet_model_path = os.path.join(model_save_dir, f'{MODEL_NAME_PREFIX}_{safe_metric_name}_model')
        final_pipeline.named_steps['model'].save_model(tabnet_model_path)
        print(f"    TabNet .zip 模型已保存至: {tabnet_model_path}.zip")

        # --- 2c. (3/3) 模型解释 (特征重要性) ---
        print("\n--- (3/3) 正在进行模型解释 (特征重要性) ---")
        plot_tabnet_importance(
            final_pipeline, 
            feature_names, 
            safe_metric_name, 
            MODEL_NAME_PREFIX, 
            OUTPUT_DIR
        )

        # 记录本指标的CV最优结果 (与你最新脚本风格一致)
        all_cv_performance_summary.append({
            '模型': MODEL_NAME_PREFIX,
            '性能指标': metric, 
            'R2 Score (平均值)': f"{best_mean_r2:.4f}"
        })

    # --- 4. 汇总并保存所有模型的交叉验证结果 ---
    if all_cv_performance_summary:
        summary_df = pd.DataFrame(all_cv_performance_summary)
        print(f"\n\n--- TabNet 交叉验证性能汇总表 ({K_FOLDS}-Fold CV) ---")
        print(summary_df.to_string(index=False))
        
        # 保持你的最新输出文件名
        summary_path = os.path.join(OUTPUT_DIR, 'performance_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n交叉验证性能汇总表已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print("--- 步骤 01a (TabNet, 含随机调参与保存最优) 全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()
