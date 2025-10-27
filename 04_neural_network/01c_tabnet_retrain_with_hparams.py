# -*- coding: utf-8 -*-
"""
文件名: 01c_tabnet_retrain_with_hparams.py
(注意: 此脚本应放置在 04_neural_network 文件夹中)

功能: 
    1. 【手动定义】已知晓的最优超参数。
    2. 加载最终数据集。
    3. [跳过] 不执行 Optuna 或 K-折调参。
    4. [重构] 将数据分为 训练集(85%) 和 验证集(15%)。
    5. 使用已知的超参数, 在 训练集 上训练最终模型, 在 验证集 上早停。
    6. 保存模型 (Pipeline + .zip)。
    7. 提取特征重要性。
"""

# --- 0. 并行与GPU控制 ---
import os
import torch

# --- GPU配置 ---
GPU_DEVICE_ID = 0  # 指定你希望使用的GPU卡号 (0, 1, 2, 3)
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_ID)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 限制Pytorch和Numpy的CPU线程数
N_CPU_WORKERS = 4
torch.set_num_threads(N_CPU_WORKERS)
os.environ["OMP_NUM_THREADS"] = str(N_CPU_WORKERS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CPU_WORKERS)
os.environ["MKL_NUM_THREADS"] = str(N_CPU_WORKERS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_CPU_WORKERS)

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# 导入 TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

# --- 1. 配置区 ---

# --- 输入/输出文件配置 ---
INPUT_DIR = '../01_read_data/results'
OUTPUT_DIR = 'results/01_tabnet_results' 
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# --- 列定义 ---
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# --- 模型配置 ---
RANDOM_STATE = 42
MODEL_NAME_PREFIX = '01a_TabNet' # 保持前缀一致
TABNET_EPOCHS = 300
TABNET_PATIENCE = 30
TABNET_BATCH_SIZE = 1024
VALIDATION_SPLIT_SIZE = 0.15 # 必须与 01a 中的设置一致

# --- Matplotlib 设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 【核心】: 手动定义的最优超参数 ---
# [2025-10-27] 从 01a 脚本的日志中复制的最优参数
BEST_HPARAMS = {
    '抗拉强度': {
        'n_d': 32, 
        'n_steps': 4, 
        'gamma': 1.2160304714974566, 
        'lambda_sparse': 0.0027313996491565135, 
        'learning_rate': 0.006713826095388964, 
        'virtual_batch_size': 128,
        'n_a': 32 # n_a 通常等于 n_d
    },
    
    '屈服Rp0.2值*': {
        # 'n_d': ..., 
        # 'n_steps': ..., 
        # ... (当 01a 跑完该指标后, 在此填入)
    },
    
    '断后伸长率': {
        # 'n_d': ..., 
        # 'n_steps': ..., 
        # ... (当 01a 跑完该指标后, 在此填入)
    }
}


# --- 2. 辅助函数 (与 01a 相同) ---

def sanitize_filename(filename):
    """清洗文件名中的非法字符。"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def plot_tabnet_importance(model, feature_names, metric_name, model_name_prefix, output_dir):
    """
    提取 TabNet 的 feature_importances_ 并可视化。
    """
    print("    - 正在提取 TabNet 内置特征重要性...")
    try:
        importance = model.named_steps['model'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance (TabNet)': importance
        }).sort_values(by='Importance (TabNet)', ascending=False)
        
        csv_filename = f'{metric_name}_feature_importance.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
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

# --- 3. 主执行函数 ---

def main():
    """主执行函数"""
    print("=" * 60)
    print("--- 启动脚本: 01c_tabnet_retrain_with_hparams.py ---")
    print(f"--- 目的: 使用已知的最优超参数, 快速训练并保存最终模型 ---")
    print(f"--- (Device={device}) ---")
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
    
    df = df.fillna(0)
    
    X_all = df.drop(columns=[ID_COLUMN] + PERFORMANCE_METRICS, errors='ignore').select_dtypes(include=np.number)
    Y_all = df[PERFORMANCE_METRICS]
    feature_names = X_all.columns
    n_features = len(feature_names)
    print(f"数据准备完成，共包含 {n_features} 个输入特征。")

    # --- 2. 循环训练已知的超参 ---
    # [2025-10-27] 我们只循环 BEST_HPARAMS 中定义了的指标
    for metric in BEST_HPARAMS.keys():
        
        # 检查超参数是否已填入
        if not BEST_HPARAMS[metric]:
            print(f"\n\n{'='*20} 跳过指标: '{metric}' (未在 BEST_HPARAMS 中定义超参数) {'='*20}")
            continue
            
        print(f"\n\n{'='*20} 正在为性能指标: '{metric}' 训练最终模型 (使用已知超参) {'='*20}")
        safe_metric_name = sanitize_filename(metric)

        # 准备数据
        metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna(subset=[metric])
        if metric_data.empty:
            print(f"    [警告] 移除 Y 为 NaN 的行后，没有剩余数据可用于评估 '{metric}'，跳过。")
            continue
        
        X_full = metric_data[X_all.columns].values.astype(np.float32)
        y_full = metric_data[metric].values.astype(np.float32)

        # [2025-10-27] 复制 01a 的数据拆分逻辑, 为最终模型提供早停验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, 
            test_size=VALIDATION_SPLIT_SIZE, 
            random_state=RANDOM_STATE
        )
        print(f"    数据拆分: 训练集 {X_train.shape[0]} 行, 验证集 {X_val.shape[0]} 行。")

        # --- 2b. (1/2) 训练最终模型并保存 ---
        print("\n--- (1/2) 正在训练最终模型并保存 ---")
        print(f"    (TabNet 正在训练 {TABNET_EPOCHS} epochs, patience={TABNET_PATIENCE}...)")
        
        # 1. 在 训练集(X_train) 上 fit scaler
        final_scaler = StandardScaler()
        X_train_scaled = final_scaler.fit_transform(X_train)
        
        # 2. 用同样的 scaler 去 transform 验证集(X_val)
        X_val_scaled = final_scaler.transform(X_val)
        
        # 3. [2025-10-27] 复制 01a 中的 Bug 修复逻辑, 分离超参数
        final_params = BEST_HPARAMS[metric].copy() 
        opt_lr = final_params.pop('learning_rate')
        fit_vbs = final_params.pop('virtual_batch_size')
        
        final_model = TabNetRegressor(
            seed=RANDOM_STATE,
            verbose=0,
            device_name=device, # 使用GPU
            optimizer_params=dict(lr=opt_lr),
            **final_params
        )
        
        final_model.fit(
            X_train_scaled, y_train.reshape(-1, 1),
            eval_set=[(X_val_scaled, y_val.reshape(-1, 1))],
            eval_metric=['rmse', 'mae'], 
            max_epochs=TABNET_EPOCHS,
            patience=TABNET_PATIENCE,
            batch_size=TABNET_BATCH_SIZE,
            virtual_batch_size=fit_vbs
        )

        final_pipeline = Pipeline([
            ('scaler', final_scaler),
            ('model', final_model)
        ])
        
        model_filename = f'{MODEL_NAME_PREFIX}_{safe_metric_name}_model.joblib'
        model_path = os.path.join(model_save_dir, model_filename)
        joblib.dump(final_pipeline, model_path)
        print(f"    Pipeline (Scaler+TabNet) 已保存至: {model_path}")
        
        tabnet_model_path = os.path.join(model_save_dir, f'{MODEL_NAME_PREFIX}_{safe_metric_name}_model')
        final_pipeline.named_steps['model'].save_model(tabnet_model_path)
        print(f"    TabNet .zip 模型已保存至: {tabnet_model_path}.zip")

        # --- 2c. (2/2) 模型解释 (特征重要性) ---
        print("\n--- (2/2) 正在进行模型解释 (特征重要性) ---")
        plot_tabnet_importance(
            final_pipeline, 
            feature_names, 
            safe_metric_name, 
            MODEL_NAME_PREFIX, 
            OUTPUT_DIR
        )

    print("\n" + "=" * 60)
    print(f"--- 步骤 01c (TabNet, 手动重训) 全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()