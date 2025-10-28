# -*- coding: utf-8 -*-
"""
文件名: 01a_tabnet_modeling.py

功能: 
    1. 加载最终数据集
    2. 启用GPU训练，支持指定GPU设备
    3. 将数据分为训练集(K折调参)和验证集(最终早停)
    4. 使用Optuna贝叶斯优化进行K折交叉验证
    5. 在调参阶段计算R2、MAE、RMSE均值
    6. 调参后保存最优超参数到txt文件作为备份
    7. 训练和保存最终模型时捕获异常，确保单个模型失败不会中断整个脚本
    8. 提取TabNet内置的特征重要性
"""

# GPU和并行控制配置
import os
import torch

# GPU设备配置
GPU_DEVICE_ID = 0  # 指定使用的GPU卡号
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_ID)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 限制CPU线程数，避免数据加载瓶颈
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
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# 评价指标
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# 导入 TabNet
from pytorch_tabnet.tab_model import TabNetRegressor
import optuna

# 配置参数
INPUT_DIR = '../01_read_data/results'
OUTPUT_DIR = 'results/01_tabnet_results' 
CLEANED_DATA_FILE = '04_data_selected.xlsx'

# 数据列定义
ID_COLUMN = '罩退钢卷号'
PERFORMANCE_METRICS = ["抗拉强度", "屈服Rp0.2值*", "断后伸长率"]

# 模型训练配置
RANDOM_STATE = 42
K_FOLDS = 5
MODEL_NAME_PREFIX = '01a_TabNet' 

# TabNet训练参数
TABNET_EPOCHS = 300         # 训练轮数上限
TABNET_PATIENCE = 30        # 早停耐心值
TABNET_BATCH_SIZE = 1024    # 批次大小

# Optuna优化配置
N_TRIALS_OPTUNA = 40        # 优化迭代次数
VALIDATION_SPLIT_SIZE = 0.15 # 验证集比例

# 图表字体设置
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


# 辅助函数

def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def plot_tabnet_importance(model, feature_names, metric_name, model_name_prefix, output_dir):
    """提取TabNet特征重要性并生成可视化图表"""
    print("    - 正在提取TabNet特征重要性...")
    try:
        # 从Pipeline中获取模型的特征重要性
        importance = model.named_steps['model'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance (TabNet)': importance
        }).sort_values(by='Importance (TabNet)', ascending=False)
        
        # 保存CSV文件
        csv_filename = f'{metric_name}_feature_importance.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 生成图表
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
        print(f"    - TabNet特征重要性分析完成")
    except Exception as e:
        print(f"    [错误] TabNet特征重要性分析失败: {e}")

# 主执行函数

def main():
    """主执行函数"""
    print("=" * 60)
    print("--- 启动脚本: 01a_tabnet_modeling.py ---")
    print(f"--- 目的: 使用Optuna进行超参数优化，训练TabNet模型 ---")
    print(f"--- 设备: {device}, 训练轮数: {TABNET_EPOCHS}, 早停耐心: {TABNET_PATIENCE}, 优化次数: {N_TRIALS_OPTUNA} ---")
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
    
    df = df.fillna(0)
    
    X_all = df.drop(columns=[ID_COLUMN] + PERFORMANCE_METRICS, errors='ignore').select_dtypes(include=np.number)
    Y_all = df[PERFORMANCE_METRICS]
    feature_names = X_all.columns
    n_features = len(feature_names)
    print(f"数据准备完成，共包含 {n_features} 个输入特征。")
    
    all_cv_performance_summary = []
    
    # 设置Optuna日志级别
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 定义超参数备份文件
    hparams_txt_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_best_hparams.txt')
    # 清空旧的备份文件
    if os.path.exists(hparams_txt_path):
        os.remove(hparams_txt_path)
        print(f"已清除旧的超参数备份文件: {hparams_txt_path}")

    # 循环训练各性能指标的模型
    for metric in Y_all.columns:
        print(f"\n\n{'='*20} 正在为性能指标: '{metric}' 进行建模 {'='*20}")
        safe_metric_name = sanitize_filename(metric)

        # 准备数据
        metric_data = pd.concat([X_all, Y_all[metric]], axis=1).dropna(subset=[metric])
        if metric_data.empty:
            print(f"    [警告] 移除Y为NaN的行后，没有剩余数据可用于评估 '{metric}'，跳过。")
            continue
        
        # 转换为float32格式
        X_full = metric_data[X_all.columns].values.astype(np.float32)
        y_full = metric_data[metric].values.astype(np.float32)

        # 创建训练集和验证集
        X_train, X_val_final, y_train, y_val_final = train_test_split(
            X_full, y_full, 
            test_size=VALIDATION_SPLIT_SIZE, 
            random_state=RANDOM_STATE
        )
        print(f"    数据拆分: 训练集 {X_train.shape[0]} 行, 验证集 {X_val_final.shape[0]} 行。")

        # Optuna超参数搜索和K折交叉验证
        print(f"\n--- 正在执行Optuna超参数搜索 (共{K_FOLDS}折, 迭代{N_TRIALS_OPTUNA}次) ---")
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        # 定义 Optuna 的目标函数
        def objective(trial):
            # 定义超参搜索空间
            params = {
                'n_d': trial.suggest_categorical('n_d', [16, 24, 32, 40]),
                'n_steps': trial.suggest_categorical('n_steps', [3, 4, 5, 6]),
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 5e-4, 7e-3, log=True),
                'virtual_batch_size': trial.suggest_categorical('virtual_batch_size', [128, 256, 512])
            }
            params['n_a'] = params['n_d']
            
            cv_scores = {'r2': [], 'mae': [], 'rmse': []}

            for fold, (train_index, test_index) in enumerate(kfold.split(X_train)):
                # 1) 标准化 (每折仅用训练集拟合)
                scaler_fold = StandardScaler()
                X_tr = scaler_fold.fit_transform(X_train[train_index])
                X_te = scaler_fold.transform(X_train[test_index])
                y_tr, y_te = y_train[train_index], y_train[test_index]

                # 2) 拟合 TabNet
                model_fold = TabNetRegressor(
                    seed=RANDOM_STATE,
                    verbose=0,
                    device_name=device, # 使用GPU
                    n_d=params["n_d"], n_a=params["n_a"],
                    n_steps=params["n_steps"],
                    gamma=params["gamma"],
                    lambda_sparse=params["lambda_sparse"],
                    optimizer_params=dict(lr=params["learning_rate"]),
                )
                
                model_fold.fit(
                    X_tr, y_tr.reshape(-1, 1),
                    eval_set=[(X_te, y_te.reshape(-1, 1))],
                    eval_metric=['rmse', 'mae'], 
                    max_epochs=TABNET_EPOCHS,
                    patience=TABNET_PATIENCE,
                    batch_size=TABNET_BATCH_SIZE,
                    virtual_batch_size=params["virtual_batch_size"]
                )

                # 3) 验证
                y_pred = model_fold.predict(X_te).flatten()
                
                cv_scores['r2'].append(r2_score(y_te, y_pred))
                cv_scores['mae'].append(mean_absolute_error(y_te, y_pred))
                cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_te, y_pred)))

            # 计算均值并存储
            mean_r2 = float(np.mean(cv_scores['r2']))
            mean_mae = float(np.mean(cv_scores['mae']))
            mean_rmse = float(np.mean(cv_scores['rmse']))
            
            trial.set_user_attr('mean_mae', mean_mae)
            trial.set_user_attr('mean_rmse', mean_rmse)
            
            # Optuna 的优化目标
            return mean_r2

        # 执行 Optuna 搜索
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=N_TRIALS_OPTUNA, show_progress_bar=True)

        # 提取所有最佳指标
        best_trial = study.best_trial
        best_mean_r2 = best_trial.value
        best_mean_mae = best_trial.user_attrs['mean_mae']
        best_mean_rmse = best_trial.user_attrs['mean_rmse']
        best_params = best_trial.params
        
        print(f"\n    [最优] '{metric}' 的最优CV均值: R2={best_mean_r2:.4f}, MAE={best_mean_mae:.4f}, RMSE={best_mean_rmse:.4f}")
        print(f"           最优超参: {best_params}")
        
        # 保存 Optuna 搜索结果 (CSV)
        search_csv_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_{safe_metric_name}_hparam_search.csv')
        study.trials_dataframe().to_csv(search_csv_path, index=False, encoding='utf-8-sig')

        # --- 【新增修改点 2】: 立即备份最优超参数至 TXT ---
        try:
            with open(hparams_txt_path, 'a', encoding='utf-8') as f:
                f.write(f"--- {metric} ---\n")
                f.write(f"# R2(mean)={best_mean_r2:.4f}, MAE(mean)={best_mean_mae:.4f}, RMSE(mean)={best_mean_rmse:.4f}\n")
                # 写入可直接用于 01c 的 Python 字典格式
                f.write(f"'{metric}': {best_params},\n\n")
            print(f"    [成功] 最优超参数已备份至: {hparams_txt_path}")
        except Exception as e_txt:
            print(f"    [严重警告] 无法保存最优超参数到 TXT 文件: {e_txt}")
        # --- 备份结束 ---

        # --- 【新增修改点 3】: 包裹后续步骤, 捕获异常 ---
        try:
            # --- 2b. (2/3) 训练最终模型并保存 (使用最优超参) ---
            print("\n--- (2/3) 正在训练最终模型并保存 ---")
            print(f"    (TabNet 正在训练 {TABNET_EPOCHS} epochs, patience={TABNET_PATIENCE}...)")
            
            # 1. 在 K折调参集(X_train) 上 fit scaler
            final_scaler = StandardScaler()
            X_train_scaled = final_scaler.fit_transform(X_train)
            
            # 2. 用同样的 scaler 去 transform 最终验证集(X_val_final)
            X_val_final_scaled = final_scaler.transform(X_val_final)
            
            # 3. [Bug修复] 分离超参数
            final_params = best_params.copy() 
            opt_lr = final_params.pop('learning_rate')
            fit_vbs = final_params.pop('virtual_batch_size')
            
            final_model = TabNetRegressor(
                seed=RANDOM_STATE,
                verbose=0,
                device_name=device, 
                optimizer_params=dict(lr=opt_lr), 
                **final_params 
            )
            
            final_model.fit(
                X_train_scaled, y_train.reshape(-1, 1),
                eval_set=[(X_val_final_scaled, y_val_final.reshape(-1, 1))],
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
            
            # 保存整个 Pipeline
            model_filename = f'{MODEL_NAME_PREFIX}_{safe_metric_name}_model.joblib'
            model_path = os.path.join(model_save_dir, model_filename)
            joblib.dump(final_pipeline, model_path)
            print(f"    Pipeline (Scaler+TabNet) 已保存至: {model_path}")
            
            # 保存 TabNet 原生 .zip
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
        
        except Exception as e_save:
            # [2025-10-27] 如果训练/保存/分析失败, 打印警告并继续
            print(f"\n\n{'!'*20} [严重警告] {'!'*20}")
            print(f"    指标 '{metric}' 的最终模型训练、保存或特征重要性分析失败。")
            print(f"    错误详情: {e_save}")
            print(f"    已跳过此模型的保存，但最优超参数已记录在 {hparams_txt_path} 中。")
            print(f"    请在所有指标完成后，使用 01c 脚本手动重训此模型。")
            print(f"{'!'*60}\n\n")
            # `pass` 会让循环继续执行到下面的 `all_cv_performance_summary.append`
            pass
        # --- 异常捕获结束 ---

        # 记录本指标的CV最优结果 (无论最终模型是否保存成功, CV结果都应被记录)
        all_cv_performance_summary.append({
            '模型': MODEL_NAME_PREFIX,
            '性能指标': metric, 
            'R2 Score (平均值)': f"{best_mean_r2:.4f}",
            'MAE (平均值)': f"{best_mean_mae:.4f}",
            'RMSE (平均值)': f"{best_mean_rmse:.4f}"
        })

    # --- 4. 汇总并保存所有模型的交叉验证结果 ---
    if all_cv_performance_summary:
        summary_df = pd.DataFrame(all_cv_performance_summary)
        print(f"\n\n--- TabNet 交叉验证性能汇总表 ({K_FOLDS}-Fold CV) ---")
        # 参照RF脚本格式打印
        print(summary_df[['性能指标', 'R2 Score (平均值)', 'MAE (平均值)', 'RMSE (平均值)']].to_string(index=False))
        
        # 参照RF脚本命名风格 (添加模型前缀)
        summary_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME_PREFIX}_performance_summary.csv')
        # 参照RF脚本格式保存 (不包含'模型'列, 尽管DataFrame中有)
        summary_df.drop(columns=['模型']).to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"\n交叉验证性能汇总表已保存至: {summary_path}")

    print("\n" + "=" * 60)
    print(f"--- 步骤 {MODEL_NAME_PREFIX} (TabNet, Optuna + GPU) 全部任务完成 ---")
    print("=" * 60)

if __name__ == '__main__':
    main()