# 钢卷性能预测 - 集成学习模型

## 项目概述

本模块专注于使用集成学习算法（随机森林、XGBoost、LightGBM）对钢卷性能指标进行预测建模。通过超参数调优和K折交叉验证，构建高性能的预测模型，并提供特征重要性分析和扰动预测功能。

## 最终结果保存位置

- **模型性能汇总**: `results/*/performance_summary.csv` - 各模型的性能指标对比
- **训练好的模型**: `results/*/saved_models/` - 保存的模型文件（.joblib格式）
- **特征重要性**: `results/*/` 文件夹中的特征重要性图表和CSV文件
- **扰动预测结果**: `results/*/` 文件夹中的PDP图表和预测结果CSV文件
- **决策树可视化**: `results/01_random_forest_results/decision_tree_visualizations/` - 决策树结构图

## 代码文件详细说明

### 1. 01a_random_forest_modeling.py - 随机森林建模

**功能**: 使用随机森林算法进行性能预测，包含完整的模型训练和解释分析

**输入文件**:
- `../01_read_data/results/04_data_selected.xlsx` - 经过特征筛选的最终数据集

**输出文件**:
- `results/01_random_forest_results/` - 随机森林结果文件夹
- `saved_models/` - 训练好的随机森林模型
- `*_feature_importance.png/csv` - 特征重要性分析
- `decision_tree_visualizations/` - 决策树结构可视化
- `random_forest_performance_summary.csv` - 性能汇总

**主要功能**:
- 使用5折交叉验证评估随机森林模型性能
- 训练最终模型并保存到磁盘
- 生成特征重要性分析（图表+CSV）
- 可视化决策树分裂节点结构
- 支持SHAP分析（可选开关控制）

**性能表现**:
- 抗拉强度: R²=0.8991, MAE=7.5121, RMSE=10.4959
- 屈服Rp0.2值*: R²=0.7670, MAE=9.7394, RMSE=13.7881
- 断后伸长率: R²=0.5044, MAE=1.9425, RMSE=2.5992

### 2. 01b_rf_perturbation_predict.py - 随机森林扰动预测

**功能**: 使用训练好的随机森林模型对扰动数据进行预测分析

**输入文件**:
- `results/01_random_forest_results/saved_models/` - 训练好的随机森林模型
- `../01_read_data/results/05_perturbed_dataset_*.csv` - 扰动数据集

**输出文件**:
- `01b_rf_prediction_results_*.csv` - 详细预测结果（ICE）
- `01b_rf_summary_stats_*.csv` - 统计汇总结果（PDP）
- `01b_pdp_plot_*.png` - 部分依赖图（带95%置信区间）

**主要功能**:
- 加载训练好的随机森林模型进行扰动预测
- 生成带置信区间的PDP效应图
- 分析关键工艺参数对性能指标的影响趋势
- 提供性能指标基线参考线

### 3. 02a_xgboost_modeling.py - XGBoost超参数调优建模

**功能**: 使用XGBoost算法进行性能预测，并进行超参数调优

**输入文件**:
- `../01_read_data/results/04_data_selected.xlsx` - 经过特征筛选的最终数据集

**输出文件**:
- `results/02_xgboost_results/` - XGBoost调优结果文件夹
- `saved_models/` - 调优后的XGBoost模型
- `*_feature_importance.png/csv` - 特征重要性分析
- `xgboost_tuned_performance_summary.csv` - 调优后性能汇总

**主要功能**:
- 使用RandomizedSearchCV对XGBoost超参数进行调优（200次迭代）
- 调优参数包括：n_estimators, learning_rate, max_depth, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda
- 5折交叉验证评估调优后的最佳XGBoost模型性能
- 生成特征重要性分析

**性能表现**:
- 抗拉强度: R²=0.9022, MAE=7.2556, RMSE=10.3182
- 屈服Rp0.2值*: R²=0.7766, MAE=9.5039, RMSE=13.5158
- 断后伸长率: R²=0.5194, MAE=1.9158, RMSE=2.5608

### 4. 02b_xgb_perturbation_predict.py - XGBoost扰动预测

**功能**: 使用调优后的XGBoost模型对扰动数据进行预测分析

**输入文件**:
- `results/02_xgboost_results/saved_models/` - 调优后的XGBoost模型
- `../01_read_data/results/05_perturbed_dataset_*.csv` - 扰动数据集

**输出文件**:
- `02b_xgb_prediction_results_*.csv` - 详细预测结果
- `02b_xgb_summary_stats_*.csv` - 统计汇总结果
- `02b_pdp_plot_*.png` - 部分依赖图

**主要功能**:
- 加载调优后的XGBoost模型进行扰动预测
- 生成PDP效应图分析关键参数影响
- 输出详细的预测结果和统计汇总

### 5. 03a_lightgbm_modeling.py - LightGBM超参数调优建模

**功能**: 使用LightGBM算法进行性能预测，并进行超参数调优

**输入文件**:
- `../01_read_data/results/04_data_selected.xlsx` - 经过特征筛选的最终数据集

**输出文件**:
- `results/03_lightgbm_results/` - LightGBM调优结果文件夹
- `saved_models/` - 调优后的LightGBM模型
- `*_feature_importance.png/csv` - 特征重要性分析
- `lightgbm_tuned_performance_summary.csv` - 调优后性能汇总

**主要功能**:
- 使用RandomizedSearchCV对LightGBM超参数进行调优（200次迭代）
- 调优参数包括：n_estimators, learning_rate, max_depth, num_leaves, subsample, colsample_bytree, reg_alpha, reg_lambda
- 5折交叉验证评估调优后的最佳LightGBM模型性能
- 生成特征重要性分析

**性能表现**:
- 抗拉强度: R²=0.8962, MAE=7.3866, RMSE=10.5968
- 屈服Rp0.2值*: R²=0.7655, MAE=9.8005, RMSE=13.7690
- 断后伸长率: R²=0.5120, MAE=1.9255, RMSE=2.5801

### 6. 03b_lgb_perturbation_predict.py - LightGBM扰动预测

**功能**: 使用调优后的LightGBM模型对扰动数据进行预测分析

**输入文件**:
- `results/03_lightgbm_results/saved_models/` - 调优后的LightGBM模型
- `../01_read_data/results/05_perturbed_dataset_*.csv` - 扰动数据集

**输出文件**:
- `03b_lgb_prediction_results_*.csv` - 详细预测结果
- `03b_lgb_summary_stats_*.csv` - 统计汇总结果
- `03b_pdp_plot_*.png` - 部分依赖图

**主要功能**:
- 加载调优后的LightGBM模型进行扰动预测
- 生成PDP效应图分析关键参数影响
- 输出详细的预测结果和统计汇总

## 模型特点与优势

### 随机森林（Random Forest）
- **集成学习**: 通过多棵决策树的投票机制提高预测准确性
- **特征重要性**: 提供直观的特征重要性排序
- **抗过拟合**: 通过随机采样和特征选择降低过拟合风险
- **可解释性**: 支持决策树可视化，便于理解模型决策过程
- **适用场景**: 适合处理高维特征和复杂非线性关系

### XGBoost（eXtreme Gradient Boosting）
- **梯度提升**: 通过梯度提升算法逐步优化预测性能
- **正则化**: 内置L1和L2正则化防止过拟合
- **并行计算**: 支持多线程并行训练，提高计算效率
- **缺失值处理**: 自动处理缺失值，无需预处理
- **适用场景**: 在结构化数据上表现优异，适合竞赛和实际应用

### LightGBM（Light Gradient Boosting Machine）
- **高效训练**: 使用基于梯度的单侧采样和互斥特征捆绑技术
- **内存优化**: 相比XGBoost占用更少内存
- **快速预测**: 训练和预测速度都很快
- **类别特征**: 原生支持类别特征，无需独热编码
- **适用场景**: 适合大规模数据集和实时预测场景

## 超参数调优策略

### XGBoost调优参数
- **n_estimators**: 树的数量 (100-1200)
- **learning_rate**: 学习率 (0.01-0.2)
- **max_depth**: 树的最大深度 (3-12)
- **subsample**: 训练样本采样率 (0.6-1.0)
- **colsample_bytree**: 特征采样率 (0.6-1.0)
- **gamma**: 节点分裂的最小损失降低 (0-0.5)
- **reg_alpha**: L1正则化 (0-1)
- **reg_lambda**: L2正则化 (0-1)

### LightGBM调优参数
- **n_estimators**: 树的数量 (100-1200)
- **learning_rate**: 学习率 (0.01-0.2)
- **max_depth**: 树的最大深度 (3-12)
- **num_leaves**: 叶子节点数 (20-60)
- **subsample**: 训练样本采样率 (0.6-1.0)
- **colsample_bytree**: 特征采样率 (0.6-1.0)
- **reg_alpha**: L1正则化 (0-1)
- **reg_lambda**: L2正则化 (0-1)

## 特征重要性分析

- **内置重要性**: 使用模型内置的特征重要性计算方法
- **排序分析**: 按重要性排序显示Top 20特征
- **可视化**: 生成水平条形图展示特征重要性
- **CSV导出**: 保存完整的重要性数据供进一步分析

## 扰动预测分析

通过PDP（部分依赖图）分析关键工艺参数对性能指标的影响：
- **热点峰值温度**: 分析最高温度点对性能的影响
- **冷点峰值温度**: 分析最低温度点对性能的影响
- **保温时长**: 分析保温时间对性能的影响

每个扰动分析都包含：
- **ICE数据**: 个体条件期望的详细预测结果
- **PDP统计**: 按扰动值分组的统计汇总（均值、中位数、95%置信区间）
- **可视化图表**: 带置信区间的PDP效应图
- **性能基线**: 显示性能指标的最小值和最大值基线

## 模型性能评估

每个模型都会生成详细的性能评估报告，包括：
- **R²分数**: 决定系数，衡量模型解释方差的能力
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **交叉验证结果**: 5折交叉验证的均值和标准差
- **调优前后对比**: 展示超参数调优的效果

## 性能对比总结

| 模型 | 抗拉强度 R² | 屈服Rp0.2值* R² | 断后伸长率 R² |
|------|-------------|-----------------|---------------|
| 随机森林 | 0.8991 | 0.7670 | 0.5044 |
| XGBoost | 0.9022 | 0.7766 | 0.5194 |
| LightGBM | 0.8962 | 0.7655 | 0.5120 |

**最佳表现**:
- 抗拉强度: XGBoost (R²=0.9022)
- 屈服Rp0.2值*: XGBoost (R²=0.7766)
- 断后伸长率: XGBoost (R²=0.5194)

## 注意事项

- 请确保已运行01_read_data模块生成最终数据集
- 所有模型都设置了随机种子确保结果可复现
- 扰动预测需要先生成扰动数据集（01_read_data模块的05脚本）
- 超参数调优过程可能需要较长时间，建议在性能较好的机器上运行
- 模型文件较大，请确保有足够的磁盘空间
- 建议按顺序运行脚本：先运行建模脚本（01a, 02a, 03a），再运行扰动预测脚本（01b, 02b, 03b）