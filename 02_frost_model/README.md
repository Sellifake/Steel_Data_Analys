# 钢卷性能预测 - 集成学习模型

## 项目概述

本模块专注于使用集成学习方法（随机森林、XGBoost、LightGBM）对钢卷性能指标进行预测建模。通过K折交叉验证评估模型性能，生成特征重要性分析，并进行扰动预测分析，为钢卷生产工艺优化提供数据支持。

## 最终结果保存位置

- **模型性能汇总**: `results/` 各子文件夹中的 `*_performance_summary.csv` - 各模型的交叉验证性能指标
- **训练好的模型**: `results/*/saved_models/` - 保存的模型文件（.joblib格式）
- **特征重要性**: `results/*/` 文件夹中的特征重要性图表和CSV文件
- **扰动预测结果**: `results/*/` 文件夹中的PDP图表和预测结果CSV文件
- **SHAP分析**: `results/01_random_forest_results/shap_dependence_plots/` - SHAP依赖图分析

## 代码文件详细说明

### 1. 01a_random_forest_modeling.py - 随机森林建模

**功能**: 使用随机森林算法对钢卷性能指标进行预测建模

**输入文件**:
- `../01_read_data/results/04_data_selected.xlsx` - 经过特征筛选的最终数据集

**输出文件**:
- `results/01_random_forest_results/` - 随机森林模型结果文件夹
- `saved_models/` - 训练好的模型文件
- `*_feature_importance.png/csv` - 特征重要性图表和数据
- `*_shap_summary.png` - SHAP概要图
- `shap_dependence_plots/` - SHAP依赖图文件夹
- `decision_tree_visualizations/` - 决策树可视化图

**主要功能**:
- 加载经过特征筛选的最终数据集
- 对三个性能指标（抗拉强度、屈服Rp0.2值、断后伸长率）分别建模
- 使用5折交叉验证评估模型性能
- 生成特征重要性分析和SHAP解释性分析
- 导出决策树结构图进行模型可解释性分析
- 保存训练好的模型供后续扰动预测使用

### 2. 01b_rf_perturbation_predict.py - 随机森林扰动预测

**功能**: 使用训练好的随机森林模型对扰动数据进行预测分析

**输入文件**:
- `results/01_random_forest_results/saved_models/` - 训练好的随机森林模型
- `../01_read_data/results/05_perturbed_dataset_*.csv` - 扰动数据集

**输出文件**:
- `01b_rf_prediction_results_*.csv` - 详细预测结果（ICE数据）
- `01b_rf_summary_stats_*.csv` - 统计汇总结果（PDP数据）
- `01b_pdp_plot_*.png` - 部分依赖图（PDP效应图）

**主要功能**:
- 加载训练好的随机森林模型
- 对热点峰值温度、冷点峰值温度、保温时长三个关键特征进行扰动预测
- 生成个体条件期望（ICE）和部分依赖图（PDP）分析
- 绘制带90%置信区间的PDP效应图
- 分析关键工艺参数对性能指标的影响趋势

### 3. 02a_xgboost_modeling.py - XGBoost建模

**功能**: 使用XGBoost算法进行性能预测建模

**输入文件**:
- `../01_read_data/results/04_data_selected.xlsx` - 经过特征筛选的最终数据集

**输出文件**:
- `results/02_xgboost_results/` - XGBoost模型结果文件夹
- `saved_models/` - 训练好的XGBoost模型
- `*_feature_importance.png/csv` - 特征重要性分析

**主要功能**:
- 使用XGBoost梯度提升算法进行建模
- 5折交叉验证评估模型性能
- 生成特征重要性分析
- 保存模型供扰动预测使用

### 4. 02b_xgb_perturbation_predict.py - XGBoost扰动预测

**功能**: 使用训练好的XGBoost模型进行扰动预测分析

**主要功能**:
- 加载XGBoost模型进行扰动预测
- 生成PDP效应图分析关键参数影响
- 输出详细的预测结果和统计汇总

### 5. 03a_lightgbm_modeling.py - LightGBM建模

**功能**: 使用LightGBM算法进行性能预测建模

**主要功能**:
- 使用LightGBM梯度提升算法
- 5折交叉验证评估性能
- 生成特征重要性分析
- 保存模型供后续使用

### 6. 03b_lgb_perturbation_predict.py - LightGBM扰动预测

**功能**: 使用LightGBM模型进行扰动预测分析

**主要功能**:
- LightGBM模型的扰动预测分析
- 生成PDP效应图
- 输出预测结果和统计汇总

### 7. 04_boosting_tuning.py - 集成模型调优

**功能**: 对XGBoost和LightGBM模型进行超参数调优

**主要功能**:
- 使用随机搜索进行超参数调优
- 比较调优前后的模型性能
- 保存最佳参数配置的模型

## 模型性能评估

每个模型都会生成详细的性能评估报告，包括：
- **R²分数**: 决定系数，衡量模型解释方差的能力
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **交叉验证结果**: 5折交叉验证的均值和标准差

## 特征重要性分析

- **随机森林**: 基于基尼不纯度减少的特征重要性
- **XGBoost/LightGBM**: 基于增益的特征重要性
- **SHAP分析**: 提供模型可解释性的详细分析

## 扰动预测分析

通过PDP（部分依赖图）分析关键工艺参数对性能指标的影响：
- **热点峰值温度**: 分析最高温度点对性能的影响
- **冷点峰值温度**: 分析最低温度点对性能的影响  
- **保温时长**: 分析保温时间对性能的影响

## 注意事项

- 请确保已运行01_read_data模块生成最终数据集
- 各模型脚本可独立运行，但建议按顺序执行
- 扰动预测需要先生成扰动数据集（01_read_data模块的05脚本）
- 所有模型都设置了随机种子确保结果可复现
