# 钢卷性能预测 - 非线性模型

## 项目概述

本模块专注于使用非线性机器学习方法（高斯过程回归GPR、支持向量回归SVR）对钢卷性能指标进行预测建模。通过超参数调优和K折交叉验证，构建高性能的非线性预测模型，并提供不确定性量化和特征重要性分析。

## 最终结果保存位置

- **调优后模型性能**: `results/*/gpr_tuned_performance_summary.csv` 和 `svr_tuned_performance_summary.csv` - 调优后的模型性能指标
- **训练好的模型**: `results/*/saved_models/` - 保存的调优后模型文件（.joblib格式）
- **特征重要性**: `results/*/` 文件夹中的特征重要性图表和CSV文件
- **扰动预测结果**: `results/*/` 文件夹中的PDP图表和预测结果CSV文件
- **超参数搜索结果**: 调优过程中的参数搜索记录

## 代码文件详细说明

### 1. 01a_gpr_tuning.py - 高斯过程回归调优建模

**功能**: 使用高斯过程回归（GPR）进行性能预测，并进行超参数调优

**输入文件**:
- `../01_read_data/results/04_data_selected.xlsx` - 经过特征筛选的最终数据集

**输出文件**:
- `results/01_gpr_tuned_results/` - GPR调优结果文件夹
- `saved_models/` - 调优后的GPR模型
- `*_feature_importance.png/csv` - 特征重要性分析
- `gpr_tuned_performance_summary.csv` - 调优后性能汇总

**主要功能**:
- 使用RandomizedSearchCV对GPR核函数超参数进行调优
- 核函数结构：ConstantKernel * RBF + WhiteKernel
- 使用Pipeline捆绑StandardScaler和GaussianProcessRegressor
- 5折交叉验证评估调优后的最佳GPR模型性能
- 使用排列重要性（Permutation Importance）作为特征重要性衡量标准
- 提供预测不确定性量化（GPR的核心优势）

### 2. 01b_gpr_perturbation_predict.py - GPR扰动预测

**功能**: 使用调优后的GPR模型对扰动数据进行预测分析

**输入文件**:
- `results/01_gpr_tuned_results/saved_models/` - 调优后的GPR模型
- `../01_read_data/results/05_perturbed_dataset_*.csv` - 扰动数据集

**输出文件**:
- `01b_gpr_prediction_results_*.csv` - 详细预测结果（包含不确定性）
- `01b_gpr_summary_stats_*.csv` - 统计汇总结果
- `01b_pdp_plot_*.png` - 部分依赖图（带不确定性区间）

**主要功能**:
- 加载调优后的GPR模型进行扰动预测
- 提供预测均值和不确定性区间
- 生成带置信区间的PDP效应图
- 分析关键工艺参数对性能指标的影响趋势
- 利用GPR的不确定性量化能力进行风险评估

### 3. 02a_svr_tuning.py - 支持向量回归调优建模

**功能**: 使用支持向量回归（SVR）进行性能预测，并进行超参数调优

**输入文件**:
- `../01_read_data/results/04_data_selected.xlsx` - 经过特征筛选的最终数据集

**输出文件**:
- `results/02_svr_tuned_results/` - SVR调优结果文件夹
- `saved_models/` - 调优后的SVR模型
- `*_feature_importance.png/csv` - 特征重要性分析
- `svr_tuned_performance_summary.csv` - 调优后性能汇总

**主要功能**:
- 使用RandomizedSearchCV对SVR的C和gamma参数进行调优
- 使用RBF核函数的SVR模型
- 使用Pipeline捆绑StandardScaler和SVR
- 5折交叉验证评估调优后的最佳SVR模型性能
- 使用排列重要性作为特征重要性衡量标准
- 处理高维特征空间中的非线性关系

### 4. 02b_svr_perturbation_predict.py - SVR扰动预测

**功能**: 使用调优后的SVR模型对扰动数据进行预测分析

**输入文件**:
- `results/02_svr_tuned_results/saved_models/` - 调优后的SVR模型
- `../01_read_data/results/05_perturbed_dataset_*.csv` - 扰动数据集

**输出文件**:
- `02b_svr_prediction_results_*.csv` - 详细预测结果
- `02b_svr_summary_stats_*.csv` - 统计汇总结果
- `02b_pdp_plot_*.png` - 部分依赖图

**主要功能**:
- 加载调优后的SVR模型进行扰动预测
- 生成PDP效应图分析关键参数影响
- 输出详细的预测结果和统计汇总
- 分析支持向量对预测结果的影响

## 模型特点与优势

### 高斯过程回归（GPR）
- **不确定性量化**: 提供预测的置信区间
- **核函数灵活性**: 通过核函数捕捉复杂的非线性关系
- **贝叶斯框架**: 基于贝叶斯推理的预测方法
- **适用场景**: 小到中等规模数据集，需要不确定性量化的场景

### 支持向量回归（SVR）
- **高维处理能力**: 在高维特征空间中表现优异
- **核技巧**: 通过核函数处理非线性关系
- **稀疏性**: 只使用支持向量进行预测，计算效率高
- **鲁棒性**: 对异常值具有一定的抗干扰能力

## 超参数调优策略

### GPR调优参数
- **ConstantKernel**: 信号方差参数
- **RBF核长度尺度**: 控制函数平滑度
- **WhiteKernel**: 噪声水平参数
- **搜索空间**: 使用对数均匀分布进行参数搜索

### SVR调优参数
- **C参数**: 控制正则化强度
- **gamma参数**: RBF核函数的带宽参数
- **搜索空间**: 使用对数均匀分布进行参数搜索

## 特征重要性分析

- **排列重要性**: 通过随机打乱特征值评估特征重要性
- **核函数分析**: 分析核函数参数对特征重要性的影响
- **支持向量分析**: 分析支持向量对模型决策的贡献

## 扰动预测分析

通过PDP（部分依赖图）分析关键工艺参数对性能指标的影响：
- **热点峰值温度**: 分析最高温度点对性能的影响
- **冷点峰值温度**: 分析最低温度点对性能的影响
- **保温时长**: 分析保温时间对性能的影响

## 模型性能评估

每个调优后的模型都会生成详细的性能评估报告，包括：
- **R²分数**: 决定系数，衡量模型解释方差的能力
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **交叉验证结果**: 5折交叉验证的均值和标准差
- **调优前后对比**: 展示超参数调优的效果

## 注意事项

- 请确保已运行01_read_data模块生成最终数据集
- GPR模型适合小到中等规模数据集，计算复杂度较高
- SVR模型对数据标准化敏感，Pipeline中已包含StandardScaler
- 扰动预测需要先生成扰动数据集（01_read_data模块的05脚本）
- 所有模型都设置了随机种子确保结果可复现
- GPR提供的不确定性量化是其他模型无法提供的独特优势
