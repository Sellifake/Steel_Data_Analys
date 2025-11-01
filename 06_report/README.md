# 研究报告

本文件夹包含完整的研究报告LaTeX源码和相关资源文件。

## 文件说明

- **`研究报告_AI技术在钢铁材料性能预测中的应用.tex`**：主LaTeX源文件
- **`研究报告_AI技术在钢铁材料性能预测中的应用.pdf`**：编译后的PDF报告（如果存在）
- **`fig/`**：所有报告使用的图片文件（已重命名为英文名称，避免编码问题）

## LaTeX编译辅助文件

以下文件为LaTeX编译时自动生成的临时文件，可忽略：
- `.aux`, `.log`, `.out`, `.fdb_latexmk`, `.fls`, `.synctex.gz`

## 使用说明

### 本地编译（推荐使用LaTeX Workshop插件）

1. 使用VS Code打开`.tex`文件
2. 安装LaTeX Workshop插件
3. 直接编译，所有图片路径已设置为相对路径`fig/`

### Overleaf在线编译

1. 将整个`06_report/`文件夹内容上传到Overleaf项目
2. 确保主文件名为`研究报告_AI技术在钢铁材料性能预测中的应用.tex`
3. 编译引擎选择：**XeLaTeX**（支持中文）
4. 直接编译即可

## 图片文件

所有图片已重命名为英文名称：
- `01_comprehensive_model_performance_comparison.png` - 模型性能综合对比
- `feature_frequency_analysis_elongation.png` - 断后伸长率特征重要性
- `feature_frequency_analysis_tensile_strength.png` - 抗拉强度特征重要性
- `feature_frequency_analysis_yield_strength.png` - 屈服强度特征重要性
- `02b_pdp_plot_holding_time_vs_tensile_strength.png` - 保温时长扰动分析
- `02b_pdp_plot_cold_spot_temp_vs_tensile_strength.png` - 冷点温度扰动分析
- `02b_pdp_plot_hot_spot_temp_vs_tensile_strength.png` - 热点温度扰动分析

## 项目链接

GitHub: https://github.com/Sellifake/Steel_Data_Analys

