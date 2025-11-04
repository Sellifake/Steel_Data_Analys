"""
生成研究总体框架流程图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 定义模块信息
modules = [
    {
        'name': '数据预处理模块',
        'desc': '从原始Excel文件中提取温度曲线数据，\n进行数据清洗和格式统一',
        'pos': (2, 8.5),
        'size': (2.5, 1.2)
    },
    {
        'name': '特征提取模块',
        'desc': '从时间序列数据中\n提取35个工艺特征',
        'pos': (5.5, 8.5),
        'size': (2.5, 1.2)
    },
    {
        'name': '数据合并模块',
        'desc': '整合工艺特征、化学成分\n和性能指标数据',
        'pos': (2, 6.5),
        'size': (2.5, 1.2)
    },
    {
        'name': '特征选择模块',
        'desc': '基于相关性和VIF分析\n删除冗余特征',
        'pos': (5.5, 6.5),
        'size': (2.5, 1.2)
    },
    {
        'name': '扰动数据生成模块',
        'desc': '生成用于敏感性分析的\n扰动数据集',
        'pos': (2, 4.5),
        'size': (2.5, 1.2)
    },
    {
        'name': '模型训练模块',
        'desc': '训练多种机器学习模型\n并优化超参数',
        'pos': (5.5, 4.5),
        'size': (2.5, 1.2)
    },
    {
        'name': '模型评估模块',
        'desc': '通过交叉验证\n评估模型性能',
        'pos': (2, 2.5),
        'size': (2.5, 1.2)
    },
    {
        'name': '结果解释模块',
        'desc': '分析特征重要性和\n工艺参数影响',
        'pos': (5.5, 2.5),
        'size': (2.5, 1.2)
    }
]

# 定义箭头连接（从模块索引到模块索引）
arrows = [
    (0, 1),  # 数据预处理 -> 特征提取
    (1, 2),  # 特征提取 -> 数据合并
    (2, 3),  # 数据合并 -> 特征选择
    (3, 4),  # 特征选择 -> 扰动数据生成
    (3, 5),  # 特征选择 -> 模型训练
    (5, 6),  # 模型训练 -> 模型评估
    (4, 6),  # 扰动数据生成 -> 模型评估
    (6, 7),  # 模型评估 -> 结果解释
]

# 绘制模块
for i, module in enumerate(modules):
    x, y = module['pos']
    w, h = module['size']
    
    # 绘制圆角矩形框
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05",
        linewidth=2,
        edgecolor='#2c3e50',
        facecolor='#ecf0f1',
        zorder=2
    )
    ax.add_patch(box)
    
    # 添加模块名称
    ax.text(x, y + 0.3, module['name'], 
            ha='center', va='center',
            fontsize=11, fontweight='bold',
            color='#2c3e50', zorder=3)
    
    # 添加描述文字
    ax.text(x, y - 0.25, module['desc'],
            ha='center', va='center',
            fontsize=9, color='#34495e',
            zorder=3)

# 绘制箭头
for start_idx, end_idx in arrows:
    start_mod = modules[start_idx]
    end_mod = modules[end_idx]
    
    start_x, start_y = start_mod['pos']
    start_w, start_h = start_mod['size']
    end_x, end_y = end_mod['pos']
    end_w, end_h = end_mod['size']
    
    # 计算起始和结束点
    # 根据位置关系确定箭头方向
    dx = end_x - start_x
    dy = end_y - start_y
    
    if abs(dx) > abs(dy):  # 水平连接
        if dx > 0:  # 向右
            start_point = (start_x + start_w/2, start_y)
            end_point = (end_x - end_w/2, end_y)
        else:  # 向左
            start_point = (start_x - start_w/2, start_y)
            end_point = (end_x + end_w/2, end_y)
    else:  # 垂直连接
        if dy > 0:  # 向上
            start_point = (start_x, start_y + start_h/2)
            end_point = (end_x, end_y - end_h/2)
        else:  # 向下
            start_point = (start_x, start_y - start_h/2)
            end_point = (end_x, end_y + end_h/2)
    
    # 绘制箭头
    arrow = FancyArrowPatch(
        start_point, end_point,
        arrowstyle='->', lw=2,
        color='#3498db',
        zorder=1,
        mutation_scale=20
    )
    ax.add_patch(arrow)

# 添加标题
ax.text(5, 9.5, '研究总体框架', 
        ha='center', va='center',
        fontsize=16, fontweight='bold',
        color='#2c3e50')

plt.tight_layout()
plt.savefig('fig/00_research_framework.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("流程图已保存到 fig/00_research_framework.png")
plt.close()

