# 项目配色方案说明

## 🎨 配色方案

本项目采用了统一的配色方案，确保所有可视化图表的视觉一致性和美观性。

### 主要颜色定义

```python
COLORS = {
    'primary': '#98CFE6',    # 浅蓝色 - 主要颜色
    'secondary': '#ADE7A8',  # 浅绿色 - 次要颜色  
    'accent': '#F39F4E',     # 橙色 - 强调色
    'highlight': '#EEB7D3',  # 粉色 - 高亮色
    'neutral': '#DBDAD3',    # 灰色 - 中性色
    'warning': '#FFDF97'     # 黄色 - 警告色
}
```

### 颜色应用规则

#### 1. 主要数据系列
- **健康人数据**: `secondary` (#ADE7A8) - 浅绿色
- **中风患者数据**: `highlight` (#EEB7D3) - 粉色

#### 2. 图表元素
- **条形图/柱状图**: `primary` (#98CFE6) - 浅蓝色
- **强调数据**: `accent` (#F39F4E) - 橙色
- **中性数据**: `neutral` (#DBDAD3) - 灰色
- **警告/注意**: `warning` (#FFDF97) - 黄色

#### 3. 多系列数据
按优先级顺序使用：
1. `primary` (#98CFE6)
2. `secondary` (#ADE7A8) 
3. `accent` (#F39F4E)
4. `highlight` (#EEB7D3)
5. `neutral` (#DBDAD3)
6. `warning` (#FFDF97)

## 📊 具体应用

### 分类结果图 (classification_summary.png)
- **模型性能对比**: 准确率(primary)、精确率(secondary)、召回率(accent)
- **特征重要性**: primary
- **样本分布饼图**: 健康人(secondary)、中风患者(highlight)

### 回归分析图 (regression_summary.png)
- **效应量排序**: primary
- **组间差异对比**: 健康人(secondary)、中风患者(highlight)
- **p值分布**: primary
- **严重程度分布**: 健康人(secondary)、中风患者(highlight)
- **模型性能**: R²(primary)、RMSE(accent)

### 临床洞察图 (clinical_insights.png)
- **雷达图**: 健康人(secondary)、中风患者(highlight)
- **康复优先级**: accent、primary、secondary、highlight、neutral
- **严重程度分级**: 轻度(secondary)、中度(warning)、重度(accent)
- **预后预测**: 轻度(secondary)、中度(warning)、重度(accent)

## 🎯 设计原则

### 1. 一致性
- 所有图表使用相同的配色方案
- 相同类型的数据使用相同颜色
- 健康人始终使用绿色系，中风患者使用粉色系

### 2. 可读性
- 颜色对比度适中，易于区分
- 避免使用过于鲜艳或刺眼的颜色
- 考虑色盲友好性

### 3. 专业性
- 选择医学/科研领域常用的配色
- 避免过于花哨的颜色组合
- 保持整体视觉的专业感

### 4. 语义化
- 绿色代表健康/正常状态
- 粉色代表异常/患病状态
- 橙色用于强调重要信息
- 黄色用于警告或注意事项

## 🔧 技术实现

### 在脚本中使用
```python
# 定义配色方案
COLORS = {
    'primary': '#98CFE6',
    'secondary': '#ADE7A8', 
    'accent': '#F39F4E',
    'highlight': '#EEB7D3',
    'neutral': '#DBDAD3',
    'warning': '#FFDF97'
}

# 使用示例
plt.bar(x, y, color=COLORS['primary'])
plt.plot(x, y1, color=COLORS['secondary'], label='健康人')
plt.plot(x, y2, color=COLORS['highlight'], label='中风患者')
```

### 配色列表
```python
# 用于多系列数据
COLOR_PALETTE = [
    COLORS['primary'], 
    COLORS['secondary'], 
    COLORS['accent'],
    COLORS['highlight'], 
    COLORS['neutral'], 
    COLORS['warning']
]
```

## 📝 更新记录

- **2025-07-04**: 初始配色方案定义
- **2025-07-04**: 应用到所有可视化脚本
- **2025-07-04**: 创建配色说明文档

## 🎨 颜色预览

| 颜色名称 | 十六进制 | 预览 | 用途 |
|---------|---------|------|------|
| Primary | #98CFE6 | 🟦 | 主要数据、条形图 |
| Secondary | #ADE7A8 | 🟩 | 健康人数据 |
| Accent | #F39F4E | 🟧 | 强调数据、重要信息 |
| Highlight | #EEB7D3 | 🟪 | 中风患者数据 |
| Neutral | #DBDAD3 | ⬜ | 中性数据、背景 |
| Warning | #FFDF97 | 🟨 | 警告、注意事项 |

---

**维护者**: Augment Agent  
**最后更新**: 2025年7月4日
