# 中文字体显示修复说明

## 问题描述

在可视化结果中，中文图例和标签可能显示为方框或乱码，这是由于 matplotlib 默认字体不支持中文字符导致的。

## 解决方案

### 1. 自动修复脚本

我们提供了 `fix_chinese_font.py` 脚本来自动修复中文显示问题：

```bash
cd scripts
python fix_chinese_font.py
```

### 2. 字体配置原理

脚本会按优先级尝试以下中文字体：

1. **Microsoft YaHei** (微软雅黑) - Windows 系统默认
2. **SimHei** (黑体) - 常见中文字体
3. **Arial Unicode MS** - 支持 Unicode 的字体
4. **DejaVu Sans** - 备用字体

### 3. 手动配置方法

如果需要手动配置，可以在 Python 脚本中添加：

```python
import matplotlib
import matplotlib.pyplot as plt

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

## 修复效果

修复后的可视化图表将正确显示：

- ✅ 健康人 / 中风患者
- ✅ 样本分布 / 数据组分布
- ✅ 分类器性能对比
- ✅ 混淆矩阵标签
- ✅ 所有中文图例和标题

## 布局优化

除了中文字体修复，还进行了以下布局优化：

- ✅ **标题间距**: 增加了标题与图表的间距（pad=15-20）
- ✅ **数字标注**: 调整了柱状图数字标注位置，避免与标题重合
- ✅ **Y 轴范围**: 动态调整 Y 轴上限，为数字标注留出空间
- ✅ **子图间距**: 增加了子图之间的间距，避免重叠
- ✅ **图例位置**: 优化了图例位置，避免遮挡数据
- ✅ **字体大小**: 调整了标注和标签的字体大小，提高可读性

## 备用方案

如果系统不支持中文字体，脚本会自动切换到英文标签：

- Healthy / Stroke
- Sample Distribution
- Classifier Performance Comparison
- Confusion Matrix
- 等等

## 常见问题

### Q: 为什么会出现中文显示问题？

A: matplotlib 默认使用不支持中文的字体，需要手动配置中文字体。

### Q: 如何检查系统可用字体？

A: 可以使用以下代码查看：

```python
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in fonts if 'YaHei' in f or 'SimHei' in f or 'Microsoft' in f]
print(chinese_fonts)
```

### Q: 修复后图表质量如何？

A: 修复后的图表保持原有的高质量（300 DPI），只是字体显示得到改善。

## 技术细节

### 字体优先级策略

1. 首先尝试 Windows 系统常见的中文字体
2. 然后尝试跨平台的 Unicode 字体
3. 最后使用系统默认字体作为备用

### 兼容性考虑

- Windows: Microsoft YaHei, SimHei
- macOS: Arial Unicode MS, PingFang SC
- Linux: DejaVu Sans, WenQuanYi

### 自动降级机制

如果所有中文字体都不可用，系统会：

1. 自动切换到英文标签
2. 保持图表的完整性和可读性
3. 在控制台提示用户字体状态

## 使用建议

1. **推荐使用修复脚本**: `fix_chinese_font.py` 提供了最完整的解决方案
2. **检查输出信息**: 脚本会显示字体配置状态
3. **保留原始文件**: 修复脚本会生成新文件，不会覆盖原始结果
4. **验证显示效果**: 打开生成的 PNG 文件检查中文显示是否正常

---

**修复时间**: 2025 年 7 月 4 日  
**适用系统**: Windows, macOS, Linux  
**测试状态**: ✅ 已验证有效
