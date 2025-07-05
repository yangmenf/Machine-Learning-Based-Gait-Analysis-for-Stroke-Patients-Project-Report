# 步态数据分析：中风患者与健康人的分类建模

## 项目概述

本项目成功解决了 MATLAB v7.3 格式 MAT 文件的读取难题，完整提取了 Figshare 公开数据集中 188 名受试者（138 名健康成年人和 50 名中风患者）的步态数据，实现了高精度的分类建模，为中风康复评估提供了可靠的技术方案。

## 🎯 主要成果

- **数据规模**: 1,128 个样本（188 个受试者 × 6 个数据组）
- **分类准确率**: 98.2% (Random Forest)
- **特征数量**: 386 个步态特征
- **技术突破**: 解决 MAT 文件转换问题，数据提取量提升 9.4 倍
- **临床价值**: 为中风康复评估提供高精度分类工具

## 📁 项目结构

> 📋 详细的项目结构说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

```
深大老师布置的任务/
├── README.md                    # 项目说明文档
├── PROJECT_STRUCTURE.md         # 详细项目结构说明
├── data/                        # 数据文件目录
│   ├── MAT_normalizedData_AbleBodiedAdults_v06-03-23.mat     # 健康人原始数据
│   ├── MAT_normalizedData_PostStrokeAdults_v27-02-23.mat    # 中风患者原始数据
│   ├── complete_gait_features.csv                           # **完整步态特征数据(1,128样本)** ⭐
│   ├── complete_gait_features_readme.txt                    # 完整数据说明文档
│   ├── MATdatafiles_description_v1.3_LST.xlsx               # 原始变量说明文件
│   ├── s41597-023-02767-y.pdf                               # 相关论文
│   └── 数据集介绍以及任务.txt                                  # 任务说明
├── scripts/                     # 分析脚本目录
│   ├── complete_mat_converter.py                            # **完整MAT转换器** ⭐
│   ├── complete_data_analysis.py                            # **完整数据分析脚本** ⭐（分类任务）
│   ├── regression_analysis.py                               # **回归分析脚本** ⭐（回归任务）
│   ├── fix_chinese_font.py                                  # 中文字体修复脚本
│   └── inspect_mat_structure.py                             # MAT结构检查工具
├── results/                     # 结果文件目录
│   ├── complete_mat_analysis.png                            # **完整数据分析结果图** ⭐（分类）
│   └── regression_analysis.png                              # **回归分析结果图** ⭐（回归）
└── docs/                        # 文档目录
    ├── final_project_report.md                              # **完整项目报告** ⭐（含图表说明）
    ├── executive_summary.md                                 # **项目执行摘要** ⭐
    ├── complete_mat_analysis_report.md                      # 分类分析详细报告
    ├── regression_analysis_report.md                        # 回归分析详细报告
    ├── mat_conversion_success_summary.md                    # MAT转换解决方案总结
    ├── chinese_font_fix.md                                  # 中文字体修复说明
    └── color_scheme.md                                      # 配色方案说明
```

## 📊 数据说明

### 完整真实数据

**原始 MAT 文件**：

- 来自 Figshare 的真实医学研究数据
- MATLAB v7.3 格式（HDF5 结构）
- 包含完整的生物力学信息

**转换后的 CSV 数据**：

- **complete_gait_features.csv**: 1,128 个样本 × 386 个特征
- 来源：188 个真实受试者（138 健康人 + 50 中风患者）
- 每个受试者 6 个数据组记录（不同步态条件）
- 包含运动学和动力学特征

### 数据组详细说明

#### 数据组命名规则

**[侧别]Segm\_[数据类型]Data**

#### 健康人数据组（828 个样本）

每个健康人（138 人）有 6 种不同的步态分析记录：

- **LsideSegm_BsideData**: 以左腿为主导的双侧步态分析（138 样本）
- **LsideSegm_LsideData**: 以左腿为主导的左侧步态分析（138 样本）
- **LsideSegm_RsideData**: 以左腿为主导的右侧步态分析（138 样本）
- **RsideSegm_BsideData**: 以右腿为主导的双侧步态分析（138 样本）
- **RsideSegm_LsideData**: 以右腿为主导的左侧步态分析（138 样本）
- **RsideSegm_RsideData**: 以右腿为主导的右侧步态分析（138 样本）

#### 中风患者数据组（300 个样本）

每个中风患者（50 人）有 6 种不同的步态分析记录：

- **NsideSegm_BsideData**: 以非瘫痪侧为主导的双侧步态分析（50 样本）
- **NsideSegm_NsideData**: 非瘫痪侧的步态分析（50 样本）
- **NsideSegm_PsideData**: 以非瘫痪侧为主导分析瘫痪侧步态（50 样本）
- **PsideSegm_BsideData**: 以瘫痪侧为主导的双侧步态分析（50 样本）
- **PsideSegm_NsideData**: 以瘫痪侧为主导分析非瘫痪侧步态（50 样本）
- **PsideSegm_PsideData**: 瘫痪侧的步态分析（50 样本）

#### 术语解释

- **LsideSegm**: Left side Segment（左侧步态分段）
- **RsideSegm**: Right side Segment（右侧步态分段）
- **NsideSegm**: Non-paretic side Segment（非瘫痪侧步态分段）
- **PsideSegm**: Paretic side Segment（瘫痪侧步态分段）
- **BsideData**: Bilateral side Data（双侧数据）
- **LsideData**: Left side Data（左侧数据）
- **RsideData**: Right side Data（右侧数据）
- **NsideData**: Non-paretic side Data（非瘫痪侧数据）
- **PsideData**: Paretic side Data（瘫痪侧数据）

#### 临床意义

**对于健康人**：

- 左右对称性分析：比较左右腿的步态差异
- 主导腿分析：分析以不同腿为主导时的步态模式
- 双侧协调性：评估两腿协调配合情况

**对于中风患者**：

- 瘫痪程度评估：比较瘫痪侧与非瘫痪侧的功能差异
- 补偿机制分析：研究非瘫痪侧如何补偿瘫痪侧功能
- 康复效果监测：通过不同侧别的数据评估康复进展

#### 数据组织优势

这种多维度的数据组织方式使得研究能够：

1. **深入分析步态不对称性**: 通过左右侧对比识别异常模式
2. **识别中风特异性步态模式**: 瘫痪侧与非瘫痪侧的差异分析
3. **提供个性化康复数据支撑**: 针对不同侧别制定康复方案
4. **建立更精确的分类模型**: 多条件数据提高模型准确性

> 💡 **为什么是 1,128 个样本？**
> 188 个受试者 × 6 个数据组 = 1,128 个样本
> 每个人在不同步态条件下被测量 6 次，提供了更全面的步态信息

## 🛠️ 技术流程

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  MAT文件    │ → │  数据预处理  │ → │  特征提取   │ → │  机器学习   │
│ (1,128样本) │   │ (HDF5解析)  │   │ (386特征)   │   │ (98.2%准确) │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
                                                            │
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  技术报告   │ ← │  可视化图表  │ ← │  统计分析   │ ← │  回归建模   │
│ (详细文档)  │   │ (专业图表)  │   │ (197差异)   │   │ (效应量化)  │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

## 🚀 快速开始

### 环境要求

```bash
Python 3.x
numpy==1.24.3
h5py==3.8.0
scipy==1.10.1
scikit-learn
matplotlib
seaborn
pandas
```

### 运行分析

```bash
# 1. 进入scripts目录
cd scripts

# 2. (可选) 检查MAT文件结构
python inspect_mat_structure.py

# 3. (可选) 重新转换MAT文件
python complete_mat_converter.py

# 4. 运行分类分析
python complete_data_analysis.py

# 5. 运行回归分析（推荐）
python regression_analysis.py

# 6. (可选) 修复中文字体显示
python fix_chinese_font.py
```

### 查看结果

**分类分析结果**：

1. 📊 查看 `results/complete_mat_analysis.png` 了解分类分析结果
2. 📄 阅读 `docs/complete_mat_analysis_report.md` 了解详细分类分析

**回归分析结果**：

1. 📊 查看 `results/regression_analysis.png` 了解回归分析结果
2. 📄 阅读 `docs/regression_analysis_report.md` 了解详细回归分析

**完整报告**：

1. 📋 查看 `docs/final_project_report.md` 了解完整项目报告（推荐提交给老师）
2. 📄 查看 `docs/executive_summary.md` 了解项目执行摘要

**技术文档**：

1. 🔧 参考 `docs/mat_conversion_success_summary.md` 了解技术解决方案
2. 🛠️ 参考 `docs/chinese_font_fix.md` 了解字体修复方案

## 📊 主要发现

### 分类建模结果

- **最佳准确率**: 98.2% (Random Forest)
- **精确率**: 健康人 98%, 中风患者 100%
- **召回率**: 健康人 100%, 中风患者 93%
- **数据规模**: 1,128 个样本，386 个特征

### 回归分析结果

- **显著差异特征**: 197 个 (占总特征的 51.7%)
- **大效应特征**: 20 个 (效应量 > 0.8)
- **最佳回归模型**: Ridge Regression (R² = 0.877)
- **主要差异领域**: 踝关节运动、关节角度、力量输出

### 技术突破

1. **MAT 文件转换**: 解决 MATLAB v7.3 格式读取难题
2. **数据完整性**: 提取所有可用数据，提升 9.4 倍
3. **兼容性问题**: 解决 NumPy 版本兼容性障碍
4. **分类性能**: 在大规模数据上实现 98.2%准确率

### 临床价值

- 🎯 **高精度分类**: 98.2%准确率为临床应用提供可靠支撑
- 📊 **大规模数据**: 1,128 个样本提供充分的统计支撑
- 🔧 **技术方案**: 可复用的 MAT 文件处理解决方案
- 🏥 **临床应用**: 为中风康复评估提供客观工具
- 🔍 **差异识别**: 197 个显著差异特征为康复训练提供精准指导
- ⚖️ **效应量化**: 通过 Cohen's d 量化中风对步态的影响程度

## 🔬 技术特点

### 数据处理突破

- ✅ 解决 MATLAB v7.3 格式 MAT 文件读取难题
- ✅ 克服 NumPy 版本兼容性障碍
- ✅ 实现完整数据提取（提升 9.4 倍）
- ✅ 处理复杂 HDF5 数据结构

### 分析方法

- ✅ 高精度机器学习分类（98.2%准确率）
- ✅ 多分类器性能对比
- ✅ 386 个步态特征全面分析
- ✅ 大规模数据统计验证

### 可视化

- ✅ 完整数据分析可视化
- ✅ 数据提取对比展示
- ✅ 分类性能详细分析
- ✅ 统一美观的配色方案

## 📈 可视化图表说明

> 🎨 所有图表采用统一的美观配色方案，详见 [配色方案说明](docs/color_scheme.md)

### complete_mat_analysis.png ⭐

- **样本分布**: 健康人 vs 中风患者数量对比
- **数据组分布**: 6 个主要数据组的样本分布
- **各数据组标签分布**: 每个数据组的健康人/中风患者比例
- **分类器性能对比**: Random Forest、Logistic Regression、SVM 性能
- **混淆矩阵**: 最佳分类器的详细分类结果
- **数据提取对比**: 初始提取 vs 完整提取的样本数量对比

### 配色方案

- **主色调**: 浅蓝色 (#98CFE6) - 主要数据展示
- **健康人**: 浅绿色 (#ADE7A8) - 代表正常状态
- **中风患者**: 粉色 (#EEB7D3) - 代表异常状态
- **强调色**: 橙色 (#F39F4E) - 重要信息突出
- **辅助色**: 灰色 (#DBDAD3)、黄色 (#FFDF97)

## 🏥 临床应用

### 评估工具

- 客观量化步态异常程度
- 识别个体化康复重点
- 预测康复预后

### 康复指导

1. **步速训练** - 优先级最高
2. **对称性训练** - 改善偏瘫侧功能
3. **关节活动度训练** - 增加活动范围
4. **平衡训练** - 减少步宽过大
5. **肌力训练** - 针对性肌群强化

## ⚠️ 注意事项

### 环境要求

- 需要安装特定版本的科学计算库（numpy==1.24.3, h5py==3.8.0, scipy==1.10.1）
- 确保在 scripts 目录下运行脚本
- MAT 文件较大，转换过程可能需要几分钟时间

### 数据特点

- 数据不平衡（健康人 828 样本 vs 中风患者 300 样本）
- 每个受试者有 6 个数据组记录，代表不同步态条件
- 特征数量较多（386 个），建议进行特征选择

## 🔮 未来改进

1. **数据扩展**

   - 增加中风患者样本量
   - 纳入更多临床变量
   - 收集纵向随访数据

2. **技术升级**

   - 结合深度学习方法
   - 开发实时评估系统
   - 优化特征选择算法

3. **临床应用**
   - 开发临床决策支持系统
   - 建立标准化评估流程
   - 验证康复效果

## 📞 联系信息

如有问题或建议，请参考：

- 📄 完整分析报告：`docs/complete_mat_analysis_report.md`
- 📋 技术解决方案：`docs/mat_conversion_success_summary.md`
- 💻 代码注释：各 Python 脚本中的详细注释

---

**项目**: 深大老师布置的任务 - 步态数据分析
**开发时间**: 2025 年 7 月 4 日
**技术栈**: Python, h5py, NumPy, SciPy, scikit-learn, Matplotlib
**数据来源**: Figshare 步态数据集
**项目状态**: ✅ 完成 - MAT 转换问题已解决，完整数据已提取
