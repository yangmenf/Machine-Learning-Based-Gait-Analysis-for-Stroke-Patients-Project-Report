#!/usr/bin/env python3
"""
集成分析流水线
统一的分析流程：特征选择 → 分类建模 → 回归分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from evolutionary_feature_selection import GeneticFeatureSelector
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """设置中文字体"""
    import matplotlib
    font_options = [
        ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
        ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'],
        ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        ['DejaVu Sans', 'Arial', 'sans-serif']
    ]
    
    for fonts in font_options:
        try:
            matplotlib.rcParams['font.sans-serif'] = fonts
            matplotlib.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.sans-serif'] = fonts
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            continue
    return False

def load_data():
    """步骤1: 加载原始数据"""
    print("="*70)
    print("步骤1: 数据加载")
    print("="*70)
    
    try:
        df = pd.read_csv("../data/complete_gait_features.csv")
        print(f"✓ 成功加载数据: {df.shape}")
        
        # 分离特征和标签
        feature_cols = [col for col in df.columns if col not in ['label', 'group', 'subject_id', 'data_group', 'subject_index']]
        
        X = df[feature_cols].copy()
        y = df['label'].copy()
        
        # 数据预处理
        X = X.replace('', np.nan)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        print(f"✓ 原始特征数量: {X.shape[1]}")
        print(f"✓ 样本数量: {X.shape[0]}")
        print(f"✓ 标签分布: 健康人 {sum(y==0)}, 中风患者 {sum(y==1)}")
        
        return X, y, feature_cols, df
        
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return None, None, None, None

def perform_feature_selection(X, y, feature_names):
    """步骤2: 遗传算法特征选择"""
    print("\n" + "="*70)
    print("步骤2: 遗传算法特征选择")
    print("="*70)
    
    # 创建特征选择器
    rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    
    selector = GeneticFeatureSelector(
        estimator=rf_estimator,
        n_features_to_select=120,  # 选择120个特征
        population_size=30,
        generations=15,
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        random_state=42
    )
    
    # 执行特征选择
    X_selected = selector.fit_transform(X.values, y.values)
    selected_indices = selector.selected_features_
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"✓ 特征选择完成")
    print(f"✓ 原始特征: {X.shape[1]} → 选择特征: {X_selected.shape[1]}")
    print(f"✓ 特征减少比例: {(1 - X_selected.shape[1]/X.shape[1])*100:.1f}%")
    print(f"✓ 最佳适应度: {selector.best_score_:.4f}")
    
    return X_selected, selected_indices, selected_feature_names, selector

def perform_classification(X_original, X_selected, y):
    """步骤3: 分类建模对比"""
    print("\n" + "="*70)
    print("步骤3: 分类建模（原始特征 vs 选择特征）")
    print("="*70)
    
    # 数据分割
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_sel, X_test_sel, _, _ = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化
    scaler_orig = StandardScaler()
    scaler_sel = StandardScaler()
    
    X_train_orig_scaled = scaler_orig.fit_transform(X_train_orig)
    X_test_orig_scaled = scaler_orig.transform(X_test_orig)
    
    X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
    X_test_sel_scaled = scaler_sel.transform(X_test_sel)
    
    # 定义分类器
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    classification_results = {}
    
    print("\n分类器性能对比:")
    print("-" * 60)
    print(f"{'分类器':<15} {'原始特征':<12} {'选择特征':<12} {'性能变化':<10}")
    print("-" * 60)
    
    for name, clf in classifiers.items():
        # 原始特征
        clf_orig = clf.__class__(**clf.get_params())
        clf_orig.fit(X_train_orig_scaled, y_train)
        y_pred_orig = clf_orig.predict(X_test_orig_scaled)
        acc_orig = accuracy_score(y_test, y_pred_orig)
        
        # 选择特征
        clf_sel = clf.__class__(**clf.get_params())
        clf_sel.fit(X_train_sel_scaled, y_train)
        y_pred_sel = clf_sel.predict(X_test_sel_scaled)
        acc_sel = accuracy_score(y_test, y_pred_sel)
        
        improvement = acc_sel - acc_orig
        
        classification_results[name] = {
            'original_accuracy': acc_orig,
            'selected_accuracy': acc_sel,
            'improvement': improvement
        }
        
        print(f"{name:<15} {acc_orig:<12.4f} {acc_sel:<12.4f} {improvement*100:<+10.2f}%")
    
    return classification_results, y_test

def perform_regression_analysis(X_original, X_selected, y, selected_feature_names):
    """步骤4: 回归分析（基于选择的特征）"""
    print("\n" + "="*70)
    print("步骤4: 回归分析（使用选择的特征）")
    print("="*70)
    
    # 使用选择的特征进行统计分析
    print("进行统计显著性分析...")
    
    # 分组数据
    healthy_indices = y == 0
    stroke_indices = y == 1
    
    statistical_results = []
    
    for i, feature_name in enumerate(selected_feature_names):
        try:
            healthy_values = X_selected[healthy_indices, i]
            stroke_values = X_selected[stroke_indices, i]
            
            # 独立样本t检验
            t_stat, p_value = stats.ttest_ind(healthy_values, stroke_values)
            
            # 计算效应量 (Cohen's d)
            pooled_std = np.sqrt(((len(healthy_values) - 1) * np.var(healthy_values, ddof=1) + 
                                (len(stroke_values) - 1) * np.var(stroke_values, ddof=1)) / 
                               (len(healthy_values) + len(stroke_values) - 2))
            
            if pooled_std > 0:
                cohens_d = (np.mean(healthy_values) - np.mean(stroke_values)) / pooled_std
            else:
                cohens_d = 0
            
            statistical_results.append({
                'feature': feature_name,
                'healthy_mean': np.mean(healthy_values),
                'stroke_mean': np.mean(stroke_values),
                'mean_difference': np.mean(healthy_values) - np.mean(stroke_values),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_size': abs(cohens_d)
            })
        except Exception as e:
            continue
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(statistical_results)
    results_df = results_df.sort_values('effect_size', ascending=False)
    
    # 显著性筛选
    significant_features = results_df[results_df['p_value'] < 0.05]
    
    print(f"✓ 分析了 {len(results_df)} 个选择的特征")
    print(f"✓ 发现 {len(significant_features)} 个显著差异特征")
    print(f"✓ 显著性比例: {len(significant_features)/len(results_df)*100:.1f}%")
    
    # 显示前10个最显著的差异
    print(f"\n前10个最显著的差异特征:")
    print("-" * 80)
    for i, row in significant_features.head(10).iterrows():
        effect_level = "大" if abs(row['cohens_d']) > 0.8 else "中" if abs(row['cohens_d']) > 0.5 else "小"
        print(f"{row['feature'][:30]:<30} | 效应量: {row['cohens_d']:.3f} ({effect_level}) | p值: {row['p_value']:.6f}")
    
    return results_df, significant_features

def create_integrated_visualization(selector, classification_results, regression_results, X_original, X_selected):
    """步骤5: 创建集成可视化"""
    print("\n" + "="*70)
    print("步骤5: 生成集成分析可视化")
    print("="*70)
    
    setup_chinese_font()
    
    # 用户偏好的颜色方案
    COLORS = ['#98CFE6', '#ADE7A8', '#F39F4E', '#EEB7D3', '#DBDAD3', '#FFDF97']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('集成分析流水线结果', fontsize=16, fontweight='bold')
    
    # 1. 特征选择进化过程
    ax1 = axes[0, 0]
    generations = range(1, len(selector.best_fitness_history) + 1)
    ax1.plot(generations, selector.best_fitness_history, 'o-', color=COLORS[0], linewidth=2, markersize=4)
    ax1.set_xlabel('进化代数')
    ax1.set_ylabel('最佳适应度')
    ax1.set_title('特征选择进化过程')
    ax1.grid(True, alpha=0.3)
    
    # 2. 特征数量对比
    ax2 = axes[0, 1]
    feature_counts = [X_original.shape[1], X_selected.shape[1]]
    labels = ['原始特征', '选择特征']
    colors = [COLORS[1], COLORS[2]]
    
    bars = ax2.bar(labels, feature_counts, color=colors, alpha=0.8)
    ax2.set_ylabel('特征数量')
    ax2.set_title('特征数量对比')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. 分类性能对比
    ax3 = axes[0, 2]
    classifiers = list(classification_results.keys())
    orig_accs = [classification_results[clf]['original_accuracy'] for clf in classifiers]
    sel_accs = [classification_results[clf]['selected_accuracy'] for clf in classifiers]
    
    x = np.arange(len(classifiers))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, orig_accs, width, label='原始特征', color=COLORS[3], alpha=0.8)
    bars2 = ax3.bar(x + width/2, sel_accs, width, label='选择特征', color=COLORS[4], alpha=0.8)
    
    ax3.set_xlabel('分类器')
    ax3.set_ylabel('准确率')
    ax3.set_title('分类性能对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels([clf.replace(' ', '\n') for clf in classifiers])
    ax3.legend()
    ax3.set_ylim(0.9, 1.0)
    
    # 4. 回归分析效应量分布
    ax4 = axes[1, 0]
    if len(regression_results) > 0:
        effect_sizes = regression_results['effect_size'].values
        ax4.hist(effect_sizes, bins=15, color=COLORS[5], alpha=0.7, edgecolor='black')
        ax4.axvline(x=0.2, color='green', linestyle='--', label='小效应 (0.2)')
        ax4.axvline(x=0.5, color='orange', linestyle='--', label='中效应 (0.5)')
        ax4.axvline(x=0.8, color='red', linestyle='--', label='大效应 (0.8)')
        ax4.set_xlabel('效应量 (Cohen\'s d)')
        ax4.set_ylabel('特征数量')
        ax4.set_title('选择特征的效应量分布')
        ax4.legend(fontsize=8)
    
    # 5. 显著性特征比例
    ax5 = axes[1, 1]
    if len(regression_results) > 0:
        significant_count = len(regression_results[regression_results['p_value'] < 0.05])
        non_significant_count = len(regression_results) - significant_count
        
        sizes = [significant_count, non_significant_count]
        labels = [f'显著\n({significant_count})', f'非显著\n({non_significant_count})']
        colors = [COLORS[0], COLORS[1]]
        
        wedges, texts, autotexts = ax5.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax5.set_title('选择特征显著性分布')
    
    # 6. 流水线总结
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 添加文本总结
    summary_text = f"""
集成分析流水线总结

步骤1: 数据加载
• 样本数量: {X_original.shape[0]}
• 原始特征: {X_original.shape[1]}

步骤2: 特征选择
• 选择特征: {X_selected.shape[1]}
• 降维比例: {(1-X_selected.shape[1]/X_original.shape[1])*100:.1f}%

步骤3: 分类建模
• 最佳准确率: {max([r['selected_accuracy'] for r in classification_results.values()]):.4f}

步骤4: 回归分析
• 显著特征: {len(regression_results[regression_results['p_value'] < 0.05]) if len(regression_results) > 0 else 0}
• 显著比例: {len(regression_results[regression_results['p_value'] < 0.05])/len(regression_results)*100:.1f}%
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor=COLORS[2], alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图表
    output_file = "../results/integrated_analysis_pipeline.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 集成分析可视化已保存: {output_file}")
    
    plt.close()

def generate_pipeline_report(selector, classification_results, regression_results, selected_feature_names, X_original, X_selected):
    """生成集成分析报告"""
    print("生成集成分析报告...")
    
    report_file = "../docs/integrated_analysis_pipeline_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 集成分析流水线报告\n\n")
        f.write("## 分析流程\n\n")
        f.write("本报告展示了完整的集成分析流水线：**特征选择 → 分类建模 → 回归分析**\n\n")
        
        f.write("### 步骤1: 数据加载\n")
        f.write(f"- **样本数量**: {X_original.shape[0]}\n")
        f.write(f"- **原始特征数**: {X_original.shape[1]}\n\n")
        
        f.write("### 步骤2: 遗传算法特征选择\n")
        f.write(f"- **选择特征数**: {X_selected.shape[1]}\n")
        f.write(f"- **降维比例**: {(1-X_selected.shape[1]/X_original.shape[1])*100:.1f}%\n")
        f.write(f"- **最佳适应度**: {selector.best_score_:.4f}\n\n")
        
        f.write("### 步骤3: 分类建模对比\n")
        f.write("| 分类器 | 原始特征准确率 | 选择特征准确率 | 性能变化 |\n")
        f.write("|--------|----------------|----------------|----------|\n")
        
        for clf_name, result in classification_results.items():
            f.write(f"| {clf_name} | {result['original_accuracy']:.4f} | {result['selected_accuracy']:.4f} | {result['improvement']*100:+.2f}% |\n")
        
        f.write("\n### 步骤4: 回归分析结果\n")
        if len(regression_results) > 0:
            significant_count = len(regression_results[regression_results['p_value'] < 0.05])
            f.write(f"- **分析特征数**: {len(regression_results)}\n")
            f.write(f"- **显著差异特征**: {significant_count}\n")
            f.write(f"- **显著性比例**: {significant_count/len(regression_results)*100:.1f}%\n\n")
            
            f.write("#### 前10个最显著差异特征\n")
            significant_features = regression_results[regression_results['p_value'] < 0.05].head(10)
            for i, row in significant_features.iterrows():
                f.write(f"{i+1}. **{row['feature']}**: 效应量 {row['cohens_d']:.3f}, p值 {row['p_value']:.6f}\n")
        
        f.write("\n## 主要优势\n\n")
        f.write("1. **统一流程**: 特征选择→分类→回归的完整分析链\n")
        f.write("2. **智能优化**: 遗传算法自动选择最优特征子集\n")
        f.write("3. **性能保持**: 大幅降维的同时保持分类性能\n")
        f.write("4. **深度分析**: 基于优化特征进行统计分析\n\n")
        
        f.write("---\n\n")
        f.write("**报告生成时间**: 2025年7月4日\n")
        f.write("**分析方法**: 集成分析流水线\n")
    
    print(f"✓ 集成分析报告已保存: {report_file}")

def main():
    """主函数 - 执行完整的集成分析流水线"""
    print("🚀 启动集成分析流水线")
    print("流程: 数据加载 → 特征选择 → 分类建模 → 回归分析 → 结果可视化")
    
    # 步骤1: 加载数据
    X, y, feature_names, df = load_data()
    if X is None:
        return
    
    # 步骤2: 特征选择
    X_selected, selected_indices, selected_feature_names, selector = perform_feature_selection(X, y, feature_names)
    
    # 步骤3: 分类建模
    classification_results, y_test = perform_classification(X.values, X_selected, y.values)
    
    # 步骤4: 回归分析
    regression_results, significant_features = perform_regression_analysis(X.values, X_selected, y.values, selected_feature_names)
    
    # 步骤5: 可视化
    create_integrated_visualization(selector, classification_results, regression_results, X.values, X_selected)
    
    # 生成报告
    generate_pipeline_report(selector, classification_results, regression_results, selected_feature_names, X.values, X_selected)
    
    # 保存选择的特征
    selected_features_df = pd.DataFrame({
        'feature_index': selected_indices,
        'feature_name': selected_feature_names
    })
    selected_features_df.to_csv("../results/pipeline_selected_features.csv", index=False)
    
    print("\n" + "="*70)
    print("🎉 集成分析流水线完成!")
    print("="*70)
    
    best_classifier = max(classification_results.keys(), key=lambda x: classification_results[x]['selected_accuracy'])
    best_accuracy = classification_results[best_classifier]['selected_accuracy']
    significant_count = len(regression_results[regression_results['p_value'] < 0.05]) if len(regression_results) > 0 else 0
    
    print(f"\n📊 流水线总结:")
    print(f"  ✓ 特征优化: {X.shape[1]} → {X_selected.shape[1]} ({(1-X_selected.shape[1]/X.shape[1])*100:.1f}% 降维)")
    print(f"  ✓ 最佳分类器: {best_classifier}")
    print(f"  ✓ 最高准确率: {best_accuracy:.4f}")
    print(f"  ✓ 显著差异特征: {significant_count}")
    print(f"  ✓ 遗传算法适应度: {selector.best_score_:.4f}")
    
    print(f"\n📁 生成的文件:")
    print(f"  📊 ../results/integrated_analysis_pipeline.png - 集成分析可视化")
    print(f"  📋 ../docs/integrated_analysis_pipeline_report.md - 集成分析报告")
    print(f"  📄 ../results/pipeline_selected_features.csv - 流水线选择的特征")

if __name__ == "__main__":
    main()
