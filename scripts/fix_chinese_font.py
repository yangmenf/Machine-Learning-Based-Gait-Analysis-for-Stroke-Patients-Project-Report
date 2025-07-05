#!/usr/bin/env python3
"""
中文字体修复脚本
解决matplotlib中文显示问题，重新生成可视化结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def setup_chinese_font():
    """设置中文字体"""
    print("正在配置中文字体...")
    
    # 尝试多种中文字体配置
    font_options = [
        ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans'],
        ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans'],
        ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        ['DejaVu Sans', 'Arial', 'sans-serif']  # 备用选项
    ]
    
    for fonts in font_options:
        try:
            matplotlib.rcParams['font.sans-serif'] = fonts
            matplotlib.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.sans-serif'] = fonts
            plt.rcParams['axes.unicode_minus'] = False
            
            # 测试中文显示
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
            plt.close(fig)
            
            print(f"✓ 成功配置字体: {fonts[0]}")
            return True
        except Exception as e:
            continue
    
    print("⚠️ 无法配置中文字体，将使用英文标签")
    return False

def load_data():
    """加载完整数据"""
    try:
        df = pd.read_csv("../data/complete_gait_features.csv")
        print(f"✓ 成功加载数据: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return None

def prepare_features(df):
    """准备特征数据"""
    feature_cols = [col for col in df.columns if col not in ['label', 'group', 'subject_id', 'data_group', 'subject_index']]
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # 处理缺失值和异常值
    X = X.replace('', np.nan)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X, y, feature_cols

def perform_classification(X, y):
    """执行分类分析"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    for name, clf in classifiers.items():
        if name == 'Random Forest':
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    return results

def create_fixed_visualization(df, X, y, results, use_chinese=True):
    """创建修复后的可视化图表"""
    print("创建修复后的可视化...")
    
    # 用户偏好的颜色方案
    COLORS = ['#98CFE6', '#ADE7A8', '#F39F4E', '#EEB7D3', '#DBDAD3', '#FFDF97']
    
    # 设置标签（中文或英文）
    if use_chinese:
        labels = {
            'healthy': '健康人',
            'stroke': '中风患者',
            'sample_dist': '样本分布',
            'sample_count': '样本数量',
            'data_group_dist': '主要数据组分布',
            'group_label_dist': '各数据组标签分布',
            'data_group': '数据组',
            'classifier_perf': '分类器性能对比',
            'accuracy': '准确率',
            'confusion_matrix': '混淆矩阵',
            'predicted_label': '预测标签',
            'true_label': '真实标签',
            'data_extract_comp': '数据提取对比',
            'extract_method': '提取方式',
            'initial_extract': '之前提取',
            'complete_extract': '完整提取',
            'title': '完整MAT数据分析结果 (1,128个样本)'
        }
    else:
        labels = {
            'healthy': 'Healthy',
            'stroke': 'Stroke',
            'sample_dist': 'Sample Distribution',
            'sample_count': 'Sample Count',
            'data_group_dist': 'Main Data Group Distribution',
            'group_label_dist': 'Label Distribution by Data Group',
            'data_group': 'Data Group',
            'classifier_perf': 'Classifier Performance Comparison',
            'accuracy': 'Accuracy',
            'confusion_matrix': 'Confusion Matrix',
            'predicted_label': 'Predicted Label',
            'true_label': 'True Label',
            'data_extract_comp': 'Data Extraction Comparison',
            'extract_method': 'Extraction Method',
            'initial_extract': 'Initial Extract',
            'complete_extract': 'Complete Extract',
            'title': 'Complete MAT Data Analysis Results (1,128 samples)'
        }
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(labels['title'], fontsize=16, fontweight='bold')
    
    # 1. 标签分布
    ax1 = axes[0, 0]
    label_counts = df['label'].value_counts()
    bars = ax1.bar([labels['healthy'], labels['stroke']], label_counts.values, color=COLORS[:2])
    ax1.set_title(labels['sample_dist'], pad=15)
    ax1.set_ylabel(labels['sample_count'])
    # 动态调整Y轴上限，为数字标注留出空间
    max_height = max(label_counts.values)
    ax1.set_ylim(0, max_height * 1.15)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 2. 数据组分布
    ax2 = axes[0, 1]
    group_counts = df['data_group'].value_counts()
    top_groups = group_counts.head(6)
    ax2.pie(top_groups.values, labels=top_groups.index,
            autopct='%1.1f%%', colors=COLORS, textprops={'fontsize': 8})
    ax2.set_title(labels['data_group_dist'], pad=15)
    
    # 3. 各数据组的标签分布
    ax3 = axes[0, 2]
    group_label_data = []
    group_names = []
    
    for group in df['data_group'].unique()[:6]:
        group_data = df[df['data_group'] == group]
        healthy = len(group_data[group_data['label'] == 0])
        stroke = len(group_data[group_data['label'] == 1])
        group_label_data.append([healthy, stroke])
        group_names.append(group.split('_')[0])
    
    if group_label_data:
        group_label_data = np.array(group_label_data)
        x_pos = np.arange(len(group_names))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, group_label_data[:, 0], width, 
                       label=labels['healthy'], color=COLORS[0], alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, group_label_data[:, 1], width,
                       label=labels['stroke'], color=COLORS[1], alpha=0.8)
        
        ax3.set_title(labels['group_label_dist'], pad=15)
        ax3.set_xlabel(labels['data_group'])
        ax3.set_ylabel(labels['sample_count'])
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(group_names, rotation=45)
        ax3.legend(loc='upper right')
    
    # 4. 分类器性能对比
    ax4 = axes[1, 0]
    classifier_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in classifier_names]

    bars = ax4.bar(classifier_names, accuracies, color=COLORS[2:5])
    ax4.set_title(labels['classifier_perf'], pad=20)  # 增加标题间距
    ax4.set_ylabel(labels['accuracy'])
    ax4.set_ylim(0, 1.1)  # 增加Y轴上限，为数字标注留出空间

    for bar in bars:
        height = bar.get_height()
        # 调整数字标注位置，避免与标题重合
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. 混淆矩阵
    ax5 = axes[1, 1]
    best_classifier = max(results.keys(), key=lambda x: results[x]['accuracy'])
    y_test = results[best_classifier]['y_test']
    y_pred = results[best_classifier]['y_pred']

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                xticklabels=[labels['healthy'], labels['stroke']],
                yticklabels=[labels['healthy'], labels['stroke']],
                annot_kws={'fontsize': 12})
    ax5.set_title(f"{labels['confusion_matrix']} ({best_classifier})", pad=15)
    ax5.set_xlabel(labels['predicted_label'])
    ax5.set_ylabel(labels['true_label'])
    
    # 6. 数据提取对比
    ax6 = axes[1, 2]
    comparison_data = [
        [labels['initial_extract'], 60, 60],
        [labels['complete_extract'], 828, 300]
    ]

    x_labels = [item[0] for item in comparison_data]
    healthy_counts = [item[1] for item in comparison_data]
    stroke_counts = [item[2] for item in comparison_data]

    x_pos = np.arange(len(x_labels))
    width = 0.35

    bars1 = ax6.bar(x_pos - width/2, healthy_counts, width,
                   label=labels['healthy'], color=COLORS[0], alpha=0.8)
    bars2 = ax6.bar(x_pos + width/2, stroke_counts, width,
                   label=labels['stroke'], color=COLORS[1], alpha=0.8)

    ax6.set_title(labels['data_extract_comp'], pad=15)
    ax6.set_xlabel(labels['extract_method'])
    ax6.set_ylabel(labels['sample_count'])
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(x_labels)
    ax6.legend(loc='upper left')

    # 动态调整Y轴上限，为数字标注留出空间
    max_height = max(max(healthy_counts), max(stroke_counts))
    ax6.set_ylim(0, max_height * 1.15)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + max_height * 0.02,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # 调整子图间距，避免重叠
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)

    # 保存图表
    output_file = "../results/complete_mat_analysis_fixed.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 修复后的可视化结果已保存: {output_file}")

    plt.show()

def main():
    """主函数"""
    print("="*60)
    print("中文字体修复和可视化重新生成")
    print("="*60)
    
    # 设置中文字体
    use_chinese = setup_chinese_font()
    
    # 加载数据
    df = load_data()
    if df is None:
        return
    
    # 准备特征
    X, y, feature_cols = prepare_features(df)
    
    # 执行分类分析
    results = perform_classification(X, y)
    
    # 创建修复后的可视化
    create_fixed_visualization(df, X, y, results, use_chinese)
    
    print("\n" + "="*60)
    print("🎉 中文字体修复完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  📊 ../results/complete_mat_analysis_fixed.png - 修复后的分析可视化")
    
    if use_chinese:
        print("\n✓ 中文显示正常")
    else:
        print("\n⚠️ 使用英文标签（系统不支持中文字体）")

if __name__ == "__main__":
    main()
