#!/usr/bin/env python3
"""
回归分析脚本
专门分析中风患者和健康人在步态特征上的差异
研究中风患者在哪些问题上和健康人有区别
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
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

def load_and_prepare_data():
    """加载和准备数据"""
    print("="*60)
    print("步态特征回归分析")
    print("="*60)
    
    try:
        df = pd.read_csv("../data/complete_gait_features.csv")
        print(f"✓ 成功加载数据: {df.shape}")
        
        # 分离特征和标签
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
        
        print(f"✓ 特征矩阵形状: {X.shape}")
        print(f"✓ 标签分布: 健康人 {len(y[y==0])}, 中风患者 {len(y[y==1])}")
        
        return df, X, y, feature_cols
        
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return None, None, None, None

def perform_statistical_analysis(df, X, y, feature_cols):
    """执行统计分析，识别显著差异特征"""
    print("\n" + "="*40)
    print("统计显著性分析")
    print("="*40)
    
    # 分组数据
    healthy_data = X[y == 0]
    stroke_data = X[y == 1]
    
    # 统计检验结果
    statistical_results = []
    
    print("进行独立样本t检验...")
    
    for feature in feature_cols:
        try:
            healthy_values = healthy_data[feature].dropna()
            stroke_values = stroke_data[feature].dropna()
            
            if len(healthy_values) > 10 and len(stroke_values) > 10:
                # 独立样本t检验
                t_stat, p_value = stats.ttest_ind(healthy_values, stroke_values)
                
                # 计算效应量 (Cohen's d)
                pooled_std = np.sqrt(((len(healthy_values) - 1) * healthy_values.var() + 
                                    (len(stroke_values) - 1) * stroke_values.var()) / 
                                   (len(healthy_values) + len(stroke_values) - 2))
                
                if pooled_std > 0:
                    cohens_d = (healthy_values.mean() - stroke_values.mean()) / pooled_std
                else:
                    cohens_d = 0
                
                # 计算基本统计量
                healthy_mean = healthy_values.mean()
                stroke_mean = stroke_values.mean()
                mean_diff = healthy_mean - stroke_mean
                
                statistical_results.append({
                    'feature': feature,
                    'healthy_mean': healthy_mean,
                    'stroke_mean': stroke_mean,
                    'mean_difference': mean_diff,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'effect_size': abs(cohens_d)
                })
        except Exception as e:
            continue
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(statistical_results)
    
    if len(results_df) > 0:
        # 按效应量排序
        results_df = results_df.sort_values('effect_size', ascending=False)
        
        # 显著性筛选 (p < 0.05)
        significant_features = results_df[results_df['p_value'] < 0.05]
        
        print(f"✓ 完成 {len(results_df)} 个特征的统计检验")
        print(f"✓ 发现 {len(significant_features)} 个显著差异特征 (p < 0.05)")
        
        # 显示前10个最显著的差异
        print(f"\n前10个最显著的差异特征:")
        print("-" * 80)
        for i, row in significant_features.head(10).iterrows():
            effect_level = "大" if abs(row['cohens_d']) > 0.8 else "中" if abs(row['cohens_d']) > 0.5 else "小"
            print(f"{row['feature'][:30]:<30} | 效应量: {row['cohens_d']:.3f} ({effect_level}) | p值: {row['p_value']:.6f}")
        
        return results_df, significant_features
    else:
        print("✗ 未能完成统计分析")
        return None, None

def perform_regression_modeling(X, y):
    """执行回归建模分析"""
    print("\n" + "="*40)
    print("回归建模分析")
    print("="*40)
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义回归模型
    regressors = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    regression_results = {}
    
    print("训练回归模型...")
    
    for name, regressor in regressors.items():
        try:
            if name == 'Random Forest':
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
            else:
                regressor.fit(X_train_scaled, y_train)
                y_pred = regressor.predict(X_test_scaled)
            
            # 计算评估指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            regression_results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'model': regressor,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            print(f"{name}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
            
        except Exception as e:
            print(f"✗ {name} 训练失败: {e}")
    
    return regression_results

def create_regression_visualizations(results_df, significant_features, regression_results):
    """创建回归分析可视化"""
    print("\n" + "="*40)
    print("创建回归分析可视化")
    print("="*40)
    
    # 用户偏好的颜色方案
    COLORS = ['#98CFE6', '#ADE7A8', '#F39F4E', '#EEB7D3', '#DBDAD3', '#FFDF97']
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('步态特征回归分析结果', fontsize=16, fontweight='bold')
    
    # 1. 效应量排序（前15个）
    ax1 = axes[0, 0]
    if len(significant_features) > 0:
        top_features = significant_features.head(15)
        feature_names = [f.split('_')[0] for f in top_features['feature']]  # 简化特征名
        
        bars = ax1.barh(range(len(feature_names)), top_features['effect_size'], color=COLORS[0])
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names, fontsize=8)
        ax1.set_xlabel('效应量 (Cohen\'s d)')
        ax1.set_title('特征效应量排序 (Top 15)', pad=15)
        ax1.invert_yaxis()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f}', ha='left', va='center', fontsize=8)
    
    # 2. p值分布
    ax2 = axes[0, 1]
    if len(results_df) > 0:
        ax2.hist(results_df['p_value'], bins=20, color=COLORS[1], alpha=0.7, edgecolor='black')
        ax2.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
        ax2.set_xlabel('p值')
        ax2.set_ylabel('特征数量')
        ax2.set_title('p值分布', pad=15)
        ax2.legend()
    
    # 3. 健康人vs中风患者特征对比（前6个显著特征）
    ax3 = axes[0, 2]
    if len(significant_features) > 0:
        top_6_features = significant_features.head(6)
        healthy_means = top_6_features['healthy_mean'].values
        stroke_means = top_6_features['stroke_mean'].values
        feature_names = [f.split('_')[0] for f in top_6_features['feature']]
        
        x_pos = np.arange(len(feature_names))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, healthy_means, width, 
                       label='健康人', color=COLORS[0], alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, stroke_means, width,
                       label='中风患者', color=COLORS[1], alpha=0.8)
        
        ax3.set_title('主要差异特征对比', pad=15)
        ax3.set_xlabel('特征')
        ax3.set_ylabel('平均值')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(feature_names, rotation=45, fontsize=8)
        ax3.legend()
    
    # 4. 回归模型性能对比
    ax4 = axes[1, 0]
    if regression_results:
        model_names = list(regression_results.keys())
        r2_scores = [regression_results[name]['r2'] for name in model_names]
        
        bars = ax4.bar(model_names, r2_scores, color=COLORS[2:6])
        ax4.set_title('回归模型性能对比', pad=15)
        ax4.set_ylabel('R² 分数')
        ax4.set_ylim(0, 1)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. 效应量分布
    ax5 = axes[1, 1]
    if len(results_df) > 0:
        ax5.hist(results_df['effect_size'], bins=20, color=COLORS[3], alpha=0.7, edgecolor='black')
        ax5.axvline(x=0.2, color='green', linestyle='--', label='小效应 (0.2)')
        ax5.axvline(x=0.5, color='orange', linestyle='--', label='中效应 (0.5)')
        ax5.axvline(x=0.8, color='red', linestyle='--', label='大效应 (0.8)')
        ax5.set_xlabel('效应量 (Cohen\'s d)')
        ax5.set_ylabel('特征数量')
        ax5.set_title('效应量分布', pad=15)
        ax5.legend()
    
    # 6. 显著性特征统计
    ax6 = axes[1, 2]
    if len(results_df) > 0:
        total_features = len(results_df)
        significant_count = len(significant_features)
        non_significant_count = total_features - significant_count
        
        sizes = [significant_count, non_significant_count]
        labels = [f'显著差异\n({significant_count})', f'无显著差异\n({non_significant_count})']
        colors = [COLORS[4], COLORS[5]]
        
        wedges, texts, autotexts = ax6.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax6.set_title('特征显著性分布', pad=15)
    
    # 调整布局
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    
    # 保存图表
    output_file = "../results/regression_analysis.png"
    try:
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 回归分析可视化已保存: {output_file}")
    except Exception as e:
        print(f"✗ 保存可视化失败: {e}")

    # 关闭图表以释放内存
    plt.close(fig)

    # 不显示图表，避免在某些环境下出现问题
    # plt.show()

def generate_regression_report(results_df, significant_features, regression_results):
    """生成回归分析报告"""
    print("\n" + "="*40)
    print("生成回归分析报告")
    print("="*40)
    
    report_file = "../docs/regression_analysis_report.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 步态特征回归分析报告\n\n")
        f.write("## 分析目标\n\n")
        f.write("通过回归分析方法，深入研究中风患者和健康人在步态特征上的差异，")
        f.write("识别中风患者在哪些方面存在显著问题，为康复治疗提供科学依据。\n\n")
        
        f.write("## 统计分析结果\n\n")
        if len(results_df) > 0:
            f.write(f"- **总特征数**: {len(results_df)}\n")
            f.write(f"- **显著差异特征数**: {len(significant_features)} (p < 0.05)\n")
            f.write(f"- **显著性比例**: {len(significant_features)/len(results_df)*100:.1f}%\n\n")
            
            # 效应量分布
            large_effect = len(results_df[results_df['effect_size'] > 0.8])
            medium_effect = len(results_df[(results_df['effect_size'] > 0.5) & (results_df['effect_size'] <= 0.8)])
            small_effect = len(results_df[(results_df['effect_size'] > 0.2) & (results_df['effect_size'] <= 0.5)])
            
            f.write("### 效应量分布\n\n")
            f.write(f"- **大效应** (d > 0.8): {large_effect} 个特征\n")
            f.write(f"- **中效应** (0.5 < d ≤ 0.8): {medium_effect} 个特征\n")
            f.write(f"- **小效应** (0.2 < d ≤ 0.5): {small_effect} 个特征\n\n")
        
        f.write("## 主要差异特征\n\n")
        if len(significant_features) > 0:
            f.write("### Top 10 显著差异特征\n\n")
            f.write("| 特征名称 | 健康人均值 | 中风患者均值 | 差异 | 效应量 | p值 |\n")
            f.write("|----------|------------|--------------|------|--------|-----|\n")
            
            for i, row in significant_features.head(10).iterrows():
                f.write(f"| {row['feature'][:30]} | {row['healthy_mean']:.3f} | {row['stroke_mean']:.3f} | ")
                f.write(f"{row['mean_difference']:.3f} | {row['cohens_d']:.3f} | {row['p_value']:.6f} |\n")
            f.write("\n")
        
        f.write("## 回归建模结果\n\n")
        if regression_results:
            f.write("| 模型 | R² 分数 | RMSE | MAE |\n")
            f.write("|------|---------|------|-----|\n")
            for name, result in regression_results.items():
                f.write(f"| {name} | {result['r2']:.3f} | {result['rmse']:.3f} | {result['mae']:.3f} |\n")
            f.write("\n")
        
        f.write("## 临床解释\n\n")
        f.write("### 中风患者的主要步态问题\n\n")
        if len(significant_features) > 0:
            f.write("基于统计分析结果，中风患者在以下方面与健康人存在显著差异：\n\n")
            
            # 分析前几个最显著的特征
            top_features = significant_features.head(5)
            for i, row in top_features.iterrows():
                feature_name = row['feature']
                if row['mean_difference'] > 0:
                    direction = "降低"
                    comparison = "低于"
                else:
                    direction = "增加"
                    comparison = "高于"
                
                f.write(f"{i+1}. **{feature_name}**: 中风患者{comparison}健康人 ")
                f.write(f"{abs(row['mean_difference']):.3f} 个单位，效应量为 {abs(row['cohens_d']):.3f}\n")
            f.write("\n")
        
        f.write("### 康复建议\n\n")
        f.write("基于回归分析结果，建议中风患者的康复训练重点关注：\n\n")
        f.write("1. **步态对称性训练**: 改善左右侧步态不平衡\n")
        f.write("2. **步速训练**: 提高行走速度和效率\n")
        f.write("3. **关节活动度训练**: 增加关节灵活性\n")
        f.write("4. **平衡训练**: 改善步态稳定性\n")
        f.write("5. **肌力训练**: 针对性加强相关肌群\n\n")
        
        f.write("## 方法学说明\n\n")
        f.write("- **统计检验**: 独立样本t检验\n")
        f.write("- **效应量**: Cohen's d\n")
        f.write("- **显著性水平**: p < 0.05\n")
        f.write("- **回归模型**: 线性回归、岭回归、Lasso回归、随机森林\n")
        f.write("- **评估指标**: R²、RMSE、MAE\n\n")
        
        f.write("## 结论\n\n")
        f.write("通过回归分析，我们成功识别了中风患者与健康人在步态特征上的显著差异，")
        f.write("为个性化康复方案的制定提供了科学依据。这些发现有助于临床医生更好地")
        f.write("理解中风对步态的影响，并制定针对性的康复训练计划。\n")
    
    print(f"✓ 回归分析报告已生成: {report_file}")

def main():
    """主函数"""
    # 设置中文字体
    setup_chinese_font()
    
    # 加载和准备数据
    df, X, y, feature_cols = load_and_prepare_data()
    if df is None:
        return
    
    # 执行统计分析
    results_df, significant_features = perform_statistical_analysis(df, X, y, feature_cols)
    if results_df is None:
        return
    
    # 执行回归建模
    regression_results = perform_regression_modeling(X, y)
    
    # 创建可视化
    create_regression_visualizations(results_df, significant_features, regression_results)
    
    # 生成报告
    generate_regression_report(results_df, significant_features, regression_results)
    
    print("\n" + "="*60)
    print("🎉 回归分析完成!")
    print("="*60)
    print("\n生成的文件:")
    print("  📊 ../results/regression_analysis.png - 回归分析可视化")
    print("  📝 ../docs/regression_analysis_report.md - 详细分析报告")
    print("\n主要发现:")
    if len(significant_features) > 0:
        print(f"  ✓ 发现 {len(significant_features)} 个显著差异特征")
        print(f"  ✓ 最大效应量: {significant_features.iloc[0]['effect_size']:.3f}")
        print(f"  ✓ 显著性比例: {len(significant_features)/len(results_df)*100:.1f}%")
    if regression_results:
        best_model = max(regression_results.keys(), key=lambda x: regression_results[x]['r2'])
        print(f"  ✓ 最佳回归模型: {best_model} (R² = {regression_results[best_model]['r2']:.3f})")

if __name__ == "__main__":
    main()
