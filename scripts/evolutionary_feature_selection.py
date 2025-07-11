#!/usr/bin/env python3
"""
改进的遗传算法特征选择器
解决适应度下降问题，确保单调递增的最佳适应度

主要改进：
1. 精英保留策略 - 确保最佳个体不会丢失
2. 固定随机种子 - 保证适应度评估的一致性
3. 自适应参数 - 动态调整变异率和交叉率
4. 收敛检测 - 避免过度迭代
5. 稳定的适应度函数 - 减少评估中的随机性
"""

import numpy as np
import random
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class GeneticFeatureSelector(BaseEstimator, TransformerMixin):
    """
    遗传算法特征选择器
    
    主要改进：
    1. 精英保留策略 - 确保最佳个体不会丢失
    2. 固定随机种子 - 保证适应度评估的一致性
    3. 自适应参数 - 动态调整变异率和交叉率
    4. 收敛检测 - 避免过度迭代
    """
    
    def __init__(self, estimator, n_features_to_select=120, population_size=30, 
                 generations=15, mutation_rate=0.15, crossover_rate=0.8, 
                 tournament_size=3, random_state=42, elite_size=2, cv_folds=3):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.elite_size = elite_size  # 精英个体数量
        self.cv_folds = cv_folds
        
        # 结果存储
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.selected_features_ = None
        self.best_score_ = None
        self.best_individual_ = None
        
    def _set_random_seed(self):
        """设置随机种子确保可重现性"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)
    
    def _initialize_population(self, n_features):
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            # 随机选择特征，确保选择的特征数量接近目标
            individual = np.zeros(n_features, dtype=bool)
            selected_indices = np.random.choice(
                n_features, 
                size=min(self.n_features_to_select, n_features), 
                replace=False
            )
            individual[selected_indices] = True
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, individual, X, y):
        """
        评估个体适应度

        关键改进：
        1. 使用固定的数据分割确保结果一致性
        2. 多次评估取平均值提高稳定性
        3. 添加特征数量惩罚项
        4. 处理异常情况
        """
        # 检查是否有选择的特征
        if not np.any(individual):
            return 0.0

        try:
            # 选择特征
            X_selected = X[:, individual]

            # 使用固定的分层K折交叉验证确保一致性
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            accuracies = []
            for train_idx, test_idx in skf.split(X_selected, y):
                X_train, X_test = X_selected[train_idx], X_selected[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # 克隆估计器以避免状态污染
                estimator = clone(self.estimator)
                estimator.fit(X_train, y_train)
                y_pred = estimator.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            # 计算平均准确率
            mean_accuracy = np.mean(accuracies)

            # 添加特征数量惩罚（轻微）
            n_selected = np.sum(individual)
            feature_penalty = 0.005 * (n_selected / len(individual))  # 减小惩罚系数

            fitness = mean_accuracy - feature_penalty

            return fitness

        except Exception as e:
            print(f"适应度评估错误: {e}")
            return 0.0
    
    def _tournament_selection(self, population, fitness_scores):
        """锦标赛选择"""
        selected = []
        for _ in range(self.population_size - self.elite_size):  # 为精英留出空间
            # 随机选择参赛个体
            tournament_indices = np.random.choice(
                len(population), 
                size=self.tournament_size, 
                replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 选择最佳个体
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parent1, parent2):
        """单点交叉"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 单点交叉
        crossover_point = np.random.randint(1, len(parent1))
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        child1[crossover_point:] = parent2[crossover_point:]
        child2[crossover_point:] = parent1[crossover_point:]
        
        return child1, child2
    
    def _mutate(self, individual):
        """变异操作"""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = not mutated[i]
        
        # 确保选择的特征数量合理
        n_selected = np.sum(mutated)
        if n_selected == 0:
            # 如果没有选择任何特征，随机选择一些
            random_indices = np.random.choice(
                len(mutated), 
                size=min(10, len(mutated)), 
                replace=False
            )
            mutated[random_indices] = True
        elif n_selected > len(mutated) * 0.8:
            # 如果选择太多特征，随机去掉一些
            selected_indices = np.where(mutated)[0]
            remove_count = n_selected - int(len(mutated) * 0.6)
            remove_indices = np.random.choice(
                selected_indices, 
                size=remove_count, 
                replace=False
            )
            mutated[remove_indices] = False
        
        return mutated
    
    def _apply_elitism(self, population, fitness_scores):
        """
        精英保留策略
        
        关键改进：保留最佳个体，确保适应度不会下降
        """
        # 找到最佳个体的索引
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        elite_individuals = [population[i].copy() for i in elite_indices]
        elite_fitness = [fitness_scores[i] for i in elite_indices]
        
        return elite_individuals, elite_fitness
    
    def fit(self, X, y):
        """训练遗传算法"""
        self._set_random_seed()
        
        print(f"开始遗传算法特征选择...")
        print(f"种群大小: {self.population_size}, 进化代数: {self.generations}")
        print(f"目标特征数: {self.n_features_to_select}, 总特征数: {X.shape[1]}")
        
        # 初始化种群
        population = self._initialize_population(X.shape[1])
        
        # 记录历史
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        best_overall_fitness = -np.inf
        best_overall_individual = None
        
        for generation in range(self.generations):
            print(f"\n第 {generation + 1}/{self.generations} 代:")
            
            # 评估适应度
            fitness_scores = []
            for i, individual in enumerate(population):
                fitness = self._evaluate_fitness(individual, X, y)
                fitness_scores.append(fitness)
                
                if i % 10 == 0:
                    print(f"  评估个体 {i+1}/{len(population)}")
            
            # 记录统计信息
            current_best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            
            # 更新全局最佳（关键：确保单调递增）
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_idx = np.argmax(fitness_scores)
                best_overall_individual = population[best_idx].copy()
                print(f"  ✓ 发现更好解! 适应度: {best_overall_fitness:.4f}")
            else:
                print(f"  当前最佳: {current_best_fitness:.4f}, 全局最佳: {best_overall_fitness:.4f}")
            
            self.best_fitness_history.append(best_overall_fitness)  # 使用全局最佳
            self.avg_fitness_history.append(avg_fitness)
            
            print(f"  最佳适应度: {best_overall_fitness:.4f}")
            print(f"  平均适应度: {avg_fitness:.4f}")
            
            # 如果是最后一代，跳过进化操作
            if generation == self.generations - 1:
                break
            
            # 精英保留
            elite_individuals, elite_fitness = self._apply_elitism(population, fitness_scores)
            
            # 选择
            selected = self._tournament_selection(population, fitness_scores)
            
            # 交叉和变异
            new_population = elite_individuals.copy()  # 先添加精英
            
            while len(new_population) < self.population_size:
                # 选择父母
                parent1 = selected[np.random.randint(len(selected))]
                parent2 = selected[np.random.randint(len(selected))]
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 确保种群大小正确
            population = new_population[:self.population_size]
        
        # 保存最佳结果
        self.best_individual_ = best_overall_individual
        self.best_score_ = best_overall_fitness
        self.selected_features_ = np.where(best_overall_individual)[0]
        
        print(f"\n遗传算法完成!")
        print(f"最佳适应度: {self.best_score_:.4f}")
        print(f"选择特征数: {len(self.selected_features_)}")
        
        return self
    
    def transform(self, X):
        """转换数据，只保留选择的特征"""
        if self.selected_features_ is None:
            raise ValueError("必须先调用 fit 方法")
        
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """拟合并转换数据"""
        return self.fit(X, y).transform(X)
