"""
KNN 预测模块
用于水位预测和分类任务
"""

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class KNNPredictor:
    """
    KNN 预测器类
    
    功能:
    - KNN 分类
    - KNN 回归
    - 模型评估
    """
    
    def __init__(self, k=5, task='classification'):
        """
        初始化 KNN 预测器
        
        参数:
            k: 邻居数量
            task: 任务类型 ('classification' 或 'regression')
        """
        self.k = k
        self.task = task
        
        if task == 'classification':
            self.model = KNeighborsClassifier(n_neighbors=k)
        elif task == 'regression':
            self.model = KNeighborsRegressor(n_neighbors=k)
        else:
            raise ValueError("task 必须是 'classification' 或 'regression'")
    
    def fit(self, X_train, y_train):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
        """
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """
        预测
        
        参数:
            X_test: 测试特征
        
        返回:
            预测结果
        """
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
        
        返回:
            评估指标
        """
        y_pred = self.predict(X_test)
        
        if self.task == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            return {'accuracy': accuracy}
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            return {'mse': mse, 'rmse': rmse}
    
    def get_nearest_neighbors(self, X):
        """
        获取最近的邻居
        
        参数:
            X: 查询点
        
        返回:
            最近的邻居索引和距离
        """
        distances, indices = self.model.kneighbors(X)
        return {'distances': distances, 'indices': indices}


def water_level_classification_example():
    """
    水位分类示例
    """
    print("\n" + "="*60)
    print("水位分类示例（KNN）")
    print("="*60)
    
    # 模拟数据
    # 特征: [流量(m³/s), 河宽(m)]
    # 标签: 0-低水位, 1-中水位, 2-高水位
    X = np.array([
        [500, 10],   # 低水位
        [600, 12],   # 低水位
        [700, 14],   # 中水位
        [800, 16],   # 中水位
        [900, 18],   # 高水位
        [1000, 20],  # 高水位
        [550, 11],   # 低水位
        [750, 15],   # 中水位
        [850, 17],   # 中水位
        [950, 19],   # 高水位
    ])
    
    y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 1, 2])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建并训练 KNN 模型
    knn = KNNPredictor(k=3, task='classification')
    knn.fit(X_train, y_train)
    
    # 预测
    y_pred = knn.predict(X_test)
    
    # 评估
    results = knn.evaluate(X_test, y_test)
    
    # 输出结果
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"准确率: {results['accuracy']:.2%}")
    
    print(f"\n测试结果:")
    for i in range(len(X_test)):
        print(f"  样本{i+1}: 流量={X_test[i][0]:.0f}m³/s, 河宽={X_test[i][1]:.0f}m, "
              f"真实标签={y_test[i]}, 预测标签={y_pred[i]}")
    
    # 获取最近的邻居
    neighbors = knn.get_nearest_neighbors(X_test)
    print(f"\n最近的邻居信息:")
    for i in range(len(X_test)):
        print(f"  样本{i+1}: 距离={neighbors['distances'][i]}")


def water_level_regression_example():
    """
    水位回归示例
    """
    print("\n" + "="*60)
    print("水位回归示例（KNN）")
    print("="*60)
    
    # 模拟数据
    # 特征: [流量(m³/s), 河宽(m)]
    # 目标: 水位高度(m)
    X = np.array([
        [500, 10],   # 水位: 50
        [600, 12],   # 水位: 50
        [700, 14],   # 水位: 50
        [800, 16],   # 水位: 50
        [900, 18],   # 水位: 50
        [1000, 20],  # 水位: 50
        [550, 11],   # 水位: 50
        [750, 15],   # 水位: 50
        [850, 17],   # 水位: 50
        [950, 19],   # 水位: 50
    ])
    
    y = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 创建并训练 KNN 模型
    knn = KNNPredictor(k=3, task='regression')
    knn.fit(X_train, y_train)
    
    # 预测
    y_pred = knn.predict(X_test)
    
    # 评估
    results = knn.evaluate(X_test, y_test)
    
    # 输出结果
    print(f"\n训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"均方误差 (MSE): {results['mse']:.2f}")
    print(f"均方根误差 (RMSE): {results['rmse']:.2f}")
    
    print(f"\n测试结果:")
    for i in range(len(X_test)):
        print(f"  样本{i+1}: 流量={X_test[i][0]:.0f}m³/s, 河宽={X_test[i][1]:.0f}m, "
              f"真实水位={y_test[i]:.2f}m, 预测水位={y_pred[i]:.2f}m")


def main():
    """
    主函数
    """
    print("\n" + "="*60)
    print("KNN 预测模块 - 智慧水利项目")
    print("="*60)
    
    # 运行分类示例
    water_level_classification_example()
    
    # 运行回归示例
    water_level_regression_example()
    
    print("\n" + "="*60)
    print("KNN 模块演示完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()