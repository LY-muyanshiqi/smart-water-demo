"""
LSTM 深度学习预测模块（优化版）
用于时序数据预测（水位预测）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LSTMPredictor:
    """LSTM时序预测器（优化版）"""
    
    def __init__(self, sequence_length=20, features=1):
        """
        初始化LSTM预测器
        
        参数:
            sequence_length: 时间步长（回溯多少个时间点）
            features: 特征数量
        """
        self.sequence_length = sequence_length
        self.features = features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
    
    def prepare_sequences(self, data, target_columns='water_level'):
        """
        准备时序数据序列（支持多特征）
        
        参数:
            data: DataFrame或数组，包含时序数据
            target_columns: 目标列名（可以是单个列名或列表）
        
        返回:
            X, y: 特征和标签序列
        """
        # 处理目标列
        if isinstance(target_columns, str):
            target_columns = [target_columns]
        
        if isinstance(data, pd.DataFrame):
            data = data[target_columns].values
        
        # 数据标准化
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            # 特征：过去sequence_length个时间点的数据
            X.append(scaled_data[i:(i + self.sequence_length), :])
            # 标签：下一个时间点的数据（所有特征）
            y.append(scaled_data[i + self.sequence_length, :])
        
        return np.array(X), np.array(y)
    
    def build_model(self, units=[128, 64, 32], dropout_rate=0.3, learning_rate=0.001):
        """
        构建LSTM模型（优化架构）
        
        参数:
            units: 各LSTM层的神经元数量列表
            dropout_rate: Dropout率
            learning_rate: 学习率
        
        返回:
            编译后的模型
        """
        model = Sequential()
        
        # 输入层
        model.add(Input(shape=(self.sequence_length, self.features)))
        
        # 第一层LSTM
        model.add(LSTM(
            units[0],
            return_sequences=True if len(units) > 1 else False,
        ))
        model.add(Dropout(dropout_rate))
        
        # 中间层LSTM
        for i, unit in enumerate(units[1:], 1):
            return_sequences = i < len(units) - 1
            model.add(LSTM(unit, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        # 输出层
        model.add(Dense(self.features))
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print("\n" + "="*60)
        print("LSTM模型架构（优化版）")
        print("="*60)
        model.summary()
        print("="*60 + "\n")
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, patience=15):
        """
        训练LSTM模型（优化版）
        
        参数:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            patience: 早停耐心值
        
        返回:
            训练历史
        """
        if self.model is None:
            raise ValueError("请先调用 build_model() 构建模型")
        
        # 回调函数
        callbacks = []
        
        # 早停
        if X_val is not None and y_val is not None:
            early_终止ping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_终止ping)
        
        # 学习率调度
        if X_val is not None and y_val is not None:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
            callbacks.append(reduce_lr)
        
        # 模型检查点
        if not os.path.exists('models'):
            os.makedirs('models')
        
        checkpoint = ModelCheckpoint(
            'models/lstm_best_model.keras',  # 使用.keras格式
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # 训练模型
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """
        预测
        
        参数:
            X: 输入特征
        
        返回:
            预测结果（已反标准化）
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 预测
        predictions_scaled = self.model.predict(X)
        
        # 反标准化
        predictions = self.scaler.inverse_transform(predictions_scaled)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        评估模型
        
        参数:
            X_test: 测试集特征
            y_test: 测试集标签
        
        返回:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 预测
        predictions_scaled = self.model.predict(X_test)
        
        # 反标准化
        predictions = self.scaler.inverse_transform(predictions_scaled)
        y_test_original = self.scaler.inverse_transform(y_test)
        
        # 计算指标
        mse = np.mean((predictions - y_test_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test_original))
        
        # 计算 MAPE（平均绝对百分比误差）
        mape = np.mean(np.abs((y_test_original - predictions) / (y_test_original + 1e-8))) * 100
        
        # 计算 R²（决定系数）
        ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
        ss_res = np.sum((y_test_original - predictions) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2
        }
        
        print("\n" + "="*60)
        print("模型评估指标（优化版）")
        print("="*60)
        for metric, value in metrics.items():
            print(f"{metric:8s}: {value:.4f}")
        print("="*60 + "\n")
        
        return metrics
    
    def plot_training_history(self):
        """
        绘制训练历史（优化版）
        """
        if self.history is None:
            print("没有训练历史记录")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(self.history.history['loss'], label='训练损失', linewidth=2)
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='验证损失', linewidth=2)
        axes[0].set_title('模型损失', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss', fontsize=11)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # MAE曲线
        axes[1].plot(self.history.history['mae'], label='训练MAE', linewidth=2)
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='验证MAE', linewidth=2)
        axes[1].set_title('平均绝对误差', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('MAE', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/lstm_training_history.png', dpi=300, bbox_inches='tight')
        print("\n训练历史图已保存到 images/lstm_training_history.png")
    
    def plot_predictions(self, y_true, y_pred, title='LSTM预测结果'):
        """
        绘制预测结果（优化版）
        
        参数:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
        """
        plt.figure(figsize=(14, 6))
        
        # 真实值 vs 预测值
        plt.plot(y_true, label='真实值', linewidth=2.5, alpha=0.8)
        plt.plot(y_pred, label='预测值', linewidth=2.5, linestyle='--', alpha=0.8)
        
        # 添加置信区间
        residual_std = np.std(y_true - y_pred)
        plt.fill_between(
            range(len(y_pred)),
            (y_pred.flatten() - residual_std),
            (y_pred.flatten() + residual_std),
            alpha=0.2,
            label='±1σ 置信区间',
            color='orange'
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('水位 (m)', fontsize=12)
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/lstm_predictions.png', dpi=300, bbox_inches='tight')
        print("预测结果图已保存到 images/lstm_predictions.png")
    
    def plot_residuals(self, y_true, y_pred):
        """
        绘制残差图（优化版）
        
        参数:
            y_true: 真实值
            y_pred: 预测值
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 残差时间序列
        axes[0, 0].plot(residuals.flatten())
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_title('残差时间序列', fontsize=11, fontweight='bold')
        axes[0, 0].set_xlabel('时间', fontsize=10)
        axes[0, 0].set_ylabel('残差', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 残差直方图
        axes[0, 1].hist(residuals.flatten(), bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_title('残差分布', fontsize=11, fontweight='bold')
        axes[0, 1].set_xlabel('残差', fontsize=10)
        axes[0, 1].set_ylabel('频数', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q图（正态性检验）', fontsize=11, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 残差散点图
        axes[1, 1].scatter(y_pred.flatten(), residuals.flatten(), alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].set_title('预测值 vs 残差', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('预测值', fontsize=10)
        axes[1, 1].set_ylabel('残差', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/lstm_residuals.png', dpi=300, bbox_inches='tight')
        print("\n残差分析图已保存到 images/lstm_residuals.png")


def main():
    """主函数 - LSTM预测演示（优化版）"""
    print("\n" + "="*60)
    print("LSTM 深度学习预测模块演示（优化版）")
    print("="*60)
    
    # 生成模拟时序数据
    print("\n生成模拟时序数据...")
    np.random.seed(42)
    n_samples = 365  # 一年的数据
    
    # 生成带有趋势和季节性的水位数据
    t = np.arange(n_samples)
    trend = 0.01 * t  # 线性趋势
    seasonality = 2 * np.sin(2 * np.pi * t / 30)  # 月度季节性
    noise = np.random.normal(0, 0.5, n_samples)  # 随机噪声
    
    water_level = 5 + trend + seasonality + noise
    
    # 模拟其他特征（流量、降雨量）
    discharge_rate = 600 + 100 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 30, n_samples)
    rainfall = np.random.exponential(2, n_samples)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=n_samples),
        'water_level': water_level,
        'discharge_rate': discharge_rate,
        'rainfall': rainfall
    })
    
    print(f"数据生成完成！共 {n_samples} 天的数据")
    print(f"水位范围: {water_level.min():.2f}m ~ {water_level.max():.2f}m")
    print(f"特征数量: 3（水位、流量、降雨量）")
    
    # 准备数据（使用多特征）
    print("\n准备时序数据（多特征）...")
    sequence_length = 20  # 使用过去20天的数据
    lstm = LSTMPredictor(sequence_length=sequence_length, features=3)
    
    X, y = lstm.prepare_sequences(df, target_columns=['water_level', 'discharge_rate', 'rainfall'])
    print(f"序列准备完成！")
    print(f"特征序列形状: {X.shape}")
    print(f"标签序列形状: {y.shape}")
    
    # 划分训练集、验证集、测试集
    print("\n划分数据集...")
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    
    # 构建模型（优化架构）
    print("\n构建LSTM模型（优化架构）...")
    lstm.build_model(units=[128, 64, 32], dropout_rate=0.3)
    
    # 训练模型
    print("\n训练LSTM模型...")
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32,
        patience=15
    )
    
    # 绘制训练历史
    lstm.plot_training_history()
    
    # 评估模型（只预测水位）
    print("\n评估模型性能...")
    metrics = lstm.evaluate(X_test, y_test)
    
    # 预测
    print("\n生成预测...")
    y_pred = lstm.predict(X_test)
    y_test_original = lstm.scaler.inverse_transform(y_test)
    
    # 只绘制水位预测结果
    lstm.plot_predictions(y_test_original[:, 0], y_pred[:, 0], 'LSTM水位预测结果（优化版）')
    
    # 绘制残差分析
    lstm.plot_residuals(y_test_original[:, 0], y_pred[:, 0])
    
    print("\n" + "="*60)
    print("LSTM预测模块演示完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()