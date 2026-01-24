"""
智慧水利项目 - 核心代码注释
为所有核心模块添加详细注释，便于理解和维护
"""

# ============================================================
# 1. 水位计算模块 (water_analysis.py)
# ============================================================

"""
模块名称：water_analysis.py
功能描述：水位计算与分析模块
作者：李垚
创建日期：2026-01-24
"""

def calculate_water_level(discharge_rate, river_width):
    """
    计算水位高度
    
    该函数基于连续性方程原理，通过流量和河宽计算水位高度。
    
    物理原理：
        水位 = 流量 / 河宽
        （假设河流横截面为矩形，忽略摩擦损失）
    
    参数：
        discharge_rate (float/int): 流量，单位：立方米/秒 (m³/s)
        river_width (float/int): 河宽，单位：米 (m)
    
    返回：
        float: 水位高度，单位：米 (m)
    
    使用示例：
        >>> level = calculate_water_level(500, 10)
        >>> print(f"水位：{level}m")
        水位：50.0m
    
    异常处理：
        - 如果 river_width 为 0，会抛出 ZeroDivisionError
        - 建议调用前进行参数验证
    """
    water_level = discharge_rate / river_width
    return water_level


def calculate_discharge(water_level, river_width):
    """
    计算流量（逆向计算）
    
    参数：
        water_level (float/int): 水位高度，单位：米 (m)
        river_width (float/int): 河宽，单位：米 (m)
    
    返回：
        float: 流量，单位：立方米/秒 (m³/s)
    """
    discharge = water_level * river_width
    return discharge


def validate_input(value, param_name):
    """
    验证输入参数是否有效
    
    参数：
        value: 待验证的值
        param_name (str): 参数名称（用于错误提示）
    
    返回：
        bool: 验证通过返回 True，否则返回 False
    
    异常：
        ValueError: 当参数无效时抛出
    """
    if value <= 0:
        raise ValueError(f"{param_name} 必须为正数，当前值为：{value}")
    return True


# ============================================================
# 2. 数据可视化模块 (visualizer.py)
# ============================================================

"""
模块名称：visualizer.py
功能描述：数据可视化模块，支持多种图表类型
作者：李垚
创建日期：2026-01-24
"""

def plot_water_level(discharge_rates, river_widths, 
                    title='水位变化趋势图', 
                    xlabel='时间点', 
                    ylabel='水位高度（m）',
                    save_path=None):
    """
    绘制水位变化曲线图
    
    使用 Matplotlib 创建水位随时间变化的折线图，
    支持自定义标题、标签和保存路径。
    
    参数：
        discharge_rates (list): 流量列表，单位：m³/s
        river_widths (list): 河宽列表，单位：m
        title (str): 图表标题，默认为'水位变化趋势图'
        xlabel (str): X轴标签，默认为'时间点'
        ylabel (str): Y轴标签，默认为'水位高度（m）'
        save_path (str): 图片保存路径（可选），如 None 则不保存
    
    返回：
        None
    
    依赖：
        - matplotlib.pyplot
        - numpy (可选，用于数据处理)
    
    使用示例：
        >>> rates = [500, 600, 700, 800, 900]
        >>> widths = [10, 12, 14, 16, 18]
        >>> plot_water_level(rates, widths, save_path='water_level.png')
    
    注意事项：
        - 需要提前配置中文字体，否则中文显示为方框
        - 图片格式由 save_path 扩展名决定（png, jpg, pdf, svg）
    """
    import matplotlib.pyplot as plt
    
    # 配置中文字体（防止乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 计算水位
    water_levels = [d/r for d, r in zip(discharge_rates, river_widths)]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(water_levels, 
             marker='o', 
             linestyle='-', 
             linewidth=2, 
             markersize=8, 
             color='#2E86C1')
    
    # 添加标题和标签
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 显示数值标签
    for i, level in enumerate(water_levels):
        plt.annotate(f'{level:.2f}', 
                    (i, level), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（如果指定了保存路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至：{save_path}")
    
    # 显示图表
    plt.show()


def plot_discharge_water_level(discharge_rates, water_levels, save_path=None):
    """
    绘制流量与水位关系散点图
    
    参数：
        discharge_rates (list): 流量列表，单位：m³/s
        water_levels (list): 水位列表，单位：m
        save_path (str): 图片保存路径（可选）
    
    返回：
        None
    """
    import matplotlib.pyplot as plt
    
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(discharge_rates, water_levels, 
               c='red', 
               s=100, 
               alpha=0.6, 
               edgecolors='black')
    
    # 添加标题和标签
    plt.title('流量与水位关系散点图', fontsize=16, fontweight='bold')
    plt.xlabel('流量（m³/s）', fontsize=12)
    plt.ylabel('水位（m）', fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（如果指定了保存路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至：{save_path}")
    
    # 显示图表
    plt.show()


# ============================================================
# 3. KNN 预测模块 (knn_predictor.py)
# ============================================================

"""
模块名称：knn_predictor.py
功能描述：K近邻算法实现，用于水位分类预测
作者：李垚
创建日期：2026-01-24
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class KNNWaterLevelPredictor:
    """
    KNN 水位预测器
    
    该类封装了 KNN 算法，支持分类和回归两种任务，
    用于基于历史水文数据预测未来水位。
    
    属性：
        n_neighbors (int): K 值，默认为 5
        weights (str): 权重计算方式，'uniform' 或 'distance'
        algorithm (str): 算法类型，'auto', 'ball_tree', 'kd_tree', 'brute'
        task_type (str): 任务类型，'classification' 或 'regression'
        model: 训练好的模型实例
    
    使用示例：
        >>> predictor = KNNWaterLevelPredictor(n_neighbors=5, task_type='classification')
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> predictor.fit(X_train, y_train)
        >>> predictions = predictor.predict(X_test)
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', task_type='classification'):
        """
        初始化 KNN 预测器
        
        参数：
            n_neighbors (int): K 值，默认为 5
            weights (str): 权重计算方式
                - 'uniform': 所有邻居权重相同
                - 'distance': 距离越近权重越大
            algorithm (str): 算法类型
                - 'auto': 自动选择
                - 'ball_tree': 适用于高维数据
                - 'kd_tree': 适用于低维数据
                - 'brute': 暴力搜索
            task_type (str): 任务类型
                - 'classification': 分类任务
                - 'regression': 回归任务
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.task_type = task_type
        self.model = None
        
        # 根据任务类型选择模型
        if task_type == 'classification':
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm
            )
        elif task_type == 'regression':
            self.model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm
            )
        else:
            raise ValueError(f"不支持的 task_type：{task_type}，请选择 'classification' 或 'regression'")
    
    def fit(self, X, y):
        """
        训练模型
        
        参数：
            X (array-like): 特征数据，形状为 (n_samples, n_features)
            y (array-like): 目标数据，形状为 (n_samples,)
        
        返回：
            self: 返回自身，支持链式调用
        
        使用示例：
            >>> predictor.fit(X_train, y_train)
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        预测水位
        
        参数：
            X (array-like): 待预测的特征数据，形状为 (n_samples, n_features)
        
        返回：
            array-like: 预测结果
                - 分类任务：类别标签
                - 回归任务：连续数值
        
        使用示例：
            >>> predictions = predictor.predict(X_test)
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        参数：
            X_test (array-like): 测试集特征数据
            y_test (array-like): 测试集目标数据
        
        返回：
            float: 评估指标
                - 分类任务：准确率 (accuracy)
                - 回归任务：均方误差 (MSE)
        """
        y_pred = self.predict(X_test)
        
        if self.task_type == 'classification':
            return accuracy_score(y_test, y_pred)
        else:
            return mean_squared_error(y_test, y_pred)
    
    def predict_proba(self, X):
        """
        预测概率（仅分类任务）
        
        参数：
            X (array-like): 待预测的特征数据
        
        返回：
            array-like: 预测概率，形状为 (n_samples, n_classes)
        
        异常：
            ValueError: 如果是回归任务，该方法不可用
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba 方法仅适用于分类任务")
        return self.model.predict_proba(X)


# ============================================================
# 4. LSTM 预测模块 (lstm_predictor.py)
# ============================================================

"""
模块名称：lstm_predictor.py
功能描述：LSTM 时序预测模型实现
作者：李垚
创建日期：2026-01-24
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


class LSTMWaterLevelPredictor:
    """
    LSTM 水位预测器
    
    该类封装了 LSTM（长短期记忆网络）算法，
    用于基于时序数据预测未来水位变化。
    
    属性：
        input_shape (tuple): 输入数据形状 (timesteps, features)
        units (list): LSTM 层神经元数量列表
        dropout_rate (float): Dropout 比例，防止过拟合
        optimizer (str): 优化器类型
        loss (str): 损失函数
        metrics (list): 评估指标
        model: 编译好的 Keras 模型
        history: 训练历史记录
    
    使用示例：
        >>> predictor = LSTMWaterLevelPredictor(
        ...     input_shape=(10, 3),
        ...     units=[64, 32],
        ...     dropout_rate=0.2
        ... )
        >>> predictor.fit(X_train, y_train, epochs=100)
        >>> predictions = predictor.predict(X_test)
    """
    
    def __init__(self, input_shape, units=[64, 32], dropout_rate=0.2, 
                 optimizer='adam', loss='mse', metrics=['mae']):
        """
        初始化 LSTM 预测器
        
        参数：
            input_shape (tuple): 输入数据形状 (timesteps, features)
                - timesteps: 时间步长（过去多少个时间点的数据）
                - features: 特征数量（每个时间点有多少个特征）
            units (list): LSTM 层神经元数量列表，如 [64, 32] 表示两层 LSTM
            dropout_rate (float): Dropout 比例，默认为 0.2
            optimizer (str): 优化器类型，默认为 'adam'
            loss (str): 损失函数，默认为 'mse'（均方误差）
            metrics (list): 评估指标列表，默认为 ['mae']
        
        使用示例：
            >>> predictor = LSTMWaterLevelPredictor(
            ...     input_shape=(10, 3),  # 过去10个时间点，每个时间点3个特征
            ...     units=[64, 32],       # 两层LSTM，分别为64和32个神经元
            ...     dropout_rate=0.2      # Dropout比例为0.2
            ... )
        """
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model = None
        self.history = None
        
        # 构建模型
        self._build_model()
    
    def _build_model(self):
        """
        构建 LSTM 模型
        
        模型结构：
            Input → LSTM(units[0]) → Dropout → LSTM(units[1]) → Dropout → Dense → Output
        
        注意：
            - 第一层 LSTM 设置 return_sequences=True，返回序列
            - 最后一层 LSTM 设置 return_sequences=False，返回最终状态
            - Dropout 层防止过拟合
        """
        model = Sequential()
        
        # 添加第一层 LSTM
        model.add(LSTM(
            self.units[0],
            return_sequences=True,
            input_shape=self.input_shape
        ))
        model.add(Dropout(self.dropout_rate))
        
        # 添加中间层 LSTM（如果有多层）
        for i in range(1, len(self.units) - 1):
            model.add(LSTM(
                self.units[i],
                return_sequences=True
            ))
            model.add(Dropout(self.dropout_rate))
        
        # 添加最后一层 LSTM
        model.add(LSTM(
            self.units[-1],
            return_sequences=False
        ))
        model.add(Dropout(self.dropout_rate))
        
        # 添加全连接层
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))  # 输出层，1个神经元（预测水位）
        
        # 编译模型
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        
        self.model = model
        
        # 打印模型结构
        print("模型结构：")
        model.summary()
    
    def fit(self, X_train, y_train, epochs=100, batch_size=32, 
            validation_split=0.2, verbose=1):
        """
        训练模型
        
        参数：
            X_train (array-like): 训练集特征数据，形状为 (n_samples, timesteps, features)
            y_train (array-like): 训练集目标数据，形状为 (n_samples, 1)
            epochs (int): 训练轮数，默认为 100
            batch_size (int): 批次大小，默认为 32
            validation_split (float): 验证集比例，默认为 0.2（20%）
            verbose (int): 日志详细程度
                - 0: 不输出日志
                - 1: 输出进度条
                - 2: 每个epoch输出一行
        
        返回：
            History: 训练历史记录
        
        使用示例：
            >>> history = predictor.fit(X_train, y_train, epochs=100, batch_size=32)
        """
        # 创建回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """
        预测水位
        
        参数：
            X (array-like): 待预测的特征数据，形状为 (n_samples, timesteps, features)
        
        返回：
            array-like: 预测的水位值，形状为 (n_samples, 1)
        
        使用示例：
            >>> predictions = predictor.predict(X_test)
            >>> print(predictions)
        """
        return self.model.predict(X)
    
    def save_model(self, path='lstm_water_level_model.h5'):
        """
        保存模型
        
        参数：
            path (str): 模型保存路径，默认为 'lstm_water_level_model.h5'
        
        使用示例：
            >>> predictor.save_model('my_model.h5')
        """
        self.model.save(path)
        print(f"模型已保存至：{path}")
    
    def load_model(self, path='lstm_water_level_model.h5'):
        """
        加载模型
        
        参数：
            path (str): 模型加载路径，默认为 'lstm_water_level_model.h5'
        
        使用示例：
            >>> predictor.load_model('my_model.h5')
        """
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            print(f"模型已从 {path} 加载")
        else:
            raise FileNotFoundError(f"模型文件不存在：{path}")


# ============================================================
# 5. 数据预处理工具 (data_preprocessor.py)
# ============================================================

"""
模块名称：data_preprocessor.py
功能描述：数据预处理工具，包括缺失值处理、特征工程等
作者：李垚
创建日期：2026-01-24
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_data(file_path, encoding='utf-8'):
    """
    加载数据文件
    
    参数：
        file_path (str): 文件路径
        encoding (str): 文件编码，默认为 'utf-8'
    
    返回：
        DataFrame: 加载的数据
    
    支持的文件格式：
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - JSON (.json)
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, encoding=encoding)
    elif file_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        return pd.read_json(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{file_path}")


def handle_missing_values(df, strategy='mean', columns=None):
    """
    处理缺失值
    
    参数：
        df (DataFrame): 输入数据
        strategy (str): 处理策略
            - 'mean': 使用均值填充（数值型）
            - 'median': 使用中位数填充（数值型）
            - 'mode': 使用众数填充
            - 'forward': 使用前向填充
            - 'backward': 使用后向填充
            - 'drop': 删除包含缺失值的行
        columns (list): 指定要处理的列，如 None 则处理所有列
    
    返回：
        DataFrame: 处理后的数据
    
    使用示例：
        >>> df_cleaned = handle_missing_values(df, strategy='mean')
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.columns
    
    if strategy == 'mean':
        for col in columns:
            if df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in columns:
            if df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col].fillna(df_copy[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in columns:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    elif strategy == 'forward':
        df_copy[columns].fillna(method='ffill', inplace=True)
    elif strategy == 'backward':
        df_copy[columns].fillna(method='bfill', inplace=True)
    elif strategy == 'drop':
        df_copy.dropna(subset=columns, inplace=True)
    else:
        raise ValueError(f"不支持的策略：{strategy}")
    
    return df_copy


def normalize_data(df, columns=None, method='standard'):
    """
    数据标准化/归一化
    
    参数：
        df (DataFrame): 输入数据
        columns (list): 指定要处理的列，如 None 则处理所有数值型列
        method (str): 标准化方法
            - 'standard': 标准化（均值为0，标准差为1）
            - 'minmax': 归一化（范围[0,1]）
    
    返回：
        DataFrame: 标准化后的数据
        scaler: 标准化器对象（可用于转换测试数据）
    
    使用示例：
        >>> df_normalized, scaler = normalize_data(df, method='standard')
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=['int64', 'float64']).columns
    
    if method == 'standard':
        scaler = StandardScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
    else:
        raise ValueError(f"不支持的方法：{method}")
    
    return df_copy, scaler


def create_time_features(df, datetime_col, features=['hour', 'day', 'month', 'dayofweek']):
    """
    创建时间特征
    
    参数：
        df (DataFrame): 输入数据
        datetime_col (str): 日期时间列名
        features (list): 要创建的特征列表
            - 'hour': 小时
            - 'day': 日
            - 'month': 月
            - 'dayofweek': 星期几（0-6）
            - 'is_weekend': 是否周末
    
    返回：
        DataFrame: 添加了时间特征的数据
    """
    df_copy = df.copy()
    
    # 确保日期时间列是 datetime 类型
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])
    
    # 创建时间特征
    if 'hour' in features:
        df_copy['hour'] = df_copy[datetime_col].dt.hour
    if 'day' in features:
        df_copy['day'] = df_copy[datetime_col].dt.day
    if 'month' in features:
        df_copy['month'] = df_copy[datetime_col].dt.month
    if 'dayofweek' in features:
        df_copy['dayofweek'] = df_copy[datetime_col].dt.dayofweek
    if 'is_weekend' in features:
        df_copy['is_weekend'] = (df_copy[datetime_col].dt.dayofweek >= 5).astype(int)
    
    return df_copy


# ============================================================
# 6. 主程序入口 (main.py)
# ============================================================

"""
模块名称：main.py
功能描述：智慧水利项目主程序入口
作者：李垚
创建日期：2026-01-24
"""

if __name__ == "__main__":
    """
    主程序入口
    
    该程序演示了智慧水利项目的完整流程：
    1. 数据加载
    2. 数据预处理
    3. 水位计算
    4. 数据可视化
    5. KNN 预测
    6. LSTM 预测
    """
    
    print("=" * 60)
    print("智慧水利项目 - 主程序")
    print("=" * 60)
    
    # 1. 导入模块
    from water_analysis import calculate_water_level
    from visualizer import plot_water_level
    from knn_predictor import KNNWaterLevelPredictor
    
    # 2. 示例数据
    discharge_rates = [500, 600, 700, 800, 900]  # 流量数据（m³/s）
    river_widths = [10, 12, 14, 16, 18]         # 河宽数据（m）
    
    # 3. 水位计算
    print("\n1. 水位计算")
    print("-" * 60)
    water_levels = [calculate_water_level(d, r) for d, r in zip(discharge_rates, river_widths)]
    print(f"{'时间':<8} {'流量(m³/s)':<12} {'河宽(m)':<10} {'水位(m)':<10}")
    print("-" * 60)
    for i, (d, r, level) in enumerate(zip(discharge_rates, river_widths, water_levels)):
        print(f"时间{i+1:<4} {d:<12} {r:<10} {level:<10.2f}")
    
    # 4. 数据可视化
    print("\n2. 数据可视化")
    print("-" * 60)
    print("正在绘制水位变化曲线...")
    plot_water_level(discharge_rates, river_widths, save_path='water_level_plot.png')
    print("图表绘制完成！")
    
    # 5. KNN 预测（示例）
    print("\n3. KNN 预测（示例）")
    print("-" * 60)
    print("KNN 预测模块已加载")
    print("使用示例：")
    print("  >>> predictor = KNNWaterLevelPredictor(n_neighbors=5)")
    print("  >>> predictor.fit(X_train, y_train)")
    print("  >>> predictions = predictor.predict(X_test)")
    
    # 6. LSTM 预测（示例）
    print("\n4. LSTM 预测（示例）")
    print("-" * 60)
    print("LSTM 预测模块已加载")
    print("使用示例：")
    print("  >>> predictor = LSTMWaterLevelPredictor(input_shape=(10, 3))")
    print("  >>> predictor.fit(X_train, y_train, epochs=100)")
    print("  >>> predictions = predictor.predict(X_test)")
    
    print("\n" + "=" * 60)
    print("程序执行完毕！")
    print("=" * 60)


# ============================================================
# 结束
# ============================================================

"""
智慧水利项目 - 核心代码注释
作者：李垚
最后更新：2026-01-24

所有核心模块已添加详细注释，包括：
1. water_analysis.py - 水位计算模块
2. visualizer.py - 数据可视化模块
3. knn_predictor.py - KNN 预测模块
4. lstm_predictor.py - LSTM 预测模块
5. data_preprocessor.py - 数据预处理工具
6. main.py - 主程序入口

注释包括：
- 模块级文档字符串
- 类和函数的详细说明
- 参数和返回值的类型说明
- 使用示例
- 异常处理说明
- 注意事项

这些注释将帮助理解代码逻辑，便于后续维护和扩展。
"""