"""
数据预处理模块
用于数据加载、清洗、特征工程和可视化探索
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class DataPreprocessor:
    """数据预处理器类"""
    
    def __init__(self, data_path=None):
        """
        初始化数据预处理器
        
        参数:
            data_path: 数据文件路径（CSV/Excel）
        """
        self.data_path = data_path
        self.df = None
        self.original_df = None
        
        # 确保必要的目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = ['data', 'images', 'models']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"已创建目录: {directory}/")
    
    def load_data(self, file_path=None, file_type='csv'):
        """
        加载数据
        
        参数:
            file_path: 数据文件路径
            file_type: 文件类型 ('csv' 或 'excel')
        
        返回:
            DataFrame
        """
        if file_path:
            self.data_path = file_path
            
        if file_type == 'csv':
            self.df = pd.read_csv(self.data_path)
        elif file_type == 'excel':
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("file_type 必须是 'csv' 或 'excel'")
            
        # 保存原始数据副本
        self.original_df = self.df.copy()
        
        print(f"数据加载成功！共 {len(self.df)} 行，{len(self.df.columns)} 列")
        return self.df
    
    def data_overview(self):
        """
        数据概览
        """
        if self.df is None:
            print("请先加载数据！")
            return
            
        print("\n" + "="*60)
        print("数据概览")
        print("="*60)
        
        # 1. 基本信息
        print("\n1. 基本信息:")
        print(f"   数据形状: {self.df.shape}")
        print(f"   数据大小: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # 2. 数据类型
        print("\n2. 数据类型:")
        print(self.df.dtypes)
        
        # 3. 缺失值统计
        print("\n3. 缺失值统计:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct
        })
        print(missing_df[missing_df['缺失数量'] > 0])
        
        # 4. 统计描述
        print("\n4. 数值型变量统计:")
        print(self.df.describe())
        
        # 5. 前5行数据
        print("\n5. 前5行数据:")
        print(self.df.head())
        
        print("="*60)
    
    def handle_missing_values(self, strategy='mean', columns=None):
        """
        处理缺失值
        
        参数:
            strategy: 处理策略 ('mean', 'median', 'mode', 'drop')
            columns: 指定处理的列，None表示处理所有列
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        if columns is None:
            # 找出有缺失值的列
            columns = self.df.columns[self.df.isnull().any()].tolist()
        
        print(f"\n处理缺失值，策略: {strategy}")
        print(f"处理列: {columns}")
        
        for col in columns:
            if strategy == 'drop':
                self.df = self.df.dropna(subset=[col])
            elif strategy == 'mean' and self.df[col].dtype in ['int64', 'float64']:
                self.df[col].fillna(self.df[col].mean(), inplace=True)
            elif strategy == 'median' and self.df[col].dtype in ['int64', 'float64']:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            elif strategy == 'mode':
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print(f"缺失值处理完成！剩余行数: {len(self.df)}")
        return self.df
    
    def detect_outliers(self, column, method='iqr', threshold=1.5):
        """
        检测异常值
        
        参数:
            column: 列名
            method: 检测方法 ('iqr' 或 'zscore')
            threshold: 阈值 (IQR默认1.5，Z-score默认3)
        
        返回:
            异常值的索引
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        if method == 'iqr':
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = self.df[(self.df[column] < lower_bound) | 
                              (self.df[column] > upper_bound)].index
        
        elif method == 'zscore':
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / 
                            self.df[column].std())
            outliers = self.df[z_scores > threshold].index
        
        else:
            raise ValueError("method 必须是 'iqr' 或 'zscore'")
        
        print(f"\n检测到 {len(outliers)} 个异常值（列: {column}）")
        return outliers
    
    def remove_outliers(self, column, method='iqr', threshold=1.5):
        """
        移除异常值
        
        参数:
            column: 列名
            method: 检测方法
            threshold: 阈值
        """
        outliers = self.detect_outliers(column, method, threshold)
        self.df = self.df.drop(outliers)
        print(f"已移除 {len(outliers)} 个异常值")
        return self.df
    
    def create_time_features(self, date_column):
        """
        创建时序特征
        
        参数:
            date_column: 日期列名
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        # 转换为日期格式
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        # 提取时间特征
        self.df['year'] = self.df[date_column].dt.year
        self.df['month'] = self.df[date_column].dt.month
        self.df['day'] = self.df[date_column].dt.day
        self.df['hour'] = self.df[date_column].dt.hour
        self.df['day_of_week'] = self.df[date_column].dt.dayofweek
        self.df['day_of_year'] = self.df[date_column].dt.dayofyear
        self.df['quarter'] = self.df[date_column].dt.quarter
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        # 季节特征（北半球）
        self.df['season'] = self.df['month'].apply(self._get_season)
        
        print(f"已创建时序特征！新增列: ['year', 'month', 'day', 'hour', 'day_of_week', 'day_of_year', 'quarter', 'is_weekend', 'season']")
        return self.df
    
    def _get_season(self, month):
        """根据月份返回季节"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def create_lag_features(self, columns, lags=[1, 3, 7]):
        """
        创建滞后特征
        
        参数:
            columns: 需要创建滞后特征的列名列表
            lags: 滞后期列表
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        for col in columns:
            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        
        print(f"已创建滞后特征！新增 {len(columns) * len(lags)} 个特征")
        return self.df
    
    def create_rolling_features(self, columns, windows=[3, 7, 14], 
                              agg_funcs=['mean', 'std', 'min', 'max']):
        """
        创建滚动窗口特征
        
        参数:
            columns: 需要创建滚动特征的列名列表
            windows: 窗口大小列表
            agg_funcs: 聚合函数列表
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        for col in columns:
            for window in windows:
                for func in agg_funcs:
                    new_col_name = f'{col}_rolling_{window}_{func}'
                    if func == 'mean':
                        self.df[new_col_name] = self.df[col].rolling(window=window).mean()
                    elif func == 'std':
                        self.df[new_col_name] = self.df[col].rolling(window=window).std()
                    elif func == 'min':
                        self.df[new_col_name] = self.df[col].rolling(window=window).min()
                    elif func == 'max':
                        self.df[new_col_name] = self.df[col].rolling(window=window).max()
        
        print(f"已创建滚动窗口特征！新增 {len(columns) * len(windows) * len(agg_funcs)} 个特征")
        return self.df
    
    def plot_time_series(self, columns):
        """
        绘制时间序列图
        
        参数:
            columns: 需要绘制的列名列表
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4*len(columns)))
        if len(columns) == 1:
            axes = [axes]
        
        for idx, col in enumerate(columns):
            axes[idx].plot(self.df.index, self.df[col])
            axes[idx].set_title(f'{col} 时间序列')
            axes[idx].set_xlabel('时间')
            axes[idx].set_ylabel(col)
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.savefig('images/time_series_plot.png', dpi=300, bbox_inches='tight')
        print("\n时间序列图已保存到 images/time_series_plot.png")
    
    def plot_correlation_heatmap(self):
        """
        绘制相关性热力图
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        # 只选择数值型列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('特征相关性热力图')
        plt.tight_layout()
        plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("\n相关性热力图已保存到 images/correlation_heatmap.png")
    
    def plot_boxplot(self, columns):
        """
        绘制箱线图
        
        参数:
            columns: 需要绘制的列名列表
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        fig, axes = plt.subplots(len(columns), 1, figsize=(10, 4*len(columns)))
        if len(columns) == 1:
            axes = [axes]
        
        for idx, col in enumerate(columns):
            self.df.boxplot(column=col, ax=axes[idx])
            axes[idx].set_title(f'{col} 箱线图')
            axes[idx].grid(True)
        
        plt.tight_layout()
        plt.savefig('images/boxplot.png', dpi=300, bbox_inches='tight')
        print("\n箱线图已保存到 images/boxplot.png")
    
    def save_processed_data(self, output_path='data/processed_data.csv'):
        """
        保存处理后的数据
        
        参数:
            output_path: 输出文件路径
        """
        if self.df is None:
            print("请先加载数据！")
            return
        
        try:
            # 获取目录路径
            dir_path = os.path.dirname(output_path)
            
            # 如果有目录路径，确保目录存在
            if dir_path:
                # 检查是否存在同名的文件
                if os.path.exists(dir_path) and not os.path.isdir(dir_path):
                    print(f"警告: 路径 '{dir_path}' 已存在且是文件，无法创建目录")
                    # 修改输出路径到当前目录
                    base_name = os.path.basename(output_path)
                    output_path = base_name
                    print(f"已将输出路径修改为: {output_path}")
                else:
                    # 创建目录（如果不存在）
                    os.makedirs(dir_path, exist_ok=True)
            
            # 保存数据
            self.df.to_csv(output_path, index=False)
            print(f"\n处理后的数据已保存到 {output_path}")
            
        except Exception as e:
            print(f"\n保存数据时出错: {e}")
            # 尝试保存到当前目录
            base_name = os.path.basename(output_path)
            self.df.to_csv(base_name, index=False)
            print(f"已将数据保存到当前目录: {base_name}")


def main():
    """主函数 - 演示数据预处理流程"""
    print("\n" + "="*60)
    print("数据预处理模块演示")
    print("="*60)
    
    # 创建数据预处理器实例
    processor = DataPreprocessor()
    
    # 生成模拟数据（如果没有真实数据）
    print("\n生成模拟数据...")
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    data = {
        'date': dates,
        'water_level': np.random.normal(5, 1, len(dates)),
        'discharge_rate': np.random.normal(600, 100, len(dates)),
        'rainfall': np.random.exponential(2, len(dates)),
        'temperature': np.random.normal(20, 5, len(dates))
    }
    processor.df = pd.DataFrame(data)
    processor.original_df = processor.df.copy()
    
    # 1. 数据概览
    processor.data_overview()
    
    # 2. 创建时序特征
    processor.create_time_features('date')
    
    # 3. 创建滞后特征
    processor.create_lag_features(['water_level', 'discharge_rate'], lags=[1, 3, 7])
    
    # 4. 创建滚动窗口特征
    processor.create_rolling_features(['water_level', 'discharge_rate'], 
                                     windows=[3, 7], 
                                     agg_funcs=['mean', 'std'])
    
    # 5. 数据可视化
    processor.plot_time_series(['water_level', 'discharge_rate'])
    processor.plot_correlation_heatmap()
    processor.plot_boxplot(['water_level', 'discharge_rate'])
    
    # 6. 保存处理后的数据
    processor.save_processed_data('data/processed_data.csv')
    
    print("\n" + "="*60)
    print("数据预处理模块演示完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()