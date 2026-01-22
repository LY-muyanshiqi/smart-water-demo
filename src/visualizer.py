"""
数据可视化模块
用于绘制水位变化曲线
"""

import matplotlib.pyplot as plt

def plot_water_level(discharge_rates, river_widths):
    """
    绘制水位变化曲线
    
    参数:
        discharge_rates: 流量列表（m³/s）
        river_widths: 河宽列表（m）
    """
    # 计算水位
    water_levels = [d/r for d, r in zip(discharge_rates, river_widths)]
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(water_levels)), water_levels, marker='o', linestyle='-', color='blue', label='水位')
    
    # 添加标题和标签
    plt.xlabel('时间')
    plt.ylabel('水位 (m)')
    plt.title('水位变化曲线')
    
    # 添加图例和网格
    plt.legend()
    plt.grid(True)
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 示例数据
    discharge_rates = [500, 600, 700, 800, 900]
    river_widths = [10, 12, 14, 16, 18]
    
    # 绘制水位变化曲线
    plot_water_level(discharge_rates, river_widths)