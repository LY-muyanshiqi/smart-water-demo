"""
智慧水利数据分析模块
用于演示GitHub版本控制功能
"""

def calculate_water_level(discharge_rate, river_width):
    """
    计算水位高度
    
    参数:
        discharge_rate: 流量（m³/s）
        river_width: 河宽（m）
        
    返回:
        水位高度（m）
    """
    water_level = discharge_rate / river_width
    return water_level

def main():
    """主函数"""
    # 示例数据（多个时间点）
    discharge_rates = [500, 600, 700, 800, 900]  # 流量数据（m³/s）
    river_widths = [10, 12, 14, 16, 18]         # 河宽数据（m）
    
    # 计算水位
    water_levels = [calculate_water_level(d, r) for d, r in zip(discharge_rates, river_widths)]
    
    # 打印计算结果
    print("=" * 50)
    print("水位计算结果")
    print("=" * 50)
    print(f"{'时间':<8} {'流量(m³/s)':<12} {'河宽(m)':<10} {'水位(m)':<10}")
    print("-" * 50)
    for i, (d, r, level) in enumerate(zip(discharge_rates, river_widths, water_levels)):
        print(f"时间{i+1:<4} {d:<12} {r:<10} {level:<10.2f}")
    print("=" * 50)
    
    # 导入可视化模块
    try:
        from visualizer import plot_water_level
        print("\n正在绘制水位变化曲线...")
        plot_water_level(discharge_rates, river_widths)
        print("图表绘制完成！")
    except ImportError:
        print("\n提示：可视化模块未安装，请先安装 matplotlib：pip install matplotlib")
    except Exception as e:
        print(f"\n绘图时出现错误：{e}")

if __name__ == "__main__":
    main()