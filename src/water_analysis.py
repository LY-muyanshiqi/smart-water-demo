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
    # 示例数据
    discharge_rate = 500  # 流量 500 m³/s
    river_width = 10      # 河宽 10 m
    
    # 计算水位
    level = calculate_water_level(discharge_rate, river_width)
    print(f"流量: {discharge_rate} m³/s")
    print(f"河宽: {river_width} m")
    print(f"水位高度: {level} m")

if __name__ == "__main__":
    main()