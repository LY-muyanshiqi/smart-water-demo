# smart-water-demo

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-深度学习-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-前端-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API-000000?style=flat-square&logo=flask&logoColor=white)

> 面向智慧水利的洪水预警与决策支持系统 — 全栈独立开发

---

## 项目简介

集成实时雨情监测、LSTM 深度学习预测、GIS 淹没模拟和应急决策支持，为流域防洪提供智能化解决方案。

## 核心功能

- **实时雨情监测** — 接入气象数据 API，实时监测降雨情况
- **洪水预测** — 基于 LSTM/GRU 深度学习模型，预测未来洪水过程
- **淹没模拟** — GIS 可视化展示淹没范围和影响区域
- **决策支持** — 人员撤离方案 + 资源配置建议
- **数据可视化** — Matplotlib 交互式图表

## 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习 | TensorFlow / Keras · LSTM · GRU |
| 数据处理 | NumPy · Pandas · Scikit-learn |
| 后端 API | Flask REST |
| 前端 | Streamlit |
| 可视化 | Matplotlib · ECharts · GIS |
| 版本控制 | Git + GitHub |

## 快速开始

```bash
# 克隆项目
git clone https://github.com/LY-muyanshiqi/smart-water-demo.git
cd smart-water-demo

# 安装依赖
pip install -r requirements.txt

# 启动
streamlit run app.py
```

## 相关项目

- [smart-water-projects](https://github.com/LY-muyanshiqi/smart-water-projects) — 智慧水利开源项目集
- [thermal-peak-shaving-pumped-storage](https://github.com/LY-muyanshiqi/thermal-peak-shaving-pumped-storage) — 抽水蓄能减碳优化
