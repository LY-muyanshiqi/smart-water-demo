# CLAUDE.md — smart-water-demo：洪水预警演示系统

## 项目简介
面向智慧水利的洪水预警与决策支持系统，集成 LSTM/GRU 预测、Flask API、Streamlit 前端。

## 技术栈
- Python 3.8+ · TensorFlow/Keras · Flask · Streamlit
- 数据处理：pandas, numpy, matplotlib, scikit-learn
- CI/CD：GitHub Actions + pre-commit hooks

## 项目结构
```
src/
├── models/          # LSTM/GRU 模型定义与训练
├── data/            # 数据处理 (雨情数据、水文资料)
├── api/             # Flask REST API
└── app/             # Streamlit 前端界面
docs/                # 文档 (架构图 .drawio、技术路线等)
```

## 关键术语
- **洪水预测**: 基于历史降雨和径流数据的时序预测
- **LSTM/GRU**: 长短期记忆 / 门控循环单元，用于时序建模
- **GIS 淹没模拟**: 基于 GIS 的淹没范围可视化

## 开发命令
```bash
pip install -r requirements.txt
python src/app.py        # Streamlit 前端
python src/api.py        # Flask API 服务
```

## 注意事项
- docs/ 下 .drawio 文件为空 (0 bytes)，待补充
- requirements.txt 无版本最低约束，生产部署时建议固定版本
- .pre-commit-config.yaml 已配置，提交前自动运行 flake8 等检查
