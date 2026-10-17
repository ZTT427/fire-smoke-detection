\# 火灾烟雾检测系统



基于 YOLOv8 的火焰与烟雾目标检测系统，包含 Streamlit 可视化预警界面。



\## 模型指标

\- mAP50: 93.9% | Precision: 92.8% | Recall: 91.0%

\- 数据集：Home Fire Dataset（6500 张，flame / smoke 两类）



\## 使用方法

pip install ultralytics streamlit pandas

streamlit run app.py



\## 文件说明

\- app.py：预警界面（上传图片 → 检测 → 三级预警 + CSV 报告）

\- train.py：训练脚本（YOLOv8n，50 epochs，batch 16）

\- predict.py：测试集批量检测脚本

\- best.pt：训练好的模型权重

\- data.yaml：数据集配置



