import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io

st.set_page_config(page_title="火焰烟雾预警系统", page_icon="🔥", layout="wide")

st.title("🔥 火焰烟雾智能预警系统")
st.caption("基于 YOLOv8 自训练模型 | Flame & Smoke Detection")

# 侧边栏控制面板
with st.sidebar:
    st.header("⚙️ 控制面板")
    conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05,
                               help="数值越高，只显示模型更有把握的检测结果")
    st.markdown("---")
    st.info("上传图片后，系统自动检测火焰和烟雾。\n\n"
            "检测到 **flame（火焰）** 或 **smoke（烟雾）** 时会触发红色预警。")


@st.cache_resource
def load_model():
    return YOLO("D:/projects/fire-smoke-detection/runs/detect/train-3/weights/best.pt")


model = load_model()

# 上传图片
uploaded = st.file_uploader("📤 上传一张图片", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原图")
        st.image(img, use_container_width=True)

    # 运行检测
    results = model.predict(source=img, conf=conf_threshold, verbose=False)
    r = results[0]

    # 画框后的图
    annotated = r.plot()
    with col2:
        st.subheader("检测结果")
        st.image(annotated[:, :, ::-1], use_container_width=True)

    # 统计检测结果
    detections = []
    for box in r.boxes:
        detections.append({
            "类别": model.names[int(box.cls)],
            "置信度": round(float(box.conf), 3),
            "位置(x,y,w,h)": [round(v, 1) for v in box.xywh[0].tolist()],
        })

    df = pd.DataFrame(detections)

    # 预警逻辑
    has_flame = (df["类别"] == "flame").any()
    has_smoke = (df["类别"] == "smoke").any()

    if has_flame:
        st.error("🚨 重大预警：检测到明火（flame）！请立即处理！")
    if has_smoke:
        st.warning("⚠️ 烟雾预警：检测到烟雾（smoke），可能是火灾早期迹象。")
    if not has_flame and not has_smoke:
        if df.empty:
            st.success("✅ 未检测到火焰或烟雾，环境安全。")
        else:
            st.info("检测到目标但非火焰/烟雾。")

    # 结果表格 + 下载
    if not df.empty:
        st.subheader("📋 检测明细")
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 下载检测报告 (CSV)", csv, "fire_report.csv", "text/csv")
else:
    st.info("👈 请在左侧上传图片开始检测（可用 test/images 里的图片测试）")
