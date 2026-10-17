from ultralytics import YOLO
import glob
import random

if __name__ == "__main__":
    model = YOLO("D:/projects/fire-smoke-detection/runs/detect/train-3/weights/best.pt")

    # 从测试集中随机抽 8 张图做检测
    images = glob.glob("D:/projects/fire-smoke-detection/test/images/*.jpg")
    samples = random.sample(images, 8)

    results = model.predict(
        source=samples,
        conf=0.25,
        save=True,
        project="D:/projects/fire-smoke-detection/runs/detect",
        name="predict_test",
    )

    for r in results:
        names = [model.names[int(c)] for c in r.boxes.cls]
        print(r.path.split("\\")[-1], "->", names if names else "未检测到")
