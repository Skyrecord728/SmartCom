import os
from ultralytics import YOLO
import Config


def main():
    Config.ensure_dirs()

    pretrained = "yolov8n.pt"

    if not os.path.exists(Config.DATA_YAML):
        raise FileNotFoundError(f"未找到数据配置文件: {Config.DATA_YAML}")

    model = YOLO(pretrained)

    model.train(
        data=Config.DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project="runs",
        name="lychee_train",
        pretrained=True
    )

    model.val(data=Config.DATA_YAML, imgsz=640)

    print("训练完成。best.pt 默认在 runs/lychee_train/weights/best.pt")
    print("请手动复制到 models/best.pt，或修改 Config.MODEL_PATH 指向该文件。")


if __name__ == "__main__":
    main()
