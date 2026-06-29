import os
import glob
import time
import cv2
from ultralytics import YOLO

import Config
from detect_tools import (
    ensure_output_dir, build_output_img_path, get_csv_path, append_csv_rows,
    parse_yolo_result, draw_boxes_chinese, get_now_str
)


def detect_image(model, img_path, conf, iou, show_label=True):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] 无法读取图片: {img_path}")
        return

    t1 = time.time()
    results = model.predict(source=img, conf=conf, iou=iou, verbose=False)
    result = results[0]
    detections, count = parse_yolo_result(result, Config.CLASS_NAMES)

    out_img = draw_boxes_chinese(img, detections, show_label=show_label)
    cost_ms = (time.time() - t1) * 1000

    out_path = build_output_img_path(img_path)
    cv2.imwrite(out_path, out_img)

    # 写CSV
    rows = []
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(v) for v in det["xyxy"]]
        rows.append([
            get_now_str(), img_path, idx, det["cls_name"], f"{det['conf']:.4f}", x1, y1, x2, y2
        ])
    append_csv_rows(get_csv_path(), rows)

    print(f"[OK] {img_path} -> 数量: {count}, 耗时: {cost_ms:.1f}ms, 保存: {out_path}")


def gather_images(input_path):
    if os.path.isfile(input_path):
        return [input_path]
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    imgs = []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(input_path, ext)))
    return sorted(imgs)


def main():
    Config.ensure_dirs()
    ensure_output_dir()

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件: {Config.MODEL_PATH}")

    model = YOLO(Config.MODEL_PATH)

    # 改成你的图片路径或文件夹路径
    input_path = r"TestFiles"

    conf = Config.DEFAULT_CONF
    iou = Config.DEFAULT_IOU
    show_label = True

    img_list = gather_images(input_path)
    if not img_list:
        print(f"[WARN] 未找到可检测图片: {input_path}")
        return

    for p in img_list:
        detect_image(model, p, conf, iou, show_label)


if __name__ == "__main__":
    main()
