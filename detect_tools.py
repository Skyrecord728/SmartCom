import os
import csv
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import Config


def get_now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def xyxy_to_int(xyxy):
    """将浮点坐标转换为整型像素坐标"""
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(x2), int(y2)


def ensure_output_dir():
    Config.ensure_dirs()
    return Config.SAVE_DIR


def build_output_img_path(src_path):
    base = os.path.basename(src_path)
    name, _ = os.path.splitext(base)
    out_name = f"{name}{Config.IMG_SUFFIX}"
    return os.path.join(Config.SAVE_DIR, out_name)


def build_output_video_path(src_path):
    base = os.path.basename(src_path)
    name, _ = os.path.splitext(base)
    out_name = f"{name}{Config.VIDEO_SUFFIX}"
    return os.path.join(Config.SAVE_DIR, out_name)


def get_csv_path():
    return os.path.join(Config.SAVE_DIR, Config.CSV_NAME)


def init_csv_if_needed(csv_path):
    """若CSV不存在则写入表头"""
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "source_path", "target_id", "class_name", "confidence", "xmin", "ymin", "xmax", "ymax"])


def append_csv_rows(csv_path, rows):
    """追加写入多行"""
    init_csv_if_needed(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def draw_boxes_chinese(image_bgr, detections, show_label=True):
    """
    使用PIL绘制中文/文本标签，detections格式:
    [
      {"cls_name":"lychee", "conf":0.91, "xyxy":[x1,y1,x2,y2]},
      ...
    ]
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    if os.path.exists(Config.FONT_PATH):
        font = ImageFont.truetype(Config.FONT_PATH, 20)
    else:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = xyxy_to_int(det["xyxy"])
        cls_name = det["cls_name"]
        conf = det["conf"]

        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 50, 50), width=Config.LINE_THICKNESS)

        if show_label:
            label = f"{cls_name} {conf:.2f}"
            text_pos = (x1, max(0, y1 - 24))
            draw.rectangle([text_pos, (x1 + 180, y1)], fill=(255, 50, 50))
            draw.text((text_pos[0] + 2, text_pos[1] + 2), label, font=font, fill=(255, 255, 255))

    out_rgb = np.array(pil_img)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


def parse_yolo_result(result, class_names):
    """
    解析ultralytics单帧结果 -> 标准结构
    返回:
      detections: list[dict]
      count: int
    """
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections, 0

    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        cls_id = cls[i]
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        detections.append({
            "cls_id": cls_id,
            "cls_name": cls_name,
            "conf": float(conf[i]),
            "xyxy": xyxy[i].tolist()
        })

    return detections, len(detections)
