import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")  # 训练后最优权重

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

SAVE_DIR = os.path.join(BASE_DIR, "save_data")
FONT_DIR = os.path.join(BASE_DIR, "Font")
FONT_PATH = os.path.join(FONT_DIR, "platech.ttf")  # 中文字体文件

# ========== 检测参数 ==========
CLASS_NAMES = ["lychee"]   # 类别名（和 data.yaml 对齐）
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70
LINE_THICKNESS = 2

# ========== 输出文件 ==========
CSV_NAME = "save_detect_data.csv"
IMG_SUFFIX = "_detect_result.jpg"
VIDEO_SUFFIX = "_detect_result.avi"

# ========== 其他 ==========
WINDOW_TITLE = "荔枝检测与计数系统（保底版）"

def ensure_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FONT_DIR, exist_ok=True)
