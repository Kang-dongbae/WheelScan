import os

DATA_DIR = r"D:\dev\WheelScan\data"
MODEL_DIR = r"D:\dev\WheelScan\models"
#DATA_DIR = r"/Users/dongbae/Dev/WheelScan/data"
#MODEL_DIR = r"/Users/dongbae/Dev/WheelScan/models"
TEST_FOLDER = DATA_DIR + "/test/images"

dataset_yaml_path = os.path.join(DATA_DIR, "data.yaml")

YOLO_MODEL = os.path.join(MODEL_DIR, "yolov9c", "weights", "best.pt")
#YOLO_MODEL = "yolov9c.pt"

CLASSES = ['crack', 'discoloration', 'flat', 'shelling', 'spalling', 'wheel']

HYPERPARAMS = {
    'epochs': 10,  # GPU....
    'batch': 8,
    'imgsz': 0,
    'augment': True,
    'exist_ok': True
}