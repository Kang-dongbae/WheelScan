import os

DATA_DIR = r"D:\dev\WheelScan\data"
MODEL_DIR = r"D:\dev\WheelScan\models"
#DATA_DIR = r"/Users/dongbae/Dev/WheelScan/data"
#MODEL_DIR = r"/Users/dongbae/Dev/WheelScan/models"
TEST_FOLDER = DATA_DIR + "/test/images"


dataset_yaml_path = os.path.join(DATA_DIR, "data.yaml")

YOLO_MODEL = os.path.join(MODEL_DIR, "yolov8n-seg", "weights", "best.pt")

CLASSES = ['Cracks-Scratches', 'Discoloration', 'Shelling', 'Wheel']

HYPERPARAMS = {
    'epochs': 10,  # GPU....
    'batch': 8,
    'imgsz': 640,
    'augment': True,
    'exist_ok': True
}