import os

#DATA_DIR = r"D:\Dev\WheelScan\data"
#MODEL_DIR = r"D:\Dev\WheelScan\models"
DATA_DIR = r"/Users/dongbae/Dev/WheelScan/data"
MODEL_DIR = r"/Users/dongbae/Dev/WheelScan/models"



dataset_yaml_path = os.path.join(DATA_DIR, "data.yaml")

YOLO_MODEL = "yolov8n-seg.pt"

CLASSES = ['Cracks-Scratches', 'Discoloration', 'Shelling', 'Wheel']

HYPERPARAMS = {
    'epochs': 10,  # GPU....
    'batch': 8,
    'imgsz': 640,
    'augment': True,
    'exist_ok': True
}