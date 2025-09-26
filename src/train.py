from yolo_utils import YOLOUtils
from config import dataset_yaml_path, YOLO_MODEL

yolo = YOLOUtils(YOLO_MODEL)
yolo.train(dataset_yaml_path)