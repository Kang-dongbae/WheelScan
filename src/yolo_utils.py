from ultralytics import YOLO
from config import MODEL_DIR, HYPERPARAMS
import pandas as pd

class YOLOUtils:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def train(self, data_yaml):
        self.model.train(data=data_yaml, project=MODEL_DIR, name='yolov8n-seg', **HYPERPARAMS)

    def validate(self, data_yaml, split='val'):
        return self.model.val(data=data_yaml, split=split)

    def predict(self, source, save_dir):
        return self.model.predict(source=source, save=True, project=save_dir, name='predict', exist_ok=True)

    def save_metrics(self, results, save_path):
        df = pd.DataFrame({
            'Class': [name for _, name in results.names.items()],
            'Box_AP': results.box.all_ap.mean(axis=1),
            'Box_Precision': results.box.p,
            'Box_Recall': results.box.r,
            'Seg_AP': results.seg.all_ap.mean(axis=1)
        })
        df.to_csv(save_path, index=False)