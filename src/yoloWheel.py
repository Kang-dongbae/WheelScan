import os
import pandas as pd
from ultralytics import YOLO
from config import dataset_yaml_path, YOLO_MODEL, MODEL_DIR, HYPERPARAMS, DATA_DIR

class YOLOPipeline:
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

    
def main():
    # training
    yolo = YOLOPipeline(YOLO_MODEL)
    yolo.train(dataset_yaml_path)

    # 테스트 데이터 평가
    print("\nEvaluating test data...")
    test_results = yolo.validate(dataset_yaml_path, split='test')

    # 메트릭 출력
    box_map = test_results.box.map
    seg_map = test_results.seg.map
    fps = 1000 / test_results.speed['inference']
    print(f"Test Bounding Box mAP: {box_map:.4f}")
    print(f"Test Segmentation mAP: {seg_map:.4f}")
    print(f"Test Inference Speed (FPS): {fps:.2f}")

    print("\nClass-wise Performance (Bounding Box):")
    for i, name in test_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {test_results.box.all_ap.mean(axis=1)[i]:.4f}")
        print(f"  Precision: {test_results.box.p[i]:.4f}")
        print(f"  Recall: {test_results.box.r[i]:.4f}")

    print("\nClass-wise Performance (Segmentation):")
    for i, name in test_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {test_results.seg.all_ap.mean(axis=1)[i]:.4f}")

    # 메트릭 저장
    yolo.save_metrics(test_results, os.path.join(MODEL_DIR, "yolov8n-seg", "test_results.csv"))

    # 테스트 데이터 예측 및 시각화
    print("\nPredicting on test data...")
    yolo.predict(os.path.join(DATA_DIR, "test", "images"), MODEL_DIR)  # DATA_DIR 사용

if __name__ == "__main__":
    main()