import os
import pandas as pd
from ultralytics import YOLO
import torch
from config import dataset_yaml_path, YOLO_MODEL, MODEL_DIR, HYPERPARAMS, DATA_DIR

class YOLOPipeline:
    def __init__(self, model_path):
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"Using device: {device}")
        self.model = YOLO(model_path)

    def train(self, data_yaml):
        self.model.train(data=data_yaml, project=MODEL_DIR, name='yolov8n-seg', device='mps', **HYPERPARAMS)
        print("MPS Memory Allocated:", torch.mps.current_allocated_memory() / 1024**2, "MB")

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

    print("PyTorch Version:", torch.__version__)
    print("MPS Available:", torch.backends.mps.is_available())
    print("MPS Built:", torch.backends.mps.is_built())
    
    # Initialize pipeline
    yolo = YOLOPipeline(YOLO_MODEL)

    # Training
    print("Training model...")
    yolo.train(dataset_yaml_path)

    # Validation
    print("\nEvaluating validation data...")
    val_results = yolo.validate(dataset_yaml_path, split='val')
    
    # Validation metrics
    val_box_map = val_results.box.map
    val_seg_map = val_results.seg.map
    val_fps = 1000 / val_results.speed['inference']
    print(f"Validation Bounding Box mAP: {val_box_map:.4f}")
    print(f"Validation Segmentation mAP: {val_seg_map:.4f}")
    print(f"Validation Inference Speed (FPS): {val_fps:.2f}")

    print("\nClass-wise Performance (Validation - Bounding Box):")
    for i, name in val_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {val_results.box.all_ap.mean(axis=1)[i]:.4f}")
        print(f"  Precision: {val_results.box.p[i]:.4f}")
        print(f"  Recall: {val_results.box.r[i]:.4f}")

    print("\nClass-wise Performance (Validation - Segmentation):")
    for i, name in val_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {val_results.seg.all_ap.mean(axis=1)[i]:.4f}")

    # Save validation metrics
    yolo.save_metrics(val_results, os.path.join(MODEL_DIR, "yolov8n-seg", "val_results.csv"))

    # Test evaluation
    print("\nEvaluating test data...")
    test_results = yolo.validate(dataset_yaml_path, split='test')
    
    # Test metrics
    test_box_map = test_results.box.map
    test_seg_map = test_results.seg.map
    test_fps = 1000 / test_results.speed['inference']
    print(f"Test Bounding Box mAP: {test_box_map:.4f}")
    print(f"Test Segmentation mAP: {test_seg_map:.4f}")
    print(f"Test Inference Speed (FPS): {test_fps:.2f}")

    print("\nClass-wise Performance (Test - Bounding Box):")
    for i, name in test_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {test_results.box.all_ap.mean(axis=1)[i]:.4f}")
        print(f"  Precision: {test_results.box.p[i]:.4f}")
        print(f"  Recall: {test_results.box.r[i]:.4f}")

    print("\nClass-wise Performance (Test - Segmentation):")
    for i, name in test_results.names.items():
        print(f"Class {name}:")
        print(f"  AP: {test_results.seg.all_ap.mean(axis=1)[i]:.4f}")

    # Save test metrics
    yolo.save_metrics(test_results, os.path.join(MODEL_DIR, "yolov8n-seg", "test_results.csv"))

    # Test data prediction and visualization
    print("\nPredicting on test data...")
    yolo.predict(os.path.join(DATA_DIR, "test", "images"), MODEL_DIR)

if __name__ == "__main__":
    main()