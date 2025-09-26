import yaml
import os
import config
from ultralytics import YOLO
import albumentations as A

dataset_yaml_path = os.path.join(config.DATA_DIR, "data.yaml")

yolo_model = YOLO('yolov8n-seg.pt')

yolo_model.train(  # train 803장
    data=dataset_yaml_path,
    epochs=2,  # GPU.............ㅠㅠ
    imgsz=640,
    batch=16,
    augment=True,
    patience=50,
    project=config.MODEL_DIR,
    name='yolov8n-seg',
    exist_ok=True
)

validation_results = yolo_model.val(data=dataset_yaml_path)  # validation : 17장
map = validation_results.box.map # 바운딩 박스
seg_map = validation_results.seg.map # 세그멘테이션
fps = 1000 / validation_results.speed['inference'] # 속도

class_names = validation_results.names  
box_p = validation_results.box.p  #Precision
box_r = validation_results.box.r  #Recall
box_ap = validation_results.box.all_ap.mean(axis=1) #average precision
seg_ap = validation_results.seg.all_ap.mean(axis=1)

print(f"Bounding Box mAP: {map:.4f}")
print(f"Segmentation mAP: {seg_map:.4f}")
print(f"Inference Speed (FPS): {fps:.2f}")

print("\nClass-wise Performance (Bounding Box):")
for i, name in class_names.items():
    print(f"Class {name}:")
    print(f"  AP: {box_ap[i]:.4f}")
    print(f"  Precision: {box_p[i]:.4f}")
    print(f"  Recall: {box_r[i]:.4f}")

print("\nClass-wise Performance (Segmentation):")
for i, name in class_names.items():
    print(f"Class {name}:")
    print(f"  AP: {seg_ap[i]:.4f}")