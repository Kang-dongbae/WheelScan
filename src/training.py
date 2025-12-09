from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

# 내부 모듈 임포트
from utils import device_str
import config as cfg

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import os
import time
import copy

# =========================================================
# [설정 영역]
# =========================================================
DATA_DIR = r"C:\Dev\WheelScan\data\cls_tiles" # 데이터 경로
MODEL_SAVE_PATH = r"C:\Dev\WheelScan\models\step0\best_classifier.pth"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================
# [데이터셋 준비]
# =========================================================
def get_data_loaders():
    # 1. 이미지 변환 (Augmentation)
    # Train에는 다양성을 주기 위해 변형을 가함
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # 좌우 반전
        transforms.RandomVerticalFlip(),   # 상하 반전 (차륜은 회전체라 유효)
        transforms.RandomRotation(15),     # 약간의 회전
        transforms.ColorJitter(brightness=0.1, contrast=0.1), # 조명 변화 대응
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Valid는 원본 그대로 (Resize만)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 데이터셋 로드
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), val_transforms)

    # 클래스 매핑 확인 (알파벳 순서: defect=0, normal=1 일 가능성 높음)
    print(f"Class Mapping: {train_dataset.class_to_idx}")
    
    # 3. [핵심] 불균형 해결: WeightedRandomSampler
    # 각 클래스의 샘플 개수 확인
    targets = train_dataset.targets
    class_counts = np.bincount(targets)
    
    print(f"Train Data Counts: {class_counts}") 
    # 예: [1485, 2227] -> 적은 쪽 가중치를 높임
    
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    
    # 샘플러 생성
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 4. 로더 생성
    # train_loader에는 sampler를 적용 (shuffle=True와 함께 쓰면 안됨)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset.class_to_idx


# =======================
# [0단계] (옵션) 원본 학습
# =======================
def stage0_cls_train():
    print(f"Using device: {DEVICE}")
    
    train_loader, val_loader, class_idx = get_data_loaders()
    
    # Defect가 어떤 인덱스인지 확인 (보통 0 아니면 1)
    defect_idx = class_idx['defect']
    
    # 모델 정의 (ResNet18 - 가볍고 성능 좋음)
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # 2진 분류
    model = model.to(DEVICE)

    class_weights = torch.tensor([1.0, 1.0]).to(DEVICE)
    class_weights[defect_idx] = 3.0 
    
    print(f"Applying Class Weights: {class_weights}") # 확인용 출력
    
    # 가중치가 적용된 CrossEntropyLoss 사용
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_recall = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        # --- Train Phase ---
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Valid Phase ---
        model.eval()
        val_loss = 0.0
        
        # 메트릭 계산 변수
        tp = 0 # True Positive (결함을 결함이라 맞춤)
        fn = 0 # False Negative (결함을 정상이라 놓침 - 치명적)
        fp = 0 # False Positive (정상을 결함이라 오해 - 괜찮음)
        tn = 0 # True Negative (정상을 정상이라 맞춤)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)

                # 결함(defect_idx)을 Positive로 간주하고 계산
                for p, l in zip(preds, labels):
                    if l == defect_idx: # 실제 결함인 경우
                        if p == defect_idx: tp += 1
                        else: fn += 1
                    else: # 실제 정상인 경우
                        if p == defect_idx: fp += 1
                        else: tn += 1

        val_loss = val_loss / len(val_loader.dataset)
        
        # 지표 계산
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        recall = tp / (tp + fn + 1e-10)       # 재현율 (중요)
        precision = tp / (tp + fp + 1e-10)    # 정밀도
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {accuracy:.4f} | Precision: {precision:.4f}")
        print(f"★ Defect Recall: {recall:.4f} (TP:{tp}, FN:{fn})") 
        print("-" * 30)

        # 모델 저장 기준: Recall이 가장 높을 때 저장 (놓치면 안되니까)
        # 만약 Recall이 같다면 Accuracy가 높은 순
        if recall > best_recall:
            best_recall = recall
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(">> Best Model Saved (Recall Updated)")

    print(f"Training Complete. Best Recall: {best_recall:.4f}")
    return best_model_wts


# =======================
# [1단계] (옵션) 원본 학습
# =======================
def stage1_train_p2(
    data_yaml: Path = cfg.DATA_YAML, 
    out_dir: Path = cfg.STAGE1_DIR
) -> Path:
    print("\n=== [1단계] 학습 시작 (yolo11m-p2) ===")
    print(f"data: {data_yaml}")
    print(f"model cfg: {cfg.MODEL_CFG}")

    model = YOLO(cfg.MODEL_CFG)
    train_args = {
        "data": str(data_yaml),
        "project": str(cfg.MODELS_ROOT),
        "name": out_dir.name,
        "device": device_str(),
        **cfg.TRAIN_CFG,
        "exist_ok": True,
    }
    results = model.train(**train_args)
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[1단계 완료] best weights: {best}")
    return best

import torch
import torch.nn.functional as F
from ultralytics.utils.loss import v8DetectionLoss
import time

def focal_bce_with_gamma(logits, targets, gamma: float = 2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    return ((1 - pt) ** gamma * bce).mean()

def focal_bce_gamma2(logits, targets):
    return focal_bce_with_gamma(logits, targets, gamma=2.0)

def patch_v8_focal_bce_gamma2():
    """pickle-safe focal patch + init에서 1회 검증 로그"""
    orig_init = v8DetectionLoss.__init__

    def new_init(self, model):
        orig_init(self, model)
        self.bce = focal_bce_gamma2
        if hasattr(self, "BCEcls"):
            self.BCEcls = focal_bce_gamma2
        if hasattr(self, "BCEobj"):
            self.BCEobj = focal_bce_gamma2
        # ✅ 여기서 바로 ‘적용 확인’ 로그 1회 출력
        print("[FOCAL] v8DetectionLoss constructed → bce:", self.bce is focal_bce_gamma2,
              "/ has BCEcls:", hasattr(self, "BCEcls"),
              "/ has BCEobj:", hasattr(self, "BCEobj"))

    v8DetectionLoss.__init__ = new_init
    print("✅ Patched: v8DetectionLoss now uses Focal BCE (γ = 2.0) [pickle-safe]")



# =======================
# [3단계] 타일 데이터 학습 (원본 학습 을 및 파인 튜닝 모두 지원)
# =======================
def stage3_train_defect_on_tiles(
    data_yaml_tiles: Path, 
    out_dir: Path,             
    weights_path: Optional[Path] = None, 
    train_cfg_override: Optional[Dict] = None
) -> Path: 
    print("\n=== [3단계] 타일 데이터로 결함 모델 학습 ===")
    
    # 1️⃣ 모델 초기화
    if weights_path and weights_path.exists():
        print(f"파인 튜닝 시작: 초기 가중치 경로: {weights_path}")
        model = YOLO(str(weights_path)) 
    else:
        print(f"초기 학습 시작: 모델 설정 파일 사용: {cfg.MODEL_CFG}")
        model = YOLO(cfg.MODEL_CFG)
    
    #patch_v8_focal_bce_gamma2()

    # 2️⃣ 학습 설정 병합
    final_train_cfg = cfg.TRAIN_CFG.copy()
    if train_cfg_override:
        final_train_cfg.update(train_cfg_override)
        print(f"설정 덮어쓰기 적용: {list(train_cfg_override.keys())}")
    else:
        print("기본 TRAIN_CFG 설정 사용")
        
    # 3️⃣ 학습 인자 조합
    train_args = {
        "data": str(data_yaml_tiles),
        "project": str(out_dir.parent), 
        "name": out_dir.name,         
        "device": device_str(),
        **final_train_cfg,
        "plots": True,
        "exist_ok": True,
    }
    
    # 4️⃣ 학습 시작
    results = model.train(**train_args)
    
    # 5️⃣ 학습 결과 저장
    save_dir = Path(results.save_dir) 
    val_results = model.val(data=str(data_yaml_tiles), split='val')
    val_results.save_dir = save_dir 
    
    try:
        val_results.save_metrics(save_dir / "val_results.csv")
        print(f"✅ Validation 결과: {save_dir / 'val_results.csv'}")
    except Exception as e:
        print(f"⚠️ Validation 결과 저장 실패: {e}")

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[3단계 완료] best weights: {best}")
    return best

# =======================
# [3.5단계] 파인튜닝 실행기
# =======================
def run_fine_tuning() -> Path:

    print("\n=== [3.5단계] 파인튜닝 시작 ===")
    
    best_defect_ft = stage3_train_defect_on_tiles(
        data_yaml_tiles=cfg.DATA_YAML_TILES, 
        out_dir=cfg.STAGE3_FT_DIR,           
        weights_path=cfg.PREV_BEST_WEIGHTS_FOR_FT,
        train_cfg_override=cfg.FT_TRAIN_CFG 
    )
    return best_defect_ft

# =======================
# [0.5단계] conf* 고정 평가 및 로그 누적
# =======================
from ultralytics import YOLO
from pathlib import Path
import csv, time

def evaluate_fixed_conf(
    model_path,
    data_yaml,
    conf_star: float,
    iou_thr: float = 0.55,
    stage_name: str = "Stage",
    log_csv: str = "C:/Dev/WheelScan/models/step3/fix_conf/stagewise_results.csv"
):
    """
    conf* (운영점) 고정으로 YOLO validation을 수행하고, 결과를 CSV 로그에 누적 기록.
    - model_path : YOLO weight (best.pt)
    - data_yaml  : dataset yaml 경로
    - conf_star  : Stage0.5에서 구한 운영점
    - iou_thr    : IoU threshold (통일)
    - stage_name : Stage 이름 (ex. Stage1_Augment)
    - log_csv    : 결과 누적 CSV 경로
    """

    model = YOLO(str(model_path))
    print(f"\n[{stage_name}] Validation @ conf={conf_star:.3f}, IoU={iou_thr:.2f}")
    results = model.val(
        data=str(data_yaml),
        split="val",
        conf=conf_star,
        iou=iou_thr,
        save_txt=False
    )

    P, R, mAP50, mAP5095 = (
        float(results.box.mp),
        float(results.box.mr),
        float(results.box.map50),
        float(results.box.map)
    )

    print(f"✅ {stage_name} 결과:")
    print(f"   Precision={P:.4f}, Recall={R:.4f}, mAP50={mAP50:.4f}, mAP50-95={mAP5095:.4f}")

    # 로그 저장
    log_path = Path(log_csv)
    row = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage_name,
        "conf_star": conf_star,
        "precision": P,
        "recall": R,
        "mAP50": mAP50,
        "mAP50-95": mAP5095,
        "model_path": str(model_path)
    }
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"📄 로그 저장 완료: {log_path}")

    return results