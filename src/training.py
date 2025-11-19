from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

# 내부 모듈 임포트
from utils import device_str
import config as cfg

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