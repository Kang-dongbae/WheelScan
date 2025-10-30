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


# =======================
# [3단계] 타일 데이터 학습 (원본 학습 및 파인 튜닝 모두 지원)
# =======================
def stage3_train_defect_on_tiles(
    data_yaml_tiles: Path, 
    out_dir: Path,             
    weights_path: Optional[Path] = None, 
    train_cfg_override: Optional[Dict] = None
) -> Path: 
    print("\n=== [3단계] 타일 데이터로 결함 모델 학습 ===")
    
    # 1. 모델 초기화
    if weights_path and weights_path.exists():
        print(f"파인 튜닝 시작: 초기 가중치 경로: {weights_path}")
        model = YOLO(str(weights_path)) 
    else:
        print(f"초기 학습 시작: 모델 설정 파일 사용: {cfg.MODEL_CFG}")
        model = YOLO(cfg.MODEL_CFG)
    
    # 2. 최종 학습 설정 준비
    final_train_cfg = cfg.TRAIN_CFG.copy()
    if train_cfg_override:
        final_train_cfg.update(train_cfg_override)
        print(f"설정 덮어쓰기 적용: {list(train_cfg_override.keys())}")
    else:
        print("기본 TRAIN_CFG 설정 사용")
        
    # 3. 학습 인자 조합 및 실행
    train_args = {
        "data": str(data_yaml_tiles),
        "project": str(out_dir.parent), 
        "name": out_dir.name,         
        "device": device_str(),
        **final_train_cfg,
        "plots": True,
        "exist_ok": True,
    }
    
    # 4. 학습 시작
    results = model.train(**train_args)
    
    #5. 학습 결과
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