from pathlib import Path

# =======================
# 파이프라인 실행 설정 (MAIN.PY 제어)
# =======================
# ----------------------------------------------------
# 0: 아무것도 실행 안 함 (STAGE_NONE)
# 1: 원본 학습 (STAGE_TRAIN_ORIGINAL)
# 2: SAHI 타일링 (STAGE_TILE)
# 3: 오버샘플링 (정상 포함) (STAGE_OVERSAMPLE_ALL)
# 4: 오버샘플링 (결함만) (STAGE_OVERSAMPLE_DEFECT)
# 5: 타일 학습 (STAGE_TRAIN_TILE)
# 6: 파인튜닝 (STAGE_FINETUNE)
# 7: SAHI 추론 (STAGE_INFER)
# ----------------------------------------------------
PIPELINE_STAGE = 7  

#==================================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)

# 데이터 및 모델 루트
DATA_DIR = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"

# 원본 데이터 경로
DATA_ROOT = DATA_DIR / "original_data"
TRAIN_IMAGES = DATA_ROOT / "train/images"
VAL_IMAGES   = DATA_ROOT / "valid/images"
TEST_IMAGES  = DATA_ROOT / "test/images"
DATA_YAML = DATA_ROOT / "data.yaml"

# 모델별 경로
STAGE1_DIR = MODELS_ROOT / "step1"
STAGE2_DIR = MODELS_ROOT / "step2"
STAGE3_DIR = MODELS_ROOT / "step3"
STAGE4_DIR = MODELS_ROOT / "step4"

# YOLO 모델 설정
MODEL_CFG = PROJECT_ROOT / "yolo11m-p2.yaml"

# 휠
CROP_ROOT = DATA_DIR / "cropped_wheels" # 가정: 크롭 데이터셋 루트
CROP_TRAIN = CROP_ROOT / "train"
CROP_VAL   = CROP_ROOT / "valid"
CROP_TEST  = CROP_ROOT / "test"

# 타일링 결과 저장 루트 
TILE_ROOT  = DATA_DIR / "tiles_out"
TILE_TRAIN = TILE_ROOT / "train"
TILE_VAL   = TILE_ROOT / "valid"
TILE_TEST  = TILE_ROOT / "test"

# 최종 분할/오버샘플링 데이터 루트
FINAL_ROOT = DATA_DIR / "final_splits"
DATA_YAML_TILES = FINAL_ROOT / "data_tiles.yaml"

ALLOWED_DEFECTS = ["spalling", "flat"]

WHEEL_MODEL_PATH = STAGE1_DIR / "weights" / "best.pt"
DEFECT_MODEL_PATH = STAGE3_DIR / "fine2" / "weights" / "best.pt"
LOGO_IMAGE_PATH = PROJECT_ROOT / "logo.png"

# =======================
# 📜 학습 설정 (TRAIN_CFG)
# =======================
TRAIN_CFG = dict(
    imgsz=640,
    epochs=300,
    batch=6,
    workers=4,
    seed=42,
    patience=30,

    box=0.08,
    cls=0.20,
    dfl=1.5,

    mosaic=0.90,
    copy_paste=0.40,
    mixup=0.1,
    erasing=0.10,
    close_mosaic=10,

    degrees=5.0,
    shear=0.0,
    perspective=0.0,
    translate=0.05,
    scale=0.60,
    hsv_h=0.015, hsv_s=0.50, hsv_v=0.40,
    fliplr=0.5, flipud=0.0,

    rect=True,
    optimizer="AdamW",
    lr0=0.003,
    lrf=0.20,
    weight_decay=0.0005,
    freeze=0,
    amp=True,
    cache=True,
    verbose=False,
    plots=True,
)

# =======================
# 🔪 SAHI 설정 (SAHI_CFG)
# =======================
SAHI_CFG = dict(
    # --- 분할 방식 ---
    # size, count_v
    SPLIT_FLAG="count_v",
    SPLIT_VALUE=6,

    overlap_h=0.10,   # 세로 겹침
    overlap_w=0.00,   # 가로 겹침 (count_v 0.0
    
    # 추론 설정
    conf_thres=0.5,
    postprocess="NMS",
    match_metric="IOU",
    match_thres=0.45
)

# =======================
# 💡 파인튜닝 설정 (FT_CFG)
# =======================
FT_TRAIN_CFG = TRAIN_CFG.copy() 
FT_TRAIN_CFG.update(dict(
    box=4.0,     
    cls=0.3,     
    lr0=0.0005,
    epochs=100,
))

# =======================
# 3단계 학습결과, 파인튜닝 경로 설정 (FT_PATHS)
PREV_BEST_WEIGHTS_FOR_FT = STAGE3_DIR / "fine2" / "weights" / "best.pt"
# 파인튜닝 결과 저장 경로
STAGE3_FT_DIR = STAGE3_DIR / "fine_tuned"

WHEEL_MODEL = STAGE1_DIR / "weights" / "best.pt"
DEFECT_MODEL = PREV_BEST_WEIGHTS_FOR_FT