from pathlib import Path
import config as cfg
from data_preparation import (
    stage2_tile_all_with_sahi, 
    create_iterative_splits, 
    oversample_tiles_for_2_loops
)
from training import (
    stage1_train_p2, 
    stage3_train_defect_on_tiles, 
    run_fine_tuning
)
from inference import stage4_infer_yolo_with_sahi

def main():
    
    stage = cfg.PIPELINE_STAGE
    
    print(f"==========================================")
    print(f"[WheelScan] 파이프라인 실행 시작")
    print(f"설정된 실행 단계 번호: {stage}")
    print(f"==========================================")

    # --- 0: 아무것도 실행 안 함 ---
    if stage == 0:
         print("실행 단계가 '0'(NONE)으로 설정되어, 아무 작업도 실행하지 않습니다.")
        
    # --- 1: 원본 학습 ---
    elif stage == 1:
        best_wheel = stage1_train_p2(
            data_yaml=cfg.DATA_YAML, 
            out_dir=cfg.STAGE1_DIR
        )
        print(f"\n[1단계] 원본 학습 완료: {best_wheel}")
        
    # --- 2: SAHI 타일링 ---
    elif stage == 2:
        stage2_tile_all_with_sahi()
        print(f"\n[2단계] 타일링 완료: {cfg.TILE_ROOT}")

    # --- 3: 오버샘플링 (정상 포함) ---
    elif stage == 3:
        final_path = create_iterative_splits(
            tile_root=cfg.TILE_ROOT,
            final_output_root=cfg.FINAL_ROOT
        )
        print(f"\n[3단계] 최종 데이터셋 생성 완료: {final_path}")

    # --- 4: 오버샘플링 (결함만) ---
    elif stage == 4:
        final_path = oversample_tiles_for_2_loops(
            tile_root=cfg.TILE_ROOT,
            final_root=cfg.FINAL_ROOT
        )
        print(f"\n[4단계] 최종 데이터셋 생성 완료: {final_path}")

    # --- 5: 타일 학습 ---
    elif stage == 5:
        best_defect = stage3_train_defect_on_tiles(
            data_yaml_tiles=cfg.DATA_YAML_TILES,
            out_dir=cfg.STAGE3_DIR,
            weights_path=None, 
            train_cfg_override=None 
        )
        print(f"\n[5단계] 타일 학습 완료: {best_defect}")

    # --- 6: 파인튜닝 ---
    elif stage == 6:
        best_defect_ft = run_fine_tuning()
        print(f"\n [6단계] 파인튜닝 완료: {best_defect_ft}")

    # --- 7: SAHI 추론 ---
    elif stage == 7:
        weights_to_use = None
        infer_path_cfg = cfg.INFER_WEIGHTS_PATH

        if infer_path_cfg == "finetune":
            weights_to_use = cfg.STAGE3_FT_DIR / "weights" / "best.pt"
        elif isinstance(infer_path_cfg, Path) and infer_path_cfg.exists():
            weights_to_use = infer_path_cfg
        elif infer_path_cfg is None:
            weights_to_use = cfg.STAGE3_DIR / "weights" / "best.pt"
        
        if weights_to_use and weights_to_use.exists():
            stage4_infer_yolo_with_sahi(
                weights_path=weights_to_use,
                cropped_test_split=cfg.CROP_TEST,
                out_dir=cfg.STAGE4_DIR,
                sahi_cfg=cfg.SAHI_CFG
            )
            print(f"\n [7단계] SAHI 추론 완료 (가중치: {weights_to_use.name})")
        else:
            print(f" 경고: 추론 가중치 파일을 찾을 수 없습니다: {weights_to_use}")

    else:
        print(f" [에러] config.PIPELINE_STAGE에 설정된 값('{stage}')이 유효하지 않습니다. 0~7 사이의 숫자를 입력하세요.")


if __name__ == "__main__":
    main()