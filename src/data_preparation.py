"""
데이터셋 준비 파이프라인 (2단계, 2.5단계, 2.6단계)
- stage2_tile_all_with_sahi: 원본 크롭 이미지를 SAHI로 타일링
- create_iterative_splits: 타일 데이터 오버샘플링 (정상 포함) loop 8회
- oversample_tiles_for_2_loops: 타일 데이터 오버샘플링 (결함만) loop 2회
"""

import shutil
import random
from pathlib import Path
from PIL import Image
from sahi.slicing import slice_image
from typing import List, Tuple

# 내부 모듈 임포트
from utils import compute_slice_params
import config as cfg # 설정값들 가져오기


# =======================
# [2단계] SAHI 타일 분할
# =======================

from pathlib import Path
from PIL import Image

def stage2_tile_all_with_sahi(
    keep_empty: bool = True,
    min_side_px: int = 2,
    min_intersection_ratio: float = 0.2,
    use_contact_band: bool = False,   # True면 contact-band dense slicing 활성화
    band_top_ratio: float = 0.25,     # H 기준 band 시작 비율 (예: 0.25 -> 25%)
    band_bottom_ratio: float = 0.75,  # H 기준 band 끝 비율 (예: 0.75 -> 75%)
    band_overlap_h: float = 0.50,     # band 내부에서 사용할 overlap_h (기본보다 촘촘)
):
    """
    2단계 SAHI 타일링 함수.
    - 기본: 전체 이미지에 대해 SAHI 1-pass (기존 동작과 동일)
    - use_contact_band=True 일 때:
        * 전체 이미지 base slicing + 중앙 contact-band 영역을 한 번 더 촘촘하게 slicing
        * 두 슬라이스를 합쳐 타일 데이터셋 생성 (train 시 recall 향상을 위한 B2 모드)
    """
    
    # 설정에서 경로 가져오기
    tile_root = cfg.TILE_ROOT
    sahi_cfg = cfg.SAHI_CFG
    
    dst_img_root = tile_root / "images"
    dst_lbl_root = tile_root / "labels"
    dst_img_root.mkdir(parents=True, exist_ok=True)
    dst_lbl_root.mkdir(parents=True, exist_ok=True)

    def tile_one_split(src_split: Path):
        src_img = src_split / "images"
        src_lbl = src_split / "labels"

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        for ip in sorted(src_img.iterdir()):
            if ip.suffix.lower() not in exts:
                continue

            im = Image.open(ip).convert("RGB")
            W, H = im.size
            
            # --- SAHI 기본 파라미터 계산 ---
            slice_h, slice_w, ovh, ovw = compute_slice_params(W, H, sahi_cfg) 

            # ---------------------------------------------------
            # 1) Base slicing (전체 이미지 대상, 기존 SAHI 동작)
            # ---------------------------------------------------
            sliced_list: list[dict] = []

            sliced_list_base = slice_image(
                image=im,
                slice_height=slice_h,
                slice_width=slice_w,
                overlap_height_ratio=ovh,
                overlap_width_ratio=ovw
            )
            for s in sliced_list_base:
                s["mode"] = "base"  # 타일 이름 구분용
            sliced_list.extend(sliced_list_base)

            # ---------------------------------------------------
            # 2) Contact-band dense slicing (옵션: use_contact_band)
            #    - 중앙 레일 접촉부만 더 촘촘하게 슬라이싱
            # ---------------------------------------------------
            if use_contact_band:
                # band 영역 정의 (H 비율 기준)
                band_y1 = int(H * band_top_ratio)
                band_y2 = int(H * band_bottom_ratio)

                # 안전하게 클램핑
                band_y1 = max(0, min(band_y1, H))
                band_y2 = max(0, min(band_y2, H))

                if band_y2 > band_y1 + 10:  # 최소 높이 10px 이상일 때만 실행
                    band_img = im.crop((0, band_y1, W, band_y2))

                    sliced_list_band = slice_image(
                        image=band_img,
                        slice_height=slice_h,
                        slice_width=slice_w,
                        overlap_height_ratio=band_overlap_h,  # 기본보다 촘촘
                        overlap_width_ratio=ovw,
                    )

                    # band 내 좌표 → 원본 이미지 좌표로 변환
                    for s in sliced_list_band:
                        x0, y0 = s["starting_pixel"]
                        s["starting_pixel"] = (x0, y0 + band_y1)
                        s["mode"] = "band"  # 타일 이름 구분용

                    sliced_list.extend(sliced_list_band)

            # ---------------------------------------------------
            # 3) 원본 YOLO 라벨(px 좌표로 읽기)
            # ---------------------------------------------------
            ypath = src_lbl / (ip.stem + ".txt")
            boxes = []
            if ypath.exists():
                with open(ypath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls = int(parts[0])
                        cx, cy, ww, hh = map(float, parts[1:5])

                        bx = cx * W
                        by = cy * H
                        bw = ww * W
                        bh = hh * H

                        x1 = bx - bw / 2
                        y1 = by - bh / 2
                        x2 = bx + bw / 2
                        y2 = by + bh / 2

                        boxes.append((cls, x1, y1, x2, y2, bw * bh))  # (cls, x1,y1,x2,y2, area)

            # ---------------------------------------------------
            # 4) 각 슬라이스 타일 저장 + 라벨 클리핑/정규화
            # ---------------------------------------------------
            for si in sliced_list:
                x0, y0 = si["starting_pixel"]

                # slice_w, slice_h 기준으로 crop 영역 정의
                x1t, y1t = x0 + slice_w, y0 + slice_h

                # 원본 이미지에서 타일 자르기
                crop = im.crop((x0, y0, x1t, y1t))
                tw, th = crop.size  # 실제 타일 크기 (경계에서 줄어들 수 있음)

                new_lines = []
                for (cls, x1, y1, x2, y2, area) in boxes:
                    ix1 = max(x1, x0)
                    iy1 = max(y1, y0)
                    ix2 = min(x2, x1t)
                    iy2 = min(y2, y1t)

                    iw = ix2 - ix1
                    ih = iy2 - iy1

                    if iw <= 0 or ih <= 0:
                        continue
                    if iw < min_side_px or ih < min_side_px:
                        continue
                    if (iw * ih) / (area + 1e-6) < min_intersection_ratio:
                        continue

                    cx_t = ((ix1 + ix2) / 2 - x0) / tw
                    cy_t = ((iy1 + iy2) / 2 - y0) / th
                    w_t = iw / tw
                    h_t = ih / th

                    w_t = min(max(w_t, 1e-6), 1.0)
                    h_t = min(max(h_t, 1e-6), 1.0)
                    if w_t <= 0 or h_t <= 0:
                        continue

                    cx_t = min(max(cx_t, 0.0), 1.0)
                    cy_t = min(max(cy_t, 0.0), 1.0)

                    new_lines.append(f"{cls} {cx_t:.6f} {cy_t:.6f} {w_t:.6f} {h_t:.6f}")

                mode = si.get("mode", "full")
                if use_contact_band:
                    tile_name = f"{ip.stem}_{mode}_{x0}_{y0}"
                else:
                    tile_name = f"{ip.stem}_{x0}_{y0}"

                crop.save(dst_img_root / f"{tile_name}.jpg", quality=95)
                with open(dst_lbl_root / f"{tile_name}.txt", "w") as f:
                    if new_lines or keep_empty:
                        f.write("\n".join(new_lines))

    # ============================
    # 실제 타일링 실행
    # ============================
    print("\n=== [2단계] SAHI 타일 분할 시작 (단일 출력 모드) =====")
    
    #tile_one_split(cfg.CROP_ROOT)
    # 필요하면 아래 두 줄도 해제해서 val/test까지 타일링
    #tile_one_split(cfg.CROP_TRAIN)
    tile_one_split(cfg.CROP_VAL)
    
    print(f"[2단계 완료] 타일 데이터셋 root: {cfg.TILE_ROOT}")
    print(f" - 출력 폴더: {dst_img_root.parent}")
    print(f" - 모드: {sahi_cfg['SPLIT_FLAG']}, 값: {sahi_cfg['SPLIT_VALUE']}")
    print(f" - overlap_h: {sahi_cfg['overlap_h']}, overlap_w: {sahi_cfg['overlap_w']}")
    if use_contact_band:
        print(f" - Contact band: [{band_top_ratio:.2f}H, {band_bottom_ratio:.2f}H], band_overlap_h={band_overlap_h}")



# ===============================
# [2.5단계] 데이터 오버샘플링 (정상 포함)
# ===============================

def create_iterative_splits(tile_root: Path = cfg.TILE_ROOT, final_output_root: Path = cfg.FINAL_ROOT, num_iterations: int = 8, train_ratio: float = 0.8) -> Path:
    
    label_dir = tile_root / "labels"
    image_dir = tile_root / "images"
    
    if final_output_root.exists():
        shutil.rmtree(final_output_root)
    
    all_label_files = list(label_dir.glob("*.txt"))
    
    class_0_files = [] 
    class_1_files = [] 
    empty_files = []   

    for label_path in all_label_files:
        content = label_path.read_text().strip()
        if not content:
            empty_files.append(label_path)
            continue
        
        classes = {int(line.split()[0]) for line in content.split('\n') if line}
        has_class_1 = 1 in classes
        has_class_0 = 0 in classes

        if has_class_1:
             class_1_files.append(label_path)
        elif has_class_0:
            class_0_files.append(label_path)
        else:
            class_0_files.append(label_path) 
    
    final_replicated_list = [] 
    
    random.shuffle(empty_files)
    num_empty_files = len(empty_files)
    empty_split_size = num_empty_files // num_iterations 
    empty_current_index = 0
    
    num_class_1 = len(class_1_files)
    class_1_total_to_extract = num_class_1 // 2 
    class_1_extract_cycle_count = num_iterations // 2 
    
    class_1_files_for_loop = list(class_1_files)
    class_1_current_index = 0
    
    for i in range(num_iterations):
        for label_path in class_0_files:
            new_name = f"{label_path.stem}_{i}"
            final_replicated_list.append((label_path, new_name))
        
        if i % 2 == 0: 
            random.shuffle(class_1_files_for_loop)
            class_1_current_index = 0
            class_1_split_size = class_1_total_to_extract // 2
            
        if class_1_current_index < num_class_1: 
            start_index = class_1_current_index
            if i % 2 == 0: 
                end_index = min(start_index + class_1_split_size, num_class_1)
            else: 
                end_index = start_index + (class_1_total_to_extract - class_1_split_size)
                end_index = min(end_index, num_class_1) 
                
            selected_class_1 = class_1_files_for_loop[start_index:end_index]
            
            replicate_idx = i // 2
            for label_path in selected_class_1:
                new_name = f"{label_path.stem}_{replicate_idx}"
                final_replicated_list.append((label_path, new_name))
                
            class_1_current_index = end_index 
            
        start_index = empty_current_index
        end_index = min(empty_current_index + empty_split_size, num_empty_files)
        
        if i == num_iterations - 1:
            end_index = num_empty_files
            
        selected_empty = empty_files[start_index:end_index]
        
        for label_path in selected_empty:
            final_replicated_list.append((label_path, label_path.stem))
            
        empty_current_index = end_index

    random.shuffle(final_replicated_list)
    
    num_total = len(final_replicated_list)
    num_train = int(num_total * train_ratio)
    
    train_replicated = final_replicated_list[:num_train]
    valid_replicated = final_replicated_list[num_train:]

    print(f"\n--- [2.5단계] 최종 Train/Valid 분할 (총 {num_total}개 파일) ---")
    print(f"Train 셋 (복제): {len(train_replicated)}개 파일 ({train_ratio*100:.0f}%)")
    print(f"Valid 셋 (복제): {len(valid_replicated)}개 파일 ({(1-train_ratio)*100:.0f}%)")

    train_output_dir = final_output_root / "train"
    valid_output_dir = final_output_root / "valid"
    (train_output_dir / "images").mkdir(parents=True)
    (train_output_dir / "labels").mkdir(parents=True)
    (valid_output_dir / "images").mkdir(parents=True)
    (valid_output_dir / "labels").mkdir(parents=True)
    
    def replicate_and_copy(replicated_list, target_dir):
        target_img_dir = target_dir / "images"
        target_lbl_dir = target_dir / "labels"
        
        for original_path, new_stem in replicated_list:
            original_stem = original_path.stem
            original_image_path = image_dir / f"{original_stem}.jpg"
            
            if original_image_path.exists() and original_path.exists():
                new_image_name = f"{new_stem}.jpg"
                new_label_name = f"{new_stem}.txt"
                shutil.copy(original_image_path, target_img_dir / new_image_name)
                shutil.copy(original_path, target_lbl_dir / new_label_name)
                
    replicate_and_copy(train_replicated, train_output_dir)
    replicate_and_copy(valid_replicated, valid_output_dir)

    print(f"\n✅ [2.5단계] 데이터셋 생성 완료! 출력 경로: {final_output_root}")
    return final_output_root


# =======================
# [2.6단계] 오버샘플링 - 결함데이터만
# =======================

def oversample_tiles_for_2_loops(tile_root: Path = cfg.TILE_ROOT, final_root: Path = cfg.FINAL_ROOT, train_ratio: float = 0.8) -> Path:

    label_dir = tile_root / "labels"
    image_dir = tile_root / "images"
    num_loops = 2

    if final_root.exists():
        shutil.rmtree(final_root)

    all_label_files: List[Path] = list(label_dir.glob("*.txt"))

    class_0_files: List[Path] = []
    class_1_files: List[Path] = []

    for label_path in all_label_files:
        content = label_path.read_text().strip()
        if not content:
            continue 

        classes = {int(line.split()[0]) for line in content.split('\n') if line}
        has_class_1 = 1 in classes
        has_class_0 = 0 in classes

        if has_class_1:
            class_1_files.append(label_path)
        elif has_class_0:
            class_0_files.append(label_path)
        
    final_replicated_list: List[Tuple[Path, str]] = []

    random.shuffle(class_1_files)
    num_class_1 = len(class_1_files)
    half_class_1 = num_class_1 // 2
    
    for i in range(num_loops):
        print(f"--- [2.6단계] 오버샘플링 루프 {i+1}/{num_loops} ---")
        
        for label_path in class_0_files:
            new_name = f"{label_path.stem}_d0_{i+1}" 
            final_replicated_list.append((label_path, new_name))
        
        if i == 0:
            selected_class_1 = class_1_files[:half_class_1]
        else: # i == 1
            selected_class_1 = class_1_files[half_class_1:]

        for label_path in selected_class_1:
            new_name = f"{label_path.stem}_d1_{i+1}"
            final_replicated_list.append((label_path, new_name))
            
    random.shuffle(final_replicated_list)
    
    num_total = len(final_replicated_list)
    num_train = int(num_total * train_ratio)
    
    train_replicated = final_replicated_list[:num_train]
    valid_replicated = final_replicated_list[num_train:]

    print(f"\n--- [2.6단계] 최종 Train/Valid 분할 (총 {num_total}개 파일) ---")
    print(f"Train 셋 (복제): {len(train_replicated)}개 파일 ({train_ratio*100:.0f}%)")
    print(f"Valid 셋 (복제): {len(valid_replicated)}개 파일 ({(1-train_ratio)*100:.0f}%)")

    train_output_dir = final_root / "train"
    valid_output_dir = final_root / "valid"
    (train_output_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (valid_output_dir / "images").mkdir(parents=True, exist_ok=True)
    (valid_output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    def replicate_and_copy(replicated_list, target_dir):
        target_img_dir = target_dir / "images"
        target_lbl_dir = target_dir / "labels"
        
        for original_path, new_stem in replicated_list:
            original_stem = original_path.stem
            original_image_path = image_dir / f"{original_stem}.jpg"
            
            if original_image_path.exists() and original_path.exists():
                new_image_name = f"{new_stem}.jpg"
                new_label_name = f"{new_stem}.txt"
                
                shutil.copy(original_image_path, target_img_dir / new_image_name)
                shutil.copy(original_path, target_lbl_dir / new_label_name)
                
    replicate_and_copy(train_replicated, train_output_dir)
    replicate_and_copy(valid_replicated, valid_output_dir)

    print(f"\n✅ [2.6단계] 데이터셋 생성 완료! 출력 경로: {final_root}")
    return final_root

# =======================
# [3단계] 오버샘플링 - 결함 + 정상 2000개 (Baseline)
# =======================

def create_balanced_baseline_splits(num_empty_to_use: int = 2000, train_ratio: float = 0.8):
    """
    결함 타일은 그대로, 정상 타일은 num_empty_to_use 수량만큼만 랜덤 추출하여
    Train/Valid로 분할하고 final_root에 복사합니다. (Baseline용)
    """
    import config as cfg # 함수 내에서 config 사용을 위해 다시 임포트
    
    tile_root = cfg.TILE_ROOT
    final_root = cfg.FINAL_ROOT
    
    label_dir = tile_root / "labels"
    image_dir = tile_root / "images"
    
    if not label_dir.exists():
        print(f"❌ 오류: 타일 라벨 폴더를 찾을 수 없습니다: {label_dir}")
        return

    all_label_paths = list(label_dir.glob("*.txt"))

    defect_paths = [] # 결함이 있는 타일 (라벨 파일 내용 있음)
    empty_paths = []  # 결함이 없는 타일 (라벨 파일 내용 없음)
    
    # --- 타일 분리 ---
    for path in all_label_paths:
        content = path.read_text().strip()
        if content:
            defect_paths.append(path)
        else:
            empty_paths.append(path)

    # --- 1. 결함 파일 (오버샘플링 없이 그대로 사용) ---
    # (original_path, new_stem) 튜플 리스트로 변환. new_stem은 원본 이름과 동일.
    all_defect_paths = [(path, path.stem) for path in defect_paths] 
    
    # --- 2. 정상 파일 수량 조정 ---
    # 사용자가 지정한 수량만큼 랜덤 추출
    random.shuffle(empty_paths)
    
    # 추출할 수량이 실제 정상 타일 수보다 많으면 전체를 사용
    num_to_select = min(num_empty_to_use, len(empty_paths))
    selected_empty_paths = empty_paths[:num_to_select]
    
    # 정상 파일은 복제하지 않으므로 new_stem도 original_stem과 동일
    all_empty_paths = [(path, path.stem) for path in selected_empty_paths]

    # --- 3. 최종 학습 목록 구성 및 분할 ---
    final_paths_to_split = all_defect_paths + all_empty_paths
    random.shuffle(final_paths_to_split) 

    split_index = int(len(final_paths_to_split) * train_ratio)
    
    train_split = final_paths_to_split[:split_index]
    valid_split = final_paths_to_split[split_index:]

    print(f"\n--- [Stage 3] Baseline 데이터 분할 결과 ---")
    print(f"결함 타일 (사용): {len(defect_paths)}개")
    print(f"정상 타일 (추출/사용): {len(selected_empty_paths)}개 (요청 수량: {num_empty_to_use}개)")
    print(f"총 학습 데이터: {len(train_split) + len(valid_split)}개")
    print(f"Train/Valid 분할: {len(train_split)}개 / {len(valid_split)}개 (Train Ratio: {train_ratio*100:.0f}%)")
    
    # --- 4. 파일 복사 및 저장 ---
    train_output_dir = final_root / "train"
    valid_output_dir = final_root / "valid"

    # 출력 디렉토리 생성
    (train_output_dir / "images").mkdir(parents=True, exist_ok=True)
    (train_output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (valid_output_dir / "images").mkdir(parents=True, exist_ok=True)
    (valid_output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    def copy_splits(split_list, target_dir):
        target_img_dir = target_dir / "images"
        target_lbl_dir = target_dir / "labels"
        
        for original_path, new_stem in split_list:
            original_stem = original_path.stem
            
            # 이미지 파일 경로 추정
            original_image_path = image_dir / f"{original_stem}.jpg"
            
            # 파일 존재 확인 및 복사
            if original_image_path.exists() and original_path.exists():
                new_image_name = f"{new_stem}.jpg"
                new_label_name = f"{new_stem}.txt"
                
                shutil.copy2(original_image_path, target_img_dir / new_image_name)
                shutil.copy2(original_path, target_lbl_dir / new_label_name)

    print("\n✅ Train 데이터셋 복사 및 정리 중...")
    copy_splits(train_split, train_output_dir)
    print("✅ Valid 데이터셋 복사 및 정리 중...")
    copy_splits(valid_split, valid_output_dir)
    
    print(f"\n✅ 데이터 준비 완료. 최종 데이터셋: {final_root}")