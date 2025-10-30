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

def stage2_tile_all_with_sahi(keep_empty: bool = True, min_side_px: int = 2, min_intersection_ratio: float = 0.2):
    
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

        exts = (".jpg",".jpeg",".png",".bmp",".tif",".tiff")
        for ip in sorted(src_img.iterdir()):
            if ip.suffix.lower() not in exts:
                continue

            im = Image.open(ip).convert("RGB")
            W, H = im.size
            
            # SAHI 파라미터 계산
            slice_h, slice_w, ovh, ovw = compute_slice_params(W, H, sahi_cfg) 

            sliced_list = slice_image(
                image=im,
                slice_height=slice_h,
                slice_width=slice_w,
                overlap_height_ratio=ovh,
                overlap_width_ratio=ovw
            )

            # 원본 YOLO 라벨(px 좌표로 변환)
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
                        bx = cx*W; by = cy*H; bw = ww*W; bh = hh*H
                        x1 = bx - bw/2; y1 = by - bh/2
                        x2 = bx + bw/2; y2 = by + bh/2
                        boxes.append((cls, x1,y1,x2,y2, bw*bh))  # (cls, x1,y1,x2,y2, area)

            # 각 슬라이스 저장 + 라벨 클리핑/정규화
            for si in sliced_list:
                
                x0, y0 = si["starting_pixel"]
                # tw_slice, th_slice = si["image"].size 로 인한 에러 대신 slice_w, slice_h 사용
                tw_slice, th_slice = slice_w, slice_h 

                x1t, y1t = x0 + tw_slice, y0 + th_slice
                
                # 원본 이미지에서 타일 자르기
                crop = im.crop((x0, y0, x1t, y1t))
                tw, th = crop.size 

                new_lines = []
                for (cls, x1,y1,x2,y2, area) in boxes:
                    ix1 = max(x1, x0); iy1 = max(y1, y0)
                    ix2 = min(x2, x1t); iy2 = min(y2, y1t)
                    iw = ix2 - ix1; ih = iy2 - iy1
                    if iw <= 0 or ih <= 0:
                        continue
                    if iw < min_side_px or ih < min_side_px:
                        continue
                    if (iw*ih) / (area + 1e-6) < min_intersection_ratio:
                        continue

                    cx_t = ((ix1 + ix2)/2 - x0) / tw
                    cy_t = ((iy1 + iy2)/2 - y0) / th
                    w_t  = iw / tw
                    w_t = min(max(w_t, 1e-6), 1.0)
                    h_t  = ih / th
                    h_t = min(max(h_t, 1e-6), 1.0)

                    if w_t <= 0 or h_t <= 0:
                        continue
                    cx_t = min(max(cx_t, 0.0), 1.0)
                    cy_t = min(max(cy_t, 0.0), 1.0)

                    new_lines.append(f"{cls} {cx_t:.6f} {cy_t:.6f} {w_t:.6f} {h_t:.6f}")

                tile_name = f"{ip.stem}_{x0}_{y0}"
                crop.save(dst_img_root / f"{tile_name}.jpg", quality=95)
                with open(dst_lbl_root / f"{tile_name}.txt", "w") as f:
                    if new_lines or keep_empty:
                        f.write("\n".join(new_lines))

    print("\n=== [2단계] SAHI 타일 분할 시작 (단일 출력 모드) =====")
    
    # 설정에서 경로 가져오기
    tile_one_split(cfg.CROP_TRAIN)
    tile_one_split(cfg.CROP_VAL)
    tile_one_split(cfg.CROP_TEST)
    
    print(f"[2단계 완료] 타일 데이터셋 root: {cfg.TILE_ROOT}")
    print(f" - 출력 폴더: {dst_img_root.parent}")
    print(f" - 모드: {sahi_cfg['SPLIT_FLAG']}, 값: {sahi_cfg['SPLIT_VALUE']}")
    print(f" - overlap_h: {sahi_cfg['overlap_h']}, overlap_w: {sahi_cfg['overlap_w']}")


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