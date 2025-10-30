import torch
from pathlib import Path
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

def ensure_dir(p: Path): # 디렉토리가 없으면 생성
    p.mkdir(parents=True, exist_ok=True)

def list_images(dir_path: Path) -> List[Path]: # 디렉토리 내 모든 이미지 파일 경로를 반환
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files: List[Path] = []
    for e in exts:
        files.extend(dir_path.glob(e))
    return sorted(files)

def device_str() -> str: # 사용 가능한 디바이스 문자열 반환 (예: "cuda:0" 또는 "cpu")
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def compute_slice_params(W: int, H: int, cfg: dict) -> Tuple[int, int, float, float]:
    """
    SAHI 설정(cfg)의 SPLIT_FLAG/SPLIT_VALUE에 따라 
    slice_height/width와 overlap 비율 반환
    """
    flag = cfg.get("SPLIT_FLAG", "size")
    val  = int(cfg.get("SPLIT_VALUE", 640))
    ovh  = float(cfg.get("overlap_h", 0.25))
    ovw  = float(cfg.get("overlap_w", 0.25))

    if flag == "count_v":  # 세로 등분
        slice_h = max(1, H // max(1, val))
        slice_w = W
        return slice_h, slice_w, ovh, 0.0
    else:                  # "size": 정사각 타일
        slice_h = val
        slice_w = val
        return slice_h, slice_w, ovh, ovw

def measure_text(draw: ImageDraw.Draw, text: str, font: ImageFont = None) -> Tuple[int, int]:
    """Pillow Draw 객체에서 텍스트의 너비와 높이를 계산"""
    # Pillow >= 8.0
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # Fallbacks
    if font is not None and hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    if font is not None and hasattr(font, "getsize"):
        return font.getsize(text)
    # 아주 보수적인 최후의 추정
    return (max(1, int(0.6 * 12 * len(text))), 12)