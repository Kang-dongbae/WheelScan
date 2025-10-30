# application.py
import os
import re
import cv2
import pandas as pd
from typing import List, Tuple, Optional, Dict

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from os import PathLike

# -------------------- 페이지 설정 --------------------
st.set_page_config(page_title="KORAIL Wheel Defect Detection", layout="wide", initial_sidebar_state="expanded")

# -------------------- Config 기본값 --------------------
try:
    from src.config import WHEEL_MODEL as CFG_WHEEL_MODEL
except Exception:
    CFG_YOLO_MODEL = "models/best.pt"  # 기본 경로 가정
try:
    from src.config import DEFECT_MODEL as CFG_YOLO_MODEL
except Exception:
    CFG_YOLO_MODEL = "models/best.pt"  # 기본 경로 가정

try:
    from src.config import TEST_IMAGES as CFG_TEST_FOLDER
except Exception:
    CFG_TEST_FOLDER = "./data/test/images"

try:
    from src.config import ALLOWED_DEFECTS as CFG_ALLOWED_DEFECTS
except Exception:
    CFG_ALLOWED_DEFECTS = ['wheel', 'spalling', 'flat']

try:
    from src.config import BATCH_SIZE as CFG_BATCH_SIZE
except Exception:
    CFG_BATCH_SIZE = 3  # 화면(좌3·우3)에 맞춰 3개 재생 단위

try:
    from src.config import TICK_MS as CFG_TICK_MS
except Exception:
    CFG_TICK_MS = 2000  # UI에서 제거, 코드 상수로만 관리

IMG_SIZE = 1280
DISPLAY_W, DISPLAY_H = 200, 300    # 썸네일 내부 리사이즈
AUTO_REFRESH_KEY = "__auto_refresh__"

# ✅ 컬럼 구성: Select, Position, Wheel ID, ...
DISPLAY_COLS = ['Position', 'Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time']

# -------------------- 외부 모듈 --------------------
from src.appDefect import (
    Detector,
    list_images,
    find_image_by_stem,
)

# -------------------- 스타일 --------------------
# -------------------- 스타일 --------------------
st.markdown("""
<style>
/* Streamlit 메인 콘텐츠 영역의 기본 상단 패딩을 제거 */
div.block-container {
    padding-top: 1.5rem; /* 기본값 3rem에서 1.5rem으로 줄이거나, 0rem으로 설정 */
    padding-bottom: 0rem;
}
/* Streamlit 헤더 영역 제거 (선택 사항: 'Deploy' 버튼 숨김) */
header {
    visibility: hidden;
    height: 0%;
}
.main-title{font-size:32px;font-weight:800;color:#003087;text-align:center;margin:0px 0 20px 0;}
.camera-box{border:1px solid #e5e7eb;border-radius:8px;background:#fff;height:300px;width:100%; margin-bottom:20px;}
.subtle{color:#6b7280;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="main-title">코레일 차륜 결함 검지 시스템(WheelNet AI, 5팀)</div>', unsafe_allow_html=True)
# -------------------- 세션 기본값 --------------------
defaults = dict(
    detector=None,
    model_path="",
    conf_threshold=0.5,

    images_all=[],
    pairs=[],
    left_paths=[],
    right_paths=[],

    preds_cache={},                 # 표시용(기존)
    preds_cache_raw={},             # 원본 rows 캐시(추가)

    cursor=0,
    cursorL=0,
    cursorR=0,
    is_running=False,
    preloaded=False,

    # 누적 테이블
    defects_table=None,             # flat/spalling (wheel 내부일 때만), 이미지당 1행
    wheels_table=None,              # wheel만 있고 flat/spalling이 전혀 없는 이미지

    batch_size=int(CFG_BATCH_SIZE),
    tick_ms=int(CFG_TICK_MS),
    test_folder="",

    # 선택 상태
    active_source=None,             # 'defects' | 'wheels' | None
    defects_select=None,
    wheels_select=None,
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- 유틸 --------------------
def resolve_abs_path(p: str) -> str:
    if p is None:
        p = ""
    if isinstance(p, (PathLike,)):
        p = os.fspath(p)
    p = os.path.expanduser((str(p) or "").strip())
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(os.getcwd(), p))

def rows_to_display_df(rows: list, path: str) -> pd.DataFrame:
    base_cols = ['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time']
    if not rows:
        return pd.DataFrame(columns=base_cols)
    df = pd.DataFrame([r.__dict__ for r in rows])
    if not df.empty:
        df['Cropped Count'] = len(st.session_state.cropped_cache.get(path, []))

    df = df.rename(columns={
        'wheel_id': 'Wheel ID',
        'defect_type': 'Defect Type',
        'confidence': 'Confidence',
        'area_ratio': 'Area Ratio',
        'time': 'Time',
    })
    for c in base_cols:
        if c not in df.columns:
            df[c] = "" if c in ('Wheel ID','Defect Type','Time') else 0.0
    df = df[base_cols]
    df['Wheel ID'] = df['Wheel ID'].astype(str)
    df['Defect Type'] = df['Defect Type'].astype(str)
    df['Time'] = df['Time'].astype(str)
    df[['Confidence','Area Ratio']] = df[['Confidence','Area Ratio']].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def cached_read_resized(abs_path: str, w: int = DISPLAY_W, h: int = DISPLAY_H):
    img = cv2.imread(abs_path)
    if img is None:
        return None
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def make_lr_pairs(paths: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    L, R = {}, {}
    for p in paths:
        name = os.path.basename(p)
        stem, _ = os.path.splitext(name)
        s = (stem or "").strip()
        if not s:
            continue
        if s[0] in ('L','l'):
            key = (s[1:] or s).lower()
            L[key] = p
        elif s[0] in ('R','r'):
            key = (s[1:] or s).lower()
            R[key] = p
        else:
            L[s.lower()] = p
    keys = sorted(set(L) | set(R))
    return [(L.get(k), R.get(k)) for k in keys]

def stem_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]

def parse_position_and_id(filename: str):
    """파일명에서 Position(Left/Right)과 4자리 Wheel ID를 추출"""
    name = os.path.basename(filename)
    stem, _ = os.path.splitext(name)
    if not stem:
        return "", ""
    pos = ""
    first_char = stem[0].upper()
    if first_char == "L":
        pos = "Left"
    elif first_char == "R":
        pos = "Right"
    m = re.search(r"[LRlr](\d{4})", stem)
    wheel_num = m.group(1) if m else stem
    return pos, wheel_num

def compute_wheel_and_crops_once(p: str):
    """해당 이미지에 대해 최초 한 번만 wheel 예측과 크롭을 수행하고 캐시에 저장"""
    if p in st.session_state.cropped_cache and p in st.session_state.preds_cache_raw:
        return  # 이미 계산됨

    print(f"[CROP] Running real-time crop for {os.path.basename(p)}", flush=True)
    det = st.session_state.wheel_detector
    results = det.model.predict(p, imgsz=IMG_SIZE, conf=st.session_state.conf_threshold)
    boxes = results[0].boxes

    # rows(=wheel rows) 계산
    rows = det.predict_rows(p, st.session_state.conf_threshold)
    wheel_rows = [r for r in rows if str(r.defect_type).lower() == 'wheel']

    # wheel_id 보정
    _, wheel_id = parse_position_and_id(p)
    for r in wheel_rows:
        if not getattr(r, 'wheel_id', None):
            r.wheel_id = wheel_id

    # 크롭 생성 (wheel 클래스만)
    original = cv2.imread(p)
    crops = []
    crop_meta = []

    if original is not None and boxes is not None:
        for box in boxes:
            if int(box.cls) == 0:  # wheel index (전용 모델이라 0 가정)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = original[y1:y2, x1:x2]
                if crop is not None and crop.size > 0:
                    crops.append(crop)
                    crop_meta.append({'xyxy': (x1, y1, x2, y2)})

    # 캐시 저장
    st.session_state.preds_cache_raw[p] = (
        pd.DataFrame([r.__dict__ for r in wheel_rows]) if wheel_rows else pd.DataFrame()
    )
    st.session_state.cropped_cache[p] = crops
    st.session_state.crop_meta[p] = crop_meta
    print(f"[CROP] {os.path.basename(p)} → {len(crops)} crops done", flush=True)

import tempfile
import numpy as np

def compute_and_update_defects_for_path(p: str):
    """
    p의 크롭들에 대해 defect 모델을 실행하고,
    flat/spalling 박스를 원본 좌표로 변환해 abs_defect_boxes[p]에 누적 저장.
    또한 최고 conf 1건으로 defects_table을 upsert.
    """
    done_set = st.session_state.setdefault("defects_done", set())
    if p in done_set:
        return

    crops = st.session_state.cropped_cache.get(p, [])
    metas = st.session_state.crop_meta.get(p, [])
    if not crops or not metas:
        return

    det2 = st.session_state.defect_detector
    names = getattr(det2.model, 'names', None)
    label_by_idx = (lambda i: names.get(int(i), str(int(i))) if isinstance(names, dict) else str(int(i)))

    abs_boxes = []   # (x1,y1,x2,y2,label,conf)
    best = None      # (conf, area, label, time)

    wheel_df = st.session_state.preds_cache_raw.get(p, pd.DataFrame())
    wheel_time = ""
    if not wheel_df.empty and "time" in wheel_df.columns and not wheel_df["time"].isna().all():
        wheel_time = str(wheel_df["time"].iloc[0])

    for idx, crop in enumerate(crops):
        wx1, wy1, wx2, wy2 = metas[idx]['xyxy']  # ✅ 원본 내 휠 오프셋
        crop_h, crop_w = crop.shape[:2]
        crop_area = float(crop_w * crop_h) if crop_w > 0 and crop_h > 0 else 1.0  # ✅ 분모 보호

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tmp_path = tf.name
            cv2.imwrite(tmp_path, crop)

        try:
            r = det2.model.predict(tmp_path, imgsz=IMG_SIZE, conf=st.session_state.conf_threshold)[0]
            bxs = getattr(r, "boxes", None)
            if bxs is None:
                continue

            # YOLO는 입력(크롭) 사이즈 기준 좌표를 돌려줌 → 오프셋만 더하면 원본 좌표
            cls = bxs.cls.cpu().numpy().astype(int) if hasattr(bxs.cls, 'cpu') else np.array(bxs.cls).astype(int)
            confs = bxs.conf.cpu().numpy() if hasattr(bxs.conf, 'cpu') else np.array(bxs.conf)
            xyxy = bxs.xyxy.cpu().numpy().astype(int) if hasattr(bxs.xyxy, 'cpu') else np.array(bxs.xyxy).astype(int)

            for j in range(len(xyxy)):
                lbl = label_by_idx(cls[j]).lower()
                if lbl not in ("flat", "spalling"):
                    continue
                dx1, dy1, dx2, dy2 = xyxy[j].tolist()
                # ✅ 원본 좌표로 변환
                ax1, ay1 = wx1 + dx1, wy1 + dy1
                ax2, ay2 = wx1 + dx2, wy1 + dy2
                c = float(confs[j])

                defect_area = float(max(0, (dx2 - dx1))) * float(max(0, (dy2 - dy1)))
                area_ratio = round(defect_area / crop_area, 6)

                abs_boxes.append((ax1, ay1, ax2, ay2, lbl, c))

                # defects_table 업데이트용 최고 conf 추적 (area는 없으면 0으로)
                cand = (c, area_ratio, lbl, wheel_time)
                if (best is None) or (c > best[0]):
                    best = cand
        finally:
            try: os.remove(tmp_path)
            except Exception: pass

    # 원본 좌표 결함 박스 누적 저장
    st.session_state.abs_defect_boxes[p] = abs_boxes

    # 최고 conf 1건을 defects_table에 upsert
    if best is not None:
        pos, wheel_short_id = parse_position_and_id(p)
        conf, area, dtyp, tstr = best
        defect_row = {
            "Select": False,
            "Position": pos,
            "Wheel ID": wheel_short_id,
            "Defect Type": dtyp,
            "Confidence": conf,
            "Area Ratio": area,
             "Time": tstr,
        }
        df = st.session_state.get("defects_table")
        st.session_state.defects_table = _upsert_by_image_key(df, defect_row)

    print(f"[DEFECT-MAP] {os.path.basename(p)} → {len(abs_boxes)} mapped boxes", flush=True)
    done_set.add(p)

def draw_with_yolo_builtin(path: str, conf: float = 0.25):
    """
    YOLO 내장 Results.plot()을 사용해 원본 이미지 위에
    1) wheel 모델 결과, 2) defect 모델 결과를 순서대로 그립니다.
    라벨과 신뢰도(conf) 표시는 YOLO가 기본 제공.
    """
    img = cv2.imread(path)
    if img is None:
        return None

    # 1) Wheel 모델
    try:
        r_w = st.session_state.wheel_detector.model.predict(path, imgsz=IMG_SIZE, conf=conf)[0]
        # YOLO 내장 plot: img=img 로 넘기면 해당 이미지 위에 그려줌
        img = r_w.plot(img=img)  # 기본적으로 클래스명 + conf 표시
    except Exception as e:
        print(f"[PLOT] wheel plot skipped: {e}", flush=True)

    # 2) Defect 모델
    try:
        r_d = st.session_state.defect_detector.model.predict(path, imgsz=IMG_SIZE, conf=conf)[0]

        # (선택) 결함만 보고 싶다면 아래 3줄 주석 해제 (flat/spalling만 남김)
        # names = [n.lower() for n in getattr(st.session_state.defect_detector.model, 'names', {}).values()]
        # keep = [i for i, n in enumerate(names) if n in ('flat', 'spalling')]
        # r_d = r_d[np.isin(r_d.boxes.cls.cpu().numpy().astype(int), keep)]

        img = r_d.plot(img=img)
    except Exception as e:
        print(f"[PLOT] defect plot skipped: {e}", flush=True)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def draw_overlays_from_cache(path: str):
    """
    - 휠: YOLO 모델의 기본 .plot() 결과 사용 (confidence/색상 자동)
    - 결함: abs_defect_boxes 캐시에서 직접 그려줌
    """
    wheel_det = st.session_state.get("wheel_detector", None)
    if wheel_det is None:
        return None

    try:
        # ✅ YOLO 기본 plot() 사용 → 휠 박스 자동 표시
        res = wheel_det.model.predict(path, conf=st.session_state.conf_threshold)
        img = res[0].plot()   # YOLO가 휠 박스 + confidence 표시
    except Exception as e:
        print(f"[WARN] wheel plot failed: {e}")
        img = cv2.imread(path)

    if img is None:
        return None

    #defect_boxes = st.session_state.abs_defect_boxes.get(path, [])
    #for (x1, y1, x2, y2, label, conf) in defect_boxes:
    #    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 16)
    #    cv2.putText(img, f"{label} {conf:.2f}", (x1, max(0, y1 - 6)),
    #                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 8, cv2.LINE_AA)

    # YOLO의 plot()은 BGR → RGB 변환 필요
    defect_boxes = st.session_state.abs_defect_boxes.get(path, [])
    for (x1, y1, x2, y2, label, conf) in defect_boxes:
        # 박스
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 16)

        # 라벨 텍스트 내용
        text = f"{label} {conf:.2f}"

        # 텍스트 크기 계산 (배경 크기용)
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 8)
        # 좌표 약간 위로 이동
        y_text = max(y1 - 10, th + 2)

        # ✅ 빨간 배경 박스 (채움)
        cv2.rectangle(img,
                    (x1, y_text - th - baseline),
                    (x1 + tw, y_text + baseline // 2),
                    (0, 0, 255),
                    -1)

        # ✅ 흰색 글씨
        cv2.putText(img, text, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 8, cv2.LINE_AA)


    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# -------------------- 사이드바 --------------------
with st.sidebar:
    st.image("korail_logo.png", width="stretch")
    st.header("Settings")
    #in_model = st.text_input("YOLO model path", value=CFG_YOLO_MODEL)
    in_wheel_model = st.text_input("Wheel Model", value=CFG_WHEEL_MODEL)  # 휠 전용 모델 경로 (기본값 적절히)
    in_defect_model = st.text_input("Defect Model", value=CFG_YOLO_MODEL)
    
    in_folder = st.text_input("Test folder", value=CFG_TEST_FOLDER)
    in_conf = st.slider("Confidence", 0.0, 1.0, st.session_state.conf_threshold, 0.01)

    colA, colB = st.columns(2)
    with colA:
        btn_run = st.button("Run Detection", type="primary")
    with colB:
        btn_reset = st.button("Reset")

st.session_state.conf_threshold = float(in_conf)

if btn_reset:
    for k in list(st.session_state.keys()):
        if k in defaults:
            st.session_state[k] = defaults[k]
    st.success("State reset.")

# -------------------- 모델/데이터 지연 로딩 --------------------
def ensure_loaded(wheel_model_input: str, defect_model_input: str, folder_input: str) -> bool:
    if 'cropped_cache' not in st.session_state:
        st.session_state.cropped_cache = {}

    wheel_mp = resolve_abs_path(wheel_model_input)
    defect_mp = resolve_abs_path(defect_model_input)
    dp = resolve_abs_path(folder_input)

    if not os.path.exists(wheel_mp):
        st.error(f"휠 모델 파일이 없습니다: {wheel_mp}")
        return False
    if not os.path.exists(defect_mp):
        st.error(f"결함 모델 파일이 없습니다: {defect_mp}")
        return False
    if not os.path.isdir(dp):
        st.error(f"테스트 폴더가 없습니다: {dp}")
        return False

    paths = list_images(dp)
    if not paths:
        st.warning(f"테스트 폴더에 이미지가 없습니다: {dp}")
        return False
    pairs = make_lr_pairs(paths)

    left_paths, right_paths = [], []
    for Lp, Rp in pairs:
        if Lp: left_paths.append(Lp)
        if Rp: right_paths.append(Rp)

    # 모델 로드만 (예측/크롭 X)
    wheel_detector = Detector(wheel_mp, ["wheel"], imgsz=IMG_SIZE, wheel_classes=["wheel"])
    defect_detector = Detector(defect_mp, CFG_ALLOWED_DEFECTS, imgsz=IMG_SIZE)
    if hasattr(wheel_detector, "warmup"):
        wheel_detector.warmup(st.session_state.conf_threshold)

    st.session_state.wheel_detector = wheel_detector
    st.session_state.defect_detector = defect_detector
    st.session_state.images_all = paths
    st.session_state.pairs = pairs
    st.session_state.left_paths = left_paths
    st.session_state.right_paths = right_paths
    st.session_state.test_folder = dp
    st.session_state.flat = [*left_paths, *right_paths]

    # ✅ 캐시 초기화만
    st.session_state.preds_cache = {}
    st.session_state.preds_cache_raw = {}
    st.session_state.cropped_cache = {}

    st.session_state.crop_meta = {}          # {path: [{'xyxy': (x1,y1,x2,y2)} , ...]}
    st.session_state.abs_defect_boxes = {}   # {path: [(x1,y1,x2,y2,label,conf), ...]}

    st.session_state.cursor = 0
    st.session_state.cursorL = 0
    st.session_state.cursorR = 0
    st.session_state.preloaded = True
    st.session_state.is_running = True

    st.session_state.defects_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])
    st.session_state.wheels_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])

    st.session_state.active_source = None
    st.session_state.defects_select = None
    st.session_state.wheels_select = None

    #st.success("이미지 로드 완료. 재생 시 실시간 크롭합니다.")
    return True


if btn_run:
    ensure_loaded(in_wheel_model, in_defect_model, in_folder)

# -------------------- 집계 로직 --------------------
TARGET_DEFECTS = {'flat', 'spalling'}

def _aggregate_for_path(p: str):
    raw = st.session_state.preds_cache_raw.get(p, pd.DataFrame())
    if raw is None or raw.empty:
        return None, None

    # 결측 방어 (그대로)

    image_key = stem_from_path(p)
    pos, wheel_short_id = parse_position_and_id(p)

    types = raw['defect_type'].astype(str).str.lower().fillna('')
    has_wheel = (types == 'wheel').any()
    present_targets = set()  # 이 단계에서는 결함 없음

    defect_row = None
    wheel_row = None

    if has_wheel:
        subw = raw[raw['defect_type'].astype(str).str.lower() == 'wheel'].copy()
        conf_w = float(subw['confidence'].max(skipna=True) if 'confidence' in subw.columns else 0.0)
        area_w = float(subw['area_ratio'].max(skipna=True) if 'area_ratio' in subw.columns else 0.0)
        time_w = str(subw['time'].astype(str).fillna('').iloc[0] if 'time' in subw.columns and not subw.empty else "")
        wheel_row = {
            "Select": False,
            "Position": pos,
            "Wheel ID": wheel_short_id,
            "Defect Type": "wheel",
            "Confidence": conf_w,
            "Area Ratio": area_w,
            "Time": time_w,
        }

    return defect_row, wheel_row  # defect_row는 None

def _upsert_by_image_key(df: pd.DataFrame, row: Optional[dict]) -> pd.DataFrame:
    """Wheel ID(숫자 4자리)를 키로 upsert. (동일 이미지 기준 중복 방지)"""
    if row is None:
        return df
    if df is None or df.empty:
        return pd.DataFrame([row])
    mask = (df["Wheel ID"].astype(str) == str(row["Wheel ID"]))
    if mask.any():
        idx = df.index[mask][0]
        for k, v in row.items():
            df.at[idx, k] = v
        df.at[idx, "Select"] = False
        return df
    else:
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# -------------------- 진행/오토리프레시 --------------------
def advance_once():
    if not st.session_state.is_running:
        return ([], []), True

    L = st.session_state.left_paths or []
    R = st.session_state.right_paths or []
    cL = st.session_state.get("cursorL", 0)
    cR = st.session_state.get("cursorR", 0)

    take = 3
    left_batch  = L[cL:cL+take]
    right_batch = R[cR:cR+take]

    # 표는 사용 전에 보장 초기화
    if st.session_state.get("defects_table") is None:
        st.session_state.defects_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])
    if st.session_state.get("wheels_table") is None:
        st.session_state.wheels_table = pd.DataFrame(columns=["Select", *DISPLAY_COLS])

    # ✅ 이번 배치만 실시간 예측/크롭 + 테이블 갱신(단 한 번)
    for p in [*left_batch, *right_batch]:
        compute_wheel_and_crops_once(p)            # wheel 크롭
        defect_row, wheel_row = _aggregate_for_path(p)
        # wheel 테이블 반영
        st.session_state.wheels_table  = _upsert_by_image_key(st.session_state.wheels_table, wheel_row)
        # 👉 크롭 기반 defect 추론 & defects_table 반영
        compute_and_update_defects_for_path(p)

    # 커서 이동
    st.session_state.cursorL = cL + len(left_batch)
    st.session_state.cursorR = cR + len(right_batch)

    finished = (st.session_state.cursorL >= len(L)) and (st.session_state.cursorR >= len(R))
    if finished:
        st.session_state.is_running = False

    return (left_batch, right_batch), finished


if st.session_state.is_running and st.session_state.preloaded:
    st_autorefresh(interval=st.session_state.tick_ms, key=AUTO_REFRESH_KEY)

# -------------------- 실시간 L/R(각 3장) --------------------
with st.container(border=True):
    left_col, right_col = st.columns(2)

    if st.session_state.is_running:
        (left_batch, right_batch), finished = advance_once()
    else:
        (left_batch, right_batch), finished = ([], []), (
            (st.session_state.get("cursorL", 0) >= len(st.session_state.get("left_paths", [])))
            and (st.session_state.get("cursorR", 0) >= len(st.session_state.get("right_paths", [])))
        )

    COLS = 3

    with st.spinner("Loading batch..."):
        left_imgs  = [cached_read_resized(p) if p else None for p in left_batch]
        right_imgs = [cached_read_resized(p) if p else None for p in right_batch]

    with left_col:
        st.markdown("### 좌측선로")
        cols = st.columns(COLS)
        for i, ct in enumerate(cols):
            with ct:
                img = left_imgs[i] if i < len(left_imgs) else None
                if img is not None:
                    st.image(img, width="stretch")
                else:
                    st.markdown('<div class="camera-box"></div>', unsafe_allow_html=True)

    with right_col:
        st.markdown("### 우측선로")
        cols = st.columns(COLS)
        for i, ct in enumerate(cols):
            with ct:
                img = right_imgs[i] if i < len(right_imgs) else None
                if img is not None:
                    st.image(img, width="stretch")
                else:
                    st.markdown('<div class="camera-box"></div>', unsafe_allow_html=True)

    if finished and st.session_state.preloaded:
        st.success("✅ 모든 이미지 재생 완료!")
        if st.button("🔁 다시 재생"):
            st.session_state.cursorL = 0
            st.session_state.cursorR = 0
            st.session_state.is_running = True

st.markdown("---")

# -------------------- 결과 테이블 + 요약 --------------------
st.markdown("### 결함 검지 결과")
left, right = st.columns(2)

def _ensure_table_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        base = pd.DataFrame(columns=["Select", *DISPLAY_COLS])
        return base
    for c in ["Select", *DISPLAY_COLS]:
        if c not in df.columns:
            df[c] = False if c == "Select" else ("" if c in ('Position','Wheel ID','Defect Type','Time') else 0.0)
    df["Select"] = df["Select"].astype(bool)
    df["Position"] = df["Position"].astype(str)
    df["Wheel ID"] = df["Wheel ID"].astype(str)
    df["Defect Type"] = df["Defect Type"].astype(str)
    df["Time"] = df["Time"].astype(str)
    df[["Confidence","Area Ratio"]] = df[["Confidence","Area Ratio"]].apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return df[["Select", *DISPLAY_COLS]].copy()

with left:
    # --- 결함 리스트 ---
    with st.container(border=True):
        df_def = _ensure_table_columns(st.session_state.get("defects_table"))
        flat_count = int(df_def["Defect Type"].str.contains(r"\bflat\b", case=False, na=False).sum())
        spalling_count = int(df_def["Defect Type"].str.contains(r"\bspalling\b", case=False, na=False).sum())
        st.subheader(f"결함 검출 : flat {flat_count}개, spalling {spalling_count}개")

        # 방어: 혹시라도 wheel만 들어왔으면 필터
        if not df_def.empty:
            df_def = df_def[
                df_def["Defect Type"].str.contains("flat", case=False) |
                df_def["Defect Type"].str.contains("spalling", case=False)
            ]
        if df_def.empty:
            st.write("No defects detected yet.")
            st.session_state.defects_select = df_def.copy()
        else:
            edited_def = st.data_editor(
                df_def,
                hide_index=True,
                height=360,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False, help="우측 요약 보기"),
                    "Position": st.column_config.TextColumn("Position", disabled=True),
                    "Wheel ID": st.column_config.TextColumn("Wheel ID", disabled=True),
                    "Defect Type": st.column_config.TextColumn("Defect Type", disabled=True),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f", disabled=True),
                    "Area Ratio": st.column_config.NumberColumn("Area Ratio", format="%.2f", disabled=True),
                    "Time": st.column_config.TextColumn("Time", disabled=True),
                },
                key="defects_editor"
            )
            if "Select" in edited_def.columns and edited_def["Select"].sum() > 1:
                first_idx = edited_def.index[edited_def["Select"]].tolist()[0]
                edited_def.loc[edited_def.index != first_idx, "Select"] = False
            st.session_state.defects_select = edited_def

    # --- 휠 리스트 ---
    with st.container(border=True):
        df_wheels = _ensure_table_columns(st.session_state.get("wheels_table"))
        wheel_only_count = int(len(df_wheels))
        st.subheader(f"차륜 검출 : {wheel_only_count}개")

        # 방어: 결함문자 포함된 행 제거(이론상 없음)
        if not df_wheels.empty:
            mask_has_def = (
                df_wheels["Defect Type"].str.contains("flat", case=False) |
                df_wheels["Defect Type"].str.contains("spalling", case=False)
            )
            df_wheels = df_wheels[~mask_has_def]

        if df_wheels.empty:
            st.write("No wheels (without defects) detected yet.")
            st.session_state.wheels_select = df_wheels.copy()
        else:
            edited_wheels = st.data_editor(
                df_wheels,
                hide_index=True,
                height=280,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False, help="우측 요약 보기"),
                    "Position": st.column_config.TextColumn("Position", disabled=True),
                    "Wheel ID": st.column_config.TextColumn("Wheel ID", disabled=True),
                    "Defect Type": st.column_config.TextColumn("Defect Type", disabled=True),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f", disabled=True),
                    "Area Ratio": st.column_config.NumberColumn("Area Ratio", format="%.2f", disabled=True),
                    "Time": st.column_config.TextColumn("Time", disabled=True),
                },
                key="wheels_editor"
            )
            if "Select" in edited_wheels.columns and edited_wheels["Select"].sum() > 1:
                first_idx = edited_wheels.index[edited_wheels["Select"]].tolist()[0]
                edited_wheels.loc[edited_wheels.index != first_idx, "Select"] = False
            st.session_state.wheels_select = edited_wheels

# --- 상호배타 선택 ---
def _enforce_mutual_exclusive_selection():
    sel_def = st.session_state.get("defects_select")
    sel_whe = st.session_state.get("wheels_select")

    def_count = int(sel_def["Select"].sum()) if (isinstance(sel_def, pd.DataFrame) and "Select" in sel_def.columns) else 0
    whe_count = int(sel_whe["Select"].sum()) if (isinstance(sel_whe, pd.DataFrame) and "Select" in sel_whe.columns) else 0

    active_source = None
    if def_count > 0:
        active_source = 'defects'
        if whe_count > 0:
            st.session_state.wheels_select.loc[:, "Select"] = False
    elif whe_count > 0:
        active_source = 'wheels'
        if def_count > 0:
            st.session_state.defects_select.loc[:, "Select"] = False
    else:
        active_source = None

    st.session_state.active_source = active_source

_enforce_mutual_exclusive_selection()

# --- 우측: 선택 요약/시각화 ---
with right:
    with st.container(border=True):
        st.subheader("결함 상세보기")
        source = st.session_state.get("active_source", None)
        raw_folder = st.session_state.get("test_folder", CFG_TEST_FOLDER)
        folder_to_use = resolve_abs_path(raw_folder)

        if source is None:
            st.info("왼쪽의 결함 리스트 또는 휠 리스트에서 항목을 하나 선택하세요.")
        else:
            sel_df = st.session_state.get("defects_select" if source == 'defects' else "wheels_select", pd.DataFrame())
            if sel_df is None or sel_df.empty or "Select" not in sel_df.columns or sel_df["Select"].sum() == 0:
                st.info("선택된 항목이 없습니다.")
            else:
                row = sel_df[sel_df["Select"]].drop(columns=["Select"]).iloc[0]
                wheel_id_disp = str(row.get("Wheel ID", "") or "")  # 4자리 숫자
                defect_type = str(row.get("Defect Type", "") or "")
                conf = float(row.get("Confidence", 0.0) or 0.0)
                area = float(row.get("Area Ratio", 0.0) or 0.0)
                pos = str(row.get("Position", "") or "")

                # 원본 이미지는 'L####' 또는 'R####...'로 시작하므로
                # find_image_by_stem(folder, '####') 만으로도 startswith 매치 가능
                img_col, info_col = st.columns([2,1])
                with img_col:
                    path = find_image_by_stem(folder_to_use, pos, wheel_id_disp)
                    if path:
                        plotted_rgb = draw_overlays_from_cache(path)
                        if plotted_rgb is not None:
                            #plotted_rgb = cv2.resize(plotted_rgb, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_AREA)
                            st.image(plotted_rgb, width="stretch")
                        else:
                            st.warning("⚠️ Unable to visualize detection")
                    else:
                        st.warning(f"⚠️ Image not found for ID {wheel_id_disp}")
                with info_col:
                    st.markdown(f"**Source:** {('Defect List' if source=='defects' else 'Wheel List')}")
                    st.markdown(f"**Position:** {pos}")
                    st.markdown(f"**Wheel ID:** {wheel_id_disp}")
                    st.markdown(f"**Defect Type:** {defect_type}")
                    st.markdown(f"**Confidence:** {conf:.2f}")
                    st.markdown(f"**Area Ratio:** {area:.2f}")