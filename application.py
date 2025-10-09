# app.py
import os
import cv2
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from ultralytics import YOLO
from streamlit_autorefresh import st_autorefresh

# ===============================================
# (선택) src.config 에서 기본값 가져오기
# ===============================================
try:
    from src.config import YOLO_MODEL as CFG_YOLO_MODEL, TEST_FOLDER as CFG_TEST_FOLDER
except Exception:
    CFG_YOLO_MODEL, CFG_TEST_FOLDER = None, None

# ===============================================
# 페이지 설정
# ===============================================
st.set_page_config(page_title="KORAIL Wheel Defect Detection", layout="wide")

# ===============================================
# CSS
# ===============================================
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #003087;
            text-align: center;
            margin-bottom: 20px;
        }
        .camera-box-content {
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #ffffff;
            height: 250px;
            margin-bottom: 5px;
            width: 100%;
        }
        .centered-text-below {
            text-align: center;
            font-size: 16px;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">KORAIL Wheel Defect Detection System</div>', unsafe_allow_html=True)

# ===============================================
# 사이드바
# ===============================================
with st.sidebar:
    st.header("Settings")
    YOLO_MODEL = st.text_input("YOLO model path or name", value=CFG_YOLO_MODEL or "yolov8n.pt")
    TEST_FOLDER = st.text_input("Test images folder", value=CFG_TEST_FOLDER or "./test_images")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    run_btn_top = st.button("Run Detection on Test Images")

# ===============================================
# 모델 로드 (캐시)
# ===============================================
@st.cache_resource
def load_model(model_path_or_name: str):
    return YOLO(model_path_or_name)

try:
    model = load_model(YOLO_MODEL)
except Exception as e:
    st.error(f"❌ YOLO 모델 로드 실패: {e}")
    st.stop()

# ===============================================
# 세션 상태
# ===============================================
columns_def = ['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time']
if 'defects' not in st.session_state:
    st.session_state.defects = pd.DataFrame(columns=columns_def)
if 'defects_select' not in st.session_state:
    st.session_state.defects_select = None

# 슬라이드쇼 상태
if 'images_all' not in st.session_state:
    st.session_state.images_all: List[str] = []    # 절대경로 전체
if 'order' not in st.session_state:
    st.session_state.order: List[str] = []         # 셔플된 재생 순서(중복없음)
if 'cursor' not in st.session_state:
    st.session_state.cursor: int = 0               # 현재 재생 시작 인덱스
if 'is_running' not in st.session_state:
    st.session_state.is_running: bool = False
if 'pred_cache' not in st.session_state:
    # key=(abs_path, round(conf,2)) -> {'rows': DataFrame, 'plotted_bgr': np.ndarray}
    st.session_state.pred_cache = {}
if 'last_selected_key' not in st.session_state:
    st.session_state.last_selected_key = None

ALLOWED_DEFECTS = ["cracks-scratches", "discoloration", "shelling"]
BATCH_SIZE = 6           # 카메라 박스 6개
TICK_MS = 3000           # 3초 간격

# ===============================================
# 유틸
# ===============================================
def resolve_path(p: str) -> str:
    if not p:
        return p
    return os.path.abspath(os.path.expanduser(p))

def list_test_images(folder: str) -> List[str]:
    folder = resolve_path(folder)
    if not os.path.exists(folder):
        return []
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    imgs = [os.path.abspath(x) for x in imgs]
    return imgs

@st.cache_data(show_spinner=False)
def predict_cached(abs_path: str, conf: float):
    """rows(결함 리스트) + 시각화 이미지(박스 포함, BGR) 캐시"""
    key = (abs_path, round(float(conf), 2))
    if key in st.session_state.pred_cache:
        c = st.session_state.pred_cache[key]
        return c['rows'], c['plotted_bgr']

    rows = []
    results = model.predict(abs_path, conf=conf, verbose=False)
    img = cv2.imread(abs_path)
    if img is None:
        rows_df = pd.DataFrame(rows)
        plotted_bgr = np.zeros((250, 250, 3), dtype=np.uint8)
        st.session_state.pred_cache[key] = {'rows': rows_df, 'plotted_bgr': plotted_bgr}
        return rows_df, plotted_bgr

    h, w = img.shape[:2]
    img_area = float(max(1, h * w))

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for b, c, k in zip(xyxy, confs, clss):
            defect_type = r.names.get(int(k), 'Unknown')
            if defect_type.strip().lower() not in ALLOWED_DEFECTS:
                continue
            x1, y1, x2, y2 = map(float, b)
            bbox_area = max(0.0, (x2-x1)) * max(0.0, (y2-y1))
            area_ratio = bbox_area / img_area
            rows.append({
                'Wheel ID': os.path.splitext(os.path.basename(abs_path))[0],
                'Defect Type': defect_type,
                'Confidence': float(c),
                'Area Ratio': float(area_ratio),
                'Time': datetime.now()
            })

    rows_df = pd.DataFrame(rows)
    # 요약에서 사용할 박스 포함된 시각화 이미지는 여기서 만들어 캐시해둔다
    plotted_bgr = results[0].plot(line_width=3, labels=True) if len(results) else img
    st.session_state.pred_cache[key] = {'rows': rows_df, 'plotted_bgr': plotted_bgr}
    return rows_df, plotted_bgr

def start_run():
    """버튼 클릭 시 실행 초기화"""
    images = list_test_images(TEST_FOLDER)
    if not images:
        st.warning("📁 테스트 폴더에 이미지가 없습니다.")
        return

    order = images[:]          # 전체를 섞되 재생 중 중복은 없음
    random.shuffle(order)

    st.session_state.images_all = images
    st.session_state.order = order
    st.session_state.cursor = 0
    st.session_state.is_running = True

    # 누적 리스트 초기화 (매런 때마다 새로 쌓기)
    st.session_state.defects = pd.DataFrame(columns=columns_def)
    st.session_state.defects_select = None
    st.session_state.last_selected_key = None

def advance_once():
    """3초마다 다음 배치를 화면에 표시하고 해당 배치의 검출 결과를 리스트에 앞에서부터 순차 누적"""
    if not st.session_state.is_running:
        return [], False

    start = st.session_state.cursor
    end = min(start + BATCH_SIZE, len(st.session_state.order))
    batch = st.session_state.order[start:end]

    # 현재 배치 이미지들에 대해 검출 → 순서대로 누적
    appended = []
    for p in batch:
        rows_df, _ = predict_cached(p, conf_threshold)  # 재생 중엔 박스 없는 원본만 보여줄 거라 시각화는 안 쓰지만 캐시에 저장됨
        if not rows_df.empty:
            appended.append(rows_df)

    if appended:
        add_df = pd.concat(appended, ignore_index=True)
        st.session_state.defects = pd.concat(
            [st.session_state.defects, add_df],
            ignore_index=True
        )

    # 커서 이동
    st.session_state.cursor = end

    # 종료 판단
    finished = (st.session_state.cursor >= len(st.session_state.order))
    if finished:
        st.session_state.is_running = False

    return batch, finished

# ===============================================
# 버튼 동작
# ===============================================
if run_btn_top and not st.session_state.is_running:
    start_run()

# 재생 중일 때만 3초 오토리프레시
if st.session_state.is_running:
    st_autorefresh(interval=TICK_MS, key="tick3s")

# ===============================================
# 실시간 뷰
#  - 초기에는 빈 박스
#  - 버튼 후 재생 시작 → 3초마다 다음 6장을 원본(박스 없이) 표출
#  - 모든 이미지 재생 후 정지
# ===============================================
with st.container(border=True):
    left_col_main, right_col_main = st.columns(2)

    #st.markdown("### Real-time Views", unsafe_allow_html=True)

    # 이번 틱에 보여줄 배치를 가져오고(검출/누적 포함), 끝났는지 반환
    if st.session_state.is_running:
        batch_paths, finished = advance_once()
    else:
        batch_paths, finished = [], False

    # 좌/우 3개씩 채우기 (원본 이미지로만 표시)
    def show_plain_image(path):
        img = cv2.imread(path)
        if img is None:
            st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, use_container_width=True)

    with left_col_main:
        st.markdown('### Left Wheel Views', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, c in enumerate([c1, c2, c3]):
            with c:
                if i < len(batch_paths):
                    show_plain_image(batch_paths[i])
                else:
                    st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="centered-text-below">Camera #{i+1}</div>', unsafe_allow_html=True)

    with right_col_main:
        st.markdown('### Right Wheel Views', unsafe_allow_html=True)
        c4, c5, c6 = st.columns(3)
        for j, c in enumerate([c4, c5, c6], start=3):
            with c:
                idx = j
                if idx < len(batch_paths):
                    show_plain_image(batch_paths[idx])
                else:
                    st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="centered-text-below">Camera #{j+1}</div>', unsafe_allow_html=True)

    if finished:
        st.success("✅ 모든 이미지 재생이 완료되었습니다.")

st.markdown('---')

# ===============================================
# Inspection Result
#  - 재생 중엔 현재까지 누적된 결함들이 순서대로 리스트에 쌓임
#  - 재생 종료 후에는 전체 결함이 리스트에 표시됨
#  - 좌측 리스트에서 선택 → 우측 요약(박스 포함) 표시
# ===============================================
st.markdown('### Inspection Result', unsafe_allow_html=True)
selected_row = None

left_bt_sub, right_bt_sub = st.columns(2)

# --- 좌: 리스트 ---
with left_bt_sub:
    with st.container(border=True):
        defect_count = len(st.session_state.defects)
        st.subheader(f'Defect List ({defect_count})')

        if st.session_state.defects.empty:
            st.write("No defects detected yet.")
        else:
            base_df = st.session_state.defects.copy()
            base_df.insert(0, 'Select', False)

            # 이전 선택 복원
            prev = st.session_state.get('defects_select')
            if prev is not None and 'Select' in prev.columns:
                key_cols = columns_def
                try:
                    base_df = base_df.merge(
                        prev[['Select'] + key_cols],
                        on=key_cols,
                        how='left',
                        suffixes=('', '_old')
                    )
                    base_df['Select'] = base_df['Select'].fillna(False)
                except Exception:
                    base_df['Select'] = False

            edited_df = st.data_editor(
                base_df,
                use_container_width=True,
                hide_index=True,
                height=400,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", help="요약을 볼 항목에 체크하세요", default=False),
                    "Wheel ID": st.column_config.TextColumn("Wheel ID", disabled=True),
                    "Defect Type": st.column_config.TextColumn("Defect Type", disabled=True),
                    "Confidence": st.column_config.NumberColumn("Confidence", format="%.2f", disabled=True),
                    "Area Ratio": st.column_config.NumberColumn("Area Ratio", format="%.2f", disabled=True),
                    "Time": st.column_config.DatetimeColumn("Time", disabled=True),
                },
                key="defects_editor"
            )

            # 단일 선택 유지
            if edited_df["Select"].sum() > 1:
                first_true_idx = edited_df.index[edited_df["Select"]].tolist()[0]
                edited_df.loc[edited_df.index != first_true_idx, "Select"] = False

            # 다음 렌더에서 선택 복원 가능하도록 저장
            st.session_state.defects_select = edited_df.copy()

            selected_mask = edited_df["Select"]
            if selected_mask.any():
                selected_row = edited_df[selected_mask].drop(columns=["Select"]).iloc[0]

# --- 우: 요약 ---
with right_bt_sub:
    with st.container(border=True):
        st.subheader('Selected Item Summary')

        if selected_row is None:
            if st.session_state.is_running:
                st.info("재생 중입니다. 좌측 리스트에 검출 결과가 순차적으로 누적됩니다.")
            else:
                st.info("왼쪽 리스트에서 항목을 선택하면 요약이 표시됩니다.")
        else:
            wheel_id = selected_row.get("Wheel ID", "-")
            defect_type = selected_row.get("Defect Type", "-")
            conf = float(selected_row.get("Confidence", 0.0))
            area = float(selected_row.get("Area Ratio", 0.0))
            time_val = selected_row.get("Time", "-")

            img_col, info_col = st.columns([2.5, 1.5])

            # 박스 포함 시각화 (요약에서만 표시)
            with img_col:
                image_path = None
                for ext in [".jpg", ".jpeg", ".png"]:
                    p = os.path.join(TEST_FOLDER, f"{wheel_id}{ext}")
                    if os.path.exists(p):
                        image_path = os.path.abspath(p)
                        break
                if image_path:
                    try:
                        _rows, plotted_bgr = predict_cached(image_path, conf_threshold)
                        plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
                        st.image(plotted_rgb, caption="Detected Defects", use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Unable to visualize detection: {e}")
                else:
                    st.warning(f"⚠️ Image not found for {wheel_id}")

            with info_col:
                st.markdown(f"""
                    <div style="border:1px solid #e5e7eb; border-radius:8px; padding:15px; background:#fafafa;">
                        <div style="font-size:20px; font-weight:700; color:#003087; margin-bottom:10px;">
                            Defect Summary
                        </div>
                        <div style="font-size:16px; line-height:1.6;">
                            <b>Detected Time:</b> {time_val}<br>
                            <b>Defect Type:</b> {defect_type}<br>
                            <b>Confidence :</b> {conf:.2f}<br>
                            <b>Area Ratio :</b> {area:.2f}<br>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
