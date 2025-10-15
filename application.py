# application.py
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ===============================================
# (선택) src.config 에서 기본값 가져오기
# ===============================================
try:
    from src.config import (
        YOLO_MODEL as CFG_YOLO_MODEL,
        TEST_FOLDER as CFG_TEST_FOLDER,
        # 아래 3개가 config에 없으면 로컬 기본값 사용
        ALLOWED_DEFECTS as CFG_ALLOWED_DEFECTS,
        BATCH_SIZE as CFG_BATCH_SIZE,
        TICK_MS as CFG_TICK_MS,
    )
except Exception:
    CFG_YOLO_MODEL = CFG_TEST_FOLDER = None
    CFG_ALLOWED_DEFECTS = CFG_BATCH_SIZE = CFG_TICK_MS = None

# ===============================================
# 추론 서비스 (새 모듈)
# ===============================================
from src.appDefect import (
    Detector,
    list_images,
    find_image_by_stem,
    start_run_state,
    advance_state,
)

# ===============================================
# 페이지 설정 & CSS
# ===============================================
st.set_page_config(page_title="KORAIL Wheel Defect Detection", layout="wide")
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
# 상수 / 기본값
# ===============================================
ALLOWED_DEFECTS = CFG_ALLOWED_DEFECTS or ["cracks-scratches", "discoloration", "shelling"]
BATCH_SIZE = int(CFG_BATCH_SIZE or 6)
TICK_MS = int(CFG_TICK_MS or 3000)

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
# 모델 로드 (캐시) - Detector 사용
# ===============================================
@st.cache_resource
def load_detector(model_path_or_name: str) -> Detector:
    return Detector(model_path_or_name, ALLOWED_DEFECTS)

try:
    detector = load_detector(YOLO_MODEL)
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
if 'images_all' not in st.session_state:
    st.session_state.images_all: List[str] = []
if 'order' not in st.session_state:
    st.session_state.order: List[str] = []
if 'cursor' not in st.session_state:
    st.session_state.cursor: int = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running: bool = False

# ===============================================
# 캐시된 예측/시각화
# ===============================================
@st.cache_data(show_spinner=False)
def cached_predict_rows(abs_path: str, conf: float):
    from src.appDefect import DefectRow  # 타입 힌트용(선택)
    rows = detector.predict_rows(abs_path, conf)
    # DataFrame으로 바로 사용할 수 있게 변환
    return pd.DataFrame([r.__dict__ for r in rows]) if rows else pd.DataFrame(columns=columns_def)

@st.cache_data(show_spinner=False)
def cached_plot(abs_path: str, conf: float):
    plotted_bgr = detector.plot_with_boxes(abs_path, conf)
    return plotted_bgr  # BGR ndarray

# ===============================================
# 실행 제어
# ===============================================
def start_run():
    images = list_images(TEST_FOLDER)
    if not images:
        st.warning("📁 테스트 폴더에 이미지가 없습니다.")
        return
    st.session_state.update(start_run_state(images))
    st.session_state.defects = pd.DataFrame(columns=columns_def)
    st.session_state.defects_select = None

def advance_once():
    """3초마다 다음 배치를 표시하고, 해당 배치의 검출 결과를 누적"""
    if not st.session_state.is_running:
        return [], False
    batch, finished, new_state = advance_state(st.session_state, BATCH_SIZE)
    st.session_state.update(new_state)

    appended = []
    for p in batch:
        rows_df = cached_predict_rows(p, conf_threshold)
        if not rows_df.empty:
            appended.append(rows_df)
    if appended:
        st.session_state.defects = pd.concat([st.session_state.defects, *appended], ignore_index=True)

    return batch, finished

# ===============================================
# 버튼 동작 & 오토리프레시
# ===============================================
if run_btn_top and not st.session_state.is_running:
    start_run()

if st.session_state.is_running:
    st_autorefresh(interval=TICK_MS, key="tick3s")

# ===============================================
# 실시간 뷰
# ===============================================
with st.container(border=True):
    left_col_main, right_col_main = st.columns(2)

    if st.session_state.is_running:
        batch_paths, finished = advance_once()
    else:
        batch_paths, finished = [], False

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

            prev = st.session_state.get('defects_select')
            if prev is not None and 'Select' in prev.columns:
                try:
                    # 키 컬럼 기준으로 선택 복원
                    key_cols = ['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time']
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

            with img_col:
                image_path = find_image_by_stem(TEST_FOLDER, wheel_id)
                if image_path:
                    try:
                        plotted_bgr = cached_plot(image_path, conf_threshold)
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
