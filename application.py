# app.py
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

import streamlit as st
from ultralytics import YOLO

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
        .stContainer {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f8f9fa;
            margin-bottom: 15px;
        }
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
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
# 사이드바: 경로/옵션
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
# 세션 상태 초기화
# ===============================================
if 'defects' not in st.session_state:
    st.session_state.defects = pd.DataFrame(columns=['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time'])
if 'defects_select' not in st.session_state:
    st.session_state.defects_select = None

# ===============================================
# 유틸
# ===============================================
def resolve_path(p: str) -> str:
    """상대 경로를 절대 경로로 보정"""
    if not p:
        return p
    return os.path.abspath(os.path.expanduser(p))

# ===============================================
# 테스트 폴더 이미지 처리
# ===============================================
def process_test_images(test_folder=TEST_FOLDER, min_conf=0.0):
    defects_list = []
    ALLOWED_DEFECTS = ["cracks-scratches", "discoloration", "shelling"]

    if not os.path.exists(test_folder):
        st.error(f"Test folder '{test_folder}' does not exist.")
        return pd.DataFrame()

    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        st.info("No images found in the test folder.")
        return pd.DataFrame()

    progress_bar = st.progress(0)
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(test_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            progress_bar.progress((idx + 1) / len(image_files))
            continue

        try:
            # ✅ conf 임계값(min_conf) 적용
            results = model.predict(img_path, conf=min_conf, verbose=False)
        except Exception as e:
            st.warning(f"Prediction failed on {img_file}: {e}")
            progress_bar.progress((idx + 1) / len(image_files))
            continue

        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)

                img_h, img_w = img.shape[:2]
                img_area = float(img_h * img_w) if img_h * img_w > 0 else 1.0

                for b, c, k in zip(xyxy, confs, clss):
                    defect_type = result.names.get(int(k), 'Unknown')
                    defect_lower = defect_type.strip().lower()

                    # ✅ Wheel 같은 비결함 클래스 제외
                    if defect_lower not in ALLOWED_DEFECTS:
                        continue

                    x1, y1, x2, y2 = map(float, b)
                    bbox_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    area_ratio = bbox_area / img_area
                    wheel_id = os.path.splitext(img_file)[0]

                    defects_list.append({
                        'Wheel ID': wheel_id,
                        'Defect Type': defect_type,
                        'Confidence': float(c),
                        'Area Ratio': float(area_ratio),
                        'Time': datetime.now()
                    })

        progress_bar.progress((idx + 1) / len(image_files))

    progress_bar.empty()
    return pd.DataFrame(defects_list)


# ===============================================
# 상단 실행 버튼
# ===============================================
if run_btn_top:
    with st.spinner("🔎 Processing test images..."):
        new_defects = process_test_images(TEST_FOLDER, conf_threshold)
        if not new_defects.empty:
            st.session_state.defects = pd.concat(
                [st.session_state.defects, new_defects],
                ignore_index=True
            )
            # 표용 데이터는 매번 새로 생성하도록 초기화
            st.session_state.defects_select = None
            st.success(f"✅ Detected {len(new_defects)} defects from test images.")
        else:
            st.info("🔍 No defects found or no images present.")

# ===============================================
# 실시간 뷰 (자리 배치용)
# ===============================================
with st.container(border=True):
    left_col_main, right_col_main = st.columns(2)

    with left_col_main:
        st.markdown('### Left Wheel Views', unsafe_allow_html=True)
        camera1, camera2, camera3 = st.columns(3)
        with camera1:
            st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #1</div>', unsafe_allow_html=True)
        with camera2:
            st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #2</div>', unsafe_allow_html=True)
        with camera3:
            st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #3</div>', unsafe_allow_html=True)

    with right_col_main:
        st.markdown('### Right Wheel Views', unsafe_allow_html=True)
        camera4, camera5, camera6 = st.columns(3)
        with camera4:
            st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #4</div>', unsafe_allow_html=True)
        with camera5:
            st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #5</div>', unsafe_allow_html=True)
        with camera6:
            st.markdown('<div class="camera-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #6</div>', unsafe_allow_html=True)

st.markdown('---')

# ===============================================
# Inspection Result
# ===============================================
st.markdown('### Inspection Result', unsafe_allow_html=True)
selected_row = None

left_bt_sub, right_bt_sub = st.columns(2)

# --- 서브그리드 1: 리스트 ---
with left_bt_sub:
    with st.container(border=True):
        st.subheader('Defect List')
        
        # ✅ defect 개수 표시
        defect_count = len(st.session_state.defects)
        st.subheader(f'Defect List ({defect_count})')

        if st.session_state.defects.empty:
            st.write("No defects detected yet.")
        else:
            # 항상 현재 defects에서부터 표 데이터 생성
            base_df = st.session_state.defects.copy()
            base_df.insert(0, 'Select', False)

            # 이전 선택 복원 (있으면)
            prev = st.session_state.get('defects_select')
            if prev is not None and 'Select' in prev.columns:
                key_cols = ['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time']
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
                height=380,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select", help="요약을 볼 항목에 체크하세요", default=False
                    ),
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

# --- 서브그리드 2: 요약 ---
with right_bt_sub:
    with st.container(border=True):
        st.subheader('Selected Item Summary')

        if selected_row is None:
            st.info("왼쪽 리스트에서 항목을 선택하면 요약이 표시됩니다.")
        else:
            wheel_id = selected_row.get("Wheel ID", "-")
            defect_type = selected_row.get("Defect Type", "-")
            conf = float(selected_row.get("Confidence", 0.0))
            area = float(selected_row.get("Area Ratio", 0.0))
            time_val = selected_row.get("Time", "-")

            # =====================================
            # 🧱 1️⃣ 이미지(왼쪽) + 요약(오른쪽)
            # =====================================
            img_col, info_col = st.columns([2.5, 1.5])  # 이미지 넓게, 정보 좁게

            # -------------------
            # 왼쪽: 결함 표시 이미지
            # -------------------
            with img_col:
                possible_exts = [".jpg", ".jpeg", ".png"]
                image_path = None
                for ext in possible_exts:
                    test_path = os.path.join(TEST_FOLDER, f"{wheel_id}{ext}")
                    if os.path.exists(test_path):
                        image_path = test_path
                        break

                if image_path:
                    try:
                        results = model.predict(image_path, conf=0.25, verbose=False)
                        result = results[0]
                        plotted_img = result.plot(line_width=3, labels=True)
                        plotted_img_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
                        st.image(plotted_img_rgb, caption="Detected Defects", use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Unable to visualize detection for this image: {e}")
                else:
                    st.warning(f"⚠️ Image not found for {wheel_id}")

            # -------------------
            # 오른쪽: 결함 요약 정보
            # -------------------
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
