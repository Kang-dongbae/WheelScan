import streamlit as st
from ultralytics import YOLO
from src.config import YOLO_MODEL
import pandas as pd

# 페이지 설정: 넓은 레이아웃, 제목
st.set_page_config(page_title="KORAIL Wheel Defect Detection", layout="wide")

# ===============================================
# 커스텀 CSS: 배경/테두리 스타일만 담당하도록 역할 분리
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
        .stContainer { /* 그리드 스타일 */
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f8f9fa; /* gray background */
            margin-bottom: 15px; 
        }
        .block-container { /* 전체 페이지 여백 조정 */
            padding-top: 2.5rem; 
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .camera-box-content { /* 동영상 박스 스타일 */
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #ffffff;
            height: 250px; 
            margin-bottom: 5px; 
            width: 100%;
        }
        .centered-text-below {  /* 동영상 박스 아래 텍스트 스타일 */
            text-align: center;
            /*font-weight: bold; */
            font-size: 16px;
            margin-top: 5px;
        }
        .left-wheel-wrapper {
            padding: 0;
            margin: 0;
        }
        .subgrid-panel-right {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 5px;
            background-color: #ffffff;
            text-align: center;
            min-height: 240px;
        }

    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">KORAIL Wheel Defect Detection System</div>', unsafe_allow_html=True)

# YOLO 모델 로드 
#model = YOLO(YOLO_MODEL) # 실제 모델 로드 주석 처리

# 세션 상태 초기화
if 'defects' not in st.session_state:
    st.session_state.defects = pd.DataFrame(columns=['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time'])
if 'running' not in st.session_state:
    st.session_state.running = False

# ===============================================
# -----------------------------
# Wheel Real-time View (수정)
# -----------------------------
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

# -----------------------------
# Inspection Result (수정)
# -----------------------------
# -----------------------------
# Inspection Result (최종 수정)
# -----------------------------
st.markdown('### Inspection Result', unsafe_allow_html=True)

selected_row = None  # 요약에서 참조할 선택행 초기화

left_bt_sub, right_bt_sub = st.columns(2)

# --- 서브그리드 1: 리스트 ---
with left_bt_sub:
    with st.container(border=True):   # ✅ HTML div 대신 컨테이너를 부모로 사용
        st.subheader('Defect List')

        if st.session_state.defects.empty:
            st.write("No defects detected yet.")
        else:
            if 'defects_select' not in st.session_state:
                st.session_state.defects_select = st.session_state.defects.copy()
                st.session_state.defects_select.insert(0, 'Select', False)

            edited_df = st.data_editor(
                st.session_state.defects_select,
                use_container_width=True,
                hide_index=True,
                height=380,  # 표가 길어져도 서브그리드 넘치지 않도록
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

            # 단일 선택 유지 (첫 번째 True만)
            if edited_df["Select"].sum() > 1:
                first_true_idx = edited_df.index[edited_df["Select"]].tolist()[0]
                edited_df.loc[edited_df.index != first_true_idx, "Select"] = False

            st.session_state.defects_select = edited_df
            selected_mask = edited_df["Select"]
            if selected_mask.any():
                selected_row = edited_df[selected_mask].drop(columns=["Select"]).iloc[0]

# --- 서브그리드 2: 요약 ---
with right_bt_sub:
    with st.container(border=True):   # ✅ 동일하게 컨테이너를 박스로 사용
        st.subheader('Selected Item Summary')

        if selected_row is None:
            st.info("왼쪽 리스트에서 항목을 선택하면 요약이 표시됩니다.")
        else:
            wheel_id = selected_row.get("Wheel ID", "-")
            defect_type = selected_row.get("Defect Type", "-")
            conf = float(selected_row.get("Confidence", 0.0))
            area = float(selected_row.get("Area Ratio", 0.0))
            time_val = selected_row.get("Time", "-")

            st.markdown(f"""
                <div style="border:1px solid #e5e7eb; border-radius:8px; padding:12px; background:#fafafa;">
                    <div style="font-size:18px; font-weight:700; margin-bottom:6px;">Wheel #{wheel_id}</div>
                    <div><b>Defect:</b> {defect_type}</div>
                    <div><b>Confidence:</b> {conf:.2f}</div>
                    <div><b>Area Ratio:</b> {area:.2f}</div>
                    <div><b>Detected At:</b> {time_val}</div>
                </div>
            """, unsafe_allow_html=True)

            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.metric("Confidence", f"{conf:.2f}")
            with mcol2:
                st.metric("Area Ratio", f"{area:.2f}")

            st.markdown("---")
            st.caption("Actions")
            st.button("상세 리포트 열기", key="open_report_btn")
            st.button("정비 작업 생성", key="create_wo_btn")


