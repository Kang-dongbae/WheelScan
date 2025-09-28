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
with st.container(border=True):
       
    # Wheel Real-time View 
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

        st.markdown('</div>', unsafe_allow_html=True)
        
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
        
        st.markdown('</div>', unsafe_allow_html=True)

  
    st.markdown('---') # 구분선
     
    
    # Wheel Inspection Result Views
    st.markdown('### Inspection Result', unsafe_allow_html=True)
    
    left_bt_sub, right_bt_sub = st.columns(2)
    
    with left_bt_sub:
        st.markdown('<div class="subgrid-panel-right">Subgrid 1 (Defect List)</div>', unsafe_allow_html=True)
    
    with right_bt_sub:
        st.markdown('<div class="subgrid-panel-right">Subgrid 2 (Summary Data)</div>', unsafe_allow_html=True)
    
    
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True) 
    