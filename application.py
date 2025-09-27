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
            color: #003087; /* KORAIL 블루 색상 */
            text-align: center;
            margin-bottom: 20px;
        }
        /* Streamlit 컨테이너에 적용할 공통 스타일 */
        .stContainer {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f8f9fa; /* 회색 배경 */
            margin-bottom: 15px; /* 섹션 간 간격 */
        }
        /* Grid 1 내부 서브 그리드 (휠 뷰) */
        .subgrid-panel-left {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #ffffff;
            text-align: center;
            min-height: 350px; 
            margin-bottom: 10px; /* 휠 뷰 아래에 작은 여백 추가 */
        }
        /* Grid 목록 서브 그리드 (결함 리스트) */
        .subgrid-panel-right {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 5px;
            background-color: #ffffff;
            text-align: center;
            min-height: 240px;
        }
        /* Streamlit의 기본 여백을 조정합니다. */
        .block-container {
            /* 🚨 수정: 상단 여백을 2.5rem으로 늘려서 제목 잘림을 방지합니다. */
            padding-top: 2.5rem; 
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .subgrid-panel-left {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #ffffff;
            min-height: 350px; /* Right Wheel의 높이에 맞추기 */
            margin-bottom: 10px;
        }

        /* 🚨 개별 Video 박스 스타일 (video-box-content) */
        .video-box-content {
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #ffffff;
            height: 250px; /* 동영상 영역 높이 고정 */
            margin-bottom: 5px; 
            width: 100%;
        }

        /* 🚨 박스 아래 텍스트 중앙 정렬 */
        .centered-text-below {
            text-align: center;
            font-weight: bold;
            font-size: 16px;
            margin-top: 5px;
        }
        
        /* 🚨 Left Wheel 내부의 텍스트와 동영상 박스를 정렬하기 위한 스타일 */
        .left-wheel-wrapper {
            /* 이 래퍼는 Left Wheel 컬럼 전체를 채우는 역할만 합니다. */
            padding: 0;
            margin: 0;
        }

    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">KORAIL Wheel Defect Detection System</div>', unsafe_allow_html=True)

# YOLO 모델 로드 (YOLO_MODEL 변수가 정의되어 있어야 합니다)
#model = YOLO(YOLO_MODEL) # 실제 모델 로드 주석 처리

# 세션 상태 초기화
if 'defects' not in st.session_state:
    st.session_state.defects = pd.DataFrame(columns=['Wheel ID', 'Defect Type', 'Confidence', 'Area Ratio', 'Time'])
if 'running' not in st.session_state:
    st.session_state.running = False

# ===============================================
# 단일 통합 그리드 (Real-Time Inspection & Data)
# ===============================================

# 🚨 전체 컨텐츠를 담는 하나의 컨테이너
with st.container(border=True):
       
    # 1. 휠 뷰 섹션 (Grid 1 역할)
    #st.markdown('### Real-time Wheel Inspection', unsafe_allow_html=True)
    
    # Left Wheel (Video 3개)과 Right Wheel을 나누는 메인 컬럼
    left_col_main, right_col_main = st.columns(2)
    
    # 🚨 Left Wheel: 3개 동영상을 포함하는 큰 흰색 박스
    with left_col_main:
        # 이 컬럼 내부에 subgrid-panel-left 스타일을 가진 div를 엽니다.
        #st.markdown('<div class="subgrid-panel-left">', unsafe_allow_html=True) 
        
        # 내부 콘텐츠 제목
        st.markdown('### Left Wheel Views', unsafe_allow_html=True)

        # 3개 컬럼으로 분할 (동영상 3개)
        video1_col, video2_col, video3_col = st.columns(3)
        
        # Video 1
        with video1_col:
            st.markdown('<div class="video-box-content"></div>', unsafe_allow_html=True) 
            st.markdown('<div class="centered-text-below">Camera #1</div>', unsafe_allow_html=True)
            
        # Video 2
        with video2_col:
            st.markdown('<div class="video-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #2</div>', unsafe_allow_html=True)
            
        # Video 3
        with video3_col:
            st.markdown('<div class="video-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #3</div>', unsafe_allow_html=True)

        # Left Wheel 박스 닫기
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 🚨 2. Right Wheel: 단일 큰 흰색 박스
    with right_col_main:
        # 이 컬럼 내부에 subgrid-panel-left 스타일을 가진 div를 엽니다.
        #st.markdown('<div class="subgrid-panel-left">', unsafe_allow_html=True)
        
        st.markdown('### Right Wheel Veiws', unsafe_allow_html=True)
        # 여기에 오른쪽 휠의 이미지/결과 위젯을 배치
        
                # 3개 컬럼으로 분할 (동영상 3개)
        video1_col, video2_col, video3_col = st.columns(3)
        
        # Video 1
        with video1_col:
            st.markdown('<div class="video-box-content"></div>', unsafe_allow_html=True) 
            st.markdown('<div class="centered-text-below">Camera #4</div>', unsafe_allow_html=True)
            
        # Video 2
        with video2_col:
            st.markdown('<div class="video-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #5</div>', unsafe_allow_html=True)
            
        # Video 3
        with video3_col:
            st.markdown('<div class="video-box-content"></div>', unsafe_allow_html=True)
            st.markdown('<div class="centered-text-below">Camera #6</div>', unsafe_allow_html=True)
        
        # Right Wheel 박스 닫기
        st.markdown('</div>', unsafe_allow_html=True)

    
    # 2. 하단 데이터/목록 섹션 (Grid 2 & 3 역할)
    st.markdown('---') # 시각적 구분선 추가
    st.markdown('### Inspection Result', unsafe_allow_html=True)
    
    # 내부 2x2 서브 그리드 (목록)
    left_bt_sub, right_bt_sub = st.columns(2)
    
    with left_bt_sub:
        st.markdown('<div class="subgrid-panel-right">Subgrid 1 (Defect List)</div>', unsafe_allow_html=True)
    
    with right_bt_sub:
        st.markdown('<div class="subgrid-panel-right">Subgrid 2 (Summary Data)</div>', unsafe_allow_html=True)
    
    # 이 div는 st.container(border=True) 안에 있으므로 필요 없음.
    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True) 
    
#st.write("Streamlit Session Initialized!") # 세션 초기화 메시지는 맨 마지막에 배치