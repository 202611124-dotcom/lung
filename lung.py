import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="환자 군집 분석기", layout="wide")

# 한글 폰트 설정 (환경에 따라 나눔고딕 등 설치 필요)
# 운영체제별 폰트 경로 설정이 필요할 수 있습니다.
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기준
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 데이터 로드 (캐싱) ---
@st.cache_data
def load_assets():
    df = pd.read_csv('lung.csv', encoding='utf-8')
    return df

df = load_assets()

# --- 3. UI 및 입력 섹션 ---
st.title("📊 환자 특성 기반 군집 분석")
st.markdown("입력한 환자 정보를 바탕으로 어떤 그룹에 속하는지 분석하고 시각화합니다.")

with st.sidebar:
    st.header("🔍 환자 정보 입력")
    # 기존 코드의 변수명은 glucose, bmi였으나 입력창 이름은 흡연/음주여부이므로 맞춰서 구성
    smoke = st.number_input("흡연 여부 (예: 0 또는 1)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    alcohol = st.number_input("음주 여부 (예: 0 또는 1)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    age = st.number_input("나이", min_value=0, max_value=120, value=30)
    run_analysis = st.button("분석하기")

# --- 4. 예측 로직 ---
if run_analysis:
    # 입력 데이터를 데이터프레임으로 변환
    new_patient = pd.DataFrame([[smoke, alcohol, age]], columns=['흡연여부', '음주여부', '나이'])

    # 제공된 모델/스케일러는 현재 데이터와 일치하지 않으므로,
    # 가장 유사한 기존 환자를 찾아 군집을 예측합니다.
    feature_cols = ['흡연여부', '음주여부', '나이']
    distances = np.linalg.norm(df[feature_cols].to_numpy() - new_patient.to_numpy(), axis=1)
    nearest_index = int(np.argmin(distances))
    pred_cluster = int(df.loc[nearest_index, 'cluster'])

    # --- 5. 결과 출력 및 시각화 ---
    col1, col2 = st.columns([1, 2])
else:
    st.info("우측 입력 창에서 값을 선택한 뒤 '분석하기' 버튼을 눌러주세요.")
    st.stop()

with col1:
    st.subheader("📌 분석 결과")
    st.info(f"이 환자는 **{pred_cluster}번 군집**에 속합니다.")
    
    # 지표 표시
    st.metric(label="예측 군집", value=f"Group {pred_cluster}")
    st.write("---")
    st.write("**입력 데이터 요약:**")
    st.dataframe(new_patient)

with col2:
    st.subheader("📈 데이터 위치 시각화")
    
    # Matplotlib 차트 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 기존 데이터 산점도 (Cluster별로 색상 구분)
    scatter = ax.scatter(
        df['흡연여부'], 
        df['음주여부'], 
        c=df['cluster'], 
        cmap='viridis', 
        alpha=0.4, 
        label='기존 환자군'
    )
    
    # 새 환자 위치 표시 (X 표시)
    ax.scatter(
        smoke, 
        alcohol, 
        c='red', 
        s=300, 
        marker='X', 
        edgecolors='black', 
        label='새 환자 위치'
    )
    
    ax.set_xlabel('흡연여부')
    ax.set_ylabel('음주여부')
    ax.set_title('흡연 및 음주 여부에 따른 군집 분포')
    ax.legend()
    
    # Streamlit에 차트 표시
    st.pyplot(fig)

# --- 6. 추가 상세 정보 ---
st.divider()
st.caption("참고: 위 시각화는 '흡연여부'와 '음주여부' 두 축을 기준으로 표현되었습니다.")