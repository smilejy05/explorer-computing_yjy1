import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def load_data():
    df = pd.read_csv("./data/cars.csv")
    return df


def cars_home():
    st.title("🚗 자동차 연비 분석 대시보드")

    st.write("""
    이 대시보드는 자동차 성능 데이터를 기반으로  
    **대륙별 평균 연비, 마력(hp)과 연비(mpg) 관계, 차량 무게와 연비 관계, 연비 예측** 등을 보여줍니다.
    """)

    st.markdown("---")

    st.subheader("📊 주요 기능")
    st.markdown("""
    1. **탐색적 자료분석 (EDA)**  
       - 대륙별 평균 연비 비교  
       - 마력과 연비 관계 분석  
       - 차량 무게와 연비 관계 분석  

    2. **연비 예측**  
       - 실린더 수, 배기량, 마력, 무게, 가속 시간을 입력하면 예상 연비를 예측합니다.
    """)

    st.caption("📁 데이터 출처: Kaggle - Auto MPG Dataset")


def continent_mpg(df):
    st.subheader("🌏 대륙별 평균 연비")

    continent_avg = df.groupby("continent")["mpg"].mean().reset_index()

    fig = px.bar(
        continent_avg,
        x="continent",
        y="mpg",
        color="mpg",
        title="대륙별 평균 연비",
        color_continuous_scale="Greens"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **분석 포인트**  
    - 대륙별로 차량 특성이 다르며, 일반적으로 미국 차량은 비교적 연비가 낮고 일본 차량은 연비가 높은 편입니다.
    """)

    st.markdown("---")


def hp_mpg(df):
    st.subheader("⚡ 마력(hp)과 연비(mpg) 관계")

    fig = px.scatter(
        df,
        x="hp",
        y="mpg",
        color="continent",
        size="weightlbs",
        hover_name="continent",
        title="마력 대비 연비 산점도"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **분석 포인트**  
    - 마력이 높을수록 연비가 낮아지는 경향이 있습니다.
    """)

    st.markdown("---")


def weight_mpg(df):
    st.subheader("⚖️ 차량 무게(weightlbs)와 연비(mpg) 관계")

    fig = px.scatter(
        df,
        x="weightlbs",
        y="mpg",
        color="continent",
        size="hp",
        hover_name="continent",
        title="차량 무게 대비 연비 산점도"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **분석 포인트**  
    - 차량 무게가 무거울수록 연비가 낮아지는 경향이 있습니다.
    """)

    st.markdown("---")

def year_mpg(df):
    st.subheader("📈 연도별 평균 연비 변화")

    year_avg = df.groupby("year")["mpg"].mean().reset_index()

    fig = px.line(
        year_avg,
        x="year",
        y="mpg",
        markers=True,
        title="연도별 평균 연비 변화"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    💡 **분석 포인트**
    - 시간이 지날수록 평균 연비가 증가하는 경향을 확인할 수 있습니다.
    - 이는 자동차 엔진 및 연비 기술의 발전과 관련이 있습니다.
    """)

    st.markdown("---")


def cars_EDA(df):
    st.title("🔍 자동차 연비 분석 (EDA)")

    st.write("""
    이 탭에서는 자동차 성능 데이터를 활용하여  
    **연비(mpg)와 주요 변수들 간의 관계, 대륙별 특성** 등을 탐색합니다.
    """)

    st.subheader("📄 데이터 미리보기")
    st.dataframe(df.head())

    st.markdown("---")

    continent_mpg(df)
    hp_mpg(df)
    weight_mpg(df)
    year_mpg(df)

def cars_predict(df):
    st.title("🤖 자동차 연비 예측")
    st.write("선형회귀 모델을 활용하여 자동차의 연비(mpg)를 예측합니다.")

    X = df[["cylinders", "cubicinches", "hp", "weightlbs", "time-to-60"]]
    y = df["mpg"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    st.write(f"📊 모델 성능 R²: **{score:.3f}**")

    st.subheader("🚗 자동차 정보 입력")

    cylinders = st.slider("실린더 수", 3, 12, 6)
    cubicinches = st.slider("배기량", 60, 500, 200)
    hp = st.slider("마력", 50, 400, 150)
    weightlbs = st.slider("무게", 1500, 6000, 3000)
    time_to_60 = st.slider("시속 60마일 도달 시간", 4.0, 25.0, 10.0)

    input_data = pd.DataFrame({
        "cylinders": [cylinders],
        "cubicinches": [cubicinches],
        "hp": [hp],
        "weightlbs": [weightlbs],
        "time-to-60": [time_to_60]
    })

    mpg_pred = model.predict(input_data)[0]

    st.success(f"예상 연비: **{mpg_pred:.2f} mpg** 🚘")


def main():
    st.set_page_config(page_title="자동차 연비 대시보드", layout="wide")

    df = load_data()

    menu = st.sidebar.radio(
        "대시보드 메뉴",
        ["홈", "탐색적 자료분석(EDA)", "연비 예측"]
    )

    if menu == "홈":
        cars_home()

    elif menu == "탐색적 자료분석(EDA)":
        cars_EDA(df)

    elif menu == "연비 예측":
        cars_predict(df)


if __name__ == "__main__":
    main()
