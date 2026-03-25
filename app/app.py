import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict

st.set_page_config(
    page_title="Прогнозування відтоку клієнтів",
    page_icon="📡",
    layout="centered"
)

st.title("Прогнозування відтоку клієнтів")
st.write("Введіть параметри клієнта, щоб оцінити ймовірність відтоку.")

YES_NO_OPTIONS = ["Ні", "Так"]


def yes_no_to_binary(value: str) -> int:
    return 1 if value == "Так" else 0


tv_subscriber = st.selectbox("Чи є підписка на ТБ?", YES_NO_OPTIONS)
movie_package = st.selectbox("Чи є підписка на кінопакет?", YES_NO_OPTIONS)

subscription_age = st.number_input(
    "Тривалість користування послугою (умовні одиниці)",
    min_value=0.0,
    value=12.0,
    step=1.0
)

bill_avg = st.number_input(
    "Середній рахунок",
    min_value=0.0,
    value=50.0,
    step=1.0
)

remaining_contract = st.number_input(
    "Залишок контракту",
    min_value=0.0,
    value=6.0,
    step=1.0
)

service_failures = st.number_input(
    "Кількість збоїв сервісу",
    min_value=0,
    value=0,
    step=1
)

download_avg = st.number_input(
    "Середній обсяг завантаження",
    min_value=0.0,
    value=20.0,
    step=1.0
)

upload_avg = st.number_input(
    "Середній обсяг відвантаження",
    min_value=0.0,
    value=5.0,
    step=1.0
)

download_over_limit = st.selectbox(
    "Чи було перевищення ліміту завантаження?",
    YES_NO_OPTIONS
)

if st.button("Розрахувати прогноз"):
    data = [
        yes_no_to_binary(tv_subscriber),
        yes_no_to_binary(movie_package),
        subscription_age,
        bill_avg,
        remaining_contract,
        service_failures,
        download_avg,
        upload_avg,
        yes_no_to_binary(download_over_limit)
    ]

    result, proba = predict(data)

    st.markdown("---")
    st.subheader("Результат прогнозу")
    st.write(f"Ймовірність відтоку: **{proba:.2%}**")

    if proba > 0.7:
        st.error("🔴 Високий ризик відтоку")
        st.write("Рекомендована дія: запропонувати знижку або персональну утримувальну пропозицію.")
    elif proba > 0.4:
        st.warning("🟡 Середній ризик відтоку")
        st.write("Рекомендована дія: запустити engagement-кампанію або зв'язатися з клієнтом.")
    else:
        st.success("🟢 Низький ризик відтоку")
        st.write("Рекомендована дія: додаткових дій не потрібно.")

    st.caption("Прогноз сформовано на основі моделі машинного навчання Random Forest.")
