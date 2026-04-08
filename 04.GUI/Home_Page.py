import requests
import streamlit as st
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="ECG Based Authentication Interface",
    page_icon=":star:",
    layout="wide",
)


def load_lottie(url):
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        return None
    return response.json()


st.write("## ECG Based Authentication Interface")
st.write("---")

animation = load_lottie("https://assets8.lottiefiles.com/packages/lf20_zw7jo1.json")
with st.container():
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(
            """
            This project presents ECG-driven biometric authentication as a practical
            security mechanism. Instead of passwords or static identifiers, the
            system learns subject-specific cardiac signatures from waveform
            morphology and signal dynamics, then exposes that pipeline through an
            interactive inference interface.
            """
        )
    with right_col:
        st_lottie(animation, height=500, width=500, key="Signature Recognition")
