from PIL import Image

import streamlit as st

st.set_page_config(
    page_title=("Home"),
    page_icon=Image.open("./assets/logo-usu.png"),
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
        <div style='text-align: center;'>
            <h2>
                EMOTION ANALYSIS ON IMBALANCE DATASET OF<br>
                INDONESIAN EMOTIONAL TEXTS USING CLASS<br>
                WEIGHTED SUPPORT VECTOR MACHINES
            </h2>
            <h2>WEB INTERFACE</h2>
            <h2>
                MUHAMMAD RAJAUL GHUFRAN<br>
                161402142
            </h2>
            <br>
        </div>
    """,
    unsafe_allow_html=True
)

cols=st.columns([5.25, 1.5, 5.25])

with cols[1]:
    st.image(
        Image.open("./assets/logo-usu.png"),
        use_column_width="always"
    )

st.markdown(
    """
        <div style='text-align: center;'>
            <br>
            <h2>
                PROGRAM STUDI S1 TEKNOLOGI INFORMASI<br>
                FAKULTAS ILMU KOMPUTER DAN TEKNOLOGI INFORMASI<br>
                UNIVERSITAS SUMATERA UTARA<br>
                MEDAN<br>
                2023
            </h2>
        </div>
    """,
    unsafe_allow_html=True
)