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
            <h3>
                EMOTION ANALYSIS ON IMBALANCE DATASET OF INDONESIAN EMOTIONAL TEXTS<br>
                USING CLASS WEIGHTED SUPPORT VECTOR MACHINES
            </h3>
            <h3>WEB APPLICATION INTERFACE</h3>
            <h3>
                MUHAMMAD RAJAUL GHUFRAN<br>
                161402142
            </h3>
            <br>
        </div>
    """,
    unsafe_allow_html=True
)

cols=st.columns([4.75, 2.5, 4.75])

with cols[1]:
    st.image(
        Image.open("./assets/logo-usu.png"),
        use_column_width="always"
    )

st.markdown(
    """
        <div style='text-align: center;'>
            <br>
            <h3>
                PROGRAM STUDI S1 TEKNOLOGI INFORMASI<br>
                FAKULTAS ILMU KOMPUTER DAN TEKNOLOGI INFORMASI<br>
                UNIVERSITAS SUMATERA UTARA<br>
                MEDAN<br>
                2023
            </h3>
        </div>
    """,
    unsafe_allow_html=True
)