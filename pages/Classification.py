from csv import Sniffer
from io import StringIO
from PIL import Image
from time import sleep

import pandas as pd
import streamlit as st

from util import delete_state

def read_dataset() -> None:
    if st.session_state["uploaded_dataset"] is None:
        delete_state("df")

    else:
        stringio = StringIO(
            st.session_state["uploaded_dataset"]
                .getvalue()
                .decode("utf-8")
            )
    
        dialect = Sniffer().sniff(stringio.readline())
        stringio.seek(0)
        
        df = pd.read_csv(
            stringio,
            sep=dialect.delimiter
        )

        st.session_state["df"] = df

def classify(tfidf_vect_hyperparams, cw_svm_hyperparams) -> None:
    if any(
        st.session_state[key] == "Select a column"
        for key in [
            "classification.texts",
            "classification.targets"
        ]
    ):
        st.session_state["classification.alert.error"] = "Please select texts to classify and the targets of classification."
    else:
        delete_state("classification.alert.error")

        with st.spinner("Classifying..."):
            for i in range(1, 10, 1):
                sleep(1)

default_tfidf_vect_hyperparams = {
    "norm": ["l1", "l2"],
    "ngram_range": [(1, 1), (1, 2), (1,3)],
    "min_df": [1, 3, 5, 10],
    "max_df": [0.2, 0.4, 0.6, 0.8, 1.0],
}

default_cw_svm_hyperparams = {
    "kernel": ["linear","rbf"],
    "C": [0.01, 0.1, 1, 10, 100, 1000],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1]
}

st.set_page_config(
    page_title=("Classification"),
    page_icon=Image.open("./assets/logo-usu.png"),
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
        <style>
            button.step-up {display: none;}
            button.step-down {display: none;}
            div[data-baseweb] {border-radius: 4px;}
        </style>
    """,
    unsafe_allow_html=True
)

st.title("Dataset")

st.file_uploader(
    "Upload a dataset",
    type="csv",
    key="uploaded_dataset",
    on_change=read_dataset
)

if "df" in st.session_state:
    st.dataframe(
        st.session_state["df"],
        use_container_width=True
    )

    st.divider()

    st.title("Classification")

    with st.form(key="classification"):
        df_column_options = ["Select a column"] + list(st.session_state["df"].columns)

        st.header("Select columns as Inputs")

        for col, label, key, help in zip(
            st.columns(2),
            ["Texts", "Targets"],
            ["classification.texts", "classification.targets"],
            ["Texts to classify", "Targets of classification"]
        ):
            with col:
                st.selectbox(
                    label,
                    df_column_options,
                    key=key,
                    help=help
                )

        if "classification.alert.error" in st.session_state:
            st.error(st.session_state["classification.alert.error"])

        st.header("Configure the hyper-parameters")
            
        st.markdown("### [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)")

        tfidf_vect_hyperparams = st.experimental_data_editor(
            pd.DataFrame.from_dict(
                default_tfidf_vect_hyperparams,
                orient="index",
                dtype="string"
            ).transpose(),
            use_container_width=True,
            num_rows="dynamic"
        )

        st.markdown("### [CW-SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)")

        cw_svm_hyperparams = st.experimental_data_editor(
            pd.DataFrame.from_dict(
                default_cw_svm_hyperparams,
                orient="index"
            ).transpose(),
            use_container_width=True,
            num_rows="dynamic"
        )

        st.form_submit_button(
            "Classify",
            on_click=classify,
            args=(tfidf_vect_hyperparams, cw_svm_hyperparams),
            type="secondary"
        )

st.divider()

st.session_state

st.divider()