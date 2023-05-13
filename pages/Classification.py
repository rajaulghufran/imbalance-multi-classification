from csv import Sniffer
from io import StringIO
from PIL import Image

import pandas as pd
import streamlit as st

from util import delete_state, init_state, instantiate_classification

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

def restore_dtype(x):
    x = x.replace(" ","")

    try:
        if '(' in x:
            return tuple(restore_dtype(y) for y in x.replace("(","").replace(")","").split(","))

        if '.' in x:
            return float(x) 

        return int(x)

    except:
        return x

def classify(tfidfvectorizer_hyperparameters, svc_hyperparameters) -> None:
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
            for k, v in tfidfvectorizer_hyperparameters.items():
                val = tuple(restore_dtype(x) for x in v[v.notnull()])
                st.session_state["clf"].set_param_grid_attr(f'tfidfvectorizer__{k}', val)

            for k, v in svc_hyperparameters.items():
                val = tuple(restore_dtype(x) for x in v[v.notnull()])
                st.session_state["clf"].set_param_grid_attr(f'svc__{k}', val)

            X = list(st.session_state["df"][st.session_state["classification.texts"]])
            y = list(st.session_state["df"][st.session_state["classification.targets"]])

            X_cleaned = st.session_state["clf"].clean(X)
            X_tokenized = st.session_state["clf"].tokenize(X_cleaned)
            X_train, X_test, y_train, y_test = st.session_state["clf"].train_test_split(X_tokenized, y)
            best_hyperparameters = st.session_state["clf"].tuning(X_train, y_train)[0]
            # model = st.session_state["clf"].train(X_train, y_train, best_hyperparameters)
            # y_pred = st.session_state["clf"].test(model, X_test)
            # _, mcc = st.session_state["clf"].score(y_test, y_pred)

            # print(mcc)

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

init_state("clf", instantiate_classification())

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

        tfidfvectorizer_hyperparameters = st.experimental_data_editor(
            pd.DataFrame.from_dict(
                st.session_state["clf"].get_params("tfidfvectorizer", val_to_str=True),
                orient="index"
            ).transpose(),
            use_container_width=True,
            num_rows="dynamic"
        )

        st.markdown("### [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)")

        svc_hyperparameters = st.experimental_data_editor(
            pd.DataFrame.from_dict(
                st.session_state["clf"].get_params("svc", val_to_str=True),
                orient="index"
            ).transpose(),
            use_container_width=True,
            num_rows="dynamic"
        )

        st.form_submit_button(
            "Classify",
            on_click=classify,
            args=(tfidfvectorizer_hyperparameters, svc_hyperparameters),
            type="secondary"
        )

st.divider()

# st.session_state

# st.divider()