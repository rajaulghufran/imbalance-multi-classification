from csv import Sniffer
from io import StringIO
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from pipeline.classification import POS
from util import delete_state, delete_states, init_state, instantiate_classification

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

def classify(selected_pos, tfidfvectorizer_hyperparameters, svc_hyperparameters) -> None:
    if any(
        st.session_state[key] == "Select a column"
        for key in [
            "classification.texts",
            "classification.targets"
        ]
    ):
        delete_states([
            "classification.y_test",
            "classification.y_pred",
            "classification.score",
            "classification.selected_classes.alert.error"
        ])
        st.session_state["classification.selected_columns.alert.error"] = "Please select texts to classify and the targets of classification."

    elif not st.session_state["classification.selected_classes"]:
        delete_states([
            "classification.y_test",
            "classification.y_pred",
            "classification.score",
            "classification.selected_columns.alert.error"
        ])
        st.session_state["classification.selected_classes.alert.error"] = "Please select classes to classify."

    else:
        delete_states([
            "classification.y_test",
            "classification.y_pred",
            "classification.score",
            "classification.selected_columns.alert.error",
            "classification.selected_classes.alert.error"
        ])

        for k, v in tfidfvectorizer_hyperparameters.items():
            val = tuple(restore_dtype(x) for x in v[v.notnull()])
            st.session_state["clf"].set_param_grid_attr(f'tfidfvectorizer__{k}', val)

        for k, v in svc_hyperparameters.items():
            val = tuple(restore_dtype(x) for x in v[v.notnull()])
            st.session_state["clf"].set_param_grid_attr(f'svc__{k}', val)

        df = st.session_state["df"]
        col_name = st.session_state["classification.targets"]
        df = df[df[col_name].isin(st.session_state["classification.selected_classes"])]

        X = list(df[st.session_state["classification.texts"]])
        y = list(df[st.session_state["classification.targets"]])

        with st.spinner("Text cleaning..."):
            X_cleaned = st.session_state["clf"].clean(X)

        with st.spinner("Tokenization..."):
            X_tokenized = st.session_state["clf"].tokenize(X_cleaned, selected_pos.copy())

        X_train, X_test, y_train, y_test = st.session_state["clf"].train_test_split(X_tokenized, y)
        st.session_state["classification.y_test"] = y_test

        # with st.spinner("Hyperparameters tuning..."):
        #     best_hyperparameters = st.session_state["clf"].tuning(X_train, y_train)[0]

        # for testing purposes
        best_hyperparameters = {
            'svc__C': 10000,
            'svc__decision_function_shape': 'ovo',
            'svc__gamma': 0.0001,
            'svc__kernel': 'rbf',
            'tfidfvectorizer__max_df': 0.2,
            'tfidfvectorizer__min_df': 1,
            'tfidfvectorizer__ngram_range': (1, 2),
            'tfidfvectorizer__norm': 'l2'
        }

        with st.spinner("Re-training model..."):
            model = st.session_state["clf"].train(X_train, y_train, best_hyperparameters)

        y_pred = st.session_state["clf"].test(model, X_test)
        st.session_state["classification.y_pred"] = y_pred

        st.session_state["classification.score"] = st.session_state["clf"].score(y_test, y_pred)

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

    if "classification.selected_columns.alert.error" in st.session_state:
        st.error(st.session_state["classification.selected_columns.alert.error"])

    if st.session_state["classification.targets"] != "Select a column":
        df = st.session_state["df"]
        col_name = st.session_state["classification.targets"]
        options = df[col_name].unique()

        st.header("Select classes to classify")
        st.multiselect("Select classes to classify", options=options, default=options, key="classification.selected_classes", label_visibility="collapsed")

        if "classification.selected_classes.alert.error" in st.session_state:
            st.error(st.session_state["classification.selected_classes.alert.error"])

        if st.session_state["classification.selected_classes"]:
            st.header(f'Value counts of texts on selected classes')

            _1, col, _2 = st.columns([1,10,1])

            with col:
                classes = df[df[col_name].isin(st.session_state["classification.selected_classes"])][col_name]
                value_counts = dict(classes.value_counts())

                fig1, ax1 = plt.subplots()
                
                ax1.pie(value_counts.values(), labels=value_counts.keys(), autopct='%1.1f%%')
                ax1.axis('equal')
                
                st.pyplot(fig1)

    st.title("Configuration")

    with st.form(key="classification"):
        st.header("Filter specific part-of-speechs")
        st.markdown("Only tokens with these pos will be used as features")

        selected_pos = st.multiselect("_", options=POS["tags"], default=POS["tags"], label_visibility="collapsed")

        st.table(
            pd.DataFrame.from_dict(POS)
        )

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
            args=(selected_pos, tfidfvectorizer_hyperparameters, svc_hyperparameters),
            type="secondary"
        )

st.divider()

if any(
    True
    for key in [
        "classification.y_test",
        "classification.y_pred",
        "classification.score"
    ]
    if key in st.session_state
):
    st.title("Results")

    # st.header("Hyper-parameters Tuning")

    st.header("Classifier")

    st.markdown("### Confusion Matrix")

    if (
        "classification.y_test" in st.session_state and
        "classification.y_pred" in st.session_state
    ):
        _1, col, _2 = st.columns([1,10,1])

        with col:
            cm = ConfusionMatrixDisplay.from_predictions(st.session_state["classification.y_test"], st.session_state["classification.y_pred"])

            st.pyplot(cm.figure_)

    st.markdown("### Score")

    if "classification.score" in st.session_state:
        accuracy, mcc = st.session_state["classification.score"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{round(accuracy*100, 2)}%")

        with col2:
            st.metric("MCC", f"{round(mcc, 2)}")

    st.divider()

# st.session_state

# st.divider()