from datetime import datetime, timedelta
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from pipeline.classification import Classification
from pipeline.data.stopwords import STOPWORDS
from util import convert_df, create_vocab_df, delete_states, filter_dataframe, filter_dataframe_single_column, get_term_doc_freq_df, init_state, instantiate_classification, read_dataset

POS = {
    "tags": ["ADJ","ADP","ADV","AUX","CONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"],
    "descriptions": ["adjective","adposition","adverb","auxiliary","conjunction","determiner","interjection","noun","numeral","particle","pronoun","proper noun","punctuation","subordinating conjunction","symbol","verb","other"],
    "examples": ["salah, pertama, besar","di, pada, dari","juga, lebih, kemudian","adalah, akan, dapat","dan, atau, tetapi","ini, itu, buah","Hai, Oh, Sayang","tahun, orang, desa","satu, dua, 1","tidak, kah, lah","yang, dia, mereka","Indonesia, kabupaten, kecamatan",", ? ()","untuk, bahwa, dengan","%, =, °","menjadi, merupakan, memiliki", "and, image, in"]
}

TF_IDF_VECTORIZER_DF = pd.DataFrame.from_dict(
    {
        "ngram_range": ("(1, 1)", "(1, 2)"),
        "min_df": ("1", "3", "5", "10"),
        "max_df": ("0.2", "0.4", "0.6", "0.8", "1.0")
    },
    orient="index"
).transpose()

SVC_DF = pd.DataFrame.from_dict(
    {
        "kernel": ("linear", "rbf"),
        "C": ("0.01", "0.1", "1", "10", "100", "1000", "10000"),
        "gamma": ("0.0001", "0.001", "0.01", "0.1", "1")
    },
    orient="index"
).transpose()

@st.cache_resource
def load_default_stopwords():
    return pd.DataFrame(np.array_split(sorted(list(STOPWORDS)), 5)).transpose()

@st.cache_data
def convert_stopwords(df):
    return [
        stopword
        for stopword in list(np.concatenate([df[col_name].values for col_name in df.columns]))
        if stopword is not None
    ]

def read_stopwords():
    uploaded_stopwords = st.session_state["training.uploaded_stopwords"]

    if uploaded_stopwords is None:
        st.session_state["training.stopwords_df"] = load_default_stopwords()

    else:
        stopwords_df = pd.DataFrame(np.array_split(sorted(uploaded_stopwords.getvalue().decode("ascii").splitlines()), 5)).transpose()
        st.session_state["training.stopwords_df"] = stopwords_df

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

def train(dataset_df, stopwords_df, tfidfvectorizer_hyperparameters, svc_hyperparameters):
    if any(
        st.session_state[key] == "Select a column"
        for key in [
            "training.texts",
            "training.targets"
        ]
    ):
        delete_states([
            "training.categories.alert.error",
            "training.train.succeed"
        ])
        st.session_state["training.selected_columns.alert.error"] = "Please select texts to classify and the targets of classification."

    elif not st.session_state["training.categories"]:
        delete_states([
            "training.selected_columns.alert.error",
            "training.train.succeed"
        ])
        st.session_state["training.categories.alert.error"] = "Please select categories to classify."

    else:
        delete_states([
            "training.selected_columns.alert.error",
            "training.categories.alert.error"
        ])

        clf: Classification = st.session_state["clf"]

        col_name = st.session_state["training.targets"]
        dataset_df = dataset_df[dataset_df[col_name].isin(st.session_state["training.categories"])]

        X = list(dataset_df[st.session_state["training.texts"]])
        y = list(dataset_df[st.session_state["training.targets"]])

        stopwords = convert_stopwords(stopwords_df)

        clf.feature_selection_pipeline.named_steps["stopword_removal"].set_params(**{"stopwords": set(stopwords)})
        clf.feature_selection_pipeline.named_steps["pos_filter"].set_params(**{"pos": set(st.session_state["training.pos"])})
        clf.feature_selection_pipeline.named_steps["document_transformer"].set_params(**{"feat_attrs": ["lemma","upos"]})

        with st.spinner("Text Preprocessing..."):
            X = clf.text_preprocessing_pipeline.transform(X)
            st.session_state["training.X.preprocessed"] = clf.feature_selection_pipeline.named_steps["document_transformer"].transform(X, verbose__=False)

        with st.spinner("Feature Selection..."):
            X = clf.feature_selection_pipeline.transform(X)
            st.session_state["training.X.feature_selected"] = X

        # Split
        X_train, X_test, y_train, y_test = clf.train_test_split(X, y)

        st.session_state["training.y_test"] = y_test

        with st.spinner("Hyperparameters tuning..."):
            hyper_parameters = {}

            for k, v in tfidfvectorizer_hyperparameters.items():
                val = tuple(restore_dtype(x) for x in v[v.notnull()])
                hyper_parameters[k] = val

            for k, v in svc_hyperparameters.items():
                val = tuple(restore_dtype(x) for x in v[v.notnull()])
                hyper_parameters[k] = val

            param_grid = []

            if "linear" in hyper_parameters["kernel"]:
                param_grid.append({
                    "tfidfvectorizer__ngram_range": hyper_parameters["ngram_range"],
                    "tfidfvectorizer__min_df": hyper_parameters["min_df"],
                    "tfidfvectorizer__max_df": hyper_parameters["max_df"],
                    "svc__kernel": ("linear",),
                    "svc__C": hyper_parameters["C"]
                })

            if "rbf" in hyper_parameters["kernel"]:
                param_grid.append({
                    "tfidfvectorizer__ngram_range": hyper_parameters["ngram_range"],
                    "tfidfvectorizer__min_df": hyper_parameters["min_df"],
                    "tfidfvectorizer__max_df": hyper_parameters["max_df"],
                    "svc__kernel": ("rbf",),
                    "svc__C": hyper_parameters["C"],
                    "svc__gamma": hyper_parameters["gamma"]
                })

            grid_search, estimation = clf.tuning(X_train, y_train, param_grid)

        st.session_state["training.grid_search"] = grid_search
        st.session_state["training.grid_search.estimation"] = estimation

        clf.classification_pipeline.set_params(**grid_search.best_params_)

        with st.spinner("Re-training model..."):
            clf.train_preprocessed(X_train, y_train)

        with st.spinner("Prediction..."):        
            y_pred = clf.test_preprocessed(X_test)

        st.session_state["training.y_pred"] = y_pred
        st.session_state["training.score"] = clf.score(y_test, y_pred)

        st.session_state["training.train.succeed"] = True

st.set_page_config(
    page_title=("Train a model"),
    page_icon=Image.open("./assets/logo-usu.png"),
    layout="centered",
    initial_sidebar_state="expanded"
)

init_state("clf", instantiate_classification())
init_state("training.stopwords_df", load_default_stopwords())

st.title("Train a model")

st.divider()

st.title("Dataset")

st.file_uploader(
    "Upload a dataset",
    type="csv",
    key="training.uploaded_dataset",
    on_change=read_dataset,
    args=("training.uploaded_dataset", "training.dataset_df")
)

if "training.dataset_df" in st.session_state:
    dataset_df = st.experimental_data_editor(
        st.session_state["training.dataset_df"],
        use_container_width=True,
        num_rows="dynamic"
    )

    st.download_button(
        "Download Dataset",
        convert_df(dataset_df),
        file_name="dataset.csv",
        mime="text/csv"
    )

    st.divider()

    st.title("Classification")

    dataset_df_column_options = ["Select a column"] + list(dataset_df.columns)

    st.header("Select columns as Inputs")

    for col, label, key, help in zip(
        st.columns(2),
        ["Texts", "Targets"],
        ["training.texts", "training.targets"],
        ["Texts to classify", "Targets of classification"]
    ):
        with col:
            st.selectbox(
                label,
                dataset_df_column_options,
                key=key,
                help=help
            )

    if "training.selected_columns.alert.error" in st.session_state:
        st.error(st.session_state["training.selected_columns.alert.error"])

    if st.session_state["training.targets"] != "Select a column":
        col_name = st.session_state["training.targets"]
        options = dataset_df[col_name].unique()

        st.header("Select categories to classify")
        st.multiselect("Select categories to classify", options=options, default=options, key="training.categories", label_visibility="collapsed")

        if "training.categories.alert.error" in st.session_state:
            st.error(st.session_state["training.categories.alert.error"])

        if st.session_state["training.categories"]:
            st.header(f'Value counts of texts on selected categories')

            _1, col, _2 = st.columns([1,10,1])

            with col:
                categories = dataset_df[dataset_df[col_name].isin(st.session_state["training.categories"])][col_name]
                value_counts = dict(categories.value_counts())

                fig1, ax1 = plt.subplots()
                
                ax1.pie(value_counts.values(), labels=value_counts.keys(), autopct='%1.1f%%')
                ax1.axis('equal')
                
                st.pyplot(fig1)

    st.title("Training Configuration")

    st.header("Filter specific part-of-speechs")
    st.markdown("Only tokens with these pos will be used as features")

    st.multiselect(
        "_",
        options=POS["tags"],
        default=POS["tags"],
        key="training.pos",
        label_visibility="collapsed"
    )

    st.table(pd.DataFrame.from_dict(POS))

    st.header("Remove stop words")

    st.file_uploader(
        "Upload stopwords list",
        type="txt",
        key="training.uploaded_stopwords",
        on_change=read_stopwords
    )

    stopwords_df = st.experimental_data_editor(
        st.session_state["training.stopwords_df"],
        use_container_width=True,
        num_rows="dynamic"
    )

    st.download_button(
        "Download Stopwords",
        "\n".join(list(convert_stopwords(stopwords_df))),
        file_name="stopwords.txt",
        mime="text/plain"
    )

    st.header("Configure the hyper-parameters")
    st.markdown("### [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)")

    tfidfvectorizer_hyperparamters = st.experimental_data_editor(
        TF_IDF_VECTORIZER_DF,
        use_container_width=True,
        num_rows="dynamic"
    )

    st.markdown("### [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)")

    svc_hyperparamters = st.experimental_data_editor(
        SVC_DF,
        use_container_width=True,
        num_rows="dynamic"
    ) 

    st.button(
        "Train",
        on_click=train,
        args=(dataset_df, stopwords_df, tfidfvectorizer_hyperparamters, svc_hyperparamters),
        type="secondary"
    )
        
if "training.train.succeed" in st.session_state:
    st.divider()
    
    st.title("Results")

    st.header("Hyper-parameters Tuning")

    if "training.grid_search" in st.session_state:
        grid_search: GridSearchCV = st.session_state["training.grid_search"]

        if "training.grid_search.estimation" in st.session_state:
            st.markdown(f'Fitted {grid_search.n_splits_} folds of {len(grid_search.cv_results_)} candidates, finished in {str(timedelta(seconds=st.session_state["training.grid_search.estimation"]))}.')

        st.markdown("### Best hyper-parameters")
        st.dataframe(
            {
                k: str(v)
                for k, v in grid_search.best_params_.items()
            },
            use_container_width=True
        )

        st.markdown("###  Parallel Coordinates")

        cv_results = grid_search.cv_results_
        
        cv_results_df = pd.DataFrame(cv_results)
        parallel_coordinates_df = cv_results_df.loc[:, [col_name for col_name in cv_results_df.columns if "param_" in col_name or col_name == "mean_test_score"]]
        parallel_coordinates_df = parallel_coordinates_df.rename(lambda col_name: "MCC" if col_name == "mean_test_score" else col_name.split("__")[-1], axis="columns")

        dimensions = []

        for col_name in parallel_coordinates_df:
            series = parallel_coordinates_df[col_name]

            if col_name == "MCC":
                dimensions.append({
                    "label": col_name,
                    "values": parallel_coordinates_df[col_name],
                    "range": [-1, 1]
                })

            else:
                unique_values = list(series.unique())

                dimensions.append({
                    "label": col_name,
                    "values": [unique_values.index(value) for value in series],
                    "ticktext": [str(value) for value in unique_values],
                    "tickvals": list(range(len(unique_values)))
                })

        fig2 = px.parallel_coordinates(
            parallel_coordinates_df,
            color="MCC",
            color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=0,
            range_color=[-1,1]
        )

        fig2.update_traces(dimensions=dimensions)
        fig2.update_layout(margin={"l": 20})

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("###  Cross Validation results")
        st.dataframe(
            filter_dataframe(
                cv_results_df,
                "training.cv_results_df"
            ),
            use_container_width=True
        )

        st.download_button(
            "Download CV results",
            convert_df(cv_results_df),
            file_name="cv_results.csv",
            mime="text/csv"
        )

    st.header("Model Evaluation")

    clf: Classification = st.session_state["clf"]

    model_attrs = clf.get_model_attrs()

    st.subheader("TF-IDF Vectorizer Stopwords")
    st.markdown("""
        Terms that were ignored by TF-IDF Vectorizer because they either:
        - occured in too few documents (min_df)
        - occured in too many documents (max_df)
    """)
    st.dataframe(
        filter_dataframe_single_column(
            pd.DataFrame(
                np.array_split(
                    sorted(model_attrs["tfidfvectorizer__stop_words"]),
                    6
                )
            ).transpose(),
            key="training.tfidfvectorizer.stopwords",
            n_splits=6
        ),
        use_container_width=True
    )

    st.subheader("TF-IDF Vectorizer Vocabulary")
    st.markdown(f"""
        Terms that were used as features to train the classifier.  
        features_shape=(1, {len(model_attrs["tfidfvectorizer__vocabulary"])})
    """)
    st.dataframe(
        filter_dataframe_single_column(
            create_vocab_df(model_attrs["tfidfvectorizer__vocabulary"]),
            key="training.tfidfvectorizer.vocabulary",
            n_splits=3
        ),
        use_container_width=True
    )

    st.subheader("Terms & Document Frequencies")

    tab1, tab2 = st.tabs(["Pre-feature selection","Post-feature selection"])

    with tab1:
        length, pre_tdf_df = get_term_doc_freq_df(st.session_state["training.X.preprocessed"])
        
        st.markdown(f"n_unique={length}")

        st.dataframe(
            filter_dataframe(
                pre_tdf_df,
                key="training.pre_tdf_df"
            ),
            use_container_width=True
        )

        st.download_button(
            "Download",
            convert_df(pre_tdf_df),
            file_name="terms_document_frequencies.csv",
            mime="text/csv",
            key="training.pre_tdf_df.download"
        )

        st.divider()

    with tab2:
        length, post_tdf_df = get_term_doc_freq_df(st.session_state["training.X.feature_selected"])
        
        st.markdown(f"n_unique={length}")

        st.dataframe(
            filter_dataframe(
                post_tdf_df,
                key="training.post_tdf_df"
            ),
            use_container_width=True
        )

        st.download_button(
            "Download",
            convert_df(post_tdf_df),
            file_name="terms_document_frequencies.csv",
            mime="text/csv",
            key="training.post_tdf_df.download"
        )

        st.divider()

    if (
        "training.y_test" in st.session_state and
        "training.y_pred" in st.session_state
    ):
        st.markdown("### Confusion Matrix")

        _1, col, _2 = st.columns([1,10,1])

        with col:
            cm = ConfusionMatrixDisplay.from_predictions(st.session_state["training.y_test"], st.session_state["training.y_pred"], normalize="true", cmap="YlGn")
            st.pyplot(cm.figure_)

    if "training.score" in st.session_state:
        st.markdown("### Score")

        accuracy, mcc = st.session_state["training.score"]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{round(accuracy*100, 3)}%")

        with col2:
            st.metric("MCC", f"{round(mcc, 3)}")

    st.download_button(
        "Download Model",
        data=st.session_state["clf"].to_bytes(),
        file_name=f'model.{datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")}.pickle',
        mime="application/octet-stream"
    )

st.divider()

# st.session_state

# st.divider()