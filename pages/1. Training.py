from datetime import datetime, timedelta
from PIL import Image

import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV

from pipeline.classification import Classification
from pipeline.data.stopwords import STOPWORDS
from util import POS, convert_df, create_vocab_df, delete_state, delete_states, filter_dataframe, filter_dataframe_single_column, get_term_doc_freq_df, init_state, instantiate_classification, read_dataset, stack_df

TF_IDF_VECTORIZER_DF = pd.DataFrame.from_dict(
    {
        "ngram_range": ("(1, 1)", "(1, 2)", "(2, 2)"),
        "min_df": ("0.01","1", "3", "5", "10"),
        "max_df": ("0.2", "0.4", "0.6", "0.8", "1.0"),
        "norm": ("None", "l1", "l2"),
        "sublinear_tf": ("True", "False")
    },
    orient="index"
).transpose()

SVC_DF = pd.DataFrame.from_dict(
    {
        "C": ("0.01", "0.1", "1", "10", "100"),
        "class_weight": ("None","balanced",),
        "decision_function_shape": ("ovo", "ovr")
    },
    orient="index"
).transpose()

@st.cache_resource
def load_default_stopwords():
    return pd.DataFrame(np.array_split(sorted(list(STOPWORDS)), 5)).transpose()

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
        if (
            x in ["None","True","False"] or
            any(y in x for y in ["(","{","."])
        ):
            return ast.literal_eval(x)

        return int(x)

    except:
        return x

def train(
    dataset_df,
    texts_col_name,
    targets_col_name,
    categories,
    pos,
    stopwords,
    feature_attrs,
    tfidfvectorizer_hyperparameters_df,
    svc_hyperparameters_df,
    n_iter,
    n_splits,
    train_size
):
    valid = (
        (
            texts_col_name != "Select a column" and
            targets_col_name != "Select a column"
        ),
        bool(categories) if targets_col_name != "Select a column" else True,
        bool(pos),
        bool(feature_attrs),
        (
            not tfidfvectorizer_hyperparameters_df["ngram_range"].dropna().empty and
            not tfidfvectorizer_hyperparameters_df["min_df"].dropna().empty and
            not tfidfvectorizer_hyperparameters_df["max_df"].dropna().empty and
            not tfidfvectorizer_hyperparameters_df["norm"].dropna().empty and
            not tfidfvectorizer_hyperparameters_df["sublinear_tf"].dropna().empty
        ),
        (
            not svc_hyperparameters_df["C"].dropna().empty and
            not svc_hyperparameters_df["class_weight"].dropna().empty and
            not svc_hyperparameters_df["decision_function_shape"].dropna().empty
        )
    )

    if all(valid):
        delete_states([
            "training.selected_columns.alert.error",
            "training.categories.alert.error",
            "training.pos.alert.error",
            "training.feature_attrs.alert.error",
            "training.tfidfvectorizer_hyperparameters.alert.error",
            "training.svc_hyperparameters.alert.error"
        ])

        clf: Classification = st.session_state["clf"]

        dataset_df = dataset_df[dataset_df[targets_col_name].isin(categories)]

        X_train = list(dataset_df[texts_col_name])
        y_train = list(dataset_df[targets_col_name])

        clf.feature_selection_pipeline.named_steps["stopword_removal"].set_params(**{"stopwords": set(stopwords)})
        clf.feature_selection_pipeline.named_steps["pos_filter"].set_params(**{"pos": set(pos)})
    
        with st.spinner("Text Preprocessing..."):
            X_train = clf.text_preprocessing_pipeline.transform(X_train)
            
            clf.feature_selection_pipeline.named_steps["document_transformer"].set_params(**{"feat_attrs": ["lemma","upos"]})
            st.session_state["training.X_train.preprocessed"] = clf.feature_selection_pipeline.named_steps["document_transformer"].transform(X_train, verbose__=False)

        clf.feature_selection_pipeline.named_steps["document_transformer"].set_params(**{"feat_attrs": feature_attrs})

        with st.spinner("Feature Selection..."):
            X_train = clf.feature_selection_pipeline.transform(X_train)
            st.session_state["training.X_train.feature_selected"] = X_train

        with st.spinner("Hyperparameters tuning..."):
            hyper_parameters = {}

            for k, v in tfidfvectorizer_hyperparameters_df.items():
                val = tuple(restore_dtype(x) for x in v[v.notnull()])
                hyper_parameters[k] = val

            for k, v in svc_hyperparameters_df.items():
                val = tuple(restore_dtype(x) for x in v[v.notnull()])
                hyper_parameters[k] = val

            param_distributions = {
                "tfidfvectorizer__ngram_range": hyper_parameters["ngram_range"],
                "tfidfvectorizer__min_df": hyper_parameters["min_df"],
                "tfidfvectorizer__max_df": hyper_parameters["max_df"],
                "tfidfvectorizer__norm": hyper_parameters["norm"],
                "tfidfvectorizer__sublinear_tf": hyper_parameters["sublinear_tf"],
                "svc__kernel": ("linear",),
                "svc__C": hyper_parameters["C"],
                "svc__class_weight": hyper_parameters["class_weight"],
                "svc__decision_function_shape": hyper_parameters["decision_function_shape"]
            }

            randomized_search, estimation = clf.tuning(X_train, y_train, param_distributions, n_iter=n_iter, n_splits=n_splits, train_size=train_size)

        st.session_state["training.randomized_search"] = randomized_search
        st.session_state["training.randomized_search.estimation"] = estimation

        clf.classification_pipeline = randomized_search.best_estimator_

        st.session_state["training.train.succeed"] = True

    else:
        delete_state("training.train.succeed")

        if valid[0]:
            delete_state("training.selected_columns.alert.error")
        else:
            st.session_state["training.selected_columns.alert.error"] = "Please select the texts to classify and the targets of classification."

        if valid[1]:
            delete_state("training.categories.alert.error")
        else:
            st.session_state["training.categories.alert.error"] = "Please select categories to classify."

        if valid[2]:
            delete_state("training.pos.alert.error")
        else:
            st.session_state["training.pos.alert.error"] = "Please select at least one part-of-speech."

        if valid[3]:
            delete_state("training.feature_attrs.alert.error")
        else:
            st.session_state["training.feature_attrs.alert.error"] = "Please select at least one word attribute."

        if valid[4]:
            delete_state("training.tfidfvectorizer_hyperparameters.alert.error")
        else:
            st.session_state["training.tfidfvectorizer_hyperparameters.alert.error"] = "Please fill at least one for each parameter."

        if valid[5]:
            delete_state("training.svc_hyperparameters.alert.error")
        else:
            st.session_state["training.svc_hyperparameters.alert.error"] = """Please fill at least one for each parameter."""

st.set_page_config(
    page_title=("Train a model"),
    page_icon=Image.open("./assets/logo-usu.png"),
    layout="centered",
    initial_sidebar_state="expanded"
)

init_state("clf", instantiate_classification())
init_state("training.dataset_df", pd.DataFrame.from_dict({"texts": [], "targets": []}, dtype=str))
init_state("training.stopwords_df", load_default_stopwords())

st.title("Train a model")

st.divider()

st.title("Training set")

st.file_uploader(
    "Upload a training set",
    type="csv",
    key="training.uploaded_dataset",
    on_change=read_dataset,
    args=("training.uploaded_dataset", "training.dataset_df")
)

dataset_df = st.experimental_data_editor(
    st.session_state["training.dataset_df"],
    use_container_width=True,
    num_rows="dynamic"
)

st.markdown(f'length = {len(dataset_df)}')

st.download_button(
    "Download Training Set",
    convert_df(dataset_df),
    file_name="training-set.csv",
    mime="text/csv"
)

st.divider()

if not dataset_df.empty:
    st.title("Training Configuration")

    dataset_df_column_options = ["Select a column"] + list(dataset_df.columns)

    st.header("Select columns as Inputs")

    col1, col2 = st.columns(2)

    with col1:
        texts_col_name = st.selectbox(
            "Texts",
            dataset_df_column_options,
            help="Texts to classify"
        )

    with col2:
        targets_col_name = st.selectbox(
            "Targets",
            dataset_df_column_options,
            help="Targets of classification"
        )

    categories = []

    if "training.selected_columns.alert.error" in st.session_state:
        st.error(st.session_state["training.selected_columns.alert.error"])

    if targets_col_name != "Select a column":
        options = dataset_df[targets_col_name].unique()

        st.header("Select categories to classify")
        categories = st.multiselect("_", options=options, default=options, label_visibility="collapsed")

        if "training.categories.alert.error" in st.session_state:
            st.error(st.session_state["training.categories.alert.error"])

        if categories:
            st.header(f'Value counts of selected categories')

            _1, col, _2 = st.columns([1,10,1])

            with col:
                categories_filtered_df = dataset_df[dataset_df[targets_col_name].isin(categories)][targets_col_name]
                value_counts = categories_filtered_df.value_counts()

                fig1, ax1 = plt.subplots()
                
                ax1.pie(value_counts, labels=value_counts.keys(), autopct=lambda x: '{:.2f}%\n({:.0f})'.format(x, round(x*value_counts.sum()/100, 0)))
                ax1.axis('equal')
                
                st.pyplot(fig1)

    st.header("Filter part-of-speechs")
    st.markdown("Only tokens with these part-of-speech will be used as features")

    pos = st.multiselect(
        "_",
        options=POS["tags"],
        default=POS["tags"],
        label_visibility="collapsed"
    )

    if "training.pos.alert.error" in st.session_state:
        st.error(st.session_state["training.pos.alert.error"])

    st.table(pd.DataFrame.from_dict(POS))

    st.header("Remove stop words")

    st.file_uploader(
        "Upload a stopwords list",
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
        "\n".join(stack_df(stopwords_df)),
        file_name="stopwords.txt",
        mime="text/plain"
    )

    st.header("Feature builder")
    st.markdown("Build the features based on [word attributes](https://stanfordnlp.github.io/stanza/data_objects.html#word).")

    feature_attrs = st.multiselect(
        "_",
        ["text", "lemma", "upos", "xpos"],
        ["lemma", "upos"],
        label_visibility="collapsed"
    )
    
    if "training.feature_attrs.alert.error" in st.session_state:
        st.error(st.session_state["training.feature_attrs.alert.error"])

    st.markdown(f'shape = {".".join(["<"+attr+">"for attr in feature_attrs])}')

    st.header("Hyper-parameters")

    st.subheader("[TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)")

    tfidfvectorizer_hyperparameters_df = st.experimental_data_editor(
        TF_IDF_VECTORIZER_DF,
        use_container_width=True,
        num_rows="dynamic"
    )

    if "training.tfidfvectorizer_hyperparameters.alert.error" in st.session_state:
        st.error(st.session_state["training.tfidfvectorizer_hyperparameters.alert.error"])

    st.subheader("[SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)")
    st.markdown("kernel=linear")

    svc_hyperparameters_df = st.experimental_data_editor(
        SVC_DF,
        use_container_width=True,
        num_rows="dynamic"
    )

    if "training.svc_hyperparameters.alert.error" in st.session_state:
        st.error(st.session_state["training.svc_hyperparameters.alert.error"])

    st.header("Tuning Method")

    st.subheader("[RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)")
    
    col1, col2 = st.columns(2)

    with col1:
        n_iter = st.number_input(
            "N Iterations",
            min_value=1,
            value=10,
            step=1,
            help="Number of parameter settings that are sampled. n_iter trades off runtime vs quality of the solution."
        )
    
    st.subheader("[StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)")
    
    col1, col2 = st.columns(2)

    with col1:
        n_splits = st.number_input(
            "N Splits",
            min_value=1,
            value=5,
            step=1,
            help="Number of re-shuffling & splitting iterations."
        )

    with col2:
        train_size = st.number_input(
            "Train Size",
            min_value=0.01,
            max_value=1.0,
            value=0.8,
            step=0.01,
            help="If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples."
        )

    st.button(
        "Train",
        on_click=train,
        args=(
            dataset_df,
            texts_col_name,
            targets_col_name,
            categories,
            pos,
            stack_df(stopwords_df),
            feature_attrs,
            tfidfvectorizer_hyperparameters_df,
            svc_hyperparameters_df,
            n_iter,
            n_splits,
            train_size
        ),
        type="secondary"
    )
        
if "training.train.succeed" in st.session_state:
    st.divider()
    
    st.title("Results")

    st.header("Hyper-parameters Tuning")

    if "training.randomized_search" in st.session_state:
        randomized_search: RandomizedSearchCV = st.session_state["training.randomized_search"]

        cv_results = randomized_search.cv_results_
        cv_results_df = pd.DataFrame(cv_results)

        if "training.randomized_search.estimation" in st.session_state:
            st.markdown(f'Fitted {randomized_search.n_splits_} folds of {len(cv_results_df)} candidates, finished in {str(timedelta(seconds=st.session_state["training.randomized_search.estimation"]))}.')

        st.subheader("Best hyper-parameters")
        st.dataframe(
            {
                k: str(v)
                for k, v in randomized_search.best_params_.items()
            },
            use_container_width=True
        )

        st.subheader("Cross Validation results")
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

        st.subheader("Parallel coordinates plot")
        st.markdown("To show the performance of every hyper-parameter combinations")

        parallel_coordinates_df = cv_results_df.loc[:, [col_name for col_name in cv_results_df.columns if any(x in col_name for x in ["param_", "mean_test_score"])]]
        parallel_coordinates_df = parallel_coordinates_df.rename(lambda col_name: col_name.split("__")[-1] if "param_" in col_name else col_name, axis="columns")
        parallel_coordinates_df = parallel_coordinates_df.reindex(columns=["ngram_range","min_df","max_df","norm","sublinear_tf","kernel","C","class_weight","decision_function_shape","mean_test_score"])
        parallel_coordinates_df["class_weight"] = parallel_coordinates_df["class_weight"].astype(str)
        parallel_coordinates_df = parallel_coordinates_df.replace({None: "None"})

        dimensions = []

        for col_name in parallel_coordinates_df:
            series = parallel_coordinates_df[col_name]

            if col_name == "mean_test_score":
                dimensions.append({
                    "label": col_name,
                    "values": parallel_coordinates_df[col_name],
                    "range": [-1, 1]
                })

            else:
                unique_values = sorted(list(series.unique()))

                dimensions.append({
                    "label": col_name,
                    "values": [unique_values.index(value) for value in series],
                    "ticktext": [str(value) for value in unique_values],
                    "tickvals": list(range(len(unique_values)))
                })

        fig2 = px.parallel_coordinates(
            parallel_coordinates_df,
            color="mean_test_score",
            color_continuous_scale=px.colors.diverging.RdYlGn,
            color_continuous_midpoint=0,
            range_color=[-1,1]
        )

        fig2.update_traces(dimensions=dimensions)
        fig2.update_layout(margin={"l": 30})

        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Best model")
    
    clf: Classification = st.session_state["clf"]

    st.markdown("""On [testing page](/._Testing), you can upload a testing set and evaluate the model.""")

    st.download_button(
        "Download Model",
        data=clf.to_bytes(),
        file_name=f'model.{datetime.now().strftime("%Y.%m.%d.%H.%M.%S.%f")}.pickle',
        mime="application/octet-stream"
    )

    model_attrs = clf.get_model_attrs()

    st.subheader("Terms & Document Frequencies")

    tab1, tab2 = st.tabs(["Pre-feature selection","Post-feature selection"])

    with tab1:
        length, pre_tdf_df = get_term_doc_freq_df([[word.split(".")[0] for word in document] for document in st.session_state["training.X_train.preprocessed"]])
        
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
        length, post_tdf_df = get_term_doc_freq_df(st.session_state["training.X_train.feature_selected"])
        
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

    st.subheader("Filtered POS")

    tab1, tab2 = st.tabs(["Keep","Removed"])

    terms = {}

    for document in st.session_state["training.X_train.preprocessed"]:
        for token in document:
            if token in terms:
                terms[token] += 1
            else:
                terms[token] = 1

    terms = {k: v for k, v in sorted(terms.items(), key=lambda item: item[1], reverse=True)}

    filtered_pos_df = pd.DataFrame.from_dict({
        "Terms": [token.split(".")[0] for token in list(terms.keys())],
        "POS": [token.split(".")[1] for token in list(terms.keys())],
        "Freq": list(terms.values())
    })

    with tab1:
        keep_pos_df = filtered_pos_df[filtered_pos_df["POS"].isin(model_attrs["pos_filter__pos"])]
        
        st.markdown(f"n_unique={len(keep_pos_df)}")

        st.dataframe(
            filter_dataframe(
                keep_pos_df,
                key="training.keep_pos_df"
            ),
            use_container_width=True
        )

        st.download_button(
            "Download",
            convert_df(keep_pos_df),
            file_name="keep_pos.csv",
            mime="text/csv",
            key="training.keep_pos_df.download"
        )

        st.divider()

    with tab2:
        removed_pos_df = filtered_pos_df[filtered_pos_df["POS"].isin(set(POS["tags"]) - set(model_attrs["pos_filter__pos"]))]
        
        st.markdown(f"n_unique={len(removed_pos_df)}")

        st.dataframe(
            filter_dataframe(
                removed_pos_df,
                key="training.removed_pos_df"
            ),
            use_container_width=True
        )

        st.download_button(
            "Download",
            convert_df(removed_pos_df),
            file_name="removed_pos.csv",
            mime="text/csv",
            key="training.removed_pos_df.download"
        )

        st.divider()

    st.subheader("TF-IDF Vectorizer Stopwords")
    
    st.markdown("""
        Terms that were ignored by TF-IDF Vectorizer because they either:
        - occured in too few documents (min_df)
        - occured in too many documents (max_df)
    """)
    
    tfidfvectorizer_stopwords_df = filter_dataframe_single_column(
        pd.DataFrame(
            np.array_split(
                sorted(model_attrs["tfidfvectorizer__stop_words"]),
                6
            )
        ).transpose(),
        key="training.tfidfvectorizer.stopwords",
        n_splits=6
    )
    
    st.dataframe(
        tfidfvectorizer_stopwords_df,
        use_container_width=True
    )

    st.download_button(
        "Download TF-IDF Vectorizer Stopwords",
        "\n".join(stack_df(tfidfvectorizer_stopwords_df)),
        file_name="tfidfvectorizer_stopwords.txt",
        mime="text/plain"
    )

    st.subheader("TF-IDF Vectorizer Vocabulary")
    
    st.markdown(f"""
        Features that were used to train the classifier.  
        length = {len(model_attrs["tfidfvectorizer__vocabulary"])}
    """)

    tfidfvectorizer_vocabulary_df = filter_dataframe_single_column(
        create_vocab_df(model_attrs["tfidfvectorizer__vocabulary"]),
        key="training.tfidfvectorizer.vocabulary",
        n_splits=3
    )
    
    st.dataframe(
        tfidfvectorizer_vocabulary_df,
        use_container_width=True
    )

    st.download_button(
        "Download TF-IDF Vectorizer Vocabulary",
        "\n".join(stack_df(tfidfvectorizer_vocabulary_df)),
        file_name="tfidfvectorizer_vocabulary.txt",
        mime="text/plain"
    )

st.divider()

# st.session_state

# st.divider()