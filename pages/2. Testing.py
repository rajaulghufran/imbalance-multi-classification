from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from pipeline.classification import Classification
from util import POS, convert_df, create_vocab_df, delete_state, delete_states, filter_dataframe, filter_dataframe_single_column, get_term_doc_freq_df, init_state, instantiate_classification, read_dataset, stack_df

def load_model():
    st.session_state["clf"].from_bytes(st.session_state["testing.uploaded_model"].getvalue())

def test(dataset_df):
    if st.session_state["testing.texts"] == "Select a column":
        delete_state("testing.test.succeed")
        st.session_state["testing.selected_columns.alert.error"] = "Please select texts to classify"

    else:
        delete_state("testing.selected_columns.alert.error")

        clf: Classification = st.session_state["clf"]

        X_test = list(dataset_df[st.session_state["testing.texts"]])

        try:
            y_test = list(dataset_df[st.session_state["testing.targets"]])
        except KeyError:
            y_test = None

        feature_attrs = clf.feature_selection_pipeline.named_steps["document_transformer"].feat_attrs

        with st.spinner("Text Preprocessing..."):
            X_test = clf.text_preprocessing_pipeline.transform(X_test)

            # clf.feature_selection_pipeline.named_steps["document_transformer"].set_params(**{"feat_attrs": ["text","upos"]})
            # st.session_state["testing.X_test.preprocessed"] = clf.feature_selection_pipeline.named_steps["document_transformer"].transform(X_test)
        
        with st.spinner("Feature Selection..."):
            clf.feature_selection_pipeline.named_steps["document_transformer"].set_params(**{"feat_attrs": feature_attrs})
            X_test = clf.feature_selection_pipeline.transform(X_test)
            # st.session_state["testing.X_test.feature_selected"] = X_test

        with st.spinner("Prediction..."):
            y_pred = clf.test_preprocessed(X_test)

        st.session_state["testing.dataset_df"] = dataset_df.assign(predictions=y_pred)
        st.session_state["testing.dataset_df"].reset_index(drop=True, inplace=True)
        st.session_state["testing.y_pred"] = y_pred

        if y_test is None:
            delete_states([
                "testing.y_test",
                "testing.score"
            ])

        else:
            st.session_state["testing.y_test"] = y_test
            st.session_state["testing.score"] = clf.score(y_test, y_pred)

        st.session_state["testing.test.succeed"] = True

st.set_page_config(
    page_title=("Test a model"),
    page_icon=Image.open("./assets/logo-usu.png"),
    layout="centered",
    initial_sidebar_state="expanded"
)

init_state("clf", instantiate_classification())
init_state("testing.dataset_df", pd.DataFrame.from_dict({"texts": [], "targets": []}, dtype=str))

st.title("Test a model")

st.divider()

st.title("Load a model")

st.file_uploader(
    "_",
    type="pickle",
    key="testing.uploaded_model",
    on_change=load_model,
    label_visibility="collapsed"
)

st.divider()

clf: Classification = st.session_state["clf"]

if clf.is_fitted():
    st.title("Model Configuration")

    model_attrs = clf.get_model_attrs()

    st.subheader("Classification categories")
    st.table(np.array_split(sorted(model_attrs["linearsvc__classes"]), 1))

    st.subheader("Hyper-parameters")
    st.markdown("These are parameters that were used to train the model.")

    hyper_parameters = {}

    for k, v in model_attrs.items():
        if k not in [
            "pos_filter__pos",
            "stopword_removal__stopwords",
            "document_transformer__feat_attrs",
            "tfidfvectorizer__vocabulary",
            "tfidfvectorizer__stop_words",
            "linearsvc__classes"
        ]:
            key = k.split("__")[-1]
            hyper_parameters[key] = str(v)

    st.table(hyper_parameters)

    st.subheader(f'Feature shape: {".".join(["<"+x+">" for x in model_attrs["document_transformer__feat_attrs"]])}')

    st.subheader("Filtered Part-of-Speech")
    st.table(np.array_split(sorted(model_attrs["pos_filter__pos"]), 3))

    st.subheader("Removed Stopwords")
    st.markdown("""
        Terms that were ignored while training the model because they either:
        - manually set by the user
        - occured in too few documents (min_df)
        - occured in too many documents (max_df)
    """)

    stopwords_df = filter_dataframe_single_column(
        pd.DataFrame(
            np.array_split(
                sorted(model_attrs["stopword_removal__stopwords"] | model_attrs["tfidfvectorizer__stop_words"]),
                3
            )
        ).transpose(),
        key="testing.stopwords",
        n_splits=6
    )

    st.dataframe(
        stopwords_df,
        use_container_width=True
    )

    st.download_button(
        "Download Stopwords",
        "\n".join(stack_df(stopwords_df)),
        file_name="stopwords.txt",
        mime="text/plain"
    )

    st.subheader("Vocabulary")
    st.markdown(f"""
        Terms that were used as features to train the classifier.  
        features_shape=(1, {len(model_attrs["tfidfvectorizer__vocabulary"])})  
        Generated by TF-IDF Vectorizer with ngram_range={model_attrs["tfidfvectorizer__ngram_range"]}
    """)

    vocabulary_df = filter_dataframe_single_column(
        create_vocab_df(model_attrs["tfidfvectorizer__vocabulary"]),
        key="testing.vocabulary",
        n_splits=3
    )

    st.dataframe(
        vocabulary_df,
        use_container_width=True
    )

    st.download_button(
        "Download Vocabulary",
        "\n".join(stack_df(vocabulary_df)),
        file_name="vocabulary.txt",
        mime="text/plain"
    )

    st.divider()
    
    st.title("Testing Set")

    st.file_uploader(
        "Upload a testing set",
        type="csv",
        key="testing.uploaded_dataset",
        on_change=read_dataset,
        args=("testing.uploaded_dataset", "testing.dataset_df")
    )

    dataset_df = st.experimental_data_editor(
        st.session_state["testing.dataset_df"],
        use_container_width=True,
        num_rows="dynamic"
    )

    st.markdown(f'length = {len(dataset_df)}')

    st.download_button(
        "Download Testing Set",
        convert_df(dataset_df),
        file_name="testing-set.csv",
        mime="text/csv"
    )

    st.divider()

    if not dataset_df.empty:
        st.title("Classification")

        df_column_options = ["Select a column"] + list(st.session_state["testing.dataset_df"].columns)

        st.header("Select columns as Inputs")

        for col, label, key, help in zip(
            st.columns(2),
            ["Texts", "Targets"],
            ["testing.texts", "testing.targets"],
            ["Texts to classify", "Targets of classification"]
        ):
            with col:
                st.selectbox(
                    label,
                    df_column_options,
                    key=key,
                    help=help
                )

        if "testing.selected_columns.alert.error" in st.session_state:
            st.error(st.session_state["testing.selected_columns.alert.error"])

        if st.session_state["testing.targets"] != "Select a column":
            col_name = st.session_state["testing.targets"]
            options = dataset_df[col_name].unique()

            st.header(f'Value counts of targets')

            _1, col, _2 = st.columns([1,10,1])

            with col:
                categories = dataset_df[col_name]
                value_counts = categories.value_counts()

                fig1, ax1 = plt.subplots()
                
                ax1.pie(value_counts, labels=value_counts.keys(), autopct=lambda x: '{:.2f}%\n({:.0f})'.format(x, round(x*value_counts.sum()/100, 0)))
                ax1.axis('equal')
                
                st.pyplot(fig1)

        st.button(
            "Test",
            on_click=test,
            args=(dataset_df,),
            type="secondary"
        )

    if "testing.test.succeed" in st.session_state:
        st.divider()

        st.title("Results")
        st.markdown("#### Predictions have been added to the dataset!")

        if "testing.y_test" in st.session_state:
            st.header("Model Evaluation")

        # st.subheader("Filtered POS")

        # tab1, tab2 = st.tabs(["Keep","Removed"])

        # terms = {}

        # for document in st.session_state["testing.X_test.preprocessed"]:
        #     for token in document:
        #         if token in terms:
        #             terms[token] += 1
        #         else:
        #             terms[token] = 1

        # terms = {k: v for k, v in sorted(terms.items(), key=lambda item: item[1], reverse=True)}

        # filtered_pos_df = pd.DataFrame.from_dict({
        #     "Terms": [token.split(".")[0] for token in list(terms.keys())],
        #     "POS": [token.split(".")[1] for token in list(terms.keys())],
        #     "Freq": list(terms.values())
        # })

        # with tab1:
        #     keep_pos_df = filtered_pos_df[filtered_pos_df["POS"].isin(model_attrs["pos_filter__pos"])]
            
        #     st.markdown(f"n_unique={len(keep_pos_df)}")

        #     st.dataframe(
        #         filter_dataframe(
        #             keep_pos_df,
        #             key="testing.keep_pos_df"
        #         ),
        #         use_container_width=True
        #     )

        #     st.download_button(
        #         "Download",
        #         convert_df(keep_pos_df),
        #         file_name="keep_pos.csv",
        #         mime="text/csv",
        #         key="testing.keep_pos_df.download"
        #     )

        #     st.divider()

        # with tab2:
        #     removed_pos_df = filtered_pos_df[filtered_pos_df["POS"].isin(set(POS["tags"]) - set(model_attrs["pos_filter__pos"]))]
            
        #     st.markdown(f"n_unique={len(removed_pos_df)}")

        #     st.dataframe(
        #         filter_dataframe(
        #             removed_pos_df,
        #             key="testing.removed_pos_df"
        #         ),
        #         use_container_width=True
        #     )

        #     st.download_button(
        #         "Download",
        #         convert_df(removed_pos_df),
        #         file_name="removed_pos.csv",
        #         mime="text/csv",
        #         key="testing.removed_pos_df.download"
        #     )

        #     st.divider()

        # st.subheader("Terms & Document Frequencies")

        # tab1, tab2 = st.tabs(["Pre-feature selection","Post-feature selection"])

        # with tab1:
        #     length, pre_tdf_df = get_term_doc_freq_df([[word.split(".")[0] for word in document] for document in st.session_state["testing.X_test.preprocessed"]])
            
        #     st.markdown(f"n_unique={length}")

        #     st.dataframe(
        #         filter_dataframe(
        #             pre_tdf_df,
        #             key="testing.pre_tdf_df"
        #         ),
        #         use_container_width=True
        #     )

        #     st.download_button(
        #         "Download",
        #         convert_df(pre_tdf_df),
        #         file_name="terms_document_frequencies.csv",
        #         mime="text/csv",
        #         key="testing.pre_tdf_df.download"
        #     )

        # with tab2:
        #     length, post_tdf_df = get_term_doc_freq_df(st.session_state["testing.X_test.feature_selected"], ngram_range=model_attrs["tfidfvectorizer__ngram_range"], stopwords=model_attrs["tfidfvectorizer__stop_words"])
            
        #     st.markdown(f"n_unique={length}")

        #     st.dataframe(
        #         filter_dataframe(
        #             post_tdf_df,
        #             key="testing.post_tdf_df"
        #         ),
        #         use_container_width=True
        #     )

        #     st.download_button(
        #         "Download",
        #         convert_df(post_tdf_df),
        #         file_name="terms_document_frequencies.csv",
        #         mime="text/csv",
        #         key="testing.post_tdf_df.download"
        #     )
        
        if "testing.y_test" in st.session_state:
            if (
                "testing.y_test" in st.session_state and
                "testing.y_pred" in st.session_state
            ):
                st.subheader("Confusion Matrix")

                normalize = st.checkbox("Normalize")

                _1, col, _2 = st.columns([1,10,1])

                with col:
                    cm = ConfusionMatrixDisplay.from_predictions(
                        st.session_state["testing.y_test"],
                        st.session_state["testing.y_pred"],
                        normalize="true" if normalize else None,
                        cmap="YlGn"
                    )

                    st.pyplot(cm.figure_)

            if "testing.score" in st.session_state:
                st.subheader("Score")

                accuracy, mcc = st.session_state["testing.score"]

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Accuracy", f"{round(accuracy*100, 3)}%")

                with col2:
                    st.metric("MCC", f"{round(mcc, 3)}")

else:
    st.subheader("Model is not fitted.")
    st.markdown("Please load a fitted model or [train a model](/._Training) first!")

st.divider()

# st.session_state

# st.divider()