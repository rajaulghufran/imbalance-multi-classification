from csv import Sniffer
from io import StringIO
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_numeric_dtype,
)

from pipeline.classification import Classification

def init_state(name: str, val: any) -> None:
    if name not in st.session_state:
        st.session_state[name] = val

def init_states(d: Dict[str, any]) -> None:
    for key, value in d.items():
        init_state(key, value)

def delete_state(name: str) -> None:
    if name in st.session_state:
        del st.session_state[name]

def delete_states(l: List[str]) -> None:
    for key in l:
        delete_state(key)

def filter_dataframe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    modify = st.checkbox("Add filters", key=f'{key}.add_filters')

    if not modify:
        return df

    df = df.copy()

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns, key=f'{key}.filter_dataframe_on')

        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            
            # Treat columns with < 20 unique values as categorical

            if is_categorical_dtype(df[column]) or df[column].nunique() < 20:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )

                df = df[df[column].isin(user_cat_input)]
            
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )

                df = df[df[column].between(*user_num_input)]

            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )

                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

# TODO: full compatibility with splitted df
def filter_dataframe_single_column(df: pd.DataFrame, key: str, n_splits: int) -> pd.DataFrame:
    modify = st.checkbox("Add filters", key=f'{key}.add_filters')

    if not modify:
        return df

    df = df.copy()
    df = pd.DataFrame([x for x in df.values.ravel('F') if x is not None])

    modification_container = st.container()

    with modification_container:
        # Treat columns with < 20 unique values as categorical

        if is_categorical_dtype(df[0]) or df[0].nunique() < 20:
            user_cat_input = st.multiselect(
                f"Values",
                df[0].unique(),
                default=list(df[0].unique()),
            )

            df = df[df[0].isin(user_cat_input)]
        
        elif is_numeric_dtype(df[0]):
            _min = float(df[0].min())
            _max = float(df[0].max())
            step = (_max - _min) / 100
            
            user_num_input = st.slider(
                f"Values",
                _min,
                _max,
                (_min, _max),
                step=step,
            )

            df = df[df[0].between(*user_num_input)]

        else:
            user_text_input = st.text_input(
                f"Substring or regex",
            )

            if user_text_input:
                df = df[df[0].str.contains(user_text_input)]

    return pd.DataFrame(np.array_split(list(df[0]), n_splits)).transpose()

@st.cache_data
def get_term_doc_freq_df(X):
    terms = {}

    for x in X:
        for t in set(x):
            if t in terms:
                terms[t] = terms[t] + 1
            else:
                terms[t] = 1

    terms = {k: v for k, v in sorted(terms.items(), key=lambda item: item[1], reverse=True)}

    t1, t2 = np.array_split(list(terms.keys()), 2)
    df1, df2 = np.array_split(list(terms.values()), 2)
    dfp1, dfp2 = np.array_split([v / len(X) for v in terms.values()], 2)

    return (
        len(terms),
        pd.DataFrame([t1, df1, dfp1, t2, df2, dfp2])
        .transpose()
        .set_axis(
            ["TERMS1", "DF1", "DF1 %", "TERMS2", "DF2", "DF2 %"],
            axis="columns"
        ).astype({
            "TERMS1": 'object',
            "DF1": 'Int32',
            "DF1 %": "Float32",
            "TERMS2": 'object',
            "DF2": 'Int32',
            "DF2 %": "Float32"
        })
    )

@st.cache_data
def convert_df(df: pd.DataFrame) -> str:
    return df.to_csv(index=False).encode('ascii', errors="ignore")

@st.cache_data
def create_vocab_df(vocab_=None):
    vocab = {k: v for k, v in sorted(vocab_.items(), key=lambda item: item[1])}

    return pd.DataFrame(
        np.array_split(list(vocab.keys()), 3)
    ).transpose()

@st.cache_resource
def instantiate_classification() -> Classification:
    return Classification()

def read_dataset(from_: str, to_: str) -> None:
    uploaded_dataset = st.session_state[from_]

    if uploaded_dataset is not None:
        stringio = StringIO(
            uploaded_dataset
                .getvalue()
                .decode("utf-8")
        )

        dialect = Sniffer().sniff(stringio.readline())
        stringio.seek(0)
        
        dataset_df = pd.read_csv(
            stringio,
            sep=dialect.delimiter
        )

        st.session_state[to_] = dataset_df
