from typing import Dict, List

import streamlit as st

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

@st.cache_resource
def instantiate_classification():
    return Classification()