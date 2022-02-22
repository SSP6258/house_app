import streamlit as st
from house_app import fn_app


def app():
    st.title('🗃️ 其它專案')
    fn_app('projects')
