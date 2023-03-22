import streamlit as st
from house_app import fn_app


def app():
    st.title('🔭️ $資料探勘$')
    fn_app('eda')
