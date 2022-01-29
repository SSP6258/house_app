import streamlit as st
from house_app import fn_app


def app():
    st.title('📚 參考資料')
    fn_app('reference')
