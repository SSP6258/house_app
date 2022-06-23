import streamlit as st
from house_app import fn_app


def app():
    st.title('👨‍🏫 網站導覽')
    fn_app('introduce')
