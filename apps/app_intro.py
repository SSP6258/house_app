import streamlit as st
from house_app import fn_app


def app():
    st.title('👨‍🏫 $網站介紹$')
    fn_app('introduce')
