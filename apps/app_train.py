import streamlit as st
from house_app import fn_app


def app():
    st.title('🏋‍♂️ $模型訓練$')
    fn_app('train')


