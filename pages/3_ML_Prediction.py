import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Prediction", page_icon="📈", layout="wide")

st.title('🤖 Machine Learning Prediction')
st.markdown('Week 9: Predicting Quality of upcoming Sleep')

st.markdown('---')

@st.cache_data
def load_data():
    return pd.read_csv('data/Sleep_health_and_lifestyle_dataset_cleaned.csv')

try:
    df = load_data()

except FileNotFoundError:
    st.error("❌ File not found: `data/Sleep_health_and_lifestyle_dataset_cleaned.csv`")

st.markdown('---')
st.caption("DataPy WiSe25/26 - Sleep Quality Prediction")