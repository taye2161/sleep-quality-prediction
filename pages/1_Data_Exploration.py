import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

st.title("📊 Data Exploration")
st.markdown("Data Cleaning and Inspection")

st.markdown("---")


@st.cache_data
def load_data():
    return pd.read_csv('data/Sleep_health_and_lifestyle_dataset_cleaned.csv')

try:
    df = load_data()

    tab1, tab2 = st.tabs(["📋 Overview", "📊 Statistics"])

    with tab1:
        st.subheader("Dataset Overview")

        st.markdown("#### Data Preview")
        n_rows = st.slider("Number of rows", 5, 374, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

        st.markdown("#### Info")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**Shape:** {df.shape}")

        with col2:
            st.write(f"**Rows:** {len(df.columns)}")

        st.write(f'**Datatypes:**')
        st.dataframe(pd.DataFrame({
            'Type': df.dtypes,
            'Count': df.count()
        }))

    with tab2:
        st.subheader('Statistical summary')
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("#### Categorical variables")
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                with st.expander(f"📊 {col}"):
                    st.write(df[col].value_counts())



except FileNotFoundError:
    st.error("❌ File not found: `data/Sleep_health_and_lifestyle_dataset_cleaned.csv`")


st.markdown('---')
st.caption("DataPy WiSe25/26 - Sleep Quality Prediction")