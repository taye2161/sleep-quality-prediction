import streamlit as st
import pandas as pd

st.set_page_config(page_title='Sleep Quality Prediction', page_icon='💤', layout='wide')

st.title('💤 Sleep Quality Prediction App')
st.markdown('### Data Analysis and ML Prediction')

st.markdown('---')

st.markdown("""
            
This App analyzes the Sleep health and a Dataset and makes predictions about your Sleep Quality.

### Pages:

1. **Data Exploration** - inspecting and cleaning the dataset
2. **Visualisation** - Explorative Data Analysis (EDA)
3. **ML Prediction** - Machine Learning Prediction
                       
""")

st.markdown('---')

st.markdown('📊 Dataset Info')

try:
    df = pd.read_csv('data/Sleep_health_and_lifestyle_dataset_cleaned.csv')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Test Subjects", len(df))

    with col2:
        st.metric("Features", len(df.columns))

    with col3:
        sick = df['Sleep Disorder'].sum() if 'Sleep Disorder' in df.columns else 0
        st.metric("with Sleep Disorder", sick)

    with col4:
        avg_age = df['Age'].mean() if 'Age' in df.columns else 0
        st.metric("Ø Age", f"{avg_age:.1f}")

    with st.expander("📋 Data Preview"):
        st.dataframe(df.head())

except FileNotFoundError:
    st.warning("⚠️ Datei nicht gefunden: `Sleep_health_and_lifestyle_dataset_cleaned.csv`")
    st.info("Bitte legen Sie die CSV-Datei in den `data/` Ordner.")

st.markdown('---')
st.caption("DataPy WiSe25/26 - Sleep Quality Prediction")

