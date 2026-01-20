import streamlit as st
import pandas as pd

st.set_page_config(page_title='Sleep Quality Prediction', page_icon='💤', layout='wide')

st.title('💤 Sleep Quality Prediction App')
st.markdown('### Data Analysis and ML Prediction')

st.markdown('---')

st.markdown("""
## Welcome!

This App analyzes the Sleep health and lifestyle Dataset and makes predictions about your Sleep Quality.

### Pages:

1. **Data Exploration** - inspecting and cleaning the dataset
2. **Visualisation** - Explorative Data Analysis (EDA)
3. **ML Prediction** - Machine Learning Prediction
                       
""")

st.markdown('---')

st.markdown('📊 Dataset Info')

try:
    df = pd.read_csv('../data/processed/Sleep_health_and_lifestyle_dataset_cleaned.csv')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patienten", len(df))

    with col2:
        st.metric("Features", len(df.columns))

    with col3:
        sick = df['condition'].sum() if 'condition' in df.columns else 0
        st.metric("Mit Herzkrankheit", sick)

    with col4:
        avg_age = df['age'].mean() if 'age' in df.columns else 0
        st.metric("Ø Alter", f"{avg_age:.1f}")

    with st.expander("📋 Daten-Vorschau"):
        st.dataframe(df.head())

except FileNotFoundError:
    st.warning("⚠️ Datei nicht gefunden: `Sleep_health_and_lifestyle_dataset_cleaned.csv`")
    st.info("Bitte legen Sie die CSV-Datei in den `data/` Ordner.")
