import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Data Visualisation", page_icon="📈", layout="wide")

st.title('📈 Data Visualisation')
st.markdown("Explorative Data Analysis (EDA)")

st.markdown('---')

@st.cache_data
def load_data():
    return pd.read_csv('data/Sleep_health_and_lifestyle_dataset_cleaned.csv')

try:
    df = load_data()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlationen", "📉 Comparisons"])

    with tab1:
        st.subheader('Distribution Analysis')

        col1, col2 = st.columns([1, 3])

        with col1:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            selected_feature = st.selectbox("Select a feature:", numeric_cols)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df[selected_feature], bins=20, edgecolor='black')

            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution: {selected_feature}')
            st.pyplot(fig)

    with tab2:
        st.subheader('Correlation Analysis')

        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, ax=ax)
        ax.set_title('Corralation-Matrix')
        st.pyplot(fig)


    with tab3:
        st.subheader('Feature Comparisons')

        col1, col2 = st.columns([1, 1])

        with col1:
            x_feature = st.selectbox("X-Axis:", numeric_cols, index=0)

        with col2:
            y_feature = st.selectbox("Y-Axis:", numeric_cols, index=1)

        #color_by = 'Quality of Sleep' if 'Quality of Sleep' in df.columns else None
        #fig = px.scatter(df, x=x_feature, y=y_feature, color=color_by,
        #                 title=f'{y_feature} vs {x_feature}',
        #                 hover_data=df.columns)
        #st.plotly_chart(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            df[x_feature],
            df[y_feature],
            c=df['Quality of Sleep'],
            alpha=0.7
        )
        fig.colorbar(scatter, ax=ax, label='Quality of Sleep')

        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'{y_feature} vs {x_feature}')
        ax.grid(True)

        st.pyplot(fig)



except FileNotFoundError:
    st.error("❌ File not found: `data/Sleep_health_and_lifestyle_dataset_cleaned.csv`")

st.markdown('---')
st.caption("DataPy WiSe25/26 - Sleep Quality Prediction")

