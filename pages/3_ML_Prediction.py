import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Prediction", page_icon="📈", layout="wide")

st.title('🤖 Machine Learning Prediction')
st.markdown('Predicting Quality of upcoming Sleep')

st.markdown('---')

@st.cache_data
def load_data():
    return pd.read_csv('data/Sleep_health_and_lifestyle_dataset_cleaned.csv')

try:
    df = load_data()

    if "model" not in st.session_state:
        st.session_state.model = None

    categorical_cols = ['Gender', 'Occupation', 'BMI Category']
    df = df.drop(columns=categorical_cols, errors='ignore')


    feature_cols = [col for col in df.columns if col != 'Quality of Sleep']
    target_col = 'Quality of Sleep'

    if target_col not in df.columns:
        st.error("❌ Target-variable 'Quality of Sleep' not found!")
        st.stop()

    X = df[feature_cols].drop(columns=['Systolic', 'Diastolic', 'Daily Steps', 'Physical Activity Level'])
    y = df[target_col]

    tab1, tab2 = st.tabs(["🎯 Training & Evaluation", "🔮 Prediction"])

    with tab1:
        st.subheader("Model Training")

        col1, col2 = st.columns([1, 3])

        with col1:
            st.markdown("#### Options")

            model_choice = st.selectbox(
                "Model:",
                ["Random Forest", "Decision Tree", "K-Nearest Neighbors"]
            )

            test_size = st.slider("Test-Set Größe:", 0.1, 0.4, 0.2, 0.05)

            train_button = st.button("🚀 Start Training", type="primary")

        with col2:
            
            if train_button:
                with st.spinner("Trainiere Modell..."):
                    # Train/Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    st.session_state.feature_cols = X_train.columns.tolist()

                    if model_choice == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
                    elif model_choice == "Decision Tree":
                        model = DecisionTreeRegressor(max_depth=5, random_state=42)
                    else:  # KNN
                        model = KNeighborsRegressor(n_neighbors=5)

                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.session_state.model = model

                    st.success("✅ Training finished!")

                    st.markdown("#### 📊 Results")

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        st.metric("RMSE", f"{rmse:.2f}")

                    with col_b:
                        st.metric("MAE", f"{mae:.2f}")

                    with col_c:
                        st.metric("R²", f"{r2:.2%}")

                    with st.expander("📋 Predicted vs Actual"):
                        comparison = pd.DataFrame({
                            "Actual": y_test,
                            "Predicted": y_pred
                        }).reset_index(drop=True)
                        st.dataframe(comparison)

                    


    with tab2:
        st.subheader("Individual Prediction")
        st.info("💡 Fill in your own data for a Prediction")

        if st.session_state.model is None:
            st.warning("⚠️ **Hinweis:** Diese Funktion benötigt ein trainiertes Modell. "
                   "Bitte trainieren Sie erst ein Modell im Training-Tab!")
            
        with st.form("prediction_form"):
            st.markdown("#### Individual data")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("Age", 20, 100, 50)
                sleep_duration = st.number_input("Sleep Duration", 3.0, 10.0, 5.0, 0.1)
                

            with col2:
                sleep_disorder = st.selectbox("Sleep Disorder", options=[0, 1], format_func=lambda x: "No Disorder" if x==0 else "Has Disorder")
                stress_lvl = st.number_input("Stress Level", 3, 9, 5, 1)

            with col3:
                heart_rate = st.number_input("Heart Rate", 60, 90, 70, 1)
                

            submitted = st.form_submit_button("🔮 Prediction")

            if submitted:
                if 'model' not in st.session_state:
                    st.warning("⚠️ Bitte trainiere zuerst ein Modell im Training-Tab!")
                else:
                    model = st.session_state.model
                    input_data = pd.DataFrame({
                        "Age": [age],
                        "Sleep Duration": [sleep_duration],
                        "Sleep Disorder": [sleep_disorder],
                        "Stress Level": [stress_lvl],
                        "Heart Rate": [heart_rate],
                    })

                    input_data = input_data[st.session_state.feature_cols]

                    prediction = model.predict(input_data)
                    st.success(f"🔮 Vorhersage der Schlafqualität: {prediction[0]:.2f}")


            





except FileNotFoundError:
    st.error("❌ File not found: `data/Sleep_health_and_lifestyle_dataset_cleaned.csv`")

st.markdown('---')
st.caption("DataPy WiSe25/26 - Sleep Quality Prediction")