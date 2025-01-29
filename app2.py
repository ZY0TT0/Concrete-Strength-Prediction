import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


@st.cache_data
def load_data():
    data = pd.read_csv('conc.csv')  
    return data


st.title("Concrete Strength Prediction using XGBoost")


data = load_data()


st.write("### Dataset Preview")
st.write(data.head())

st.write("""
This app predicts the compressive strength of concrete based on various features.
Fill in the values below and press predict to see the concrete strength prediction.
""")


cement = st.number_input("Cement (kg in a m³ mixture):", min_value=0.0, max_value=500.0, value=150.0)
blast_furnace_slag = st.number_input("Blast Furnace Slag (kg in a m³ mixture):", min_value=0.0, max_value=500.0, value=100.0)
fly_ash = st.number_input("Fly Ash (kg in a m³ mixture):", min_value=0.0, max_value=500.0, value=50.0)
water = st.number_input("Water (kg in a m³ mixture):", min_value=0.0, max_value=300.0, value=200.0)
superplasticizer = st.number_input("Superplasticizer (kg in a m³ mixture):", min_value=0.0, max_value=20.0, value=5.0)
coarse_aggregate = st.number_input("Coarse Aggregate (kg in a m³ mixture):", min_value=0.0, max_value=1200.0, value=900.0)
fine_aggregate = st.number_input("Fine Aggregate (kg in a m³ mixture):", min_value=0.0, max_value=1000.0, value=800.0)
age = st.number_input("Age (days):", min_value=1, max_value=365, value=28)


@st.cache_data
def train_model(data):
    # Replace the target column with the actual name
    X = data.drop('Concrete compressive strength(MPa, megapascals) ', axis=1)
    y = data['Concrete compressive strength(MPa, megapascals) ']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the XGBoost Regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

# Predict Button
if st.button("Predict Strength"):
    # Train the model and get the scaler
    model, scaler = train_model(data)

    # Prepare the input data for the model
    input_data = np.array([[cement, blast_furnace_slag, fly_ash, water, superplasticizer, coarse_aggregate, fine_aggregate, age]])
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    st.success(f"The predicted concrete strength is {prediction[0]:.2f} MPa")
