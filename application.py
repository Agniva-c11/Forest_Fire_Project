import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained scaler and model
scaler = pickle.load(open('scaler.pkl', 'rb'))
ridge = pickle.load(open('ridge.pkl', 'rb'))

st.title("Forest Fire Weather Index (FWI) Prediction")

def user_input_features():
    # Collect user input from sidebar
    Temperature = st.sidebar.slider('Temperature', 0, 100, 50)
    RH = st.sidebar.slider('Relative Humidity (%)', 0, 100, 50)
    Ws = st.sidebar.slider('Wind Speed (km/h)', 0, 50, 10)
    Rain = st.sidebar.slider('Rain (mm/m^2)', 0.0, 10.0, 0.1)
    FFMC = st.sidebar.slider('Fine Fuel Moisture Code', 0.0, 100.0, 85.0)
    DMC = st.sidebar.slider('Duff Moisture Code', 0.0, 300.0, 150.0)
    ISI = st.sidebar.slider('Initial Spread Index', 0.0, 50.0, 10.0)
    Classes = st.sidebar.selectbox('Fire Class (Not fire=0, Fire=1)', [0, 1])
    Region = st.sidebar.selectbox('Region (North=0, South=1)', [0, 1])
    
# Create a dictionary of input features
    data = {
        'Temperature': Temperature,
        'RH': RH,
        'Ws': Ws,
        'Rain': Rain,
        'FFMC': FFMC,
        'DMC': DMC,
        'ISI': ISI,
        'Classes': Classes,
        'Region': Region,
    }
    
# Convert the dictionary into a dataframe
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Display input data
st.subheader('User Input Parameters')
st.write(input_df)

# Scale the input data
input_scaled = scaler.transform(input_df)

# Make prediction using the trained model
prediction = ridge.predict(input_scaled)

# Display the prediction result
st.subheader('Predicted Fire Weather Index (FWI)')
st.write(prediction[0])