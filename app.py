# Use this code to execute the file: streamlit run app.py 
# To exit: Ctrl+C
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and scaler
with open("linear_regression_model.pkl", "rb") as file: # read binary
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file: # read binary
    scaler = pickle.load(file)

# Define the Streamlit app
st.title("Tip Prediction App")
st.write("Enter the details below to predict the tip amount: ")

# Input fields for the features
total_bill = st.number_input("Total Bill", min_value=0.0, step=0.01)
size = st.number_input("Size of the party", min_value=1, step=1)
sex = st.selectbox("Gender", ["Male", "Female"])
smoker = st.selectbox("Smoker", ["Yes", "No"])
day = st.selectbox("Day of the week", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("Time of the day", ["Lunch", "Dinner"])

# Encode the categorical features
sex_male = 1 if sex == "Male" else 0
sex_female = 1 if sex == "Female" else 0
smoker_yes = 1 if smoker == "Yes" else 0
smoker_no = 1 if smoker == "No" else 0
day_mapping = {"Thur": [1,0,0,0], "Fri": [0,1,0,0], "Sat": [0,0,1,0], "Sun": [0,0,0,1]}
day_encoded = day_mapping[day]
time_lunch = 1 if time == "Lunch" else 0
time_dinner = 1 if time == "Dinner" else 0

# print(sex_male, sex_female, smoker_yes, smoker_no, day_encoded, time_lunch, time_dinner)

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    "total_bill": [total_bill],
    "size": [size],
    "sex_Female": [sex_female],
    "smoker_No": [smoker_no],
    "day_Fri": [day_encoded[1]],
    "day_Sat": [day_encoded[2]],
    "day_Sun": [day_encoded[3]],
    "time_Dinner": [time_dinner],
})
# print(input_data.head())

# Align the input_data with the training data's feature order
input_data = input_data[["total_bill", "size", "sex_Female", "smoker_No", "day_Fri", "day_Sat", "day_Sun", "time_Dinner"]]

# Scale the input data
input_data_scaled = scaler.transform(input_data)
# print(input_data_scaled)

# Predict the tip amount
predicted_tip = model.predict(input_data_scaled)
# print(predicted_tip)

# Display the predicted tip amount
# st.write("Predicted Tip Amount:", predicted_tip)
st.write("Predicted Tip Amount:", predicted_tip[0])