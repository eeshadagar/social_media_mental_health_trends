import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

# Load models and encoders
pca = joblib.load('model/pca.pkl')
model = joblib.load('model/model.pkl')
risk_model = joblib.load('model/risk_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# Streamlit UI
st.title("Mental Health Prediction Based on Social Media Use")
st.markdown("Fill the form to get a prediction")

with st.form("user_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    relationship = st.selectbox("Relationship Status", ["Single", "In a relationship", "Married", "Divorced"])
    avg_time = st.selectbox("Average Time Spent on Social Media per Day", [
        "Less than an Hour", 
        "Between 1 and 2 hours", 
        "Between 2 and 3 hours", 
        "Between 3 and 4 hours", 
        "Between 4 and 5 hours", 
        "More than 5 hours"
    ])

    st.markdown("Which platforms do you use?")
    Pinterest = st.checkbox("Pinterest")
    YouTube = st.checkbox("YouTube")
    Reddit = st.checkbox("Reddit")
    TikTok = st.checkbox("TikTok")
    Instagram = st.checkbox("Instagram")
    Facebook = st.checkbox("Facebook")
    Snapchat = st.checkbox("Snapchat")
    Discord = st.checkbox("Discord")

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Prepare input
        input_data = {
            "age": age,
            "Pinterest": float(Pinterest),
            "YouTube": float(YouTube),
            "Reddit": float(Reddit),
            "TikTok": float(TikTok),
            "Instagram": float(Instagram),
            "Facebook": float(Facebook),
            "Snapchat": float(Snapchat),
            "Discord": float(Discord),
            "gender": gender,
            "relationship": relationship,
            "avg_time_per_day": avg_time
        }

        df_input = pd.DataFrame([input_data])
        
        # Predict risk first
        enc_risk = pd.get_dummies(df_input[["gender", "relationship", "avg_time_per_day"]])
        df_risk = pd.concat([df_input.drop(columns=["gender", "relationship", "avg_time_per_day"]), enc_risk], axis=1)
        feature_cols = ['age', 'Pinterest', 'YouTube', 'Reddit', 'TikTok', 'Instagram', 'Facebook', 'Snapchat', 'Discord', 'gender_Female', 'gender_Male', 'gender_other', 'relationship_Divorced', 'relationship_In a relationship', 'relationship_Married', 'relationship_Single', 'avg_time_per_day_Between 1 and 2 hours', 'avg_time_per_day_Between 2 and 3 hours', 'avg_time_per_day_Between 3 and 4 hours', 'avg_time_per_day_Between 4 and 5 hours', 'avg_time_per_day_Less than an Hour', 'avg_time_per_day_More than 5 hours']
        # One-hot encode
        input_encoded = pd.get_dummies(df_input)
        input_encoded = input_encoded.reindex(columns=feature_cols, fill_value=0)
        
        # Predict
        prediction = model.predict(input_encoded)[0]
        result = label_encoder.inverse_transform([prediction])[0]
        labell = "lower" if result <= 2 else "higher"

        if labell == "higher":
            st.warning(f"⚠️ Prediction: {labell}")
        else:
            st.success(f"✅ Prediction: {labell}")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
