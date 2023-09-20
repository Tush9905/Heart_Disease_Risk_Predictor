import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load(filename="Model.pkl")

st.write("# Heart Disease Risk Predictor")

with st.form("Details_Form"):
    st.write("Please fill in your details")
    age = st.number_input(label="Age")
    sex = st.radio(label="Sex", options=["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", options=["None", "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    restBP = st.number_input(label="Resting Blood Pressure")
    chol = st.number_input(label="Serum Cholestrol in mg/dL")
    fbs = st.toggle(label="Fasting Blood Sugar more than > 120 mg/dl")
    restEcg = st.selectbox("resting electrocardiographic results", options=["Normal", "ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)", "Showing probable or definite left ventricular hypertrophy by Estes' criteria"])
    thalach = st.number_input("Max Heart Rate / Pulse")
    exang = st.toggle(label="Exercise induced angina")
    oldpeak = st.number_input("ST depression induced by exercise relative to rest")
    slope = st.number_input("the slope of the peak exercise ST segment")
    ca = st.slider("number of major vessels (0-3) colored by flourosopy", min_value=0, max_value=3)
    thal = st.selectbox(label="Thal", options=["Normal", "Fixed defect", "Reversable defect"])
    submitted = st.form_submit_button("Submit")

    Non_number_Features = {"sex" : sex, "fbs" : fbs, "restEcg" : restEcg, "exang" : exang, "thal" : thal, "cp" : cp}
    conv = {}

    if submitted:
        Map = {"Male" : 1, "Female" : 0, True : 1, False : 0, "Normal" : 0, "Fixed defect" : 1, "Reversable defect" : 2, 
               "ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)" : 1, "Showing probable or definite left ventricular hypertrophy by Estes' criteria" : 2, 
               "None" : 0, "Typical Angina" : 1, "Atypical Angina" : 2, "Non-Anginal Pain" : 3, "Asymptomatic" : 4}
        
        for i in Non_number_Features:
            conv[f"{i}"] = Map.get(Non_number_Features[i])
            
        pred = model.predict([[age, conv["sex"], conv["cp"], restBP, chol, conv["fbs"], conv["restEcg"], thalach, conv["exang"], oldpeak, slope, ca, conv["thal"]]])

        if pred == 1:
            st.write("You may be at risk of having heart disease")
        else:
            st.write("You have no risk of heart disease!")

st.write("Plese Note:- This is not a diagnosis tool and is only made for educational purposes.")
            

        
        





