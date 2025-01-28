import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle


def load_data():
    df = pd.read_csv("diabetes.csv")
    return df


def train_models():
    df = load_data()

    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    
    pickle.dump(rf_model, open("rf_model.pkl", "wb"))
    pickle.dump(lr_model, open("lr_model.pkl", "wb"))
    pickle.dump(dt_model, open("dt_model.pkl", "wb"))


def predict_diabetes(model_name, input_data):
    
    model = pickle.load(open(f"{model_name}.pkl", "rb"))
    prediction = model.predict(input_data)
    return prediction[0]


st.title("Diabetes Prediction App")
st.write(" ")
st.write(" ")

col1, col2 = st.columns([1, 1])

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, value=None, step=1)
    glucose = st.number_input("Glucose Concentration", min_value=0, value=None)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, value=None)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, value=None)
with col2:
    insulin = st.number_input("Insulin", min_value=0, value=None)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, value=None, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=None, step=0.01)
    age = st.number_input("Age", min_value=0, value=None, step=1)



model_option = st.selectbox("Select a model", ("Random Forest", "Logistic Regression", "Decision Tree"))


if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    model_map = {
        "Random Forest": "rf_model",
        "Logistic Regression": "lr_model",
        "Decision Tree": "dt_model"
    }
    model_name = model_map[model_option]
    result = predict_diabetes(model_name, input_data)

    if any(val == 0 or val is None for val in [glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]):
        st.error("All input fields are required. Please fill in all fields.")

    if result == 1:
        st.error("The model predicts that you are diabetic.")
    else:
        st.success("The model predicts that you are not diabetic.")

# Uncomment the line to train models on initial run
train_models()

