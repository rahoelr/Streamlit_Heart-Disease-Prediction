import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Memuat model yang sudah dibuat
model = joblib.load('rf_heart.model')

# Streamlit UI
st.title('Heart Disease Prediction')

# Create form for user inputs
with st.form(key="information", clear_on_submit=True):
    age = st.number_input('Age', min_value=1, max_value=120, value=40)
    sex = st.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.number_input('Resting BP', min_value=0, max_value=300, value=120)
    cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
    fasting_bs = st.selectbox('Fasting BS', (0, 1))
    resting_ecg = st.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.number_input('Max HR', min_value=0, max_value=220, value=120)
    exercise_angina = st.selectbox('Exercise Angina', ('N', 'Y'))
    oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=0.0)
    st_slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))

    # Submit button inside the form
    submit_button = st.form_submit_button(label='Predict')

# Check if the form has been submitted
if submit_button:
    # Convert input data to dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # Encode the input data automatically using LabelEncoder
    for column in input_data.columns:
        if input_data[column].dtype == 'object':
            le = LabelEncoder()
            input_data[column] = le.fit_transform(input_data[column])

    # Predict using the model
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write('The model predicts that this person **has heart disease**.')
    else:
        st.write('The model predicts that this person **does not have heart disease**.')
