import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Memuat model yang sudah dibuat
model = joblib.load('rf_heart.model')

# Fungsi untuk menyimpan prediksi ke dalam file CSV
def save_prediction(data, prediction):
    csv_filename = 'prediction_history.csv'
    # Cek jika file sudah ada
    if not os.path.isfile(csv_filename):
        # Jika tidak ada, buat DataFrame kosong dengan kolom-kolom yang sesuai
        df = pd.DataFrame(columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'Prediction'])
        df.to_csv(csv_filename, index=False)
    
    # Buka file CSV untuk menulis
    with open(csv_filename, 'a') as f:
        # Buat DataFrame untuk data yang akan ditulis
        data['Prediction'] = prediction
        df = pd.DataFrame(data, index=[0])
        # Tulis baris data ke file CSV
        df.to_csv(f, header=False, index=False)

# Streamlit UI
st.title('Heart Disease Prediction')

col1, col2 = st.columns(2)

# Create form for user inputs
with st.form(key="information", clear_on_submit=True):
    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=40)
        chest_pain_type = st.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
        cholesterol = st.number_input('Cholesterol', min_value=0, max_value=600, value=200)
        resting_ecg = st.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
        exercise_angina = st.selectbox('Exercise Angina', ('N', 'Y'))
        st_slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))
    with col2:
        sex = st.selectbox('Sex', ('M', 'F'))
        resting_bp = st.number_input('Resting BP', min_value=0, max_value=300, value=120)
        fasting_bs = st.selectbox('Fasting BS', (0, 1))
        max_hr = st.number_input('Max HR', min_value=0, max_value=220, value=120)
        oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=10.0, value=0.0)

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
    prediction = model.predict(input_data)[0]
    
    # Save the prediction
    save_prediction(input_data.iloc[0].to_dict(), prediction)

    # Display the prediction result
    if prediction == 1:
        st.write('The model predicts that this person **has heart disease**.')
    else:
        st.write('The model predicts that this person **does not have heart disease**.')

# Display Prediction History
if st.checkbox('Show Prediction History'):
    if os.path.isfile('prediction_history.csv'):
        history = pd.read_csv('prediction_history.csv')
        st.write(history)
    else:
        st.write("No prediction history available.")
