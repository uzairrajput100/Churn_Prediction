import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Telecommunication Company')
st.title('Customer Churn Prediction Model')

# Input fields
monthly_charge = st.number_input('Monthly Charge', min_value=0.0, max_value=1000.0, value=50.0)
total_charges = st.number_input('Total Charges', min_value=0.0, max_value=10000.0, value=500.0)
gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
married = st.selectbox('Married', ['No', 'Yes'])
dependents = st.selectbox('Dependents', ['No', 'Yes'])
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['No', 'Yes'])
online_security = st.selectbox('Online Security', ['No', 'Yes'])
online_backup = st.selectbox('Online Backup', ['No', 'Yes'])
device_protection_plan = st.selectbox('Device Protection Plan', ['No', 'Yes'])
premium_tech_support = st.selectbox('Premium Tech Support', ['No', 'Yes'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'])
contract = st.selectbox('Contract', ['Month-to-Month', 'One Year', 'Two Year'])
paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
payment_method = st.selectbox('Payment Method', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])
tenure_group = st.selectbox('Tenure Group', ['1 - 12', '13 - 24', '25 - 36', '37 - 48', '49 - 60', '61 - 72'])

# Prepare the input data
input_data = pd.DataFrame({
    'Monthly Charge': [monthly_charge],
    'Total Charges': [total_charges],
    f'Gender_{gender}': [1],
    f'Senior Citizen_{senior_citizen}': [1],
    f'Married_{married}': [1],
    f'Dependents_{dependents}': [1],
    f'Phone Service_{phone_service}': [1],
    f'Multiple Lines_{multiple_lines}': [1],
    f'Internet Service_{internet_service}': [1],
    f'Online Security_{online_security}': [1],
    f'Online Backup_{online_backup}': [1],
    f'Device Protection Plan_{device_protection_plan}': [1],
    f'Premium Tech Support_{premium_tech_support}': [1],
    f'Streaming TV_{streaming_tv}': [1],
    f'Streaming Movies_{streaming_movies}': [1],
    f'Contract_{contract.replace(" ", "-")}': [1],
    f'Paperless Billing_{paperless_billing}': [1],
    f'Payment Method_{payment_method.replace(" ", "-")}': [1],
    f'Tenure group_{tenure_group.replace(" ", "-")}': [1],
})

# Fill missing columns with 0
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder the columns to match the model's input
input_data = input_data[model.feature_names_in_]

# Make predictions
if st.button('Predict'):
    prediction = model.predict(input_data)
    churn_label = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'Based on the provided data, the prediction indicates that the customer will continue using the services : {churn_label}')

st.divider()
st.write("This model will assist in predicting Customer behaviour, Improve customer retention, Customer satifaction, Data-Driven decision and Save the cost")
st.write("This prediction model has developed by Uzair Ahmed")
st.write("Email: uzairrajput100@gmail.com")