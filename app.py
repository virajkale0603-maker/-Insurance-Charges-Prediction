import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title='Insurance Cost Predictor', page_icon='🏥', layout='centered'
)


# Load the trained model
@st.cache_resource
def load_artifacts():
  model = joblib.load('insurance_model.pkl')
  return model


model = load_artifacts()

st.title('🏥 Insurance Charges Prediction App')
st.write(
    'Enter your details below to estimate your medical insurance costs using'
    ' your trained machine learning model.'
)

st.divider()

# Input Form Layout
with st.form('prediction_form'):
  col1, col2 = st.columns(2)

  with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input(
        'Number of Children', min_value=0, max_value=10, value=0
    )

  with col2:
    sex = st.selectbox('Sex', ['female', 'male'])
    smoker = st.selectbox('Smoker', ['no', 'yes'])
    region = st.selectbox(
        'Region', ['northeast', 'northwest', 'southeast', 'southwest']
    )

  submitted = st.form_submit_button('Predict Cost')

# Handle Prediction Logic on Form Submit
if submitted:
  # Re-create the exact engineered features from your notebook
  is_female_val = 1 if sex == 'female' else 0
  is_smoker_val = 1 if smoker == 'yes' else 0
  bmi_category_obese_val = 1 if bmi >= 30.0 else 0

  # Build the input DataFrame with the exact column names expected by the model
  input_data = {
      'age': [age],
      'bmi': [bmi],
      'children': [children],
      'is_female': [is_female_val],
      'is_smoker': [is_smoker_val],
      'bmi_category_Obese': [bmi_category_obese_val],
      # If your model also expects region_southwest or other regions, include them here:
      'region_southwest': [1 if region == 'southwest' else 0],
  }

  input_df = pd.DataFrame(input_data)

  # Align columns strictly to what the model was fitted on (safeguard against extra/missing columns)
  # If your model has a feature_names_in_ attribute, we can enforce it:
  if hasattr(model, 'feature_names_in_'):
    # Reindex ensures columns match training order and fills any missing columns with 0
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

  # Make prediction
  prediction = model.predict(input_df)

  # Display Result
  st.success(
      f'### Estimated Medical Insurance Cost: ${prediction[0]:,.2f} USD'
  )