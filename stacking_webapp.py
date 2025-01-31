# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from streamlit.components.v1 import components
import logging
from typing import Tuple, List
import base64


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

variable_descriptions = {
    # Numerical features
    'liggetid ⏳': 'Length of stay in the hospital (in days)',
    'medication_count_per_patient 💊': 'Total number of medications prescribed to the patient',
    'charlson_comorbidity_index ⚕️': 'Index measuring patient\'s comorbidity level',
    'dbi_drug_count 💉': 'Number of drugs with anticholinergic or sedative effects',
    'count_labtest_kpr 🔬': 'Number of laboratory tests performed',
    'days_since_last_gp_visit 📅': 'Days elapsed since the last general practitioner visit',
    'one_year_admissions 📆': 'Number of hospital admissions in the past year',
    'one_month_visits 🛏️': 'Number of healthcare visits in the past month',
    'previous_admissions 🏥': 'Total number of previous hospital admissions',
    'time_since_last_admission_days 📌': 'Days elapsed since the last hospital admission',
    'diag_importance ❗': 'Importance score of the primary diagnosis',
    
    # Categorical features
    'visit_trend 📈': 'Pattern of healthcare visits over time',
    'atc_category 🫀': 'Anatomical Therapeutic Chemical classification of medications',
    'omsnivahenv 🚑': 'Level of care needed',
    'aktivitetskategori 📝': 'Category of healthcare activity',
    'interaction_flag ⚠️': 'Presence of severe drug interaction in medication list',
    'pasient_kjonn_verdi ♀️♂️': 'Patient gender',
    'omsorgsniva 👥': 'Level of care provided',
    'module_number_atc 🔢': 'Module number based comedicaion pattern',
    'admissions_trend 📉': 'Pattern of hospital admissions over time',
    'utlevering_resepttype_verdi 💼': 'Type of prescription',
    'patient_overall_adherence_status ✅': 'Overall patient adherence'
}

# Load ATC modules mapping
@st.cache_resource
def load_atc_modules() -> pd.DataFrame:
    """Load ATC modules mapping from a CSV file."""
    try:
        return pd.read_csv('models/ATC_modules.csv')
    except FileNotFoundError:
        st.error("ATC_modules.csv file not found. Please upload it.")
        return pd.DataFrame(columns=['atc', 'module_number_atc'])

# Map ATC code to module number
def get_module_number(atc_code: str, atc_modules_df: pd.DataFrame) -> str:
    """Return the module number for the given ATC code, or 'unknown' if not found."""
    module_row = atc_modules_df[atc_modules_df['atc'] == atc_code]
    if not module_row.empty:
        return str(module_row['module_number_atc'].iloc[0])
    return 'unknown'

# Load model and preprocessors with caching
@st.cache_resource
def load_model():
    """Load the trained MLPClassifier model."""
    try:
        return joblib.load('models/mlp_model_deploy.joblib') #
    except FileNotFoundError:
        st.error("Model file not found.")
        return None

@st.cache_resource
def load_preprocessing_objects():
    """Load preprocessing objects: imputer, scaler, encoder."""
    try:
        num_imputer = joblib.load('models/num_imputer_deploy.joblib')
        scaler = joblib.load('models/scaler_deploy.joblib')
        cat_imputer = joblib.load('models/cat_imputer_deploy.joblib')
        encoder = joblib.load('models/encoder_deploy.joblib')
        return num_imputer, scaler, cat_imputer, encoder
    except FileNotFoundError as e:
        st.error(f"Preprocessing file not found: {e}")
        return None

@st.cache_resource
def load_shap_explainer():
    """Load the SHAP explainer."""
    try:
        return joblib.load('models/shap_explainer_deploy.joblib')
    except FileNotFoundError:
        st.error("SHAP explainer file not found.")
        return None

# Load feature names
def load_feature_names() -> List[str]:
    """Load feature names from file."""
    try:
        with open('models/selected_features.txt', 'r') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        st.error("selected_features.txt not found.")
        return []

# Load resources
model = load_model()
preprocessing_objects = load_preprocessing_objects()
if preprocessing_objects:
    num_imputer, scaler, cat_imputer, encoder = preprocessing_objects
explainer = load_shap_explainer()
feature_names = load_feature_names()
atc_modules_df = load_atc_modules()

if not all([model, preprocessing_objects, explainer, feature_names, not atc_modules_df.empty]):
    st.stop()

# Define numerical and categorical features
numerical_features = [
    'liggetid', 'medication_count_per_patient', 'charlson_comorbidity_index',
    'dbi_drug_count', 'count_labtest_kpr', 'days_since_last_gp_visit',
    'one_year_admissions', 'one_month_visits', 'previous_admissions',
    'time_since_last_admission_days', 'diag_importance'
]

categorical_features = [
    'visit_trend', 'atc_category', 'omsnivahenv', 'aktivitetskategori',
    'interaction_flag', 'pasient_kjonn_verdi', 'omsorgsniva',
    'module_number_atc', 'admissions_trend', 'utlevering_resepttype_verdi',
    'patient_overall_adherence_status'
]

def preprocess_categorical_value(value: str) -> str:
    """Return consistent categorical input as string."""
    return str(value) if value != 'unknown' else 'unknown'

feature_emojis = {
    # Numerical features
    "liggetid": "⏳",  
    "medication_count_per_patient": "💊",  
    "charlson_comorbidity_index": "⚕️",  
    "dbi_drug_count": "💉",  
    "count_labtest_kpr": "🔬",  
    "days_since_last_gp_visit": "📅",  
    "one_year_admissions": "📆",  
    "one_month_visits": "🛏️",  
    "previous_admissions": "🏥",  
    "time_since_last_admission_days": "📌",  
    "diag_importance": "❗",  

    # Categorical features
    "visit_trend": "📈",  
    "atc_category": "🫀",  
    "omsnivahenv": "🚑",  
    "aktivitetskategori": "📝",  
    "interaction_flag": "⚠️",  
    "pasient_kjonn_verdi": "♀️♂️",  
    "omsorgsniva": "👥",  
    "module_number_atc": "🔢",  
    "admissions_trend": "📉",  
    "utlevering_resepttype_verdi": "💼",  
    "patient_overall_adherence_status": "✅"  
}

def user_input_features() -> pd.DataFrame:
    """Collect user input for numerical and categorical features."""
    input_data = {}

    # Numerical features
    for feature in numerical_features:
        emoji = feature_emojis.get(feature, "")
        feature_label = f"{emoji} {feature}"
        input_data[feature] = st.sidebar.number_input(feature_label, value=0.0)

    # Categorical features with predefined options
    categorical_options = {
        'visit_trend': ['Fluctuating', 'Increasing', 'Unknown'],
        'atc_category': [
            'alimentary', 'antineoplastic', 'antiinfectives', 'blood',
            'cardiovascular', 'dermatologicals', 'genitourinary', 'immunomodulating',
            'musculoskeletal system', 'nervous', 'respiratory', 'sensory',
            'sys hormonal', 'various', 'unknown'
        ],
        'omsnivahenv': ['Level 1', 'Level 2', 'Level 3', 'Level 8', 'unknown'],
        'aktivitetskategori': ['Category 1', 'Category 2', 'Category 3', 'unknown'],
        'interaction_flag': ['No', 'Yes', 'unknown'],
        'pasient_kjonn_verdi': ['Male', 'Female', 'unknown'],
        'omsorgsniva': ['Level 1', 'Level 2', 'Level 3', 'Level 8'],
        'admissions_trend': ['Fluctuating', 'Increasing', 'Unknown'],
        'utlevering_resepttype_verdi': [f'Recipe Type {i}' for i in range(1, 7)] + ['unknown'],
        'patient_overall_adherence_status': ['No', 'Yes', 'unknown']
    }

    # Dynamically render selectbox for each categorical feature
    for feature in categorical_features:
        emoji = feature_emojis.get(feature, "")
        
        if feature == 'module_number_atc':
            # Keep module_number_atc as a free text input for user to type their ATC code
            atc_code = st.sidebar.text_input(f"{emoji} Enter ATC code for {feature}", value='unknown')
            module_number = get_module_number(atc_code, atc_modules_df)
            st.sidebar.write(f"Module Number: {module_number}")
            input_data[feature] = module_number
        elif feature in categorical_options:
            label = f"{emoji} {feature}"
            input_data[feature] = st.sidebar.selectbox(label, options=categorical_options[feature])
        else:
            label = f"{emoji} {feature}"
            input_data[feature] = st.sidebar.text_input(label, value='unknown')

    return pd.DataFrame([input_data])



@st.cache_data
def preprocess_input(input_df):
    """Preprocess the user input data."""
    input_df = input_df[numerical_features + categorical_features]

    # Numerical preprocessing
    X_num = scaler.transform(num_imputer.transform(input_df[numerical_features]))

    # Categorical preprocessing
    X_cat = encoder.transform(cat_imputer.transform(input_df[categorical_features]))

    # Combine features
    X_processed = np.hstack([X_num, X_cat])

    selected_feature_names = np.concatenate([numerical_features, encoder.get_feature_names_out(categorical_features)])
    return X_processed, selected_feature_names

def st_shap(plot, height=None):
    """Display a SHAP force plot in Streamlit with enforced horizontal scrolling."""
    shap_html = f"""
    <head>{shap.getjs()}</head>
    <div style="overflow-x: scroll; width: 100%; height: {height or 500}px;">
        {plot.html()}
    </div>
    """
    st.components.v1.html(shap_html, height=height + 50 if height else 550, scrolling=False)

st.title('Norwegian Older Patients Admission Prediction App')

# 1) Variable Descriptions in a Sidebar Expander
with st.sidebar.expander("**Variable Descriptions**", expanded=False):
    for var_name, description in variable_descriptions.items():
        st.markdown(f"**{var_name}:** {description}")
        
input_df = user_input_features()

if st.sidebar.button('Predict'):
    X_input, selected_feature_names = preprocess_input(input_df)

    prediction_proba = model.predict_proba(X_input)[0][1]
    prediction = model.predict(X_input)[0]

    st.write(f"**Prediction:** {'Admitted' if prediction == 1 else 'Not Admitted'}")
    st.write(f"**Probability of Admission:** {prediction_proba:.2f}")

    st.subheader('SHAP Force Plot')

    try:
        # Get SHAP values
        shap_values = explainer.shap_values(X_input)
        
        # Determine base value and SHAP values for single instance
        if isinstance(shap_values, list):
            # Multi-output (e.g., binary classification)
            base_value = explainer.expected_value[1]  # Positive class expected value
            shap_vals = shap_values[1][0]  # SHAP values for the positive class, first sample
        else:
            # Single-output model
            base_value = explainer.expected_value
            shap_vals = shap_values[0]

        # Generate force plot
        force_plot = shap.force_plot(
            base_value,
            shap_vals,
            X_input[0],  # First instance's feature values
            feature_names=selected_feature_names,
            matplotlib=False  # Generate interactive HTML
        )

        # Display the force plot using st.components.v1.html
        st_shap(force_plot)

    except Exception as e:
        st.error(f"Error generating SHAP force plot: {str(e)}")

        # Fallback: Show SHAP summary plot
        st.write("Falling back to feature importance summary plot...")
        fig, ax = plt.subplots()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_input, feature_names=selected_feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_input, feature_names=selected_feature_names, show=False)
        st.pyplot(fig)

# Convert the image to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load your image from a local path
image_path = (r"models/cartoon.JPG")
# Get the base64 string of the image
image_base64 = image_to_base64(image_path)

# Display your image and name in the top right corner
st.markdown(
    f"""
    <style>
    .header {{
        position: absolute;  /* Fix the position */
        top: 10px;  /* Adjust as needed */
        right: 10px;  /* Align to the right */
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding: 10px;
        flex-direction: column; /* Stack items vertically */
        text-align: center; /* Ensures text is centrally aligned */
    }}
    .header img {{
        border-radius: 50%;
        width: 50px;
        height: 50px;
        margin-bottom: 5px; /* Space between image and text */
    }}
    .header-text {{
        font-size: 12px;
        font-weight: normal; /* Regular weight for text */
        text-align: center;
    }}
    </style>
    <div class="header">
        <img src="data:image/jpeg;base64,{image_base64}" alt="Mohsen Askar">
        <div class="header-text">Developed by: Mohsen Askar</div>
    </div>
    """,
    unsafe_allow_html=True
)

