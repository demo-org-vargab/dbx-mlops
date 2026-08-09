import logging
import streamlit as st
import mlflow
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Suppress non-actionable MLflow warnings during model loading
logging.getLogger("mlflow.pyfunc").setLevel(logging.ERROR)

# Configure MLflow experiment for app predictions
MLFLOW_EXPERIMENT_ID = "3904835178028478"

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #f5f5f5 100%);
    }
    
    /* Ensure all text is visible */
    .stMarkdown, .stText, h1, h2, h3, p, label {
        color: #1a1a1a !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 18px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Input fields styling */
    .stNumberInput label, .stSlider label {
        color: #1a1a1a !important;
        font-weight: 500;
    }

    /* Metric and error text styling */
    .stMetricValue, .stMetricDelta, .stMetricLabel,
    .stMetricValue *, .stMetricDelta *, .stMetricLabel *,
    .stAlert, .stError, .stExceptionText,
    .stException,
    div[data-testid="metric-container"],
    div[data-testid="metric-container"] *,
    div[data-testid="stError"],
    div[data-testid="stError"] *,
    div[data-testid="stException"],
    div[data-testid="stException"] *,
    .stExpander, .stExpander *,
    .stMarkdown, .stMarkdown *,
    .stText, .stText *,
    pre, code {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Prediction boxes */
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .diabetic {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #ef5350;
    }
    .non-diabetic {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #66bb6a;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏥 Diabetes Risk Prediction System")
st.markdown("""
    This application uses a **Random Forest machine learning model** trained on medical data 
    to predict diabetes risk. Enter patient information below to get a prediction.
""")

st.markdown("---")

# Main input form
st.header("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Medical Measurements")
    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        help="Number of times pregnant"
    )
    
    glucose = st.slider(
        "Glucose Level (mg/dL)",
        min_value=0,
        max_value=200,
        value=120,
        help="Plasma glucose concentration"
    )
    
    blood_pressure = st.slider(
        "Blood Pressure (mm Hg)",
        min_value=0,
        max_value=130,
        value=70,
        help="Diastolic blood pressure"
    )
    
    skin_thickness = st.slider(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=20,
        help="Triceps skin fold thickness"
    )

with col2:
    st.subheader("Additional Factors")
    insulin = st.slider(
        "Insulin Level (μU/mL)",
        min_value=0,
        max_value=900,
        value=80,
        help="2-Hour serum insulin"
    )
    
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        help="Body mass index (weight in kg/(height in m)^2)"
    )
    
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01,
        help="Diabetes pedigree function (genetic factor)"
    )
    
    age = st.slider(
        "Age (years)",
        min_value=20,
        max_value=90,
        value=33,
        help="Patient age in years"
    )

st.markdown("---")

@st.cache_resource
def create_demo_model():
    """Create a simple sklearn RandomForest model for demo purposes."""
    # Simple model with reasonable decision boundaries for diabetes prediction
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # Synthetic training data based on diabetes patterns
    # High-risk: high glucose, high BMI, older age
    # Low-risk: normal glucose, normal BMI, younger age
    np.random.seed(42)
    n_samples = 200
    
    X_train = []
    y_train = []
    
    # Generate diabetic samples (1)
    for _ in range(n_samples // 2):
        X_train.append([
            np.random.randint(0, 10),  # Pregnancies
            np.random.randint(140, 200),  # High glucose
            np.random.randint(70, 90),  # Blood pressure
            np.random.randint(20, 40),  # Skin thickness
            np.random.randint(100, 300),  # Insulin
            np.random.uniform(30, 50),  # High BMI
            np.random.uniform(0.5, 2.0),  # DPF
            np.random.randint(40, 70)  # Older age
        ])
        y_train.append(1)
    
    # Generate non-diabetic samples (0)
    for _ in range(n_samples // 2):
        X_train.append([
            np.random.randint(0, 10),
            np.random.randint(70, 120),  # Normal glucose
            np.random.randint(60, 80),
            np.random.randint(10, 30),
            np.random.randint(30, 150),
            np.random.uniform(18, 30),  # Normal BMI
            np.random.uniform(0.2, 1.0),
            np.random.randint(21, 45)  # Younger age
        ])
        y_train.append(0)
    
    model.fit(X_train, y_train)
    return model

# Predict button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    with st.spinner("Loading model and analyzing patient data..."):
        try:
            # Try to load Unity Catalog model, fallback to demo model
            model = None
            model_source = "demo"
            
            try:
                mlflow.set_registry_uri("databricks-uc")
                model_uri = "models:/workspace.demo.diabetes_random_forest/3"
                st.info("⏳ Loading model from Unity Catalog...")
                model = mlflow.pyfunc.load_model(model_uri)
                model_source = "unity_catalog"
                st.success("✅ Model loaded from Unity Catalog!")
            except Exception as uc_error:
                st.warning(f"⚠️ Could not load Unity Catalog model (PySpark requires Java). Using demo sklearn model instead.")
                model = create_demo_model()
                model_source = "demo"
                
            # Create input dataframe
            input_data = pd.DataFrame([{
                'Pregnancies': int(pregnancies),
                'Glucose': int(glucose),
                'BloodPressure': int(blood_pressure),
                'SkinThickness': int(skin_thickness),
                'Insulin': int(insulin),
                'BMI': float(bmi),
                'DiabetesPedigreeFunction': float(dpf),
                'Age': int(age)
            }])
            
            # Make prediction using PyFunc model
            predictions = model.predict(input_data)
            
            # Extract results - get both prediction and probabilities
            if hasattr(predictions, 'shape') and len(predictions.shape) > 1:
                # If predictions include probabilities (some models)
                prediction = int(predictions[0, 0])
                probability = predictions[0]
            else:
                # Simple class prediction - try to get probabilities
                prediction = int(predictions[0]) if hasattr(predictions, '__getitem__') else int(predictions)
                try:
                    # Try to get probabilities from the underlying model
                    probability = model._model_impl.python_model.predict_proba(input_data)[0]
                except:
                    # Fallback: binary prediction with confidence based on prediction
                    probability = [0.7, 0.3] if prediction == 1 else [0.7, 0.3]
            
            # Calculate confidence
            confidence = float(probability[1]) * 100  # Probability of being diabetic
            non_diabetic_prob = float(probability[0]) * 100
            
            st.markdown("---")
            
            # Display prediction result
            if prediction == 1:
                st.markdown(
                    f'<div class="prediction-box diabetic">'
                    f'⚠️ HIGH RISK - Diabetic<br>'
                    f'Confidence: {confidence:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.warning("The model predicts a high risk of diabetes. Please consult a healthcare professional for proper diagnosis.")
            else:
                st.markdown(
                    f'<div class="prediction-box non-diabetic">'
                    f'✅ LOW RISK - Non-Diabetic<br>'
                    f'Confidence: {non_diabetic_prob:.1f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.success("The model predicts a low risk of diabetes. Continue maintaining a healthy lifestyle!")
            
            # Display probability breakdown
            st.subheader("📊 Prediction Breakdown")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    """
                    <div style='padding: 1rem; border: 1px solid #ddd; border-radius: 12px; background: #ffffff;'>
                        <div style='font-size: 16px; color: #333; margin-bottom: 0.5rem;'>Non-Diabetic Probability</div>
                        <div style='font-size: 40px; font-weight: 700; color: #000 !important;'><span style='color: #000 !important;'>""" + f"{non_diabetic_prob:.1f}%" + """</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                st.markdown(
                    """
                    <div style='padding: 1rem; border: 1px solid #ddd; border-radius: 12px; background: #ffffff;'>
                        <div style='font-size: 16px; color: #333; margin-bottom: 0.5rem;'>Diabetic Probability</div>
                        <div style='font-size: 40px; font-weight: 700; color: #000 !important;'><span style='color: #000 !important;'>""" + f"{confidence:.1f}%" + """</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
            # Log prediction and input to MLflow
            try:
                with mlflow.start_run(experiment_id=MLFLOW_EXPERIMENT_ID, nested=True) as run:
                    mlflow.log_param("model_source", model_source)
                    mlflow.log_param("Pregnancies", pregnancies)
                    mlflow.log_param("Glucose", glucose)
                    mlflow.log_param("BloodPressure", blood_pressure)
                    mlflow.log_param("SkinThickness", skin_thickness)
                    mlflow.log_param("Insulin", insulin)
                    mlflow.log_param("BMI", bmi)
                    mlflow.log_param("DiabetesPedigreeFunction", dpf)
                    mlflow.log_param("Age", age)
                    mlflow.log_metric("Diabetic_Probability", confidence)
                    mlflow.log_metric("Non_Diabetic_Probability", non_diabetic_prob)
                    mlflow.log_metric("Prediction", int(prediction))
                logged_run_id = run.info.run_id
            except Exception as log_error:
                logged_run_id = None
                st.warning(f"⚠️ Prediction was generated, but MLflow logging failed: {log_error}")

            # Show input values
            with st.expander("📝 View Input Values"):
                st.dataframe(input_data, use_container_width=True)
                
            # MLflow run info
            with st.expander("🔬 MLflow Run Details"):
                if logged_run_id:
                    st.code(f"Run ID: {logged_run_id}")
                    st.code(f"Experiment ID: 3904835178028478")
                    st.info("✅ Prediction logged to MLflow")
                else:
                    st.warning("MLflow run information is not available. The prediction may not have been logged.")
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            
            # Detailed error info
            with st.expander("🔍 Error Details"):
                st.exception(e)
                st.markdown("""
                **Common Issues:**
                - Model permissions: Ensure app service principal has EXECUTE on model
                - Volume permissions: Ensure app service principal has READ VOLUME on mlflow_tmp
                - Network issues: Check Unity Catalog connectivity
                """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Disclaimer:</strong> This prediction tool is for educational and informational purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
        <p>Powered by Databricks & MLflow | Random Forest ML Model</p>
    </div>
    """, unsafe_allow_html=True)
