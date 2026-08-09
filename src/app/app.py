import logging
import os
import streamlit as st
import mlflow
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np


# Suppress non-actionable MLflow warnings during model loading
logging.getLogger("mlflow.pyfunc").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
logging.getLogger("py4j").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configure MLflow experiment for app predictions (optional)
# MLflow logging will use default experiment

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

# Demo model creation removed — app expects a model registered in Unity Catalog.

# Predict button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    with st.spinner("Loading model and analyzing patient data..."):
        try:
            # Try to load Unity Catalog model, fallback to demo model
            model = None
            model_source = None
            
            # Try loading a pyfunc (python) model registered in Unity Catalog first
            # This avoids requiring Java/Spark in the app runtime.
            try:
                mlflow.set_registry_uri("databricks-uc")
                pyfunc_uri = "models:/workspace.demo.diabetes_random_forest_pyfunc/1"
                st.info("⏳ Attempting to load pyfunc model from Unity Catalog...")
                model = mlflow.pyfunc.load_model(pyfunc_uri)
                model_source = "unity_catalog_pyfunc"
                st.success("✅ Pyfunc model loaded from Unity Catalog!")
            except Exception:
                # If pyfunc model isn't available, attempt to load the Spark model
                try:
                    # Only attempt Spark model load when Java is available
                    if os.environ.get("JAVA_HOME"):
                        model_uri = "models:/workspace.demo.diabetes_random_forest/3"
                        st.info("⏳ Loading Spark model from Unity Catalog (requires Java/Spark)...")
                        model = mlflow.pyfunc.load_model(model_uri)
                        model_source = "unity_catalog_spark"
                        st.success("✅ Spark model loaded from Unity Catalog!")
                    else:
                        raise EnvironmentError("JAVA_HOME is not set")
                except Exception as uc_error:
                    st.error(
                        "⚠️ Could not load the registered Unity Catalog model.\n"
                        "This app requires a model registered in Unity Catalog that can be loaded in the runtime.\n"
                        f"Reason: {uc_error}\n\n"
                        "Possible fixes:\n"
                        " - Ensure Java + Spark are available and JAVA_HOME is set if the registered model is a Spark model.\n"
                        " - Or register a pyfunc/sklearn model for this app (no Java required).\n"
                    )
                    st.stop()
                
            # Create input dataframe (pandas)
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

            # If we loaded a pyfunc (sklearn) model, use it directly (no Java/pyspark required)
            if model_source == "unity_catalog_pyfunc" and model is not None:
                try:
                    # mlflow pyfunc models usually accept a pandas DataFrame
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_data)
                        prediction = int(model.predict(input_data)[0])
                        probability = list(proba[0])
                    else:
                        # Fallback when probability is not available
                        prediction = int(model.predict(input_data)[0])
                        # Construct a best-effort probability vector
                        probability = [1.0 - float(prediction), float(prediction)]

                    confidence = float(probability[1]) * 100
                    non_diabetic_prob = float(probability[0]) * 100
                except Exception as e:
                    st.error(f"Failed to run pyfunc model prediction: {e}")
                    st.stop()
            else:
                # Use Spark model for prediction (requires Java + pyspark)
                try:
                    from pyspark.sql import SparkSession
                    from pyspark.ml.feature import VectorAssembler
                except Exception as e:
                    st.error(f"pyspark is not available in the runtime: {e}")
                    st.stop()

                if not os.environ.get("JAVA_HOME"):
                    st.error("JAVA_HOME is not set. Spark model loading requires Java in the runtime.")
                    st.stop()

                try:
                    spark = SparkSession.builder.getOrCreate()
                    tmpdir = os.environ.get("MLFLOW_DFS_TMPDIR", "/tmp/mlflow_tmp")
                    model_uri = "models:/workspace.demo.diabetes_random_forest/3"
                    st.info("⏳ Loading Spark model from Unity Catalog (requires Java/Spark)...")
                    spark_model = mlflow.spark.load_model(model_uri, dfs_tmpdir=tmpdir)
                    model_source = "unity_catalog_spark"
                    st.success("✅ Spark model loaded from Unity Catalog!")
                except Exception as e:
                    st.error(f"Failed to load Spark model: {e}")
                    st.stop()

                # Convert pandas input to Spark DataFrame and assemble features
                input_sdf = spark.createDataFrame(input_data)
                feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
                assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                input_sdf = assembler.transform(input_sdf)

                # Run prediction with Spark model
                pred_df = spark_model.transform(input_sdf).select('prediction', 'probability').collect()
                if len(pred_df) == 0:
                    st.error("Spark model returned no predictions.")
                    st.stop()

                pred_row = pred_df[0]
                prediction = int(pred_row['prediction'])
                prob_vector = pred_row['probability']
                # probability may be DenseVector — convert to list
                probability = list(prob_vector)

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
                
            # Log prediction and input to MLflow (optional - silent failure)
            logged_run_id = None
            try:
                # Use default experiment, don't require specific experiment ID
                with mlflow.start_run() as run:
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
                # Silent failure - MLflow logging is optional
                pass

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
