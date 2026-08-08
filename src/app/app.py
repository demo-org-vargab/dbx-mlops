import streamlit as st
import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import plotly.graph_objects as go
from datetime import datetime

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
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 18px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
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

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model Information:**
    - Type: Random Forest Classifier
    - Accuracy: 81.51%
    - AUC-ROC: 0.8599
    - Training samples: 768
    
    **Top Predictive Features:**
    1. Glucose (26.12%)
    2. BMI (16.66%)
    3. Age (14.28%)
    """)
    
    st.markdown("---")
    st.markdown("**Model Registry:**")
    st.code("workspace.demo.diabetes_random_forest")
    
    st.markdown("---")
    st.markdown("**MLflow Experiment:**")
    st.code("3904835178028478")

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

# Predict button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    with st.spinner("Loading model and analyzing patient data..."):
        try:
            # Initialize MLflow
            mlflow.set_tracking_uri("databricks")
            mlflow.set_experiment(experiment_id="3904835178028478")
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Load Spark session
                spark = SparkSession.builder.appName("DiabetesPrediction").getOrCreate()
                
                # Load model (lazy loading - only when button is clicked)
                mlflow.set_registry_uri("databricks-uc")
                model_uri = "models:/workspace.demo.diabetes_random_forest/1"
                dfs_tmpdir = "/Volumes/workspace/demo/mlflow_tmp"
                
                st.info("⏳ Loading model from Unity Catalog...")
                model = mlflow.spark.load_model(model_uri, dfs_tmpdir=dfs_tmpdir)
                st.success("✅ Model loaded successfully!")
                
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
                
                # Log input parameters to MLflow
                mlflow.log_params({
                    'pregnancies': int(pregnancies),
                    'glucose': int(glucose),
                    'blood_pressure': int(blood_pressure),
                    'skin_thickness': int(skin_thickness),
                    'insulin': int(insulin),
                    'bmi': float(bmi),
                    'dpf': float(dpf),
                    'age': int(age)
                })
                
                # Convert to Spark DataFrame
                spark_df = spark.createDataFrame(input_data)
                
                # Prepare features using VectorAssembler
                feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
                features_df = assembler.transform(spark_df)
                
                # Make prediction
                prediction_result = model.transform(features_df)
                
                # Extract results
                result = prediction_result.select("prediction", "probability").collect()[0]
                prediction = int(result['prediction'])
                probability = result['probability'].toArray()
                
                # Calculate confidence
                confidence = probability[1] * 100  # Probability of being diabetic
                non_diabetic_prob = probability[0] * 100
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'prediction': prediction,
                    'diabetic_probability': confidence,
                    'non_diabetic_probability': non_diabetic_prob
                })
                
                # Log prediction timestamp
                mlflow.set_tag("prediction_timestamp", datetime.now().isoformat())
                
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
                    st.metric(
                        label="Non-Diabetic Probability",
                        value=f"{non_diabetic_prob:.1f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="Diabetic Probability",
                        value=f"{confidence:.1f}%",
                        delta=None
                    )
                
                # Feature Importance Visualization
                st.subheader("🎯 Top Risk Factors")
                st.markdown("Based on the Random Forest model's feature importance:")
                
                # Top 3 features with their importance
                features = ['Glucose', 'BMI', 'Age']
                importance = [26.12, 16.66, 14.28]
                colors = ['#0066cc', '#0088ff', '#00aaff']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=importance,
                        y=features,
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{i:.1f}%" for i in importance],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Feature Importance (Top 3 Predictors)",
                    xaxis_title="Importance (%)",
                    yaxis_title="Feature",
                    height=300,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show input values
                with st.expander("📝 View Input Values"):
                    st.dataframe(input_data, use_container_width=True)
                
                # MLflow run info
                with st.expander("🔬 MLflow Run Details"):
                    run_id = mlflow.active_run().info.run_id
                    st.code(f"Run ID: {run_id}")
                    st.code(f"Experiment ID: 3904835178028478")
                    st.info("✅ Prediction logged to MLflow")
            
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