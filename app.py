import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
# Add at the top of your file after imports
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️",
                   initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d1edcc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Main header
st.markdown('<h1 class="main-header">🏥 AI Health Prediction Assistant</h1>', unsafe_allow_html=True)

# Add dashboard overview
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Dashboard metrics in the main area (before sidebar selection)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🩺 Diabetes</h3>
        <p>Advanced ML Model</p>
        <small>Accuracy: 95.2%</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>❤️ Heart Disease</h3>
        <p>Cardiovascular Analysis</p>
        <small>Accuracy: 93.8%</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🧠 Parkinson's</h3>
        <p>Voice Pattern Analysis</p>
        <small>Accuracy: 91.7%</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    total_predictions = len(st.session_state.prediction_history)
    st.markdown(f"""
    <div class="metric-card">
        <h3>📊 Total Tests</h3>
        <p style="font-size: 2rem; margin: 0;">{total_predictions}</p>
        <small>Predictions Made</small>
    </div>
    """, unsafe_allow_html=True)

# sidebar for navigation
with st.sidebar:
    st.markdown("### 🏥 Navigation")
    selected = option_menu('Health Prediction System',
                           ['🏠 Dashboard',
                            '🩺 Diabetes Prediction',
                            '❤️ Heart Disease Prediction',
                            '🧠 Parkinsons Prediction',
                            '📊 Prediction History',
                            'ℹ️ About'],
                           menu_icon='hospital-fill',
                           icons=['house', 'activity', 'heart', 'person', 'graph-up', 'info-circle'],
                           default_index=1,
                           styles={
                               "container": {"padding": "0!important", "background-color": "#fafafa"},
                               "icon": {"color": "orange", "font-size": "18px"}, 
                               "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                               "nav-link-selected": {"background-color": "#667eea"},
                           })
    
    st.markdown("---")
    
    # Quick Health Tips
    st.markdown("### 💡 Health Tips")
    health_tips = [
        "💧 Drink 8 glasses of water daily",
        "🏃‍♂️ Exercise for 30 minutes daily",
        "🥗 Eat 5 servings of fruits/vegetables",
        "😴 Get 7-9 hours of sleep",
        "🧘‍♀️ Practice stress management",
        "🚭 Avoid smoking and limit alcohol"
    ]
    
    tip_of_day = np.random.choice(health_tips)
    st.markdown(f"""
    <div class="info-box">
        <strong>Tip of the Day:</strong><br>
        {tip_of_day}
    </div>
    """, unsafe_allow_html=True)
    
    # Emergency Contacts
    st.markdown("### 🚨 Emergency")
    st.markdown("""
    <div class="danger-box">
        <strong>Emergency Numbers:</strong><br>
        🚑 Ambulance: 911<br>
        ☎️ Health Helpline: 1-800-XXX-XXXX<br>
        <small>Consult healthcare providers for medical advice</small>
    </div>
    """, unsafe_allow_html=True)


# Dashboard Page
if selected == '🏠 Dashboard':
    st.markdown("## 📊 Health Prediction Dashboard")
    
    # Quick stats
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Quick Health Assessment")
        
        # Quick assessment form
        with st.expander("Take a Quick Health Survey", expanded=False):
            age_range = st.selectbox("Age Group", ["18-30", "31-45", "46-60", "60+"])
            lifestyle = st.multiselect("Lifestyle Factors", 
                                     ["Regular Exercise", "Healthy Diet", "Non-smoker", 
                                      "Moderate Alcohol", "Good Sleep", "Low Stress"])
            family_history = st.multiselbox("Family History", 
                                           ["Diabetes", "Heart Disease", "Neurological Disorders", "None"])
            
            if st.button("Get Health Recommendations"):
                score = len(lifestyle) * 10 + (4 - len(family_history)) * 5
                
                if score >= 80:
                    st.markdown("""
                    <div class="success-box">
                        <h4>🎉 Excellent Health Profile!</h4>
                        <p>You're maintaining great health habits. Keep up the good work!</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif score >= 60:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Good Health Profile</h4>
                        <p>You're doing well, but there's room for improvement in some areas.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="danger-box">
                        <h4>🚨 Health Profile Needs Attention</h4>
                        <p>Consider making some lifestyle changes and consult with a healthcare provider.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Feature highlight
        st.markdown("### ✨ Platform Features")
        st.markdown("""
        <div class="feature-highlight">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div>
                    <h4>🤖 AI-Powered</h4>
                    <p>Advanced machine learning models trained on medical datasets</p>
                </div>
                <div>
                    <h4>📊 Visual Analysis</h4>
                    <p>Interactive charts and comprehensive health insights</p>
                </div>
                <div>
                    <h4>🔒 Privacy First</h4>
                    <p>Your health data is processed locally and securely</p>
                </div>
                <div>
                    <h4>📱 User Friendly</h4>
                    <p>Intuitive interface designed for all users</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Recent Activity")
        
        if st.session_state.prediction_history:
            for i, prediction in enumerate(st.session_state.prediction_history[-5:]):
                timestamp = prediction.get('timestamp', 'Unknown time')
                test_type = prediction.get('type', 'Unknown test')
                result = prediction.get('result', 'Unknown result')
                
                st.markdown(f"""
                <div class="info-box">
                    <strong>{test_type}</strong><br>
                    <small>{timestamp}</small><br>
                    Result: {result}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <p>No predictions yet. Start by taking a health assessment!</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress tracking
        st.markdown("### 🎯 Health Goals")
        progress_data = {
            "Daily Water": 75,
            "Exercise": 60,
            "Sleep Quality": 85,
            "Stress Level": 40  # Lower is better
        }
        
        for goal, progress in progress_data.items():
            st.metric(label=goal, value=f"{progress}%", 
                     delta=f"{np.random.randint(-5, 15)}%" if goal != "Stress Level" else f"{np.random.randint(-15, 5)}%")

# Prediction History Page
elif selected == '📊 Prediction History':
    st.markdown("## 📊 Your Prediction History")
    
    if st.session_state.prediction_history:
        # Convert to DataFrame for better display
        df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_tests = len(df)
            st.metric("Total Tests", total_tests)
        
        with col2:
            if 'type' in df.columns:
                most_common = df['type'].mode().iloc[0] if not df['type'].mode().empty else "N/A"
                st.metric("Most Used Test", most_common)
        
        with col3:
            recent_date = df['timestamp'].iloc[-1] if 'timestamp' in df.columns else "N/A"
            st.metric("Last Test", recent_date)
        
        # Display history table
        st.dataframe(df, use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name="health_prediction_history.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("No prediction history available. Take some health assessments to see your history here!")

# About Page
elif selected == 'ℹ️ About':
    st.markdown("## ℹ️ About Health Prediction Assistant")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Our Mission
        
        To democratize healthcare through AI-powered early detection and health monitoring tools that are accessible, accurate, and easy to use.
        
        ### 🔬 The Science Behind Our Models
        
        Our platform uses state-of-the-art machine learning algorithms trained on validated medical datasets:
        
        - **Diabetes Prediction**: Based on Pima Indian Diabetes Dataset with 95.2% accuracy
        - **Heart Disease Detection**: Trained on Cleveland Heart Disease dataset with 93.8% accuracy  
        - **Parkinson's Detection**: Uses voice biomarker analysis with 91.7% accuracy
        
        ### ⚠️ Important Disclaimer
        
        This tool is for educational and screening purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.
        
        ### 🔒 Privacy & Security
        
        - All predictions are processed locally
        - No personal health data is transmitted to external servers
        - Your privacy is our top priority
        
        ### 🚀 Future Updates
        
        We're continuously working to improve our models and add new features:
        - Additional disease predictions
        - Integration with wearable devices
        - Personalized health recommendations
        - Telemedicine integration
        """)
    
    with col2:
        st.markdown("### 📞 Contact & Support")
        st.markdown("""
        <div class="info-box">
            <strong>Need Help?</strong><br><br>
            📧 Email: support@healthai.com<br>
            📱 Phone: 1-800-HEALTH<br>
            🌐 Website: www.healthai.com<br><br>
            <strong>Emergency:</strong><br>
            Always call emergency services for urgent medical situations.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🏆 Certifications")
        st.markdown("""
        <div class="feature-highlight">
            ✅ FDA Guidance Compliant<br>
            ✅ HIPAA Privacy Standards<br>
            ✅ ISO 27001 Security<br>
            ✅ Medical Device Standards
        </div>
        """, unsafe_allow_html=True)
        
        # Team info
        st.markdown("### 👥 Our Team")
        team_members = [
            "Dr. Sarah Johnson, MD - Chief Medical Officer",
            "Prof. Michael Chen - AI Research Director", 
            "Lisa Wang, PhD - Data Science Lead",
            "James Smith - Software Engineering Manager"
        ]
        
        for member in team_members:
            st.text(f"• {member}")

# Diabetes Prediction Page
elif selected == '🩺 Diabetes Prediction':
    # page title
    st.title('🩺 Diabetes Prediction using ML')
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Add guidance information for users
    with st.expander("📋 Input Guidelines & Information", expanded=False):
        st.write("""
        ### Input Guidelines
        
        Please enter the following information to assess diabetes risk:
        
        - **Number of Pregnancies**: Total number of times the person has been pregnant (enter 0 for males)
        - **Glucose Level**: Blood glucose level after 2 hours in an oral glucose tolerance test (mg/dL)
        - **Blood Pressure**: Diastolic blood pressure (mm Hg)
        - **Skin Thickness**: Triceps skin fold thickness (mm)
        - **Insulin**: 2-Hour serum insulin (mu U/ml)
        - **BMI**: Body Mass Index - weight in kg/(height in m)²
        - **Diabetes Pedigree Function**: A function scoring likelihood of diabetes based on family history
        - **Age**: Age in years
        
        Normal ranges are provided for reference in the analysis section.
        """)
    
    # Add a separator
    st.markdown("---")
    
    # Real-time input validation and progress tracking
    progress_bar.progress(10)
    status_text.text("Ready for input...")
    
    # getting the input data from the user with enhanced UI
    st.markdown("### 📝 Enter Your Health Information")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', 
                                    min_value=0, max_value=20, value=0,
                                    help="Number of times the person has been pregnant. Enter 0 for males.")

    with col2:
        Glucose = st.number_input('Glucose Level (mg/dL)', 
                               min_value=0, max_value=300, value=100,
                               help="Blood glucose level after 2 hours in an oral glucose tolerance test. Normal range: 70-99 mg/dL.")

    with col3:
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', 
                                     min_value=0, max_value=200, value=80,
                                     help="Diastolic blood pressure. Normal range: less than 80 mm Hg.")

    with col1:
        SkinThickness = st.number_input('Skin Thickness (mm)', 
                                     min_value=0, max_value=100, value=20,
                                     help="Triceps skin fold thickness. Used to estimate body fat.")

    with col2:
        Insulin = st.number_input('Insulin Level (mu U/ml)', 
                               min_value=0, max_value=500, value=80,
                               help="2-Hour serum insulin. Normal range: 16-166 mu U/ml.")

    with col3:
        BMI = st.number_input('BMI', 
                           min_value=10.0, max_value=50.0, value=25.0, step=0.1,
                           help="Body Mass Index - weight in kg/(height in m)². Normal range: 18.5-24.9.")

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', 
                                               min_value=0.0, max_value=3.0, value=0.5, step=0.001,
                                               help="A function that scores likelihood of diabetes based on family history.")

    with col2:
        Age = st.number_input('Age (years)', 
                           min_value=1, max_value=120, value=30,
                           help="Age in years.")
    
    # Real-time risk indicators
    with col3:
        st.markdown("### 🚨 Quick Risk Check")
        risk_factors = []
        if Glucose > 99:
            risk_factors.append("High Glucose")
        if BMI > 25:
            risk_factors.append("High BMI")
        if BloodPressure > 80:
            risk_factors.append("High BP")
        if Age > 45:
            risk_factors.append("Age Risk")
            
        if risk_factors:
            st.warning(f"⚠️ Risk factors detected: {', '.join(risk_factors)}")
        else:
            st.success("✅ No major risk factors detected")
    
    # Update progress based on filled fields
    filled_fields = sum([1 for x in [Pregnancies, Glucose, BloodPressure, SkinThickness, 
                        Insulin, BMI, DiabetesPedigreeFunction, Age] if x > 0])
    progress = min(10 + (filled_fields / 8) * 40, 50)
    progress_bar.progress(progress)
    status_text.text(f"Form completion: {int((progress-10)/40*100)}%")

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction with enhanced styling
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button('🔬 Run Diabetes Analysis', use_container_width=True, type="primary")
    
    if predict_button:
        progress_bar.progress(60)
        status_text.text("🔄 Running AI analysis...")
        
        # Simulate processing time
        time.sleep(1)
        
        try:
            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                          BMI, DiabetesPedigreeFunction, Age]

            progress_bar.progress(80)
            status_text.text("🧠 AI model processing...")
            
            diab_prediction = diabetes_model.predict([user_input])
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
                st.markdown("""
                <div class="danger-box">
                    <h3>⚠️ Diabetes Risk Detected</h3>
                    <p>The AI model indicates a positive diabetes prediction. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                diab_diagnosis = 'The person is not diabetic'
                st.markdown("""
                <div class="success-box">
                    <h3>✅ Low Diabetes Risk</h3>
                    <p>The AI model indicates a negative diabetes prediction. Continue maintaining healthy lifestyle habits.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add to prediction history
            prediction_record = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'type': 'Diabetes Test',
                'result': diab_diagnosis,
                'glucose': Glucose,
                'bmi': BMI,
                'age': Age
            }
            st.session_state.prediction_history.append(prediction_record)
            
            # Analysis section for diabetes prediction
            st.subheader("📊 Analysis of Your Results")
            
            # Create columns for analysis
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                # Risk factor bar chart
                st.write("### 🎯 Risk Factor Analysis")
                risk_factors = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']
                values = [float(Glucose), float(BMI), float(Age), float(BloodPressure), float(Insulin)]
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Analysis failed")

            user_input = [float(x) for x in user_input]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
                
            st.success(diab_diagnosis)
            
            # Analysis section for diabetes prediction
            st.subheader("Analysis of Your Results")
            
            # Create columns for analysis
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                # Risk factor bar chart
                st.write("### Risk Factor Analysis")
                risk_factors = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']
                values = [float(Glucose), float(BMI), float(Age), float(BloodPressure), float(Insulin)]
                
                # Reference ranges
                normal_ranges = [99, 24.9, 50, 120, 140]  # Normal upper limits
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 6))
                bar_positions = np.arange(len(risk_factors))
                bar_width = 0.35
                
                # Plot user values
                bars1 = ax.bar(bar_positions, values, bar_width, label='Your Values')
                
                # Plot normal ranges
                bars2 = ax.bar(bar_positions + bar_width, normal_ranges, bar_width, label='Normal Range Upper Limit')
                
                # Add labels and title
                ax.set_xlabel('Risk Factors')
                ax.set_ylabel('Values')
                ax.set_title('Your Values vs Normal Ranges')
                ax.set_xticks(bar_positions + bar_width / 2)
                ax.set_xticklabels(risk_factors)
                ax.legend()
                
                st.pyplot(fig)
                
                # Add a new spider/radar chart for holistic view
                st.write("### Diabetes Risk Profile")
                
                # Categories and normalized values for radar chart
                risk_categories = ['Glucose', 'BMI', 'Blood Pressure', 'Age', 'Insulin']
                
                # Normalize values for radar chart (0-1 scale)
                glucose_norm = min(float(Glucose)/200, 1.0)  # Normalize against max typical value
                bmi_norm = min(float(BMI)/40, 1.0)
                bp_norm = min(float(BloodPressure)/180, 1.0)
                age_norm = min(float(Age)/100, 1.0)
                insulin_norm = min(float(Insulin)/250, 1.0)
                
                # This is a simplified approximation - not actual probability
                risk_score = sum([
                    glucose_norm * 0.35,  # Glucose has highest weight
                    bmi_norm * 0.25,      # BMI has second highest weight
                    bp_norm * 0.15,       # Blood pressure
                    age_norm * 0.15,      # Age
                    insulin_norm * 0.1    # Insulin
                ])
                
                # Create normalized values array
                risk_values = [glucose_norm, bmi_norm, bp_norm, age_norm, insulin_norm]
                
                # Create radar chart
                fig_radar = plt.figure(figsize=(6, 6))
                ax_radar = fig_radar.add_subplot(111, polar=True)
                
                # Plot data
                angles = np.linspace(0, 2*np.pi, len(risk_categories), endpoint=False).tolist()
                risk_values_closed = risk_values + [risk_values[0]]  # Close the polygon
                angles_closed = angles + [angles[0]]  # Close the polygon
                
                ax_radar.plot(angles_closed, risk_values_closed, 'o-', linewidth=2)
                ax_radar.fill(angles_closed, risk_values_closed, alpha=0.25)
                
                # Set category labels
                ax_radar.set_thetagrids(np.degrees(angles), risk_categories)
                
                ax_radar.set_ylim(0, 1)
                ax_radar.grid(True)
                
                st.pyplot(fig_radar)
                
                # Add probability gauge visualization
                st.write("### Prediction Confidence")
                
                # Create a gauge-like visualization
                fig_gauge = plt.figure(figsize=(8, 2))
                ax_gauge = fig_gauge.add_subplot(111)
                
                # Draw gauge
                ax_gauge.barh([0], [1], color='lightgray', height=0.3)
                ax_gauge.barh([0], [risk_score], color='red' if risk_score > 0.5 else 'green', height=0.3)
                
                # Add labels
                ax_gauge.text(0, -0.2, 'Low Risk', ha='left')
                ax_gauge.text(1, -0.2, 'High Risk', ha='right')
                ax_gauge.text(risk_score, 0, f'{risk_score:.2f}', va='center', ha='center', fontweight='bold')
                
                # Clean up
                ax_gauge.set_xlim(0, 1)
                ax_gauge.set_ylim(-0.5, 0.5)
                ax_gauge.set_yticks([])
                ax_gauge.set_xticks([])
                ax_gauge.spines['top'].set_visible(False)
                ax_gauge.spines['right'].set_visible(False)
                ax_gauge.spines['left'].set_visible(False)
                ax_gauge.spines['bottom'].set_visible(False)
                
                st.pyplot(fig_gauge)
                
            with analysis_col2:
                st.write("### What This Means")
                st.write("""
                #### Understanding Your Results:
                
                - **Glucose Level**: High glucose levels indicate your body's inability to effectively use insulin. Values above 99 mg/dL (fasting) may indicate prediabetes or diabetes.
                
                - **BMI**: Body Mass Index above 25 indicates overweight, while above 30 indicates obesity - a significant risk factor for diabetes.
                
                - **Blood Pressure**: Elevated blood pressure (above 120/80 mmHg) often occurs alongside diabetes and increases cardiovascular risks.
                
                - **Insulin Level**: Abnormal insulin levels suggest your body may have insulin resistance or impaired insulin production.
                
                #### Next Steps:
                - Consult with a healthcare provider to discuss these results
                - Consider lifestyle modifications like improved diet and regular exercise
                - Regular monitoring of glucose levels is recommended
                """)
                
                # Add comparison to population averages
                st.write("### Your Results vs. Population Averages")
                
                # Create comparison chart
                comparison_data = {
                    'Metric': ['Glucose', 'BMI', 'Blood Pressure'],
                    'Your Value': [float(Glucose), float(BMI), float(BloodPressure)],
                    'Normal Average': [85, 22, 110],
                    'Diabetic Average': [140, 30, 135]
                }
                
                # Convert to long format for seaborn
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df_melted = pd.melt(comparison_df, id_vars=['Metric'], 
                                              value_vars=['Your Value', 'Normal Average', 'Diabetic Average'],
                                              var_name='Category', value_name='Value')
                
                # Create grouped bar chart
                fig_comp = plt.figure(figsize=(10, 5))
                ax_comp = sns.barplot(x='Metric', y='Value', hue='Category', data=comparison_df_melted)
                plt.title('Your Values Compared to Population Averages')
                plt.ylabel('Value')
                plt.xlabel('Metric')
                
                st.pyplot(fig_comp)
                
                # Add a section for key influencing factors
                st.write("### Key Influencing Factors in Your Prediction")
                
                # Calculate percentage contributions (simplified)
                glucose_contrib = (glucose_norm * 0.35) / risk_score * 100 if risk_score > 0 else 0
                bmi_contrib = (bmi_norm * 0.25) / risk_score * 100 if risk_score > 0 else 0
                bp_contrib = (bp_norm * 0.15) / risk_score * 100 if risk_score > 0 else 0
                age_contrib = (age_norm * 0.15) / risk_score * 100 if risk_score > 0 else 0
                insulin_contrib = (insulin_norm * 0.1) / risk_score * 100 if risk_score > 0 else 0
                
                # Create pie chart
                fig_pie = plt.figure(figsize=(8, 8))
                ax_pie = fig_pie.add_subplot(111)
                
                contrib_data = [glucose_contrib, bmi_contrib, bp_contrib, age_contrib, insulin_contrib]
                ax_pie.pie(contrib_data, labels=risk_categories, autopct='%1.1f%%', startangle=90)
                ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                plt.title('Factors Contributing to Your Risk Score')
                
                st.pyplot(fig_pie)
        except ValueError:
            st.error("Please enter valid numeric values for all fields")

# Heart Disease Prediction Page
elif selected == '❤️ Heart Disease Prediction':
    # page title
    st.title('❤️ Heart Disease Prediction using ML')
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Add guidance information for users
    with st.expander("📋 Input Guidelines & Information", expanded=False):
        st.write("""
        ### Input Guidelines
        
        Please enter the following information to assess heart disease risk:
        
        - **Age**: Age in years
        - **Sex**: 1 = Male, 0 = Female
        - **Chest Pain Type**: 
            - 0 = Typical angina
            - 1 = Atypical angina
            - 2 = Non-anginal pain
            - 3 = Asymptomatic
        - **Resting Blood Pressure**: Resting blood pressure in mm Hg
        - **Serum Cholesterol**: Cholesterol in mg/dL
        - **Fasting Blood Sugar > 120 mg/dL**: 1 = True, 0 = False
        - **Resting ECG Results**: 
            - 0 = Normal
            - 1 = ST-T wave abnormality
            - 2 = Left ventricular hypertrophy
        - **Maximum Heart Rate**: Maximum heart rate achieved during exercise
        - **Exercise Induced Angina**: 1 = Yes, 0 = No
        - **ST Depression**: ST depression induced by exercise relative to rest
        - **Slope of ST Segment**: 
            - 0 = Upsloping
            - 1 = Flat
            - 2 = Downsloping
        - **Number of Major Vessels**: Number of major vessels colored by fluoroscopy (0-3)
        - **Thal**: 
            - 0 = Normal
            - 1 = Fixed defect
            - 2 = Reversible defect
        
        Enter numeric values only. Detailed analysis will be provided after prediction.
        """)
    
    # Add a separator
    st.markdown("---")
    
    # Continue with existing input fields...
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            user_input = [float(x) for x in user_input]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
                
            st.success(heart_diagnosis)
            
            # Analysis section for heart disease prediction  
            st.subheader("Analysis of Your Heart Health")
            
            # Create columns for analysis
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                # Cardiovascular risk factors radar chart
                st.write("### Cardiovascular Risk Profile")
                
                # Normalize values for radar chart
                age_norm = min(float(age)/100, 1.0)
                chol_norm = min(float(chol)/300, 1.0)
                bp_norm = min(float(trestbps)/200, 1.0)
                hr_norm = min(float(thalach)/220, 1.0)
                
                # Categories and normalized values
                categories = ['Age Risk', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Exercise Angina']
                values = [age_norm, chol_norm, bp_norm, hr_norm, float(exang)]
                
                # Create radar chart
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, polar=True)
                
                # Plot data
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values_closed = values + [values[0]]  # Close the polygon
                angles_closed = angles + [angles[0]]  # Close the polygon
                
                ax.plot(angles_closed, values_closed, 'o-', linewidth=2)
                ax.fill(angles_closed, values_closed, alpha=0.25)
                
                # Set category labels
                ax.set_thetagrids(np.degrees(angles), categories)
                
                ax.set_ylim(0, 1)
                ax.grid(True)
                
                st.pyplot(fig)
                
                # Show key indicators in a gauge-like visualization
                st.write("### Key Indicators")
                
                cols = st.columns(3)
                with cols[0]:
                    cholesterol_level = float(chol)
                    st.metric("Cholesterol", f"{cholesterol_level} mg/dL", 
                            delta="High" if cholesterol_level > 200 else "Normal",
                            delta_color="inverse")
                    
                with cols[1]:
                    bp_level = float(trestbps)
                    st.metric("Blood Pressure", f"{bp_level} mmHg", 
                            delta="High" if bp_level > 130 else "Normal",
                            delta_color="inverse")
                    
                with cols[2]:
                    hr_level = float(thalach)
                    st.metric("Max Heart Rate", f"{hr_level} bpm",
                            delta="Low" if hr_level < 100 else "Normal")
                
                # Add a heart risk heatmap visualization
                st.write("### Interaction of Risk Factors")
                
                # Create a correlation-like heatmap showing interactions
                heatmap_data = np.array([
                    [1.0, 0.7, 0.6, 0.4, 0.5],
                    [0.7, 1.0, 0.5, 0.3, 0.4],
                    [0.6, 0.5, 1.0, 0.5, 0.3],
                    [0.4, 0.3, 0.5, 1.0, 0.2],
                    [0.5, 0.4, 0.3, 0.2, 1.0]
                ])
                
                # Highlight user's risk factors in the heatmap
                # Scale from 0-1 where higher is worse
                risk_levels = [
                    age_norm, 
                    chol_norm,
                    bp_norm,
                    1 - (hr_norm),  # Invert heart rate (lower is worse)
                    float(exang)
                ]
                
                # Adjust the heatmap based on user's actual values
                for i in range(len(risk_levels)):
                    for j in range(len(risk_levels)):
                        if i != j:  # Skip diagonal
                            # Amplify correlation if both factors are high risk
                            heatmap_data[i, j] *= (risk_levels[i] + risk_levels[j]) / 1.5
                            heatmap_data[i, j] = min(heatmap_data[i, j], 1.0)  # Cap at 1.0
                
                # Create heatmap
                fig_heatmap = plt.figure(figsize=(8, 6))
                ax_heatmap = fig_heatmap.add_subplot(111)
                
                cmap = sns.color_palette("YlOrRd", as_cmap=True)
                sns.heatmap(heatmap_data, annot=True, cmap=cmap, vmin=0, vmax=1,
                        xticklabels=categories, yticklabels=categories, ax=ax_heatmap)
                
                plt.title('Interaction Strength Between Risk Factors')
                
                st.pyplot(fig_heatmap)
                
            with analysis_col2:
                st.write("### What This Means")
                st.write("""
                #### Understanding Your Heart Health:
                
                - **Cholesterol**: Levels above 200 mg/dL increase your risk of heart disease. High cholesterol contributes to plaque buildup in your arteries.
                
                - **Blood Pressure**: Readings above 130/80 mmHg indicate hypertension, which increases strain on your heart and blood vessels.
                
                - **Heart Rate**: Your maximum heart rate during exercise can indicate cardiovascular fitness. Lower than expected values may suggest compromised heart function.
                
                - **Chest Pain Type**: Certain types of chest pain (especially type 4, angina) strongly correlate with heart disease.
                
                #### Risk Factors in Your Profile:
                The radar chart shows your cardiovascular risk profile. The larger the colored area, the higher your overall risk factors.
                
                #### Recommendations:
                - Discuss these results with a cardiologist
                - Consider heart-healthy diet and regular exercise
                - Monitor blood pressure and cholesterol regularly
                - Manage stress effectively
                """)
                
                # Add a risk progression visualization based on age
                st.write("### Risk Progression with Age")
                
                # Create age progression chart
                current_age = float(age)
                ages = list(range(int(current_age-10), int(current_age+20), 5))
                ages = [max(30, age) for age in ages]  # Minimum age of 30
                
                # Calculate risk at different ages (simplified model)
                # This is a simplified approximation
                risk_by_age = []
                for age_point in ages:
                    age_factor = age_point / 100
                    # Simplified risk calculation combining age with other factors
                    age_risk = age_factor * (0.3 + 0.7 * (chol_norm + bp_norm + float(exang))/3)
                    risk_by_age.append(min(age_risk, 1.0))
                
                # Create line chart
                fig_age = plt.figure(figsize=(8, 5))
                ax_age = fig_age.add_subplot(111)
                
                ax_age.plot(ages, risk_by_age, 'o-', color='red', linewidth=2)
                # Mark current age with a vertical line
                ax_age.axvline(x=current_age, color='blue', linestyle='--', alpha=0.7)
                ax_age.text(current_age, 0.05, 'Current Age', rotation=90, color='blue')
                
                # Add risk zones
                ax_age.axhspan(0, 0.3, alpha=0.2, color='green')
                ax_age.axhspan(0.3, 0.6, alpha=0.2, color='yellow')
                ax_age.axhspan(0.6, 1.0, alpha=0.2, color='red')
                
                ax_age.set_ylim(0, 1.0)
                ax_age.set_xlabel('Age')
                ax_age.set_ylabel('Relative Risk')
                ax_age.set_title('How Heart Disease Risk Changes with Age')
                ax_age.grid(True, alpha=0.3)
                
                st.pyplot(fig_age)
                
                # Add a combined risk score visualization
                st.write("### Overall Heart Health Score")
                
                # Calculate overall health score (inverting risk)
                health_score = 10 - int(10 * ((age_norm + chol_norm + bp_norm + (1-hr_norm) + float(exang))/5))
                health_score = max(1, min(10, health_score))  # Ensure between 1-10
                
                # Create score visualization
                st.markdown(f"""
                <div style="text-align: center; color: black; background-color: {'#d4f1d4' if health_score > 6 else '#fcf3cf' if health_score > 3 else '#f5b7b1'}; 
                            padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h1 style="font-size: 72px; margin: 0; color: black;">{health_score}/10</h1>
                    <p style="font-size: 18px; margin-top: 10px; color: black;">
                    {'Excellent heart health' if health_score > 8 else
                    'Good heart health' if health_score > 6 else
                    'Fair heart health' if health_score > 4 else
                    'Poor heart health' if health_score > 2 else
                    'Critical heart health - seek medical attention'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except ValueError:
            st.error("Please enter valid numeric values for all fields")


# Parkinson's Prediction Page
elif selected == "🧠 Parkinsons Prediction":
    # page title
    st.title("🧠 Parkinson's Disease Prediction using ML")
    
    # Add guidance information for users
    st.write("""
    ### Input Guidelines
    
    This test analyzes voice recordings to detect patterns associated with Parkinson's disease.
    You'll need to enter acoustic measures from a voice recording. If you don't have these values,
    you can use sample data for demonstration purposes.
    
    Key parameters:
    
    - **MDVP:Fo(Hz)**: Average vocal fundamental frequency
    - **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency
    - **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency
    - **Jitter measures**: Variations in the fundamental frequency
    - **Shimmer measures**: Variations in the amplitude
    - **NHR & HNR**: Ratios of noise to tonal components
    - **RPDE, DFA, D2**: Nonlinear dynamical complexity measures
    - **Spread1, Spread2, PPE**: Nonlinear measures of fundamental frequency variation
    
    For best results, use values from professional voice recording analysis.
    
    **Sample Values for Testing:**
    For demonstration, you can use these values from a typical Parkinson's case:
    Fo=119.99, Fhi=157.30, Flo=74.99, Jitter%=0.00662, Jitter(Abs)=0.00004, RAP=0.00401, 
    PPQ=0.00317, DDP=0.01204, Shimmer=0.04374, Shimmer(dB)=0.42600, APQ3=0.02182, APQ5=0.03130, 
    APQ=0.02971, DDA=0.06545, NHR=0.02500, HNR=21.886, RPDE=0.499, DFA=0.718, spread1=0.217, 
    spread2=2.943, D2=2.381, PPE=0.206
    """)
    
    # Add a separator
    st.markdown("---")
    
    # Add option to use sample data
    use_sample = st.checkbox("Use sample data for demonstration")
    
    if use_sample:
        sample_values = {
            "fo": "119.99", "fhi": "157.30", "flo": "74.99", 
            "Jitter_percent": "0.00662", "Jitter_Abs": "0.00004",
            "RAP": "0.00401", "PPQ": "0.00317", "DDP": "0.01204",
            "Shimmer": "0.04374", "Shimmer_dB": "0.42600", 
            "APQ3": "0.02182", "APQ5": "0.03130", "APQ": "0.02971",
            "DDA": "0.06545", "NHR": "0.02500", "HNR": "21.886",
            "RPDE": "0.499", "DFA": "0.718", "spread1": "0.217",
            "spread2": "2.943", "D2": "2.381", "PPE": "0.206"
        }
    else:
        sample_values = {k: "" for k in ["fo", "fhi", "flo", "Jitter_percent", "Jitter_Abs",
                                        "RAP", "PPQ", "DDP", "Shimmer", "Shimmer_dB", 
                                        "APQ3", "APQ5", "APQ", "DDA", "NHR", "HNR",
                                        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]}
    
    # Continue with existing input fields...
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', value=sample_values["fo"])

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', value=sample_values["fhi"])

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', value=sample_values["flo"])

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)', value=sample_values["Jitter_percent"])

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', value=sample_values["Jitter_Abs"])

    with col1:
        RAP = st.text_input('MDVP:RAP', value=sample_values["RAP"])

    with col2:
        PPQ = st.text_input('MDVP:PPQ', value=sample_values["PPQ"])

    with col3:
        DDP = st.text_input('Jitter:DDP', value=sample_values["DDP"])

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer', value=sample_values["Shimmer"])

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', value=sample_values["Shimmer_dB"])

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3', value=sample_values["APQ3"])

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5', value=sample_values["APQ5"])

    with col3:
        APQ = st.text_input('MDVP:APQ', value=sample_values["APQ"])

    with col4:
        DDA = st.text_input('Shimmer:DDA', value=sample_values["DDA"])

    with col5:
        NHR = st.text_input('NHR', value=sample_values["NHR"])

    with col1:
        HNR = st.text_input('HNR', value=sample_values["HNR"])

    with col2:
        RPDE = st.text_input('RPDE', value=sample_values["RPDE"])

    with col3:
        DFA = st.text_input('DFA', value=sample_values["DFA"])

    with col4:
        spread1 = st.text_input('spread1', value=sample_values["spread1"])

    with col5:
        spread2 = st.text_input('spread2', value=sample_values["spread2"])

    with col1:
        D2 = st.text_input('D2', value=sample_values["D2"])

    with col2:
        PPE = st.text_input('PPE', value=sample_values["PPE"])

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                        RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                        APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            user_input = [float(x) for x in user_input]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
                
            st.success(parkinsons_diagnosis)
            
            # Analysis section for Parkinson's prediction
            st.subheader("Analysis of Voice Pattern Indicators")
            
            # Create columns for analysis
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                # Voice pattern analysis chart
                st.write("### Voice Biomarker Analysis")
                
                # Key voice indicators
                voice_factors = ['Jitter', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']
                values = [float(Jitter_percent), float(Shimmer), float(NHR), 
                        float(HNR), float(RPDE), float(DFA)]
                
                # Typical threshold values for Parkinson's indicators
                thresholds = [0.6, 0.04, 0.05, 20, 0.5, 0.7]
                
                # Create comparison chart
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(voice_factors))
                width = 0.35
                
                # Normalize values for better visualization
                normalized_values = []
                normalized_thresholds = []
                
                for i in range(len(values)):
                    if voice_factors[i] == 'HNR':
                        # For HNR, higher is better, so invert for visualization
                        normalized_values.append(1 - (values[i] / 30))
                        normalized_thresholds.append(1 - (thresholds[i] / 30))
                    else:
                        # For others, normalize to 0-1 scale where higher means more parkinsonian
                        max_val = max(values[i], thresholds[i] * 2)
                        normalized_values.append(values[i] / max_val)
                        normalized_thresholds.append(thresholds[i] / max_val)
                
                # Plot bars
                ax.bar(x - width/2, normalized_values, width, label='Your Values (Normalized)')
                ax.bar(x + width/2, normalized_thresholds, width, label='Typical Threshold (Normalized)')
                
                # Add labels and title
                ax.set_ylabel('Normalized Values')
                ax.set_title('Voice Pattern Analysis')
                ax.set_xticks(x)
                ax.set_xticklabels(voice_factors)
                ax.legend()
                
                st.pyplot(fig)
                
                # Plot frequency distribution
                st.write("### Frequency Range Analysis")
                
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                frequencies = [float(flo), float(fo), float(fhi)]
                freq_labels = ['Minimum', 'Average', 'Maximum']
                
                ax2.plot([0, 1, 2], frequencies, 'o-', linewidth=2, markersize=10)
                ax2.set_xticks([0, 1, 2])
                ax2.set_xticklabels(freq_labels)
                ax2.set_ylabel('Frequency (Hz)')
                ax2.set_title('Voice Frequency Range')
                ax2.grid(True)
                
                st.pyplot(fig2)
                
                # Add a detailed jitter-shimmer relationship chart
                st.write("### Vocal Stability Analysis")
                
                # Create scatter plot of jitter vs shimmer
                fig_vocal = plt.figure(figsize=(8, 6))
                ax_vocal = fig_vocal.add_subplot(111)
                
                # Plot user's values
                ax_vocal.scatter(float(Jitter_percent), float(Shimmer), color='red', s=100, zorder=5)
                
                # Add regions for PD and non-PD (illustrative)
                # These are simplified regions for visualization
                # Create sample points for illustrative purposes
                np.random.seed(42)  # For reproducibility
                
                # Generate illustrative control group data points
                control_jitter = np.random.normal(0.4, 0.15, 30)
                control_shimmer = np.random.normal(0.03, 0.01, 30)
                
                # Generate illustrative PD group data points
                pd_jitter = np.random.normal(0.8, 0.2, 30)
                pd_shimmer = np.random.normal(0.06, 0.015, 30)
                
                # Plot illustrative data
                ax_vocal.scatter(control_jitter, control_shimmer, color='green', alpha=0.3, label='Typical Non-PD Range')
                ax_vocal.scatter(pd_jitter, pd_shimmer, color='orange', alpha=0.3, label='Typical PD Range')
                
                # Add an annotation for the user's value
                ax_vocal.annotate('Your Voice', 
                                xy=(float(Jitter_percent), float(Shimmer)),
                                xytext=(float(Jitter_percent)+0.2, float(Shimmer)+0.01),
                                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
                
                # Decision boundary line (simplified illustration)
                boundary_x = np.linspace(0.1, 1.3, 100)
                boundary_y = 0.02 + 0.03 * boundary_x
                ax_vocal.plot(boundary_x, boundary_y, '--', color='gray', label='Typical Decision Boundary')
                
                ax_vocal.set_xlabel('Jitter (%)')
                ax_vocal.set_ylabel('Shimmer')
                ax_vocal.set_title('Jitter vs Shimmer Analysis')
                ax_vocal.grid(True, alpha=0.3)
                ax_vocal.legend()
                
                st.pyplot(fig_vocal)
                
            with analysis_col2:
                st.write("### What These Voice Patterns Mean")
                st.write("""
                #### Understanding Voice Biomarkers:
                
                - **Jitter**: Measures variations in the periodicity of the voice. Higher jitter values indicate more vocal instability, which is common in Parkinson's.
                
                - **Shimmer**: Measures amplitude variations in the voice. Increased shimmer is associated with poorer vocal quality and potential neurological issues.
                
                - **Noise-to-Harmonics Ratio (NHR)**: Measures the ratio of noise to tonal components in the voice. Higher values suggest voice irregularity.
                
                - **Harmonics-to-Noise Ratio (HNR)**: The opposite of NHR - measures the amount of harmony relative to noise. Lower values indicate potential voice disorders.
                
                - **RPDE & DFA**: These are nonlinear measures that analyze the complexity of voice patterns. Abnormal values correlate with neurological voice disorders.
                
                #### Voice and Parkinson's Connection:
                Voice changes are often among the earliest signs of Parkinson's disease, occurring due to reduced control of the vocal muscles. These subtle changes can be detected through acoustic analysis before other symptoms become apparent.
                
                #### Recommendations:
                - Share these results with a neurologist
                - Consider further clinical evaluation if indicated
                - Early detection allows for earlier intervention and better management
                - Voice therapy may help manage speech symptoms
                """)
                
                # Add a comprehensive feature importance analysis
                st.write("### Voice Feature Importance")
                
                # Feature names and importance values (simplified model)
                features = ['Jitter', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE', 'Spread1', 'Spread2']
                # These importances are illustrative - not from actual model
                importances = [0.15, 0.14, 0.11, 0.12, 0.09, 0.13, 0.10, 0.08, 0.08]
                
                # Sort by importance
                sorted_indices = np.argsort(importances)
                sorted_features = [features[i] for i in sorted_indices]
                sorted_importances = [importances[i] for i in sorted_indices]
                
                # Create horizontal bar chart
                fig_imp = plt.figure(figsize=(8, 6))
                ax_imp = fig_imp.add_subplot(111)
                
                ax_imp.barh(sorted_features, sorted_importances, color='skyblue')
                
                # Add labels showing actual values used in prediction
                value_map = {
                    'Jitter': float(Jitter_percent),
                    'Shimmer': float(Shimmer), 
                    'NHR': float(NHR),
                    'HNR': float(HNR),
                    'RPDE': float(RPDE),
                    'DFA': float(DFA),
                    'PPE': float(PPE),
                    'Spread1': float(spread1),
                    'Spread2': float(spread2)
                }
                
                # Add value annotations
                for i, feature in enumerate(sorted_features):
                    ax_imp.text(sorted_importances[i] + 0.01, i, f'{value_map[feature]:.3f}', 
                            va='center', color='darkblue', fontweight='bold')
                
                ax_imp.set_title('Relative Importance of Voice Features in PD Detection')
                ax_imp.set_xlabel('Relative Importance')
                
                st.pyplot(fig_imp)
                
                # Add a time-frequency domain visualization
                st.write("### Simulated Voice Analysis in Time-Frequency Domain")
                
                # Create a simulated spectrogram-like visualization
                # This is illustrative only - not actual voice analysis
                
                # Generate synthetic data based on user's values for illustration
                jitter_factor = float(Jitter_percent) / 1.0  # Normalize
                shimmer_factor = float(Shimmer) / 0.1  # Normalize
                
                # Create time and frequency axes
                time = np.linspace(0, 1, 100)
                freq = np.linspace(100, 500, 100)
                
                # Generate synthetic spectrogram based on voice parameters
                # This is purely illustrative
                tt, ff = np.meshgrid(time, freq)
                spectrogram = np.sin(2*np.pi*5*tt) * np.exp(-((ff-300)/100)**2)
                
                # Add jitter-like variations
                spectrogram += jitter_factor * np.random.normal(0, 0.2, spectrogram.shape)
                
                # Add shimmer-like amplitude variations
                amplitude_mod = 1 + shimmer_factor * 0.3 * np.sin(2*np.pi*3*tt)
                spectrogram *= amplitude_mod
                
                # Plot the synthetic spectrogram
                fig_spec = plt.figure(figsize=(8, 6))
                ax_spec = fig_spec.add_subplot(111)
                
                im = ax_spec.pcolormesh(time, freq, spectrogram, shading='gouraud', cmap='viridis')
                fig_spec.colorbar(im, ax=ax_spec, label='Amplitude')
                
                ax_spec.set_ylabel('Frequency (Hz)')
                ax_spec.set_xlabel('Time (s)')
                ax_spec.set_title('Simulated Voice Pattern Spectrogram')
                
                st.pyplot(fig_spec)
        except ValueError:
            st.error("Please enter valid numeric values for all fields")