import os
import pandas as pd
import numpy as np
import datetime
import xgboost as xgb
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ============================================
# Data Processing and Model Training
# ============================================

def train_model():
    """Load data, preprocess, and train the salary prediction model"""
    
    # Load dataset
    df = pd.read_csv('1-employee_salary_dataset.csv')
    
    print("Dataset Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    # Data Preprocessing
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Gender', 'Education', 'Department', 'Job_Level', 'Remote_Work', 'City']
    
    df_processed = df.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
        print(f"\n{col} encoding:")
        for i, class_name in enumerate(le.classes_):
            print(f"  {class_name}: {i}")
    
    # Drop Employee_ID as it's not useful for prediction
    X = df_processed.drop(['Employee_ID', 'Annual_Salary_LPA'], axis=1)
    y = df_processed['Annual_Salary_LPA']
    
    print("\nFeatures used for training:")
    print(X.columns.tolist())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train XGBoost Model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate Model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print("="*50)
    
    # Save model
    model.save_model('salary_model.json')
    print("\nModel saved as 'salary_model.json'")
    
    # Save label encoders for later use
    import pickle
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("Label encoders saved as 'label_encoders.pkl'")
    
    return model, label_encoders


# ============================================
# Streamlit Web Application
# ============================================

def main():
    """Main Streamlit application for employee salary prediction"""
    
    # Load model and encoders
    model = xgb.XGBRegressor()
    model.load_model("salary_model.json")
    
    import pickle
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    html_temp = """<h1>Employee Salary Prediction</h1>"""
    
    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown("This app will help you predict an employee's annual salary based on their profile")
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        
        gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        
        education = st.selectbox("Education Level", label_encoders['Education'].classes_)
        education_encoded = label_encoders['Education'].transform([education])[0]
        
        experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=5, step=1)
        
        department = st.selectbox("Department", label_encoders['Department'].classes_)
        department_encoded = label_encoders['Department'].transform([department])[0]
    
    with col2:
        job_level = st.selectbox("Job Level", label_encoders['Job_Level'].classes_)
        job_level_encoded = label_encoders['Job_Level'].transform([job_level])[0]
        
        performance_rating = st.slider("Performance Rating (1-5)", 1, 5, 3)
        
        certifications = st.number_input("Number of Certifications", min_value=0, max_value=20, value=3, step=1)
        
        overtime_hours = st.number_input("Overtime Hours (Monthly)", min_value=0, max_value=100, value=20, step=5)
        
        remote_work = st.selectbox("Remote Work", label_encoders['Remote_Work'].classes_)
        remote_work_encoded = label_encoders['Remote_Work'].transform([remote_work])[0]
    
    col3, col4 = st.columns(2)
    
    with col3:
        city = st.selectbox("City", label_encoders['City'].classes_)
        city_encoded = label_encoders['City'].transform([city])[0]
    
    with col4:
        company_tenure = st.number_input("Company Tenure (Years)", min_value=0, max_value=40, value=3, step=1)
    
    col5, col6 = st.columns(2)
    
    with col5:
        projects_completed = st.number_input("Projects Completed", min_value=0, max_value=100, value=10, step=1)
    
    with col6:
        skill_score = st.number_input("Skill Score (0-100)", min_value=0, max_value=100, value=75, step=5)
    
    # Prepare data for prediction
    data_new = pd.DataFrame({
        'Age': age,
        'Gender': gender_encoded,
        'Education': education_encoded,
        'Experience_Years': experience,
        'Department': department_encoded,
        'Job_Level': job_level_encoded,
        'Performance_Rating': performance_rating,
        'Certifications': certifications,
        'Overtime_Hours': overtime_hours,
        'Remote_Work': remote_work_encoded,
        'City': city_encoded,
        'Company_Tenure': company_tenure,
        'Projects_Completed': projects_completed,
        'Skill_Score': skill_score
    }, index=[0])
    
    if st.button("Predict Salary"):
        prediction = model.predict(data_new)
        st.success(f"Predicted Annual Salary: ₹{prediction[0]:.2f} LPA")
        
        # Display additional insights
        st.markdown("---")
        st.subheader("Employee Profile Summary")
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.write(f"**Age:** {age} years")
            st.write(f"**Experience:** {experience} years")
            st.write(f"**Department:** {department}")
            st.write(f"**Job Level:** {job_level}")
        
        with col_summary2:
            st.write(f"**Education:** {education}")
            st.write(f"**Performance Rating:** {performance_rating}/5")
            st.write(f"**Skill Score:** {skill_score}/100")
            st.write(f"**Company Tenure:** {company_tenure} years")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Train the model
        train_model()
    else:
        # Run the Streamlit app
        main()

