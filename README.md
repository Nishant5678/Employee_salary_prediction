# Employee Salary Prediction Application

A machine learning-based web application that predicts employee annual salaries based on their professional profile and work characteristics.

## Overview

This application uses **XGBoost Regression** to predict employee salaries with high accuracy. It provides an interactive Streamlit interface where users can input employee details and receive instant salary predictions based on a trained model.

## Features

- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Real-time Predictions**: Get instant salary predictions by entering employee details
- **Comprehensive Input Parameters**: 
  - Personal: Age, Gender, Education, Experience
  - Professional: Department, Job Level, Performance Rating
  - Work Details: Certifications, Overtime Hours, Remote Work Status, City
  - Career: Company Tenure, Projects Completed, Skill Score
- **Employee Profile Summary**: View detailed profile information after prediction
- **High-Performance Model**: 
  - Train R² Score: 95.93%
  - Test R² Score: 75.18%
  - Test MAE: ₹3.64 LPA (average prediction error)

## Dataset

- **Total Records**: 5,020 employee records
- **Features**: 16 columns including demographics, work experience, and performance metrics
- **Target Variable**: Annual_Salary_LPA (Annual Salary in Lakhs per annum)

### Dataset Columns:
- `Employee_ID`: Unique employee identifier
- `Age`: Employee age
- `Gender`: Male/Female
- `Education`: Bachelor, Diploma, Master, PhD
- `Experience_Years`: Years of professional experience
- `Department`: HR, IT, Marketing, Operations, Sales
- `Job_Level`: Junior, Mid, Manager, Lead, Senior
- `Performance_Rating`: 1-5 scale
- `Certifications`: Number of professional certifications
- `Overtime_Hours`: Monthly overtime hours
- `Remote_Work`: Yes/No
- `City`: Chennai, Delhi, Hyderabad, Mumbai
- `Company_Tenure`: Years with current company
- `Projects_Completed`: Number of completed projects
- `Skill_Score`: Technical skill rating (0-100)
- `Annual_Salary_LPA`: Target salary (Lakhs Per Annum)

## Technologies Used

- **Python 3.x**
- **XGBoost**: Machine learning model
- **Scikit-learn**: Data preprocessing and model evaluation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Streamlit**: Web application framework

## Installation

### Prerequisites
- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Step 1: Clone/Download the Project
```bash
cd c:\Users\nisha\OneDrive\Desktop\python\emp_salary\employee_predict
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost streamlit
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the Streamlit Web Application

```bash
cd c:\Users\nisha\OneDrive\Desktop\python\emp_salary\employee_predict
streamlit run employee_salary_prediction.py
```

The app will open in your browser at `http://localhost:8501`

### Option 2: Retrain the Model

```bash
cd c:\Users\nisha\OneDrive\Desktop\python\emp_salary\employee_predict
python employee_salary_prediction.py train
```

This will:
- Load the dataset from `1-employee_salary_dataset.csv`
- Preprocess and encode categorical variables
- Train the XGBoost model
- Generate `salary_model.json` and `label_encoders.pkl`
- Display model performance metrics

## Project Structure

```
employee_predict/
├── employee_salary_prediction.py      # Main application file
├── salary_model.json                  # Trained XGBoost model
├── label_encoders.pkl                 # Categorical variable encoders
├── 1-employee_salary_dataset.csv      # Training dataset
├── 1-employee_salary_dataset.ipynb    # EDA notebook
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Model Details

### Algorithm: XGBoost Regressor

**Hyperparameters:**
- `n_estimators`: 100
- `max_depth`: 6
- `learning_rate`: 0.1
- `random_state`: 42

### Performance Metrics:

| Metric | Train | Test |
|--------|-------|------|
| MAE (Mean Absolute Error) | ₹1.85 LPA | ₹3.64 LPA |
| RMSE (Root Mean Squared Error) | ₹2.90 LPA | ₹7.26 LPA |
| R² Score | 0.9593 | 0.7518 |

### Data Processing:

The model uses the following features for predictions:
1. Age
2. Gender (encoded: Female=0, Male=1)
3. Education (Bachelor=0, Diploma=1, Master=2, PhD=3)
4. Experience_Years
5. Department (HR=0, IT=1, Marketing=2, Operations=3, Sales=4)
6. Job_Level (Junior=0, Lead=1, Manager=2, Mid=3, Senior=4)
7. Performance_Rating
8. Certifications
9. Overtime_Hours
10. Remote_Work (No=0, Yes=1)
11. City (Chennai=0, Delhi=1, Hyderabad=2, Mumbai=3)
12. Company_Tenure
13. Projects_Completed
14. Skill_Score

## How to Use the Web Application

1. **Launch the App**: Run `streamlit run employee_salary_prediction.py`

2. **Enter Employee Details**:
   - Fill in all required fields in the left and right columns
   - Use sliders and dropdowns for categorical variables
   - Enter numeric values for continuous variables

3. **Get Prediction**:
   - Click the **"Predict Salary"** button
   - View the predicted annual salary in LPA (Lakhs Per Annum)
   - See the employee profile summary below

4. **Interpret Results**:
   - Salary is displayed in Indian Rupees (LPA format)
   - Profile summary shows the entered employee details
   - Use predictions for salary benchmarking and HR decisions

## Example Prediction

**Input:**
- Age: 35 years
- Experience: 10 years
- Education: Master's degree
- Department: IT
- Job Level: Mid-level
- Performance Rating: 4/5
- Skill Score: 85/100

**Output:** Predicted Annual Salary: ₹52.45 LPA

## Troubleshooting

### Port Already in Use
If port 8501 is already in use, run:
```bash
streamlit run employee_salary_prediction.py --server.port 8502
```

### Module Not Found Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Model Files Missing
If `salary_model.json` or `label_encoders.pkl` are missing, retrain the model:
```bash
python employee_salary_prediction.py train
```

## Model Interpretation

### Feature Importance (Based on XGBoost):
The model considers these factors most important for salary prediction:
1. **Experience_Years** - Years of professional experience
2. **Skill_Score** - Technical competency level
3. **Job_Level** - Position hierarchy
4. **Performance_Rating** - Work performance
5. **Department** - Organizational unit
6. **Education** - Educational qualification
7. **Age** - Employee age
8. Other factors (certifications, company tenure, overtime, etc.)

## Future Enhancements

- [ ] Add more visualization capabilities
- [ ] Implement ensemble models for better accuracy
- [ ] Add salary trend analysis
- [ ] Include company size as a feature
- [ ] Add comparative salary analysis
- [ ] Deploy to cloud platform (Heroku, AWS)
- [ ] Add salary negotiation recommendations

## Author

Created as a Machine Learning project for salary prediction and HR analytics.

## License

This project is open source and available for educational purposes.

## Support

For issues or questions, please refer to the project files or documentation within the application.

---

**Last Updated**: July 15, 2026  
**Model Version**: 1.0  
**Python Version**: 3.x
