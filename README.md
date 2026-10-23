# Health Insurance Premium Prediction

## Project Overview
This mini project predicts health insurance premiums based on user details using Machine Learning and FastAPI.

## Technologies Used
- Python
- FastAPI
- Scikit-learn
- NumPy
- HTML
- CSS
- Machine Learning — Linear Regression

## Features
- Takes user details as input.
- Predicts the estimated health insurance premium using a trained ML model.
- Provides a simple web interface.
- Uses FastAPI for backend processing.

## Input Features
- Age
- Gender
- BMI (Body Mass Index)
- Number of Children
- Smoking Status

## Project Structure
```text
health-insurance-prediction/
├── main.py
├── model_Health_insurance.pkl
├── requirements.txt
└── frontend/
    ├── index.html
    └── style.css
```

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone YOUR_REPOSITORY_URL
   ```

2. Open the project folder:
   ```bash
   cd health-insurance-prediction
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI application:
   ```bash
   uvicorn main:app --reload
   ```

5. Open the application in your browser:
   `http://127.0.0.1:8000`

## Note
This project is developed for educational purposes as a Machine Learning mini project.
