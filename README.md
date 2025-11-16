# Multi-Disease Prediction System

A web-based machine learning application that predicts the risk of three common diseases: **Diabetes**, **Heart Disease**, and **Parkinson's Disease**.

## 🎯 Problem Statement

Patients often lack early detection tools for common diseases like diabetes, heart disease, and Parkinson's. This system provides a quick and accessible way to assess health risks using machine learning models.

## 🚀 Features

- **Multi-Disease Prediction**: Assess risk for three different diseases
- **User-Friendly Interface**: Clean, modern web interface with tabbed navigation
- **Real-Time Results**: Instant risk assessment with probability scores
- **Visual Feedback**: Color-coded results (Low/High Risk) with probability bars

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn, XGBoost
- **Models Used**:
  - Diabetes → Logistic Regression
  - Heart Disease → Random Forest / XGBoost
  - Parkinson's → SVM
- **Frontend**: HTML, CSS, JavaScript

## 📦 Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Select a disease tab** (Diabetes, Heart Disease, or Parkinson's)

4. **Fill in the required health parameters**

5. **Click the predict button** to get your risk assessment

## 📊 Disease Parameters

### Diabetes
- Number of Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI
- Diabetes Pedigree Function
- Age

### Heart Disease
- Age, Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope, CA, Thal

### Parkinson's Disease
- Jitter, Shimmer
- NHR, HNR
- RPDE, DFA
- Spread1, Spread2
- D2, PPE

## ⚠️ Important Disclaimer

This application is for **educational and informational purposes only**. It is **NOT a substitute** for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📝 Notes

- Models are trained on sample data for demonstration purposes
- For production use, train models on real, validated medical datasets
- Results should be interpreted as risk indicators, not definitive diagnoses

## 🎓 Impact

- Helps users quickly cross-check health risks
- Demonstrates understanding of classification models
- Showcases full-stack ML deployment capabilities

