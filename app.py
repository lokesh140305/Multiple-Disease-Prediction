from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize models (we'll train them on first request if not saved)
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def train_diabetes_model():
    """Train diabetes prediction model using Logistic Regression"""
    # Sample training data (in production, use real dataset)
    # Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.rand(n_samples, 8)
    X[:, 0] = np.random.randint(0, 17, n_samples)  # Pregnancies
    X[:, 1] = np.random.randint(0, 200, n_samples)  # Glucose
    X[:, 2] = np.random.randint(0, 130, n_samples)  # BloodPressure
    X[:, 3] = np.random.randint(0, 100, n_samples)  # SkinThickness
    X[:, 4] = np.random.randint(0, 850, n_samples)  # Insulin
    X[:, 5] = np.random.uniform(15, 50, n_samples)  # BMI
    X[:, 6] = np.random.uniform(0, 2.5, n_samples)  # DiabetesPedigreeFunction
    X[:, 7] = np.random.randint(20, 80, n_samples)  # Age
    
    # Create target based on some logic
    y = ((X[:, 1] > 140) | (X[:, 5] > 30) | (X[:, 7] > 50)).astype(int)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model

def train_heart_disease_model():
    """Train heart disease prediction model using Random Forest"""
    # Features: Age, Sex, CP, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak, Slope, Ca, Thal
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.rand(n_samples, 13)
    X[:, 0] = np.random.randint(29, 80, n_samples)  # Age
    X[:, 1] = np.random.randint(0, 2, n_samples)  # Sex
    X[:, 2] = np.random.randint(0, 4, n_samples)  # CP
    X[:, 3] = np.random.randint(94, 200, n_samples)  # Trestbps
    X[:, 4] = np.random.randint(126, 564, n_samples)  # Chol
    X[:, 5] = np.random.randint(0, 2, n_samples)  # Fbs
    X[:, 6] = np.random.randint(0, 3, n_samples)  # Restecg
    X[:, 7] = np.random.randint(71, 202, n_samples)  # Thalach
    X[:, 8] = np.random.randint(0, 2, n_samples)  # Exang
    X[:, 9] = np.random.uniform(0, 6.2, n_samples)  # Oldpeak
    X[:, 10] = np.random.randint(0, 3, n_samples)  # Slope
    X[:, 11] = np.random.randint(0, 4, n_samples)  # Ca
    X[:, 12] = np.random.randint(0, 4, n_samples)  # Thal
    
    # Create target
    y = ((X[:, 3] > 140) | (X[:, 4] > 240) | (X[:, 9] > 2)).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_parkinsons_model():
    """Train Parkinson's prediction model using SVM"""
    # Features: MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), etc.
    # Using simplified features: Jitter, Shimmer, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
    np.random.seed(42)
    n_samples = 1000
    
    X = np.random.rand(n_samples, 10)
    X[:, 0] = np.random.uniform(0, 0.1, n_samples)  # Jitter
    X[:, 1] = np.random.uniform(0, 0.5, n_samples)  # Shimmer
    X[:, 2] = np.random.uniform(0, 0.5, n_samples)  # NHR
    X[:, 3] = np.random.uniform(0, 50, n_samples)  # HNR
    X[:, 4] = np.random.uniform(0, 1, n_samples)  # RPDE
    X[:, 5] = np.random.uniform(0, 1, n_samples)  # DFA
    X[:, 6] = np.random.uniform(-10, 10, n_samples)  # spread1
    X[:, 7] = np.random.uniform(0, 1, n_samples)  # spread2
    X[:, 8] = np.random.uniform(0, 10, n_samples)  # D2
    X[:, 9] = np.random.uniform(0, 1, n_samples)  # PPE
    
    # Create target
    y = ((X[:, 0] > 0.05) | (X[:, 1] > 0.3) | (X[:, 9] > 0.5)).astype(int)
    
    model = SVC(probability=True, random_state=42)
    model.fit(X, y)
    return model

def load_or_train_model(model_name, train_func):
    """Load saved model or train new one"""
    model_path = os.path.join(MODELS_DIR, f'{model_name}.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        model = train_func()
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        results = {}
        
        # Diabetes Prediction
        if 'diabetes' in data:
            diabetes_data = data['diabetes']
            features = np.array([[
                float(diabetes_data.get('pregnancies', 0)),
                float(diabetes_data.get('glucose', 0)),
                float(diabetes_data.get('blood_pressure', 0)),
                float(diabetes_data.get('skin_thickness', 0)),
                float(diabetes_data.get('insulin', 0)),
                float(diabetes_data.get('bmi', 0)),
                float(diabetes_data.get('diabetes_pedigree', 0)),
                float(diabetes_data.get('age', 0))
            ]])
            
            model = load_or_train_model('diabetes', train_diabetes_model)
            prob = model.predict_proba(features)[0][1]
            results['diabetes'] = {
                'risk': 'High' if prob > 0.5 else 'Low',
                'probability': float(prob * 100)
            }
        
        # Heart Disease Prediction
        if 'heart' in data:
            heart_data = data['heart']
            features = np.array([[
                float(heart_data.get('age', 0)),
                float(heart_data.get('sex', 0)),
                float(heart_data.get('cp', 0)),
                float(heart_data.get('trestbps', 0)),
                float(heart_data.get('chol', 0)),
                float(heart_data.get('fbs', 0)),
                float(heart_data.get('restecg', 0)),
                float(heart_data.get('thalach', 0)),
                float(heart_data.get('exang', 0)),
                float(heart_data.get('oldpeak', 0)),
                float(heart_data.get('slope', 0)),
                float(heart_data.get('ca', 0)),
                float(heart_data.get('thal', 0))
            ]])
            
            model = load_or_train_model('heart', train_heart_disease_model)
            prob = model.predict_proba(features)[0][1]
            results['heart'] = {
                'risk': 'High' if prob > 0.5 else 'Low',
                'probability': float(prob * 100)
            }
        
        # Parkinson's Prediction
        if 'parkinsons' in data:
            parkinsons_data = data['parkinsons']
            features = np.array([[
                float(parkinsons_data.get('jitter', 0)),
                float(parkinsons_data.get('shimmer', 0)),
                float(parkinsons_data.get('nhr', 0)),
                float(parkinsons_data.get('hnr', 0)),
                float(parkinsons_data.get('rpde', 0)),
                float(parkinsons_data.get('dfa', 0)),
                float(parkinsons_data.get('spread1', 0)),
                float(parkinsons_data.get('spread2', 0)),
                float(parkinsons_data.get('d2', 0)),
                float(parkinsons_data.get('ppe', 0))
            ]])
            
            model = load_or_train_model('parkinsons', train_parkinsons_model)
            prob = model.predict_proba(features)[0][1]
            results['parkinsons'] = {
                'risk': 'High' if prob > 0.5 else 'Low',
                'probability': float(prob * 100)
            }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

