from flask import Flask, request, jsonify
from flask_cors import CORS

import xgboost as xgb
import sklearn
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import json

import os
import random
import joblib


app = Flask(__name__)
CORS = CORS(app, origins="*")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "dataset")

# Load the saved models and preprocessing tools
exercise_model = joblib.load("../model/exercise/regressor_model.pkl")
classifier = joblib.load("../model/exercise/classifier_model.pkl")
exercise_scaler = joblib.load("../model/exercise/scaler.pkl")
label_encoder = joblib.load("../model/exercise/label_encoder.pkl")

diabetes_model = joblib.load("../model/diabetes/xgb_model_diabetes.pkl")
diabetes_scaler = joblib.load("../model/diabetes/scaler_diabetes.pkl")

@app.route("/")
def index():
    return "Pranking My Self"

# diabetes prediction
# todo : fix scaller
@app.route("/diabetes_predict", methods=["POST"])
def diabetes_predict():
    """
    parameters:
        - features : [gender, age, heart_desease, smoking_history, bmi]
    
    Return: 
        - Prediction Percentage
        - Note
    """
    

    data = request.json
    gender = data.get('gender')
    age = data.get('age')
    heart_desease = data.get('heart_desease')
    smoking_history = data.get('smoking_history')
    bmi = data.get('bmi')

    if None in [gender, age, heart_desease, smoking_history, bmi]:
        return jsonify({"error": "all fields must be filled"}), 400
    
    gender = 1 if gender.lower() == 'male' else 0
    smoking_history_mapping = {'never': 0, 'former': 1, 'current': 2}
    smoking_history = smoking_history_mapping.get(smoking_history.lower(), 0)

    features = np.array([[gender, age, heart_desease, smoking_history, bmi]])
    features = [gender, age, heart_desease, smoking_history, bmi]
    
    try:
        # Rescale the input data
        features_scaled = diabetes_scaler.transform([features])

        # prediction_prob = model_diabet.predict(features_scaled)[:, 1][0]
        prediction_prob = diabetes_model.predict(features_scaled) 
        prediction_percentage = round(prediction_prob * 100, 2)

        response = {
            "percentage": prediction_percentage,
            "note": "The patient may be prone to diabetes. Please consult a doctor." if prediction_percentage > 50 else "Patients may not be prone to diabetes."
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#exercise recommendation
@app.route("/exercise_recommendation", methods=["POST"])
def exercise_recomendation():
    """
    Parameters:
        - features: [Gender, Age, Height, Diabetes, BMI]
    
    Returns:
        - Regression predictions (Calories Burned, Exercise Duration)
        - Classification prediction (Predicted Exercise Category)
    """
    data = request.json
    gender = data.get('gender')
    age = data.get('age')
    height = data.get('height')
    diabetes = data.get('diabetes')
    bmi = data.get('bmi')

    if None  in [gender, age, height, diabetes, bmi]:
        return jsonify({"error": "all fields must be filled"}), 400

    features = np.array([[gender, age, height, diabetes, bmi]])

    # Scale the input data
    scaled_data = exercise_scaler.transform(features)
    
    # Regression Predictions
    regression_results = exercise_model.predict(scaled_data)
    
    # Classification Predictions
    classification_results = classifier.predict(scaled_data)
    decoded_class = label_encoder.inverse_transform(classification_results)

    response = {
        "calories_burned": regression_results[0][0],
        "exercise_duration": round(regression_results[0][1], 2),
        "exercise_category": decoded_class[0]
    }    

    return jsonify(response)

# todo : not completed
@app.route("/food_recommendation", methods=["POST"])
def food_recommendation():
    """
    Parameters:
        - diabet_diagnose : true or false
    
    Returns:
        - Food recommendation based on diabetes or not diabetes
    """

    data = request.json
    diagnose = data.get('diabetes')

    if None in [diagnose]:
        return jsonify({"error": "fields must be filled"}), 400

    diagnose = 1 if diagnose >= 0.5 else 0

    df = pd.read_csv(os.path.join(DATA_DIR, "diabet_food_recomendation_clean.csv"))
    return df.to_json(orient="records")


if __name__ == "__main__":
    app.run(debug=True)
