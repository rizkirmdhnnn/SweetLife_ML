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

from data import filter_food, generate_combinations


app = Flask(__name__)
CORS = CORS(app, origins="*")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "dataset")

# Load the saved models and preprocessing tools
exercise_model = joblib.load("./model/exercise/regressor_model.pkl")
classifier = joblib.load("./model/exercise/classifier_model.pkl")
exercise_scaler = joblib.load("./model/exercise/scaler.pkl")
label_encoder = joblib.load("./model/exercise/label_encoder.pkl")

diabetes_model = joblib.load("./model/diabetes/xgb_model_diabetes.pkl")
diabetes_scaler = joblib.load("./model/diabetes/scaler_diabetes.pkl")

@app.route("/")
def index():
    return "Pranking My Self"

# diabetes prediction
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
    smoking_history_mapping = {"never": 0, "current": 1, "former": 2, "ever": 3, not "current": 4}
    smoking_history = smoking_history_mapping.get(smoking_history.lower(), 0)

    features = np.array([[gender, age, heart_desease, smoking_history, bmi]])
    
    try:
        # Rescale the input data
        features_scaled = diabetes_scaler.transform(features)

        prediction_prob = diabetes_model.predict_proba(features_scaled)[0][1]
        prediction_percentage = float(round(prediction_prob * 100, 2))

        response = {
            "percentage": float(f"{prediction_percentage:.2f}"),
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

    # labelencoder
    gender = 1 if gender.lower() == 'male' else 0

    if None  in [gender, age, height, diabetes, bmi]:
        return jsonify({"error": "all fields must be filled"}), 400

    features = [[gender, age, height, diabetes, bmi]]

    # Scale the input data
    scaled_data = exercise_scaler.transform(features)

    try: 
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/food_recommendation", methods=["POST"])
def food_recommendation():
    """
    Parameters:
        - diabetes_percentage : 0-100 (percentage)

    Returns:
        - Food recommendation based on diabetes or not diabetes
    """
    data = request.json
    diagnoses = data.get("diabetes_percentage")

    if diagnoses is None:
        return jsonify({"error": "The 'diabetes' field must be provided"}), 400

    try:
        diagnoses = float(diagnoses) / 100
        diagnoses = 1 if diagnoses >= 0.5 else 0

        print(diagnoses)
    except (ValueError, TypeError):
        return jsonify({"error": "The 'diabetes' field must be a numeric value"}), 400

    try:
        diabet_food_path = os.path.join(DATA_DIR, "diabet_food_recomendation_clean.csv")
        nutrition_path = os.path.join(DATA_DIR, "nutrition.csv")

        if not os.path.exists(diabet_food_path) or not os.path.exists(nutrition_path):
            return jsonify({"error": "Required data files not found"}), 500

        diabet_food_df = pd.read_csv(diabet_food_path)
        food_df = pd.read_csv(nutrition_path)

        max_calories = diabet_food_df['Calories'].max()
        max_protein = diabet_food_df['Protein'].max()
        max_fat = diabet_food_df['Fat'].max()
        max_carbs = diabet_food_df['Carbohydrates'].max()

        diabetes_food = filter_food(
            food_df, max_calories=max_calories,
            max_carbohydrate=max_carbs, max_fat=max_fat, max_protein=max_protein
        )
        normal_food = food_df

        diabetes_combinations = generate_combinations(diabetes_food)
        normal_combinations = generate_combinations(normal_food)

        food_recommendation = [
            combo[["name", "calories", "carbohydrate", "fat", "proteins", "image"]].to_dict(orient="records")
            for combo in (diabetes_combinations if diagnoses else normal_combinations)
        ]

        response = {
            "diabetes": bool(diagnoses),
            "food_recommendation": food_recommendation
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
