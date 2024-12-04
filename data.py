import requests
import numpy as np
from bs4 import BeautifulSoup
from ultralytics import YOLO

from PIL import Image
from io import BytesIO

# conver to gram
def convert_weight_to_grams(weight):
    """
    Parameters:
        - weight (str): The weight string, e.g., "500mg", "1.2kg", "3t".
        
    Returns:
        - float: The converted weight in grams.
    """
    try:
        # Normalize input (lowercase and strip spaces)
        weight = weight.lower().strip()

        if "µg" in weight or "ug" in weight:
            value = float(weight.replace("µg", "").replace("ug", "").strip())
            return value / 1_000_000

        elif "mg" in weight:
            value = float(weight.replace("mg", "").strip())
            return value / 1000
        
        elif "kg" in weight:
            value = float(weight.replace("kg", "").strip())
            return value * 1000

        elif "g" in weight:
            value = float(weight.replace("g", "").strip())
            return value

        elif "t" in weight:
            value = float(weight.replace("t", "").strip())
            return value * 1_000_000

        else:
            raise ValueError("Unit not recognized. Please use mg, g, kg, or t.")

    except Exception as e:
        raise ValueError(f"Invalid weight format: {e}")


# filter the food for diabetic patient
def filter_food(df, max_calories=None, max_carbohydrate=None, max_fat=None, max_protein=None):
    """
    Filter foods based on calorie, carbohydrate, and fat limits.
    """
    filtered_df = df.copy()
    if max_calories is not None:
        filtered_df = filtered_df[filtered_df['calories'] <= max_calories]
    if max_carbohydrate is not None:
        filtered_df = filtered_df[filtered_df['carbohydrate'] <= max_carbohydrate]
    if max_fat is not None:
        filtered_df = filtered_df[filtered_df['fat'] <= max_fat]
    if max_protein is not None:
        filtered_df = filtered_df[filtered_df['proteins'] <= max_protein]

    return filtered_df

# generate a random food combination
def generate_combinations(food_df, num_combinations=2, items_per_combination=5):
    """
    Generate a random food combination.
    """
    combinations = []
    for _ in range(num_combinations):
        random_selection = food_df.sample(min(len(food_df), items_per_combination))

        combinations.append(random_selection)
    return combinations


# load_model
def load_yolo_model(model_path, image_url):
    """Load the saved YOLO model and make predictions on a local image."""
    
    # Download the image from the URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("Failed to download the image from the URL.")
    
    # Open the image
    image = Image.open(BytesIO(response.content))
    
    # Load the YOLO model
    model = YOLO(model_path)

    # Perform prediction on the input image
    results = model.predict(source=image, save=False, conf=0.25)

    # Format the results
    detections = {}
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.cpu().numpy())  # Class index
            name = model.names[cls]  # Class name
            # Count the detected objects by name
            detections[name] = detections.get(name, 0) + 1

    # Convert the detections dictionary into the desired format
    formatted_results = [{"name": name, "unit": count} for name, count in detections.items()]

    return formatted_results


# fetch the nutritions of the food
def fetch_nutritions(prediction):
    """
    Fetch the nutrition values of the food.
    """
    try:
        proteins_url = 'https://www.google.com/search?&q=proteins in ' + prediction
        calories_url = 'https://www.google.com/search?&q=calories in ' + prediction
        carb_url = 'https://www.google.com/search?&q=carbohydrate in' + prediction
        fat_url = 'https://www.google.com/search?&q=fat in ' + prediction
        sugar_url = 'https://www.google.com/search?q=sugar in ' + prediction
        
        proteins_req = requests.get(proteins_url).text
        calories_req = requests.get(calories_url).text
        carb_req = requests.get(carb_url).text
        fat_req = requests.get(fat_url).text
        sugar_req = requests.get(sugar_url).text

        proteins_scrap = BeautifulSoup(proteins_req, 'html.parser')
        calories_scrap = BeautifulSoup(calories_req, 'html.parser')
        carb_scrap = BeautifulSoup(carb_req, 'html.parser')
        fat_scrap = BeautifulSoup(fat_req, 'html.parser')
        sugar_scrap = BeautifulSoup(sugar_req, 'html.parser')

        # Extracting data or assigning default value (0)
        def extract_value(scrap, class_name="BNeawe iBp4i AP7Wnd"):
            element = scrap.find("div", class_=class_name)
            return element.text if element else "0 g"  # Default value

        proteins = extract_value(proteins_scrap)
        calories = extract_value(calories_scrap)
        carbohydrates = extract_value(carb_scrap)
        fat = extract_value(fat_scrap)
        sugar = extract_value(sugar_scrap)

        return proteins, calories, carbohydrates, fat, sugar
    except Exception as e:
        return f"error: {e}"

def safe_convert(value, unit):
    try:
        return float(value.replace(unit, "").strip().replace(",", "."))
    except (ValueError, AttributeError):
        return 0.0  