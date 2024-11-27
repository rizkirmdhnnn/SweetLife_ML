import requests
import numpy as np
from bs4 import BeautifulSoup
# from keras.preprocessing.image import load_img, img_to_array

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

# fetch the image of the food        
# def processed_image(img_path, model):
#     img = load_img(img_path, target_size=(224, 224, 3))
#     img = img_to_array(img)
#     img = img / 255
#     img = np.expand_dims(img, [0])
#     answer = model.predict(img)
#     y_class = answer.argmax(axis=-1)
#     print(y_class)
#     y = " ".join(str(x) for x in y_class)
#     y = int(y)
#     res = labels[y]
#     print(res)
#     return res.capitalize()

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
        
        proteins_req = requests.get(proteins_url).text
        calories_req = requests.get(calories_url).text
        carb_req = requests.get(carb_url).text
        fat_req = requests.get(fat_url).text

        proteins_scrap = BeautifulSoup(proteins_req, 'html.parser')
        calories_scrap = BeautifulSoup(calories_req, 'html.parser')
        carb_scrap = BeautifulSoup(carb_req, 'html.parser')
        fat_scrap = BeautifulSoup(fat_req, 'html.parser')

        results = {}

        for scrap, key in zip([proteins_scrap, calories_scrap, carb_scrap, fat_scrap], ["proteins", "calories", "carbohydrates", "fat"]):   
            results[key] = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text
            
        return (
            results["proteins"],
            results["calories"],
            results["carbohydrates"],
            results["fat"]
        )
    except Exception as e:
        return f"error: {e}"