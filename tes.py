import requests
from bs4 import BeautifulSoup



# fetch the nutritions of the food
def fetch_nutritions(prediction):
    """
    Fetch the nutrition values of the food.
    """
    try:
        proteins_url = 'https://www.google.com/search?&q=proteins+in+' + prediction
        calories_url = 'https://www.google.com/search?&q=calories+in+' + prediction
        carb_url = 'https://www.google.com/search?&q=carbohydrate in' + prediction
        fat_url = 'https://www.google.com/search?&q=fat in ' + prediction
        sugar_url = 'https://www.google.com/search?q=sugar in ' + prediction
        
        print(proteins_url) 
        
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
    
if __name__ == "__main__":
    food_name = "chicken"
    tes = fetch_nutritions(food_name)
    print(tes) 