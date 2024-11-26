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

def generate_combinations(food_df, num_combinations=2, items_per_combination=5):
    """
    Generate a random food combination.
    """
    combinations = []
    for _ in range(num_combinations):
        random_selection = food_df.sample(min(len(food_df), items_per_combination))

        combinations.append(random_selection)
    return combinations