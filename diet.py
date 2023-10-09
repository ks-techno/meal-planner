
import openai
import os
import json
import pandas as pd
pd.set_option('display.max_columns', 500)
from fuzzywuzzy import process
import numpy as np
import pulp
from typing import List
from pydantic import BaseModel, Field





secret = 'key'
openai.api_key = secret




class Ingredient(BaseModel):
    name: str
    calories: int
    mass: str = Field(..., description="Ingredient mass in grams")
    food_group: str = Field(..., description="Type of food like meat, vegetable, cereals, dairy product")
class Recipe(BaseModel):
    meal_type: str = Field(..., description="Break fast, lunch, dinner or Snacks")
    title: str
    steps: List[str]
    meal_calories_sum: int = Field(..., description="sum of individual ingredient calories in meal")
    Ingrediets: List[Ingredient]
class daily_meal(BaseModel):
    Meal: List[Recipe]

class Ingredient:
    def __init__(self, name, protein, fat, carb):
        self.name = name
        self.protein = protein
        self.fat = fat
        self.carb = carb


def cal_bmr(weight_kg, height_cm, age_years, gender, activity_factor):
    gender_fac = { 'Male':-5, 
                  'Female':161 
                 }
    moderate_fac = {
                    "Sedentary": 1.2,
                    "Lightly Active": 1.375,
                    "Moderately Active": 1.55,
                    "Very Active": 1.725,
                    "Extremely Active": 1.9
                    }
    return (            (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) - gender_fac.get(gender,-5)           )*moderate_fac.get(activity_factor,1.55)




def calculate_bmi(weight_kg, height_cm):
    # Convert height from centimeters to meters
    height_m = height_cm / 100

    # Calculate BMI
    bmi = weight_kg / (height_m ** 2)
    return bmi



def weight_from_bmi(bmi, height_cm):
    # Convert height from centimeters to meters
    height_m = height_cm / 100

    # Calculate BMI
    weight_kg = bmi * (height_m ** 2)
    return weight_kg


def goal_selection(weight_kg, height_cm):
    bmi = calculate_bmi(weight_kg, height_cm)
    if bmi>24.9:
        goal = weight_from_bmi(24.9, height_cm)
    elif bmi<18.5:
        goal = weight_from_bmi(18.5, height_cm)
    else:
        goal = weight_kg
    return goal-weight_kg


macros_ratio = {
  "Sedentary": {
    "Protein": 20,
    "Carbohydrates": 60,
    "Fat": 20
  },
  "Lightly Active": {
    "Protein": 25,
    "Carbohydrates": 50,
    "Fat": 25
  },
  "Moderately Active": {
    "Protein": 30,
    "Carbohydrates": 45,
    "Fat": 25
  },
  "Very Active": {
    "Protein": 35,
    "Carbohydrates": 40,
    "Fat": 25
  },
  "Extremely Active": {
    "Protein": 40,
    "Carbohydrates": 40,
    "Fat": 20
  }
}



def adjust_ingredients(ingredients, required_protein, required_fat, required_carb, min_max_bounds):
    prob = pulp.LpProblem("MacronutrientAdjustment", pulp.LpMinimize)
    ingredient_vars = {ingredient: pulp.LpVariable(ingredient.name,
    min_max_bounds[ingredient.name])
    for ingredient in ingredients}
    protein_diff = pulp.LpVariable("protein_diff", 0)
    fat_diff = pulp.LpVariable("fat_diff", 0)
    carb_diff = pulp.LpVariable("carb_diff", 0)
    # Objective: Minimize the total difference from the required macros
    prob += protein_diff + fat_diff + carb_diff, "TotalDifference"
    # Constraints to compute the difference
    prob += sum(ingredient.protein * ingredient_vars[ingredient] for ingredient in ingredients) - required_protein <= protein_diff, "ProteinDifference1"
    prob += required_protein - sum(ingredient.protein * ingredient_vars[ingredient] for ingredient in ingredients) <= protein_diff, "ProteinDifference2"
    prob += sum(ingredient.fat * ingredient_vars[ingredient] for ingredient in ingredients) - required_fat <= fat_diff, "FatDifference1"
    prob += required_fat - sum(ingredient.fat * ingredient_vars[ingredient] for ingredient in ingredients) <= fat_diff, "FatDifference2"
    prob += sum(ingredient.carb * ingredient_vars[ingredient] for ingredient in ingredients) - required_carb <= carb_diff, "CarbDifference1"
    prob += required_carb - sum(ingredient.carb * ingredient_vars[ingredient] for ingredient in ingredients) <= carb_diff, "CarbDifference2"
    prob.solve()
    results = {}
    for ingredient in ingredients:
        results[ingredient.name] = pulp.value(ingredient_vars[ingredient])
    return results


def verify_results(adjusted_ingredients, ingredients_data, required_protein, required_fat, required_carb, tolerance_percent=5):
    total_protein = 0
    total_fat = 0
    total_carb = 0
    for ingredient in ingredients_data:
        amount = adjusted_ingredients.get(ingredient.name, 0)
        total_protein += ingredient.protein * amount
        total_fat += ingredient.fat * amount
        total_carb += ingredient.carb * amount

    protein_range = (required_protein * (1 - tolerance_percent/100), required_protein * (1 + tolerance_percent/100))
    fat_range = (required_fat * (1 - tolerance_percent/100), required_fat * (1 + tolerance_percent/100))
    carb_range = (required_carb * (1 - tolerance_percent/100), required_carb * (1 + tolerance_percent/100))
    
    if protein_range[0] <= total_protein <= protein_range[1] and         fat_range[0] <= total_fat <= fat_range[1] and         carb_range[0] <= total_carb <= carb_range[1]:
        return True
    return False



def ratios_adjustment_data_frame(diet_frame, calories):
    diet_frame.loc[:, 'pr_cl'] = diet_frame['Protein (g)'] * 4
    diet_frame.loc[:, 'ft_cl'] = diet_frame['Fat (g)'] * 9
    diet_frame.loc[:, 'cb_cl'] = diet_frame['Carbohydrate (g)'] * 4
    diet_frame['prot_rt'] = diet_frame['pr_cl']/(diet_frame['ft_cl']+diet_frame['pr_cl']+diet_frame['cb_cl'])
    diet_frame['fat_rt'] = diet_frame['ft_cl']/(diet_frame['ft_cl']+diet_frame['pr_cl']+diet_frame['cb_cl'])
    diet_frame['carb_rt'] = diet_frame['cb_cl']/(diet_frame['ft_cl']+diet_frame['pr_cl']+diet_frame['cb_cl'])
    ingredients_data = [Ingredient(diet_frame['name'][i],
                                   diet_frame['prot_rt'][i],
                                   diet_frame['fat_rt'][i],
                                   diet_frame['carb_rt'][i])
                        for i in range(len(diet_frame))]
    required_protein = (macros_ratio[activity_factor]['Protein']/100)*(calories)
    required_fat = (macros_ratio[activity_factor]['Fat']/100)*(calories)
    required_carb = (macros_ratio[activity_factor]['Carbohydrates']/100)*(calories)
    min_max_bounds = {diet_frame['name'][i]: diet_frame['calories'][i]*0.5
                        for i in range(len(diet_frame))}
    tolerance = 5
    adjusted_ingredients = adjust_ingredients(ingredients_data, 
                                              required_protein, required_fat, required_carb, 
                                              min_max_bounds)
    for name, amount in adjusted_ingredients.items():
        diet_frame.loc[diet_frame['name']==name, 'calories'] = amount
        

    diet_frame['1cal_gm'] = 100/diet_frame['Calories']
    temp_cal = sum(diet_frame['calories'])
    diet_frame['cal_ratio'] = diet_frame['calories']/temp_cal
    diet_frame['required_calories'] = ((calories)*diet_frame['cal_ratio'])
    diet_frame['per_meal_gms'] = diet_frame['required_calories'] * diet_frame['1cal_gm']
    diet_frame['taste'] = ''
    diet_frame.loc[diet_frame['required_calories']==0.0, 'taste'] = 'As per taste'
    df = pd.DataFrame(diet_frame[['Name', 'per_meal_gms']])
    result = []
    for index, row in df.iterrows():
        result.append({k: v for k, v in row.items()})
    return result


# Function to get user input with default values
def get_user_input(prompt, default_value):
    try:
        user_input = input(prompt)
        if user_input:
            return user_input
        else:
            return default_value
    except Exception:
        return default_value


def main(gender, height_cm, weight_kg, age_years, \
    diet_type, activity_factor, avoid, include, \
    cuisine_type, max_number_of_ingredients, \
    max_cooking_time, country_available, request, \
    number_of_meals, time_of_goal):
    """
    The main function can also be imported from any python script by passing the arguments.
    """
    how_much_weight_kg_diff_goal = goal_selection(weight_kg, height_cm)
    # BMR Calculation (Mifflin-St. Jeor equation)
    d_bmr = cal_bmr(weight_kg, height_cm, age_years,  gender, activity_factor)


    cal_deficit_total = how_much_weight_kg_diff_goal*7700 #1kg for 7700 calories and 1 pound for 3500

    if time_of_goal == 0:
        time_of_goal = int(abs((cal_deficit_total/3500)*7))


    cal_deficit_daily = cal_deficit_total/time_of_goal

    required_calories_daily =  int(d_bmr+cal_deficit_daily)

    # Display the results
    print(f"User needs to take total of {required_calories_daily} in a day to achieve goal within {time_of_goal} days.")


    if diet_type == 'paleo diet':
        prompt = f"Generate a recipe containing {required_calories_daily} calories according to paleo diet."
    elif diet_type == 'keto diet':
        prompt = f"Generate a recipe containing {required_calories_daily} calories according to keto diet."
    else:
        prompt = f"Generate a recipe containing {required_calories_daily} calories with ratio of protein={macros_ratio[activity_factor]['Protein']}, carbohydrates={macros_ratio[activity_factor]['Carbohydrates']} and fats={macros_ratio[activity_factor]['Fat']}."

    prompt+=f"Daily meal plan has to be generated including {number_of_meals} meals in a day. "
    prompt+=f"Ingredients should be available in {country_available} and suitable for cuisine type {cuisine_type}. Ingredients included in recipe should not exceed count of {max_number_of_ingredients} with maximum cooking time {max_cooking_time} minutes. And diet should follow special request of client ({request})."
    if len(avoid)>0:
        prompt+=f" Recipe should avoid {', '.join(avoid)}."
    if len(include)>0:
        prompt+=f" Recipe should include {', '.join(include)}."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
        {"role": "user", "content": prompt}
        ],
        functions=[
            {
            "name": "get_steps_and_ingredients_query",
            "description": "Get user answer in series of steps",
            "parameters": daily_meal.schema()
            }
        ],
        function_call={"name": "get_steps_and_ingredients_query"}
    )

    output = json.loads(response.choices[0]["message"]["function_call"]["arguments"])
    # output = [{'meal_type': 'breakfast', 'title': 'Greek Yogurt Bowl', 'steps': ['In a bowl, mix together Greek yogurt, honey, and vanilla extract.', 'Top the yogurt mixture with granola, berries, and sliced almonds.', 'Serve immediately.'], 'meal_calories_sum': 600, 'Ingrediets': [{'name': 'Greek yogurt', 'calories': 150, 'mass': '100g', 'food_group': 'Dairy'}, {'name': 'Honey', 'calories': 60, 'mass': '2 tbsp', 'food_group': 'Condiments'}, {'name': 'Vanilla extract', 'calories': 10, 'mass': '1 tsp', 'food_group': 'Condiments'}, {'name': 'Granola', 'calories': 150, 'mass': '1/4 cup', 'food_group': 'Grains'}, {'name': 'Berries', 'calories': 60, 'mass': '1/4 cup', 'food_group': 'Fruits'}, {'name': 'Sliced almonds', 'calories': 170, 'mass': '2 tbsp', 'food_group': 'Nuts and Seeds'}]}, {'meal_type': 'lunch', 'title': 'Quinoa Salad', 'steps': ['Cook quinoa according to package instructions and let it cool.', 'In a large bowl, combine quinoa, diced cucumbers, diced bell peppers, chopped parsley, and crumbled feta cheese.', 'In a small bowl, whisk together olive oil, lemon juice, garlic powder, salt, and pepper to make the dressing.', 'Pour the dressing over the quinoa mixture and toss to combine.', 'Serve chilled.'], 'meal_calories_sum': 800, 'Ingrediets': [{'name': 'Quinoa', 'calories': 120, 'mass': '1/2 cup', 'food_group': 'Grains'}, {'name': 'Cucumbers', 'calories': 20, 'mass': '1/2 cup', 'food_group': 'Vegetables'}, {'name': 'Bell peppers', 'calories': 30, 'mass': '1/2 cup', 'food_group': 'Vegetables'}, {'name': 'Parsley', 'calories': 10, 'mass': '2 tbsp', 'food_group': 'Herbs'}, {'name': 'Feta cheese', 'calories': 120, 'mass': '1/4 cup', 'food_group': 'Dairy'}, {'name': 'Olive oil', 'calories': 120, 'mass': '2 tbsp', 'food_group': 'Oils'}, {'name': 'Lemon juice', 'calories': 10, 'mass': '1 tbsp', 'food_group': 'Condiments'}, {'name': 'Garlic powder', 'calories': 10, 'mass': '1 tsp', 'food_group': 'Condiments'}, {'name': 'Salt', 'calories': 0, 'mass': 'to taste', 'food_group': 'Condiments'}, {'name': 'Pepper', 'calories': 0, 'mass': 'to taste', 'food_group': 'Condiments'}]}, {'meal_type': 'dinner', 'title': 'Grilled Chicken with Vegetables', 'steps': ['Preheat grill to medium heat.', 'Season chicken breasts with salt, pepper, and your choice of seasonings.', 'Grill chicken for 6-8 minutes on each side or until cooked through.', 'In a separate grill pan, cook sliced zucchini, bell peppers, and onions with olive oil, salt, and pepper until tender.', 'Serve grilled chicken with grilled vegetables.'], 'meal_calories_sum': 1000, 'Ingrediets': [{'name': 'Chicken breasts', 'calories': 200, 'mass': '1 piece', 'food_group': 'Meat'}, {'name': 'Salt', 'calories': 0, 'mass': 'to taste', 'food_group': 'Condiments'}, {'name': 'Pepper', 'calories': 0, 'mass': 'to taste', 'food_group': 'Condiments'}, {'name': 'Seasonings', 'calories': 0, 'mass': 'to taste', 'food_group': 'Condiments'}, {'name': 'Zucchini', 'calories': 20, 'mass': '1/2 cup', 'food_group': 'Vegetables'}, {'name': 'Bell peppers', 'calories': 30, 'mass': '1/2 cup', 'food_group': 'Vegetables'}, {'name': 'Onions', 'calories': 10, 'mass': '1/4 cup', 'food_group': 'Vegetables'}, {'name': 'Olive oil', 'calories': 120, 'mass': '2 tbsp', 'food_group': 'Oils'}]}]
    print('gpt out put:\n', output)


    food_db_org = pd.read_csv("food_db.csv")
    ignored = list(food_db_org.loc[(food_db_org['200 Calorie Weight (g)'].isnull()) | (food_db_org['200 Calorie Weight (g)'] == 0.0) | (food_db_org['Calories'].isnull()) | (food_db_org['Calories'] == 0.0) ]['Name'])
    food_db = food_db_org.loc[~((food_db_org['200 Calorie Weight (g)'].isnull()) | (food_db_org['200 Calorie Weight (g)'] == 0.0) | (food_db_org['Calories'].isnull()) | (food_db_org['Calories'] == 0.0)) ]



    meals_list = []
    for meal in output:
        diet_frame = pd.DataFrame()
        for ingreient in meal['Ingrediets']:
            query = ingreient['name']

            best_match, score, index = process.extractOne(query, food_db['Name'])
            ingreient['similarity_score'] = score
            if (score < 50) or (best_match not in ignored):
                best_match, score, index = process.extractOne(ingreient['food_group'], food_db['Food Group'])
            result = pd.concat([pd.DataFrame(ingreient, index=[0]), 
                                food_db.iloc[[index]][['Name','Calories', 'Food Group', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)', '200 Calorie Weight (g)']].reset_index(drop=True)
                            ], axis=1)
            diet_frame = pd.concat([diet_frame, result], axis=0)
        diet_frame.reset_index(drop=True, inplace=True)
        meals_list.append({'calories': meal['meal_calories_sum'], 
                        'meal_type': meal['meal_type'], 
                        'title': meal['title'],
                        'steps': meal['steps'],
                        'data_frame': diet_frame})

    updated_meal_data_frames = {}
    for diet_frame in meals_list:
        updated_meal_data_frames[diet_frame['meal_type']] = \
        {'title': diet_frame['title'],
        'steps': diet_frame['steps'],
        'Ingrediets': ratios_adjustment_data_frame(diet_frame = diet_frame['data_frame'], 
                                                    calories = diet_frame['calories'])}
    return updated_meal_data_frames




if __name__ == '__main__':
    # Default values
    gender = "Male"
    height_cm = 170
    weight_kg = 81
    age_years = 26
    diet_type = 'ratio based diet'
    activity_factor = 'Moderately Active'
    avoid = ['Tomatoes']
    include = ['Greek Yogurt']
    cuisine_type = 'any'
    max_number_of_ingredients = 5
    max_cooking_time = 30
    country_available = 'globally'
    request = 'diet should almost be balanced'
    number_of_meals = 3
    time_of_goal = 90
    use_recommended_time = 1

    # Assign values based on user input or default values
    gender = get_user_input("Enter your gender (Male/Female): ", gender)
    height_cm = int(get_user_input("Enter your height in cm: ", height_cm))
    weight_kg = int(get_user_input("Enter your weight in kg: ", weight_kg))
    age_years = int(get_user_input("Enter your age in years: ", age_years))
    diet_type = get_user_input("Enter your diet type: ", diet_type)
    activity_factor = get_user_input("Enter your activity factor: ", activity_factor)
    avoid = get_user_input("Enter foods to avoid (comma-separated): ", ','.join(avoid)).split(',')
    include = get_user_input("Enter foods to include (comma-separated): ", ','.join(include)).split(',')
    cuisine_type = get_user_input("Enter your preferred cuisine type: ", cuisine_type)
    max_number_of_ingredients = int(get_user_input("Enter max number of ingredients: ", max_number_of_ingredients))
    max_cooking_time = int(get_user_input("Enter max cooking time in minutes: ", max_cooking_time))
    country_available = get_user_input("Enter available countries: ", country_available)
    request = get_user_input("Enter your dietary request: ", request)
    number_of_meals = int(get_user_input("Enter number of meals between 2 to 4 in a day: ", number_of_meals))
    use_recommended_time = int(get_user_input("Enter 1 if want to use recommended time or 2 if want to use self defined time period", use_recommended_time))
    if use_recommended_time and use_recommended_time == 1:
        time_of_goal = 0
    else:
        time_of_goal = int(get_user_input("Enter your goal time in days: ", time_of_goal))
    meal_plan_suggested = main(gender, height_cm, weight_kg, age_years, \
                               diet_type, activity_factor, avoid, include, \
                               cuisine_type, max_number_of_ingredients, \
                               max_cooking_time, country_available, request, \
                               number_of_meals, time_of_goal)
    print("meal_plan_suggested:\n", meal_plan_suggested)
