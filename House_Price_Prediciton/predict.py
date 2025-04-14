import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("house_price_model.pkl")

def predict_price(area, bedrooms, age):
    features = pd.DataFrame([[area, bedrooms, age]], columns=['Area', 'Bedrooms', 'Age'])  # Adding column names
    predicted_price = model.predict(features)[0]
    return predicted_price

# Example prediction
area = 25000
bedrooms = 3
age = 10
predicted_price = predict_price(area, bedrooms, age)
print(f"Predicted House Price for Area={area} sqft, Bedrooms={bedrooms}, Age={age} years: {predicted_price:.2f}K")
