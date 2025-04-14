import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    data = {
        'Area': [750, 1200, 1800, 2400, 3000, 3600, 4200, 5000, 6000, 7000],
        'Bedrooms': [1, 2, 3, 3, 4, 4, 5, 5, 6, 7],
        'Age': [10, 5, 15, 20, 7, 3, 30, 25, 18, 10],
        'Price': [75, 150, 200, 250, 300, 350, 400, 450, 500, 600]
    }
    
    df = pd.DataFrame(data)

    X = df[['Area', 'Bedrooms', 'Age']]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print("Data Preprocessing Completed!")
