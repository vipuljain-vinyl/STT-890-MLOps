import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from pathlib import Path

def create_model_2():
    # Load the dataset
    current_path = Path(__file__)
    data_path = current_path.parents[2]/"data"/"sampregdata.csv"
    data = pd.read_csv(data_path)

    # Use two features, x1 and x2
    X = data[['x1', 'x2']]  # This is the new model with two features
    y = data['y']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model_2 = LinearRegression()
    model_2.fit(X_train, y_train)

    # Make predictions
    y_pred_2 = model_2.predict(X_test)

    # Calculate the mean squared error for the new model
    mse_2 = mean_squared_error(y_test, y_pred_2)
    print(f'Mean Squared Error (Model 2): {mse_2}')

    # Save the second model
    joblib.dump(model_2, 'model_2_features.pkl')  # Saving the model with two features
    print("Model 2 saved as 'model_2_features.pkl'")

if __name__ == "__main__":
    create_model_2()
