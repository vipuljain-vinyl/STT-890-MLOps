import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from pathlib import Path

def create_model_1():
    # Load the dataset
    current_path = Path(__file__)
    data_path = current_path.parents[2]/"data"/"sampregdata.csv"
    data = pd.read_csv(data_path)

    
    X = data[['x1']]  # We are using 'x1' as the feature for the model
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the linear regression model
    model_1 = LinearRegression()
    model_1.fit(X_train, y_train)

    # Make predictions
    y_pred_1 = model_1.predict(X_test)

    # Calculate the mean squared error
    mse_1 = mean_squared_error(y_test, y_pred_1)
    print(f'Mean Squared Error (Model 1): {mse_1}')

    # Save the model using joblib
    joblib.dump(model_1, 'model_1_feature.pkl')  # Saving the model as a pickle file
    print("Model 1 saved as 'model_1_feature.pkl'")

if __name__ == "__main__":
    create_model_1()
