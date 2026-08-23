# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
Product_sales_predictor_api = Flask("SuperKart Product Sales Prediction")

# Load the trained machine learning model
model = joblib.load("deployment_files/superkart_model.joblib")

# Define a route for the home page (GET request)
@product_sales_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Product Sales Prediction API!"

# Define an endpoint for single property prediction (POST request)
@product_sales_predictor_api.post('/v1/rental')
def predict_rental_price():
    """
    This function handles POST requests to the '/v1/rental' endpoint.
    It expects a JSON payload containing property details and returns
    the predicted rental price as a JSON response.
    """
    # Get the JSON data from the request body
    property_data = request.get_json()

    # Extract relevant features from the JSON data
    sample = {
        'Product_Weight': property_data['Product_Weight'],
        'Product_Sugar_Content': property_data['Product_Sugar_Content'],
        'Product_Allocated_Area': property_data['Product_Allocated_Area'],
        'Product_MRP': property_data['Product_MRP'],
        'Store_Size': property_data['Store_Size'],
        'Store_Location_City_Type': property_data['Store_Location_City_Type'],
        'Store_Type': property_data['Store_Type'],
        'Store_Age': property_data['Store_Age'],
        'Product_Id_char': property_data['Product_Id_char'],
        'Product_Type_Category': property_data['Product_Type_Category']
           }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_sales to Python float
    predicted_sales = round(float(predicted_sales), 2)
    
    # Return the predicted sales
    return jsonify({'Predicted Sales (in dollars)': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@product_sales_predictor_api.post('/v1/rentalbatch')
def predict_rental_price_batch():
    """
    This function handles POST requests to the '/v1/rentalbatch' endpoint.
    It expects a CSV file containing property details for multiple properties
    and returns the predicted rental prices as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all properties in the DataFrame
    predicted_sales_batch = model.predict(input_data).tolist()

    # Convert predicted_sales_batch to Python floats and round
    predicted_sales_batch = [round(float(sales), 2) for sales in predicted_sales_batch]

    # Create a dictionary of predictions with row indices as keys (assuming no specific ID column in batch data for now)
    output_dict = {f'Row_{i}': sales for i, sales in enumerate(predicted_sales_batch)}

    # Return the predictions dictionary as a JSON response
    return output_dict

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    product_sales_predictor_api.run(debug=True)
