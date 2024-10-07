# FastAPI Model Prediction API

This FastAPI-based application provides an API for making predictions using a trained RandomForest model. The application accepts POST requests with input data, applies the model, and returns the prediction.

## Project Structure

- `api.py`: The main FastAPI application script that defines the endpoints for the API.
- `random_forest_model.pkl`: The pre-trained RandomForest model that is loaded for predictions (make sure to train and save this model beforehand).
- `requirements.txt`: The list of Python dependencies required to run the project.

## Endpoints

### 1. Root Endpoint
- **URL**: `/`
- **Method**: GET
- **Description**: Returns a welcome message with instructions on using the API.

### 2. Prediction Endpoint
- **URL**: `/predict/`
- **Method**: POST
- **Description**: Accepts input data as JSON, uses the trained RandomForest model to make predictions, and returns the prediction.
  
#### Example Request:

```json
{
    "service_type": 6,
    "actual_weight_amount": 9.4,
    "dim_length": 3,
    "dim_width": 4,
    "dim_height": 5,
    "tracking_id_charge_description_1": 3,
    "tracking_id_charge_description_3": 5,
    "tracking_id_charge_description_4": 2,
    "distance": 177.05
}
