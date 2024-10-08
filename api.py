from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import nest_asyncio
import uvicorn

# Load your trained RandomForest model (assuming you saved it previously)
# Use joblib to load the model
model = joblib.load('random_forest_model.pkl')  # Make sure to save your trained model

# Define the FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mappings for encoded columns
mappings = {
    "Service Type": {
        'FedEx 2Day': 0, 'FedEx Priority Overnight': 1, 'FedEx Standard Overnight': 2, 'Ground': 3,
        'Ground - Return': 4, 'Home Delivery': 5, 'SmartPost': 6
    },
    "Tracking ID Charge Description 1": {
        'AHS - Weight': 0, 'Additional Handling': 1, 'Address Correction': 2, 'DAS Alaska Resi': 3,
        'DAS Comm': 4, 'DAS Extended Comm': 5, 'DAS Extended Resi': 6, 'DAS Remote Resi': 7, 'DAS Resi': 8,
        'Delivery Area Surcharge': 9, 'Delivery Area Surcharge Alaska': 10, 'Delivery Area Surcharge Extended': 11,
        'Delivery and Returns': 12, 'Discount': 13, 'Earned Discount': 14, 'Fuel Surcharge': 15,
        'Peak - AHS Charge': 16, 'Performance Pricing': 17, 'Print Return Label': 18, 'Residential': 19,
        'Residential Delivery': 20, 'Return Email Label': 21, 'Return Pickup Fee': 22, 'Saturday Delivery': 23,
        'USPS Non-Mach Surcharge': 24, 'Weekday Delivery': 25
    },
    "Tracking ID Charge Description 3": {
        'AHS - Weight': 0, 'Additional Handling': 1, 'Additional Handling Charge - Package': 2, 'Address Correction': 3,
        'Adult Signature': 4, 'DAS Alaska Resi': 5, 'DAS Comm': 6, 'DAS Extended Comm': 7, 'DAS Extended Resi': 8,
        'DAS Hawaii Resi': 9, 'DAS Remote Resi': 10, 'DAS Resi': 11, 'Delivery Area Surcharge': 12,
        'Delivery Area Surcharge Alaska': 13, 'Delivery Area Surcharge Extended': 14, 'Delivery and Returns': 15,
        'Discount': 16, 'Earned Discount': 17, 'Fuel Surcharge': 18, 'Peak - AHS Charge': 19, 'Performance Pricing': 20,
        'Print Return Label': 21, 'Residential': 22, 'Residential Delivery': 23, 'Return Email Label': 24,
        'Return Pickup Fee': 25, 'Saturday Delivery': 26, 'USPS Non-Mach Surcharge': 27, 'Weekday Delivery': 28
    },
    "Tracking ID Charge Description 4": {
        'Additional Handling': 0, 'Additional Handling Charge - Package': 1, 'Address Correction': 2, 'DAS Comm': 3,
        'DAS Extended Resi': 4, 'DAS Remote Resi': 5, 'DAS Resi': 6, 'Delivery Area Surcharge': 7,
        'Delivery Area Surcharge Alaska': 8, 'Delivery Area Surcharge Extended': 9, 'Delivery Area Surcharge Hawaii': 10,
        'Delivery and Returns': 11, 'Discount': 12, 'Earned Discount': 13, 'Fuel Surcharge': 14, 'Peak - AHS Charge': 15,
        'Performance Pricing': 16, 'Print Return Label': 17, 'Residential': 18, 'Residential Delivery': 19,
        'Return Email Label': 20, 'Return Pickup Fee': 21, 'Saturday Delivery': 22, 'USPS Non-Mach Surcharge': 23,
        'Weekday Delivery': 24
    }
}

class ModelInput(BaseModel):
    service_type: int
    actual_weight_amount: float
    dim_length: float
    dim_width: float
    dim_height: float
    tracking_id_charge_description_1: int
    tracking_id_charge_description_3: int
    tracking_id_charge_description_4: int
    distance: float

# Define the root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI model prediction API. Use /predict/ to make predictions or /mappings/ to see mappings."}

# Define the prediction endpoint
@app.post("/predict/")
async def predict(input_data: ModelInput):
    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data.dict().values()], columns=input_data.dict().keys())

    # Rename the columns to match the feature names used during model training
    input_df.columns = [
        'Service Type', 
        'Actual Weight Amount', 
        'Dim Length', 
        'Dim Width', 
        'Dim Height', 
        'Tracking ID Charge Description 1', 
        'Tracking ID Charge Description 3', 
        'Tracking ID Charge Description 4', 
        'Distance'
    ]

    # Predict the result using the trained model
    prediction = model.predict(input_df)

    return {"prediction": prediction[0]}

# Endpoint to get the mappings
@app.get("/mappings/")
async def get_mappings():
    return mappings

# Apply the patch for asyncio to work in Jupyter
nest_asyncio.apply()

# Now run the FastAPI app using uvicorn within the notebook
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)