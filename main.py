from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load your trained model
with open("congestion_model.pkl", "rb") as f:
    model = pickle.load(f)

# List all expected input features in order
input_features = [
    'is_holiday', 'is_weekend', 'Friday', 'Monday', 'Saturday', 'Sunday',
    'Thursday', 'Tuesday', 'Wednesday',
    'Cloudy', 'Rainy', 'Sunny', 'Windy',
    '10_00_10_30', '10_30_11_00', '11_00_11_30', '11_30_12_00',
    '12_00_12_30', '12_30_13_00', '13_00_13_30', '13_30_14_00',
    '14_00_14_30', '14_30_15_00', '15_00_15_30', '15_30_16_00',
    '16_00_16_30', '16_30_17_00', '17_00_17_30', '17_30_18_00',
    '18_00_18_30', '18_30_19_00', '19_00_19_30', '19_30_20_00',
    '8_00_8_30', '8_30_9_00', '9_00_9_30', '9_30_10_00'
]


# Input schema for FastAPI
class InputData(BaseModel):
    is_holiday: int
    is_weekend: int
    Friday: int
    Monday: int
    Saturday: int
    Sunday: int
    Thursday: int
    Tuesday: int
    Wednesday: int
    Cloudy: int
    Rainy: int
    Sunny: int
    Windy: int
    _10_00_10_30: int
    _10_30_11_00: int
    _11_00_11_30: int
    _11_30_12_00: int
    _12_00_12_30: int
    _12_30_13_00: int
    _13_00_13_30: int
    _13_30_14_00: int
    _14_00_14_30: int
    _14_30_15_00: int
    _15_00_15_30: int
    _15_30_16_00: int
    _16_00_16_30: int
    _16_30_17_00: int
    _17_00_17_30: int
    _17_30_18_00: int
    _18_00_18_30: int
    _18_30_19_00: int
    _19_00_19_30: int
    _19_30_20_00: int
    _8_00_8_30: int
    _8_30_9_00: int
    _9_00_9_30: int
    _9_30_10_00: int


# Initialize FastAPI
app = FastAPI()

# Enable CORS for all origins (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Home route
@app.get("/")
def read_root():
    return {"message": "Congestion Prediction API is running 🚦"}


# Prediction route
@app.post("/predict")
def predict_congestion(data: InputData):
    # Convert input to dictionary and fix time slot keys
    input_dict = data.dict()

    # Fix keys to match original feature names
    formatted_input = {
        k.replace("_", ":", 1).replace("_", "-") if k[0] == "_" else k.replace("_", "-") for k in input_dict
    }

    # Align features in expected order
    values = [input_dict.get(feat, 0) for feat in input_features]

    # Convert to numpy array and predict
    prediction = model.predict([values])[0]

    return {"congestion_percentage": round(prediction, 2)}
