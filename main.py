from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Load your trained model
with open("congestion_model.pkl", "rb") as f:
    model = pickle.load(f)




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
    input_features = np.array([[
        data.is_holiday, data.is_weekend, data.Friday, data.Monday, data.Saturday, data.Sunday, data.Thursday,
        data.Tuesday, data.Wednesday, data.Cloudy,
        data.Rainy, data.Sunny, data.Windy, data._10_00_10_30, data._10_30_11_00, data._11_00_11_30, data._11_30_12_00,
        data._12_00_12_30, data._12_30_13_00, data._13_00_13_30, data._13_30_14_00, data._14_00_14_30,
        data._14_30_15_00,
        data._15_00_15_30, data._15_30_16_00, data._16_00_16_30, data._16_30_17_00, data._17_00_17_30,
        data._17_30_18_00,
        data._18_00_18_30, data._18_30_19_00, data._19_00_19_30, data._19_30_20_00, data._8_00_8_30, data._8_30_9_00,
        data._9_00_9_30, data._9_30_10_00
    ]])

    # Predict
    prediction = model.predict(input_features)
    return {"congestion_percentage": float(prediction[0])}


