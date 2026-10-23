
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Health Insurance Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_Health_insurance.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


class InsuranceData(BaseModel):
    name: str
    age: int = Field(gt=0, le=100)
    gender: str
    bmi: float = Field(gt=0)
    no_of_children: int = Field(ge=0)
    isSmoker: bool


def preprocess_data(data: InsuranceData):
    gender = data.gender.strip().lower()

    if gender not in ["male", "female"]:
        raise HTTPException(
            status_code=422,
            detail="Gender must be 'male' or 'female'."
        )

    # Encoding must match the model's training data
    raw_data = pd.DataFrame([{
        "age": data.age,
        "sex": 0 if gender == "male" else 1,
        "bmi": data.bmi,
        "children": data.no_of_children,
        "smoker": int(data.isSmoker)
    }])

    return raw_data


@app.get("/")
def home():
    return {
        "message": "Health Insurance Prediction API is running",
        "docs": "/docs",
        "prediction_endpoint": "/predict"
    }


@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/predict")
def predict(data: InsuranceData):
    processed_data = preprocess_data(data)

    try:
        prediction = float(model.predict(processed_data)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    return {
        "name": data.name,
        "predicted_charges": round(prediction, 2)
    }
