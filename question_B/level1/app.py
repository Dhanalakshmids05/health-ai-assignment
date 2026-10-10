import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "level2")
)

from database import create_tables, save_prediction, get_prediction_stats
create_tables()

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Health Risk Prediction API",
    description="API for predicting health risk using a machine learning model",
    version="1.0.0"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "health_risk_model.joblib"

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
imputer = model_data["imputer"]
scaler = model_data["scaler"]
feature_names = model_data["feature_names"]


# ============================================================
# INPUT DATA MODEL
# ============================================================


class PatientData(BaseModel):
    age: float = Field(gt=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0)
    chol: float = Field(gt=0)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0)
    slope: int = Field(ge=1, le=3)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(ge=3, le=7)




# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Health Risk Prediction API is running"
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    input_data = pd.DataFrame(
        [[
            patient.age,
            patient.sex,
            patient.cp,
            patient.trestbps,
            patient.chol,
            patient.fbs,
            patient.restecg,
            patient.thalach,
            patient.exang,
            patient.oldpeak,
            patient.slope,
            patient.ca,
            patient.thal
        ]],
        columns=feature_names
    )

    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)

    prediction = int(model.predict(input_data)[0])

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    # Save the result in SQLite
    record_id = save_prediction(
        prediction,
        probability
    )

    return {
        "id": record_id,
        "prediction": prediction,
        "risk_probability": probability,
        "message": "Prediction saved successfully"
    }
    
@app.get("/stats")
def stats():
    return get_prediction_stats()