from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

pipeline = joblib.load("models/final_pipeline.pkl")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    PaperlessBilling: str
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    InternetService: str
    Contract: str
    PaymentMethod: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])
    try:
        prediction = pipeline.predict(df)
        probability = pipeline.predict_proba(df)[:, 1]
        return {"prediction": int(prediction[0]), 
                "probability": float(probability[0])}
    except Exception as e:
        return {"error": str(e)}