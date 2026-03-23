from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

try:
    model = joblib.load('models/final_model.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

pipeline = model['model']
threshold = model['threshold']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    return {"message": "Welcome to the Customer Churn Prediction API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])
    try:
        proba = pipeline.predict_proba(df)[:, 1][0]
    except Exception as e:
        return {"error": str(e)}

    return {"churn_probability": float(proba), 
            "churn_prediction": int(proba >= threshold)}