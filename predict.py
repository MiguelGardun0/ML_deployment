import joblib 
import pandas as pd 
from typing import Dict, Any
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="predict")
pipeline = joblib.load("model.pkl")


def predict_single(customer):
    X_new = pd.DataFrame([customer])
    result = round(pipeline.predict_proba(X_new)[0,1], 3)
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    result = predict_single(customer)
    return{
        'churn_prob': result,
        'churn': bool(result >= 0.5)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)