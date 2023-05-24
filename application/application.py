import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load


class ContextualPrediction(BaseModel):
    day: int
    raw_price: float


app = FastAPI()
mdl = load("assets/models/model.joblib")


@app.get("/")
async def root():
    return {"message": "Contextual Bandits"}


@app.post("/predict/")
async def predict(UserInput: ContextualPrediction):
    X = pd.json_normalize(UserInput.__dict__)
    prediction = mdl.predict(X)

    return {"prediction": int(prediction)}
