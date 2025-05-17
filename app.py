from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import os
import numpy as np
import pandas as pd
from typing import Optional

from src.pipeline.predict_pipeline import PredictPipeline,EditData

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get('/')
def index(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.get('/predictdata')
def predict_data(request:Request, make: Optional[str]=None):
    edit = EditData()
    predict_pipeline=PredictPipeline()
    print("Mid Prediction")
    data_path = os.path.join("src/data", "inputdata.csv")
    df = edit.changes(data_path)
    results=predict_pipeline.predict(df)
    print("after Prediction")
    if make:
        results = [car for car in results if car.get("make", "").lower() == make.lower()]

    available_makes = sorted({car.get("make", "") for car in results})
    return templates.TemplateResponse("home.html", {"request": request, "cars": results, "available_makes": available_makes,
        "selected_make": make}) 