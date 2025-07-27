from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from app.preprocessing import extract_and_clean_city_features  
from sklearn.preprocessing import FunctionTransformer
from prometheus_fastapi_instrumentator import Instrumentator
import os

# Load pipeline
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'catboost_pipeline_v1.pkl')

try:
    with open(model_path, "rb") as f:
        model_pipeline = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# Define FastAPI app
app = FastAPI(
    title="Delivery Time Prediction API",
    version="1.0",
    description="Predicts delivery time using a trained CatBoostRegressor pipeline"
)

# Define request schema
class DeliveryInput(BaseModel):
    source_name: str
    destination_name: str
    route_type: str
    start_scan_to_end_scan: float
    cutoff_factor: float
    actual_distance_to_destination: float
    osrm_distance: float
    segment_actual_time: float
    segment_osrm_distance: float
    segment_factor: float

# Define response schema
#class PredictionResponse(BaseModel):
#    predicted_delivery_time: float

@app.get("/health")
def health_check():
    return{"status":"ok", "model_loaded":model_pipeline is not None}

@app.post("/predict")
def predict(input_data:DeliveryInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        prediction =model_pipeline.predict(df)
        return {"predicted_actual_time":prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add Prometheus Instrumentation
Instrumentator().instrument(app).expose(app)

