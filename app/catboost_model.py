import pandas as pd
import numpy as np
import json
import pickle
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from app.preprocessing import extract_and_clean_city_features
import os

# Load and preprocess data
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'delhivery.csv')
df = pd.read_csv(csv_path)

city_cleaner = FunctionTransformer(extract_and_clean_city_features)

# preprocessing
numerical_features = ['start_scan_to_end_scan', 'cutoff_factor', 'actual_distance_to_destination', 'osrm_distance', 'segment_actual_time', 'segment_osrm_distance', 'segment_factor']
categorical_features = ['source_city', 'destination_city','route_type']

X = df.drop(['actual_time', 'data', 'route_schedule_uuid', 'trip_uuid',
             'source_center', 'destination_center', 'cutoff_timestamp', 'osrm_time', 
             'segment_osrm_time','factor'], axis=1)

y = df['actual_time']

# Preprocessing pipeline
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Create full pipeline: Preprocessing + CatBoost
full_pipeline = Pipeline(steps=[
    ("city_cleaning", city_cleaner),            # Step 1: Feature Engineering
    ('preprocessing', preprocessor),            # Step 2: Transform
    ('model', CatBoostRegressor(verbose=0))     # Step 3: Model
])

# Train CatBoost model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train pipeline
full_pipeline.fit(X_train, y_train)

# Save the entire pipeline (preprocessing + model)
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'catboost_pipeline_v1.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(full_pipeline, f)

# Save version metadata
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'model_metadata.json')
metadata = {
    "model_name": "catboost_delivery_time_predictor",
    "version": "1.0",
    "model_file": "catboost_pipeline_v1.pkl",
    "date_trained": "2025-07-24"
}
with open(config_path, "w") as f:
    json.dump(metadata, f, indent=4)
