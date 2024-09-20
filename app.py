from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import joblib

# Load the trained SVM model and scaler
svm_model = joblib.load('./svm_model.joblib')
scaler = joblib.load('./scaler.joblib')

# Create the FastAPI app
app = FastAPI()

# Define the request body model using Pydantic
class WaveformData(BaseModel):
    data: list

# Function to calculate statistical features for each input waveform
def calculate_stats(data):
    stats = {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'skewness': skew(data),
        'kurtosis': kurtosis(data)
    }
    return stats

# Define the POST endpoint
@app.post("/predict/")
async def predict(waveform: WaveformData):
    try:
        # Extract data and compute statistical features
        waveform_data = np.array(waveform.data)
        stats = calculate_stats(waveform_data)
        
        # Convert to DataFrame for consistency with training process
        stats_df = pd.DataFrame([stats])
        
        # Standardize features
        X_test_scaled = scaler.transform(stats_df)
        
        # Make prediction
        prediction = svm_model.predict(X_test_scaled)
        
        # Return the prediction as a response
        return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# To run the server, use the following command
# uvicorn filename:app --host 0.0.0.0 --port 8000 --reload
