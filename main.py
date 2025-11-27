import os
import joblib
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

# -------- CONFIG --------
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

# -------- LOAD ARTIFACTS --------
# XGBoost
xgb_model = None
try:
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_classifier_model.pkl"))
except Exception:
    xgb_model = None


# Preprocessors
import logging
logging.basicConfig(level=logging.INFO)
scaler = None
ohe = None
try:
    scaler_path = os.path.join(MODEL_DIR, "standard_scaler.pkl")
    logging.info(f"Trying to load scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load scaler: {e}")
    scaler = None

try:
    ohe_path = os.path.join(MODEL_DIR, "one_hot_encoder.pkl")
    logging.info(f"Trying to load one-hot encoder from: {ohe_path}")
    ohe = joblib.load(ohe_path)
    logging.info("One-hot encoder loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load one-hot encoder: {e}")
    ohe = None

# LSTM Autoencoder (PyTorch)
ae_path = os.path.join(MODEL_DIR, "lstm_autoencoder_model.pth")
if os.path.exists(ae_path):
    try:
        ae_model = torch.load(ae_path, map_location="cpu")
        ae_model.eval()
    except Exception:
        ae_model = None
else:
    ae_model = None

app = FastAPI(title="Vitals Inference API")

class FeaturesPayload(BaseModel):
    user_id: str
    age: int
    gender: str
    weight: float
    height: float
    resting_hr: float
    resting_spo2: float
    # features dict should follow the exact order your scaler expects
    features: Dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict_vitals")
def predict(payload: FeaturesPayload):
    # 1) prepare feature vector in consistent order
    # IMPORTANT: ensure the feature order matches the scaler used during training.
    expected_order = [
        "hr_mean","hr_std","hr_min","hr_max","hr_slope","rmssd","sdnn",
        "spo2_mean","spo2_std","spo2_min","spo2_max","spo2_slope","spo2_drop",
        "motion_mean","motion_std","percent_high_motion","sqi"
    ]
    try:
        feat_vec = np.array([payload.features[f] for f in expected_order], dtype=float).reshape(1, -1)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    # 2) scale
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler not available on server")
    feat_scaled = scaler.transform(feat_vec)

    # 3) classifier
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="XGBoost model not available on server")
    prob = float(xgb_model.predict_proba(feat_scaled)[0,1])
    pred = int(prob >= 0.5)

    # 4) autoencoder score (optional) — depends on how you trained AE
    ae_score = None
    if ae_model is not None:
        # if AE expects sequences, convert/reshape accordingly; below assumes AE was trained on feature vectors
        with torch.no_grad():
            x = torch.tensor(feat_scaled, dtype=torch.float32)
            recon = ae_model(x)  # adapt if your AE API differs
            ae_score = float(((x - recon)**2).mean().item())

    # 5) combine logic if you saved weights (optional)
    # If you have weight files, load them similarly and combine; otherwise return both scores
    return {
        "xgb_probability": prob,
        "xgb_prediction": int(pred),
        "ae_reconstruction_error": ae_score,
        "final_prediction": int(pred),
        "alert_message": "Alert" if pred==1 else "Normal"
    }


# Note: adjust expected_order to exactly match how you trained standard_scaler.pkl.
# If you used one-hot encoding for categorical vars, apply ohe.transform(...) first and then concatenate.
