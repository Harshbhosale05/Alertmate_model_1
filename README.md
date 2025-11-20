# Vitals Inference API

This repository contains a minimal FastAPI service that loads ML artifacts and exposes a `/predict_vitals` endpoint for inference.

## Repository structure

```
/
├─ models/
│  ├─ lstm_autoencoder_model.pth
│  ├─ xgboost_classifier_model.pkl
│  ├─ one_hot_encoder.pkl
│  └─ standard_scaler.pkl
├─ main.py
├─ requirements.txt
├─ Dockerfile
├─ .gitignore
└─ README.md
```

## Quick start (local)

1. Create virtualenv and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run with Uvicorn (development):

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

3. Health check:

```powershell
curl -X GET "http://127.0.0.1:8000/health"
```

4. Example POST (ensure `sample.json` matches the `expected_order` keys):

```powershell
curl -X POST "http://127.0.0.1:8000/predict_vitals" -H "Content-Type: application/json" -d @sample.json
```

## Docker (build & run)

```powershell
docker build -t vitals-inference:latest .
docker run -p 8000:10000 -e PORT=10000 vitals-inference:latest
```

## Notes
- Ensure `models/` contains the files listed above (or set `MODEL_DIR` env var to point to your model folder).
- Adjust `expected_order` inside `main.py` to match how `standard_scaler.pkl` was trained.
- If model files are large, consider using Git LFS or storing models in cloud storage and downloading at startup.

## Deploy to Render

1. Push this repo to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repo
4. Choose Python environment or Docker
5. Set start command: `gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
6. Deploy and test the public endpoint
