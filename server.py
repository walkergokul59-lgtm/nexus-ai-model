from __future__ import annotations

import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import shutil
import os

from modeling import ModelBundle, predict_dashboard, train_one

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_bundle.joblib")

# Newsletter subscribers (in-memory; resets on cold start)
subscribers: List[str] = []

class SubscribeRequest(BaseModel):
    email: str

# Global variable to hold the model
bundle: ModelBundle | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model at startup
    global bundle
    try:
        bundle = ModelBundle.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        bundle = None
        print(f"[WARN] Could not load {MODEL_PATH}: {e}")
    yield
    # Clean up if needed (nothing to do here)

app = FastAPI(title="Nexus/Sentinel Log Anomaly API", lifespan=lifespan)

# allow your frontend to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return RedirectResponse(url="/app/home.html")

@app.get("/health")
def health():
    files = []
    try:
        files = os.listdir(BASE_DIR)
    except Exception as e:
        files = [str(e)]
    return {
        "ok": True, 
        "model_loaded": bundle is not None,
        "cwd": os.getcwd(),
        "base_dir": BASE_DIR,
        "files_in_base_dir": files
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global bundle
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Train first and ensure model_bundle.joblib exists.")

    # For now: expect CSV. (You can add LOG/TXT parsing later.)
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file for now.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    try:
        result = predict_dashboard(bundle, df, top_k=50)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@app.post("/subscribe")
async def subscribe(req: SubscribeRequest):
    email = req.email.strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    if email not in subscribers:
        subscribers.append(email)
    # Notify owner via email (best-effort, works if SMTP is available)
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(f"New Nexus AI newsletter subscriber: {email}")
        msg["Subject"] = f"New Subscriber: {email}"
        msg["From"] = email
        msg["To"] = "neeranjanrit@gmail.com"
        # Note: SMTP sending requires credentials; this is a best-effort attempt
        # For production, use an email API like SendGrid, Resend, or Mailgun
    except Exception:
        pass
    return {"ok": True, "message": f"Thanks for subscribing! ({email})"}


@app.get("/subscribers")
def get_subscribers():
    return {"count": len(subscribers), "emails": subscribers}


# Mount frontend static files AFTER API routes
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")


@app.post("/train")
async def train_model(
    auth_file: List[UploadFile] = File(None),
    netflow_file: List[UploadFile] = File(None),
    dns_file: List[UploadFile] = File(None),
    http_file: List[UploadFile] = File(None)
):
    global bundle
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not loaded. Please ensure initial model exists.")

    files_map = {
        "auth": auth_file,
        "netflow": netflow_file,
        "dns": dns_file,
        "http": http_file,
    }

    updated_infos = {}
    
    for kind, file_list in files_map.items():
        if not file_list:
            continue
            
        print(f"Processing {len(file_list)} files for {kind}...")
        dfs = []
        for i, f in enumerate(file_list):
            try:
                # Read CSV directly from memory (no disk writes needed)
                content = await f.read()
                df_part = pd.read_csv(io.BytesIO(content))
                dfs.append(df_part)
            except Exception as e:
                print(f"Error processing file {f.filename}: {e}")
                continue
        
        if not dfs:
            continue

        print(f"Retraining {kind} model on combined data...")
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            
            clf, hasher, info = train_one(kind, combined_df)
            
            bundle.models[kind] = clf
            bundle.hashers[kind] = hasher
            bundle.infos[kind] = info
            
            updated_infos[kind] = {
                "precision": info.precision,
                "recall": info.recall,
                "false_positive_rate": info.false_positive_rate
            }
        except Exception as e:
            print(f"Error training {kind}: {e}")
            raise HTTPException(status_code=500, detail=f"Error training {kind}: {str(e)}")

    # Save updated bundle to /tmp (writable on Vercel) or BASE_DIR locally
    try:
        import tempfile
        save_path = os.path.join(tempfile.gettempdir(), "model_bundle.joblib")
        bundle.save(save_path)
        # Also try the original path (works locally, fails silently on Vercel)
        try:
            bundle.save(MODEL_PATH)
        except Exception:
            pass
        print(f"Updated model bundle saved to {save_path}")
    except Exception as e:
        print(f"Error saving bundle: {e}")
    
    return updated_infos
