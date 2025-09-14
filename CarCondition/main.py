# main.py
import os
import shutil
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from inference import analyze_image, debug_analyze_image

app = FastAPI(title="AI-inDrive Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class Prediction(BaseModel):
    filename: str
    dirty: bool
    dirty_prob: float
    damaged: bool
    damaged_prob: float
    predicted_class: Optional[str] = None         # новое: класс повреждения
    confidence: Optional[float] = None            # новое: уверенность модели
    probabilities: Optional[Dict[str, float]] = None  # новое: вероятности классов
    dirt_metrics: Optional[Dict[str, float]] = None   # новое: метрики грязи
    dirt_status: Optional[str] = None             # новое: статус загрязненности
    dirt_emoji: Optional[str] = None              # новое: эмодзи загрязненности
    model_available: Optional[bool] = None        # новое: доступность модели
    expert_recommendations: Optional[List[str]] = None  # новое: экспертные рекомендации
    error: Optional[str] = None                   # новое: код ошибки
    message: Optional[str] = None                 # новое: сообщение об ошибке
    parts: Optional[Dict[str, int]] = None        # старое: для совместимости
    debug: Optional[dict] = None                  # только для /predict-debug

class PredictResponse(BaseModel):
    results: List[Prediction]

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
async def predict(files: List[UploadFile] = File(...)):
    out = []
    for f in files:
        try:
            data = await f.read()
            p = analyze_image(data)                   # из inference.py (новая версия с экспертным анализом)
            out.append(Prediction(filename=f.filename, **p))
        except Exception as e:
            # В случае ошибки возвращаем базовую структуру с ошибкой
            error_result = {
                "dirty": False,
                "dirty_prob": 0.0,
                "damaged": False,
                "damaged_prob": 0.0,
                "error": "processing_error",
                "message": f"Ошибка при обработке {f.filename}: {str(e)}",
                "model_available": False
            }
            out.append(Prediction(filename=f.filename, **error_result))
    return {"results": out}

@app.post("/predict-debug", response_model=PredictResponse)
async def predict_debug(files: List[UploadFile] = File(...)):
    out = []
    for f in files:
        try:
            data = await f.read()
            p = debug_analyze_image(data)             # расширенный ответ с debug информацией
            out.append(Prediction(filename=f.filename, **p))
        except Exception as e:
            # В случае ошибки возвращаем базовую структуру с ошибкой и debug информацией
            error_result = {
                "dirty": False,
                "dirty_prob": 0.0,
                "damaged": False,
                "damaged_prob": 0.0,
                "error": "processing_error",
                "message": f"Ошибка при обработке {f.filename}: {str(e)}",
                "model_available": False,
                "debug": {"error": str(e), "filename": f.filename}
            }
            out.append(Prediction(filename=f.filename, **error_result))
    return {"results": out}
