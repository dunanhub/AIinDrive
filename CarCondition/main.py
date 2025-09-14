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
    
    # Основные поля (совместимость)
    dirty: bool
    dirty_prob: float
    damaged: bool
    damaged_prob: float
    
    # Классификация повреждений
    predicted_class: Optional[str] = None         
    confidence: Optional[float] = None            
    probabilities: Optional[Dict[str, float]] = None  
    
    # Продвинутый анализ загрязнения v2.0
    dirt_status: Optional[str] = None             # статус загрязненности
    dirt_emoji: Optional[str] = None              # эмодзи загрязненности
    dirt_score: Optional[float] = None            # оценка загрязнения 0-10
    dirt_metrics: Optional[Dict] = None           # детальные метрики грязи
    dirt_recommendation: Optional[str] = None     # рекомендация по мойке
    
    # Качество анализа
    quality_metrics: Optional[Dict] = None        # метрики качества изображения
    
    # Экспертная оценка v2.0
    expert_assessment: Optional[List[str]] = None # экспертное заключение
    
    # Оценка для такси
    taxi_status: Optional[str] = None             # статус пригодности для такси
    taxi_recommendations: Optional[List[str]] = None  # рекомендации для такси
    economic_info: Optional[Dict] = None          # экономическая оценка
    
    # Технические поля
    model_available: Optional[bool] = None        
    analysis_version: Optional[str] = None        # версия анализа
    cv_available: Optional[bool] = None           # доступность OpenCV
    
    # Обработка ошибок
    error: Optional[str] = None                   
    message: Optional[str] = None                 
    
    # Совместимость со старыми версиями
    parts: Optional[Dict[str, int]] = None        
    expert_recommendations: Optional[List[str]] = None  # deprecated, используйте expert_assessment
    
    # Debug информация (только для /predict-debug)
    debug: Optional[dict] = None

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
