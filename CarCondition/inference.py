# CarCondition/inference.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import io, os
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from dotenv import load_dotenv

# архитектура из твоего файла
from multiclass_damage_model import MulticlassDamageModel  # <-- теперь есть

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/finetuned_best_model.pth")
DEVICE     = os.getenv("DEVICE", "cpu")

# === предобработка ровно как в твоём тесте ===
_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ---- твой анализ грязи (укороченная версия для инференса) ----
def analyze_dirt_level(image: Image.Image) -> Tuple[float, Dict[str, float]]:
    """
    Анализ загрязненности с детальными метриками.
    Возвращает (dirt_prob [0..1], метрики).
    """
    import numpy as np
    from PIL import ImageFilter, ImageStat

    img_array = np.array(image)
    
    # Цветовое разнообразие
    unique_colors_r = len(np.unique(img_array[:,:,0]))
    unique_colors_g = len(np.unique(img_array[:,:,1])) 
    unique_colors_b = len(np.unique(img_array[:,:,2]))
    color_diversity = (unique_colors_r + unique_colors_g + unique_colors_b) / 3
    
    # Контраст
    gray = image.convert('L')
    contrast = ImageStat.Stat(gray).stddev[0]
    
    # Насыщенность
    hsv = image.convert('HSV')
    hsv_array = np.array(hsv)
    saturation = float(np.mean(hsv_array[:,:,1]))
    
    # Коричневые оттенки
    brown_mask = (
        (img_array[:,:,0] > img_array[:,:,2]) &
        (img_array[:,:,1] > img_array[:,:,2]) &
        (img_array[:,:,0] < 150) &
        (img_array[:,:,1] < 120)
    )
    brown_ratio = float(np.sum(brown_mask)) / (img_array.shape[0] * img_array.shape[1])
    
    # Четкость краев
    edge_image = gray.filter(ImageFilter.FIND_EDGES)
    edge_intensity = float(np.mean(np.array(edge_image)))
    
    # Яркость
    brightness = float(np.mean(img_array))
    
    # Подсчет итогового индекса грязи
    dirt_score = 0.0
    
    if color_diversity < 80:
        dirt_score += 2
    elif color_diversity < 120:
        dirt_score += 1
    
    if contrast < 25:
        dirt_score += 2
    elif contrast < 40:
        dirt_score += 1
    
    if saturation < 60:
        dirt_score += 1.5
    elif saturation < 100:
        dirt_score += 0.5
    
    if brown_ratio > 0.15:
        dirt_score += 2
    elif brown_ratio > 0.08:
        dirt_score += 1
    
    if edge_intensity < 15:
        dirt_score += 1.5
    elif edge_intensity < 25:
        dirt_score += 0.5
    
    if brightness < 90:
        dirt_score += 1
    elif brightness < 110:
        dirt_score += 0.5
    
    # Нормализуем в диапазон [0..1] (максимум по шкале ≈ 10)
    dirt_prob = float(np.clip(dirt_score / 10.0, 0.0, 1.0))

    metrics = {
        'color_diversity': float(color_diversity),
        'contrast': float(contrast),
        'saturation': float(saturation),
        'brown_ratio': float(brown_ratio),
        'edge_intensity': float(edge_intensity),
        'brightness': float(brightness),
        'dirt_score': float(dirt_score)
    }
    
    return dirt_prob, metrics

@lru_cache(maxsize=1)
def get_model() -> Optional[torch.nn.Module]:
    """Загружает модель из checkpoint с поддержкой разных форматов"""
    p = Path(MODEL_PATH)
    if not p.exists():
        print(f"[inference] MODEL_PATH '{p}' not found")
        return None
    
    device = torch.device(DEVICE)
    model = MulticlassDamageModel(num_classes=3)
    
    try:
        # Загружаем checkpoint
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
        print(f"[inference] Loaded checkpoint type: {type(ckpt)}")
        
        # Определяем формат checkpoint и извлекаем state_dict
        state_dict = None
        
        if isinstance(ckpt, dict):
            # Вариант 1: model_state_dict (формат из экспертного скрипта)
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                print(f"[inference] Found model_state_dict format")
                if "epoch" in ckpt:
                    print(f"[inference] Model epoch: {ckpt['epoch']}")
                if "f1_score" in ckpt:
                    print(f"[inference] Model F1-score: {ckpt['f1_score']:.4f}")
            
            # Вариант 2: state_dict
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                print(f"[inference] Found state_dict format")
            
            # Вариант 3: прямой state_dict на верхнем уровне
            else:
                # Проверяем, есть ли ключи модели на верхнем уровне
                model_keys = list(model.state_dict().keys())
                if any(key in ckpt for key in model_keys[:3]):  # проверяем первые 3 ключа
                    state_dict = ckpt
                    print(f"[inference] Found direct state_dict format")
                else:
                    print(f"[inference] Unknown checkpoint format. Keys: {list(ckpt.keys())}")
                    return None
        else:
            # Вариант 4: сам checkpoint является state_dict
            state_dict = ckpt
            print(f"[inference] Treating checkpoint as direct state_dict")
        
        if state_dict is None:
            print(f"[inference] Could not extract state_dict from checkpoint")
            return None
        
        # Загружаем веса
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"[inference] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[inference] Unexpected keys: {unexpected_keys}")
        
        print(f"[inference] Model loaded successfully (missing: {len(missing_keys)}, unexpected: {len(unexpected_keys)})")
        
        model.eval().to(device)
        return model
        
    except Exception as e:
        print(f"[inference] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def _damage_from_logits(logits: torch.Tensor) -> Dict[str, float]:
    """
    Анализирует logits модели и возвращает подробную информацию о повреждениях.
    logits -> softmax(prob_no, prob_minor, prob_major)
    """
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    prob_no, prob_minor, prob_major = float(probs[0]), float(probs[1]), float(probs[2])
    
    # Основная логика: damaged_prob = 1 - prob_no
    damaged_prob = 1.0 - prob_no
    
    # Определяем класс и уверенность
    class_names = ['no_damage', 'minor_damage', 'major_damage']
    predicted_idx = int(np.argmax(probs))
    predicted_class = class_names[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    return {
        'damaged_prob': round(damaged_prob, 4),
        'damaged': damaged_prob >= 0.5,
        'predicted_class': predicted_class,
        'confidence': round(confidence, 4),
        'probabilities': {
            'no_damage': round(prob_no, 4),
            'minor_damage': round(prob_minor, 4),
            'major_damage': round(prob_major, 4)
        }
    }

def analyze_image(img_bytes: bytes) -> Dict[str, float]:
    """
    Главный вход: возвращает поля для фронта с подробными данными модели.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)

    # --- грязь ---
    dirt_prob, dirt_metrics = analyze_dirt_level(img)

    # --- damage через твою модель ---
    model = get_model()
    if model is None:
        # если модели нет — отдаём только грязь
        return {
            "dirty": dirt_prob >= 0.5,
            "dirty_prob": round(float(dirt_prob), 4),
            "damaged": False,
            "damaged_prob": 0.0,
            "predicted_class": "unknown",
            "confidence": 0.0,
            "probabilities": {
                "no_damage": 0.0,
                "minor_damage": 0.0,
                "major_damage": 0.0
            },
            "dirt_metrics": dirt_metrics,
            "model_available": False
        }

    x = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(x)
    
    # Получаем подробную информацию о повреждениях
    damage_info = _damage_from_logits(logits)

    return {
        "dirty": dirt_prob >= 0.5,
        "dirty_prob": round(float(dirt_prob), 4),
        "damaged": damage_info["damaged"],
        "damaged_prob": damage_info["damaged_prob"],
        "predicted_class": damage_info["predicted_class"],
        "confidence": damage_info["confidence"],
        "probabilities": damage_info["probabilities"],
        "dirt_metrics": dirt_metrics,
        "model_available": True
    }

# удобный локальный тест
def debug_analyze_image(img_bytes: bytes) -> Dict[str, float]:
    """
    Расширенная версия analyze_image с дополнительной отладочной информацией.
    """
    result = analyze_image(img_bytes)
    
    # Добавляем дополнительную отладочную информацию
    result["debug"] = {
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "model_loaded": result.get("model_available", False)
    }
    
    return result
