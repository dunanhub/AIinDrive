# CarCondition/inference.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import io, os
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageStat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from dotenv import load_dotenv

# архитектура из твоего файла
from multiclass_damage_model import MulticlassDamageModel

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pth")
DEVICE     = os.getenv("DEVICE", "cpu")

# === предобработка ровно как в твоём тесте ===
_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# === Валидация автомобиля ===
# Load a pre-trained model for vehicle detection (e.g., ResNet)
vehicle_detection_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
vehicle_detection_model.eval()

# Define a transformation pipeline for the input image
vehicle_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def is_vehicle(image: Image.Image) -> bool:
    """Check if the image contains a vehicle."""
    # Apply transformations
    input_tensor = vehicle_transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = vehicle_detection_model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Define vehicle-related classes (example: ImageNet class IDs for vehicles)
    vehicle_classes = list(range(817, 829))  # Example range for vehicle classes in ImageNet

    # Check if the top prediction corresponds to a vehicle
    top_class = torch.argmax(probabilities).item()
    if top_class in vehicle_classes:
        return True
    return False

def analyze_dirt_level(image: Image.Image) -> Tuple[float, Dict[str, float], str, str]:
    """
    Анализ загрязненности с детальными метриками.
    Возвращает (dirt_prob [0..1], метрики, статус, эмодзи).
    """
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
    
    # Определяем уровень загрязнения
    if dirt_score >= 6:
        status = "очень грязная"
        emoji = "🟤"
    elif dirt_score >= 4:
        status = "грязная"
        emoji = "🟫"
    elif dirt_score >= 2:
        status = "слегка грязная"
        emoji = "🟨"
    elif dirt_score >= 1:
        status = "достаточно чистая"
        emoji = "🟩"
    else:
        status = "очень чистая"
        emoji = "✨"
    
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
    
    return dirt_prob, metrics, status, emoji

def determine_repairability(predicted_class: str, confidence: float, major_damage_prob: float) -> Tuple[str, Tuple[str, ...], str]:
    """
    Определяет пригодность автомобиля для работы в сервисе такси
    на основе процентных порогов безопасности и стандартов перевозок
    
    Возвращает:
    - repairability_status: "taxi_ready", "conditional_taxi", "repair_required", "taxi_banned"
    - repairability_message: детальное описание для таксопарка
    - economic_assessment: оценка для коммерческого использования
    """
    
    # Более строгие пороги для сервиса такси (безопасность пассажиров приоритет!)
    TAXI_BAN_THRESHOLD = 75.0       # > 75% серьезных повреждений = ЗАПРЕТ на работу в такси
    REPAIR_REQUIRED_THRESHOLD = 50.0 # 50-75% = ОБЯЗАТЕЛЬНЫЙ ремонт перед допуском
    CONDITIONAL_THRESHOLD = 25.0     # 25-50% = условно допустимо с ограничениями
    MINOR_DAMAGE_TAXI_LIMIT = 40.0   # даже мелкие повреждения лимитированы для имиджа
    
    if predicted_class == 'major_damage':
        if confidence > 0.8 and major_damage_prob > TAXI_BAN_THRESHOLD:
            return "taxi_banned", (
                "🚫 АВТОМОБИЛЬ ЗАПРЕЩЕН ДЛЯ РАБОТЫ В ТАКСИ!",
                f"📊 Вероятность критических повреждений: {major_damage_prob:.1f}%",
                f"🚨 Превышен предельный порог безопасности ({TAXI_BAN_THRESHOLD}%)",
                "⚠️ РИСКИ: Угроза безопасности пассажиров и водителя",
                "📉 РЕПУТАЦИЯ: Серьезный ущерб имиджу таксопарка",
                "⚖️ ПРАВО: Нарушение требований к коммерческим перевозкам",
                "🎯 РЕШЕНИЕ: Исключить из парка, продать или утилизировать"
            ), "safety_violation"
            
        elif confidence > 0.6 or major_damage_prob > REPAIR_REQUIRED_THRESHOLD:
            return "repair_required", (
                "🔧 ОБЯЗАТЕЛЬНЫЙ РЕМОНТ ПЕРЕД ДОПУСКОМ К РАБОТЕ",
                f"📊 Вероятность серьезных повреждений: {major_damage_prob:.1f}%",
                f"⚖️ Превышен порог допуска к перевозкам ({REPAIR_REQUIRED_THRESHOLD}%)",
                "🚫 СТАТУС: ВРЕМЕННО ИСКЛЮЧЕН из эксплуатации",
                "🔧 ТРЕБОВАНИЯ: Капитальный ремонт + техосмотр",
                "💰 Ожидаемые затраты: 150-500 тыс. руб.",
                "📋 Обязательна сертификация после ремонта",
                "⏱️ Время простоя: 2-4 недели"
            ), "mandatory_repair"
            
        elif major_damage_prob > CONDITIONAL_THRESHOLD:
            return "conditional_taxi", (
                "⚠️ УСЛОВНО ДОПУСТИМ С ОГРАНИЧЕНИЯМИ",
                f"📊 Вероятность серьезных повреждений: {major_damage_prob:.1f}%",
                f"🔶 В пограничной зоне ({CONDITIONAL_THRESHOLD}-{REPAIR_REQUIRED_THRESHOLD}%)",
                "🚗 ОГРАНИЧЕНИЯ: Только внутригородские поездки",
                "🚫 ЗАПРЕТ: Междугородние рейсы и VIP-клиенты",
                "🔍 КОНТРОЛЬ: Еженедельные техосмотры",
                "💼 СТРАХОВАНИЕ: Повышенные тарифы",
                "⏰ ПЛАН: Плановый ремонт в течение месяца"
            ), "restricted_operation"
        else:
            return "conditional_taxi", (
                "🔧 КОСМЕТИЧЕСКИЙ РЕМОНТ РЕКОМЕНДОВАН",
                f"📊 Вероятность серьезных повреждений: {major_damage_prob:.1f}%",
                "✅ Допустимо для работы в такси",
                "🎨 ИМИДЖ: Желательно устранить видимые дефекты",
                "💰 Затраты: 50-150 тыс. руб. на косметику",
                "🏆 РЕЙТИНГ: Поможет поддержать высокие оценки"
            ), "cosmetic_repair"
    
    elif predicted_class == 'minor_damage':
        minor_damage_prob = 100 - major_damage_prob  # примерная оценка
        if confidence > 0.6 and minor_damage_prob > MINOR_DAMAGE_TAXI_LIMIT:
            return "conditional_taxi", (
                "🔧 КОСМЕТИЧЕСКИЙ РЕМОНТ ЖЕЛАТЕЛЕН ДЛЯ ТАКСИ",
                f"🎨 Заметные косметические дефекты: {minor_damage_prob:.1f}%",
                "✅ БЕЗОПАСНОСТЬ: Не влияет на безопасность движения",
                "📱 ИМИДЖ: Может снижать рейтинг и привлекательность для клиентов",
                "💰 Затраты на устранение: 30-100 тыс. руб.",
                "📱 ОТЗЫВЫ: Возможны негативные комментарии о внешнем виде",
                "🎯 РЕКОМЕНДАЦИЯ: Плановый косметический ремонт"
            ), "image_improvement"
        else:
            return "taxi_ready", (
                "✅ ПРИГОДЕН ДЛЯ РАБОТЫ В ТАКСИ",
                "🔧 Минимальные косметические дефекты",
                "🚗 Полностью пригоден для коммерческих перевозок",
                "💰 Затраты: 10-50 тыс. руб. на мелкий ремонт",
                "⏱️ Время ремонта: 1-3 дня",
                "🏆 Сохранение хорошего рейтинга сервиса"
            ), "minor_maintenance"
    
    else:  # no_damage
        return "taxi_ready", (
            "🏆 ИДЕАЛЕН ДЛЯ ПРЕМИУМ ТАКСИ-СЕРВИСА",
            "✨ Автомобиль в отличном состоянии",
            "🚗 КЛАСС: Подходит для VIP и бизнес-клиентов",
            "📈 РЕЙТИНГ: Обеспечит максимальные оценки пассажиров",
            "💎 ТАРИФЫ: Возможность работы в премиум-сегменте",
            "🎯 СТАТУС: Эталон качества таксопарка"
        ), "premium_ready"

def generate_expert_recommendations(predicted_class: str, confidence: float, probabilities: np.ndarray, 
                                  dirt_status: str, dirt_score: float, dirt_metrics: Dict[str, float]) -> List[str]:
    """Генерирует экспертные рекомендации на основе анализа"""
    recommendations = []
    
    no_damage_prob = probabilities[0] * 100
    minor_damage_prob = probabilities[1] * 100
    major_damage_prob = probabilities[2] * 100
    
    # Получаем оценку ремонтопригодности
    repairability_status, repairability_msgs, economic_status = determine_repairability(
        predicted_class, confidence, major_damage_prob
    )
    
    # Основные рекомендации по повреждениям
    if predicted_class == 'major_damage':
        if confidence > 0.8:
            recommendations.append("🚨 Обнаружены серьезные повреждения! Автомобиль НЕ ПРИГОДЕН для работы в такси без капитального ремонта.")
            recommendations.append("⚖️ Нарушение требований безопасности для коммерческих перевозок.")
        elif confidence > 0.6:
            recommendations.append("⚠️ Подозрение на серьезные повреждения. Требуется СРОЧНАЯ профессиональная экспертиза!")
    elif predicted_class == 'minor_damage':
        if confidence > 0.7:
            recommendations.append("🔧 Обнаружены мелкие повреждения (царапины, потертости). Косметический ремонт желателен.")
            recommendations.append("💰 Ориентировочные затраты: 30-100 тыс. руб.")
    else:  # no_damage
        if confidence > 0.85:
            recommendations.append("✨ Отличное состояние! Автомобиль идеален для премиум такси-сервиса.")
            recommendations.append("🏆 Подходит для VIP и бизнес-клиентов.")
        elif confidence > 0.7:
            recommendations.append("✅ Хорошее состояние. Автомобиль пригоден для работы в такси.")
    
    # Рекомендации по загрязненности
    if dirt_score > 6:
        recommendations.append("🧼 КРИТИЧЕСКОЕ ЗАГРЯЗНЕНИЕ: Автомобиль слишком грязный для перевозки пассажиров.")
        recommendations.append("📉 Нарушение стандартов имиджа такси-сервиса. Требуется немедленная профессиональная мойка.")
    elif dirt_score > 4:
        recommendations.append("🧽 Рекомендуется комплексная мойка перед выходом на линию.")
        recommendations.append("💰 Затраты: 1.5-3 тыс. руб. на качественную мойку.")
    elif dirt_score < 2:
        recommendations.append("✨ Превосходная чистота! Автомобиль содержится в идеальном состоянии.")
    
    # Добавляем основные рекомендации по ремонтопригодности
    for msg in repairability_msgs[:2]:  # Берем первые 2 сообщения
        recommendations.append(msg)
    
    # Общие рекомендации по уверенности модели
    if confidence < 0.6:
        recommendations.append("❓ Низкая уверенность ИИ-анализа. Рекомендуется дополнительная экспертная оценка.")
    
    return recommendations[:6]  # Ограничиваем до 6 рекомендаций

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

def _damage_from_logits(logits: torch.Tensor) -> Dict[str, any]:
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
        },
        'probs_array': probs
    }

def analyze_image(img_bytes: bytes) -> Dict[str, any]:
    """
    Главный вход: возвращает поля для фронта с подробными данными модели и экспертными рекомендациями.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)

    # Проверяем, есть ли автомобиль на изображении
    if not is_vehicle(img):
        return {
            "error": "vehicle_not_detected",
            "message": "⚠️ На фото, похоже, нет автомобиля. Загрузите снимок машины (желательно 3/4 спереди или сбоку).",
            "dirty": False,
            "dirty_prob": 0.0,
            "damaged": False,
            "damaged_prob": 0.0,
            "model_available": False
        }

    # --- грязь ---
    dirt_prob, dirt_metrics, dirt_status, dirt_emoji = analyze_dirt_level(img)

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
            "dirt_status": dirt_status,
            "dirt_emoji": dirt_emoji,
            "model_available": False,
            "expert_recommendations": [
                "⚠️ ИИ-модель временно недоступна. Результат основан только на анализе загрязненности.",
                f"🧼 Состояние чистоты: {dirt_emoji} {dirt_status}"
            ]
        }

    x = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(x)
    
    # Получаем подробную информацию о повреждениях
    damage_info = _damage_from_logits(logits)
    
    # Генерируем экспертные рекомендации
    expert_recommendations = generate_expert_recommendations(
        damage_info["predicted_class"], 
        damage_info["confidence"], 
        damage_info["probs_array"],
        dirt_status, 
        dirt_metrics["dirt_score"], 
        dirt_metrics
    )

    return {
        "dirty": dirt_prob >= 0.5,
        "dirty_prob": round(float(dirt_prob), 4),
        "damaged": damage_info["damaged"],
        "damaged_prob": damage_info["damaged_prob"],
        "predicted_class": damage_info["predicted_class"],
        "confidence": damage_info["confidence"],
        "probabilities": damage_info["probabilities"],
        "dirt_metrics": dirt_metrics,
        "dirt_status": dirt_status,
        "dirt_emoji": dirt_emoji,
        "model_available": True,
        "expert_recommendations": expert_recommendations
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