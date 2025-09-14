# CarCondition/inference.py
"""
Система ИИ-диагностики автомобилей v2.0 для таксопарков
Продвинутый анализ загрязненности и повреждений с экономической оценкой
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import io, os
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageStat, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from dotenv import load_dotenv

# Импорт библиотек для продвинутого анализа
try:
    import cv2
    from scipy import ndimage
    from skimage import feature, measure
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
    print("⚠️ OpenCV/scikit-image не найдены. Используется базовый анализ грязи.")

# архитектура из твоего файла
from multiclass_damage_model import MulticlassDamageModel

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pth")
DEVICE     = os.getenv("DEVICE", "cpu")

# === предобработка ровно как в новом скрипте ===
_tf = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def advanced_dirt_analysis(image: Image.Image) -> Tuple[str, str, float, Dict, str]:
    """Продвинутый анализ загрязненности с использованием компьютерного зрения"""
    if not CV_AVAILABLE:
        # Fallback к базовому анализу
        return _basic_dirt_analysis(image)
    
    img_array = np.array(image)
    
    # Конвертируем в OpenCV формат для дополнительных операций
    cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # 1. ТЕКСТУРНЫЙ АНАЛИЗ (основной показатель грязи)
    # Local Binary Patterns для анализа текстуры
    try:
        lbp = feature.local_binary_pattern(gray, P=24, R=3, method='uniform')
        texture_variance = np.var(lbp)
    except:
        texture_variance = np.var(gray)
    
    # Анализ градиентов (грязь создает хаотичные градиенты)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_chaos = np.std(gradient_magnitude) / (np.mean(gradient_magnitude) + 1e-8)
    
    # 2. ЦВЕТОВОЙ АНАЛИЗ (улучшенный)
    # Анализ в разных цветовых пространствах
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    
    # Насыщенность (S в HSV)
    saturation = np.mean(hsv[:,:,1])
    saturation_std = np.std(hsv[:,:,1])
    
    # Анализ L-канала (яркость в LAB)
    lightness = np.mean(lab[:,:,0])
    lightness_uniformity = 1 / (np.std(lab[:,:,0]) + 1e-8)
    
    # 3. ДЕТЕКЦИЯ ГРЯЗНЫХ ПЯТЕН
    # Создаем маску для потенциально грязных областей
    # Грязь обычно имеет низкую насыщенность и среднюю яркость
    dirt_mask = (
        (hsv[:,:,1] < 80) &  # Низкая насыщенность
        (hsv[:,:,2] > 30) &  # Не слишком темно
        (hsv[:,:,2] < 200)   # Не слишком светло
    )
    
    # Анализ связанных компонентов грязных областей
    labeled_dirt = measure.label(dirt_mask)
    dirt_regions = measure.regionprops(labeled_dirt)
    
    # Подсчет крупных грязных пятен
    large_dirt_areas = sum(1 for region in dirt_regions if region.area > 100)
    total_dirt_area = np.sum(dirt_mask) / (img_array.shape[0] * img_array.shape[1])
    
    # 4. АНАЛИЗ ПОВЕРХНОСТИ АВТОМОБИЛЯ
    # Ищем металлические отражения (чистый автомобиль их больше)
    # Высокие значения в RGB каналах одновременно = отражение
    reflection_mask = (
        (img_array[:,:,0] > 200) & 
        (img_array[:,:,1] > 200) & 
        (img_array[:,:,2] > 200)
    )
    reflection_ratio = np.sum(reflection_mask) / (img_array.shape[0] * img_array.shape[1])
    
    # Анализ однородности цвета (чистая поверхность более однородна)
    color_uniformity = []
    for channel in range(3):
        channel_data = img_array[:,:,channel]
        uniformity = 1 / (np.std(channel_data) + 1e-8)
        color_uniformity.append(uniformity)
    avg_color_uniformity = np.mean(color_uniformity)
    
    # 5. СПЕЦИФИЧЕСКИЕ ПРИЗНАКИ ГРЯЗИ
    # Детекция пыли (высокочастотный шум)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dust_noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))
    
    # Детекция грязевых разводов (низкочастотные пятна)
    kernel = np.ones((15,15), np.float32) / 225
    filtered = cv2.filter2D(gray, -1, kernel)
    mud_patterns = np.std(gray.astype(float) - filtered.astype(float))
    
    # 6. РАСЧЕТ КОМПЛЕКСНОГО ИНДЕКСА ЗАГРЯЗНЕНИЯ
    dirt_score = 0.0
    confidence_factors = []
    
    # Текстурные признаки (вес 30%)
    if texture_variance > 15:
        texture_points = min(3.0, texture_variance / 10)
        dirt_score += texture_points * 0.3
        confidence_factors.append(f"Текстура: +{texture_points:.1f}")
    
    # Градиентный хаос (вес 25%)
    if gradient_chaos > 0.8:
        chaos_points = min(2.5, gradient_chaos * 2)
        dirt_score += chaos_points * 0.25
        confidence_factors.append(f"Градиенты: +{chaos_points:.1f}")
    
    # Цветовые характеристики (вес 20%)
    saturation_penalty = max(0, (100 - saturation) / 25)
    dirt_score += saturation_penalty * 0.2
    if saturation_penalty > 0.5:
        confidence_factors.append(f"Насыщенность: +{saturation_penalty:.1f}")
    
    # Грязные пятна (вес 15%)
    dirt_area_points = min(2.0, total_dirt_area * 10)
    dirt_score += dirt_area_points * 0.15
    if dirt_area_points > 0.3:
        confidence_factors.append(f"Пятна: +{dirt_area_points:.1f}")
    
    # Отсутствие отражений (вес 10%)
    reflection_penalty = max(0, (0.05 - reflection_ratio) * 20)
    dirt_score += reflection_penalty * 0.1
    if reflection_penalty > 0.2:
        confidence_factors.append(f"Отражения: +{reflection_penalty:.1f}")
    
    # Нормализуем итоговый счет к шкале 0-10
    dirt_score = min(10.0, max(0.0, dirt_score))
    
    # 7. ОПРЕДЕЛЕНИЕ СТАТУСА С УЧЕТОМ CONFIDENCE
    confidence = len(confidence_factors) / 5.0  # Максимум 5 факторов
    
    if dirt_score >= 7.5 and confidence > 0.6:
        status = "критически грязная"
        emoji = "🟫"
        recommendation = "СРОЧНАЯ профессиональная мойка + детейлинг"
    elif dirt_score >= 6.0 and confidence > 0.4:
        status = "очень грязная" 
        emoji = "🟤"
        recommendation = "Необходима тщательная мойка"
    elif dirt_score >= 4.0:
        status = "грязная"
        emoji = "🔶"
        recommendation = "Рекомендуется мойка"
    elif dirt_score >= 2.5:
        status = "слегка грязная"
        emoji = "🟨"
        recommendation = "Легкая мойка желательна"
    elif dirt_score >= 1.0:
        status = "достаточно чистая"
        emoji = "🟩"
        recommendation = "В хорошем состоянии"
    else:
        status = "очень чистая"
        emoji = "✨"
        recommendation = "Идеальное состояние чистоты"
    
    # 8. ДЕТАЛЬНЫЕ МЕТРИКИ
    detailed_metrics = {
        'dirt_score': dirt_score,
        'confidence': confidence,
        'texture_variance': texture_variance,
        'gradient_chaos': gradient_chaos,
        'saturation': saturation,
        'lightness': lightness,
        'total_dirt_area': total_dirt_area * 100,  # в процентах
        'large_dirt_spots': large_dirt_areas,
        'reflection_ratio': reflection_ratio * 100,  # в процентах
        'dust_noise': dust_noise,
        'mud_patterns': mud_patterns,
        'color_uniformity': avg_color_uniformity,
        'analysis_factors': confidence_factors,
        'recommendation': recommendation
    }
    
    return status, emoji, dirt_score, detailed_metrics, recommendation

def _basic_dirt_analysis(image: Image.Image) -> Tuple[str, str, float, Dict, str]:
    """Базовый анализ загрязненности без OpenCV"""
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
    
    # Подсчет итогового индекса грязи (базовый алгоритм)
    dirt_score = 0.0
    confidence_factors = []
    
    if color_diversity < 80:
        dirt_score += 2
        confidence_factors.append("Низкое цветовое разнообразие")
    elif color_diversity < 120:
        dirt_score += 1
        confidence_factors.append("Умеренное цветовое разнообразие")
    
    if contrast < 25:
        dirt_score += 2
        confidence_factors.append("Низкий контраст")
    elif contrast < 40:
        dirt_score += 1
        confidence_factors.append("Средний контраст")
    
    if saturation < 60:
        dirt_score += 1.5
        confidence_factors.append("Низкая насыщенность")
    elif saturation < 100:
        dirt_score += 0.5
        confidence_factors.append("Умеренная насыщенность")
    
    if brown_ratio > 0.15:
        dirt_score += 2
        confidence_factors.append("Много коричневых оттенков")
    elif brown_ratio > 0.08:
        dirt_score += 1
        confidence_factors.append("Заметны коричневые оттенки")
    
    if edge_intensity < 15:
        dirt_score += 1.5
        confidence_factors.append("Размытые края")
    elif edge_intensity < 25:
        dirt_score += 0.5
        confidence_factors.append("Нечеткие края")
    
    if brightness < 90:
        dirt_score += 1
        confidence_factors.append("Низкая яркость")
    elif brightness < 110:
        dirt_score += 0.5
        confidence_factors.append("Средняя яркость")

    # Определение статуса для базового анализа
    if dirt_score >= 7.5:
        status = "критически грязная"
        emoji = "🟫"
        recommendation = "СРОЧНАЯ профессиональная мойка + детейлинг"
    elif dirt_score >= 6.0:
        status = "очень грязная" 
        emoji = "🟤"
        recommendation = "Необходима тщательная мойка"
    elif dirt_score >= 4.0:
        status = "грязная"
        emoji = "🔶"
        recommendation = "Рекомендуется мойка"
    elif dirt_score >= 2.5:
        status = "слегка грязная"
        emoji = "🟨"
        recommendation = "Легкая мойка желательна"
    elif dirt_score >= 1.0:
        status = "достаточно чистая"
        emoji = "🟩"
        recommendation = "В хорошем состоянии"
    else:
        status = "очень чистая"
        emoji = "✨"
        recommendation = "Идеальное состояние чистоты"

    detailed_metrics = {
        'dirt_score': dirt_score,
        'confidence': len(confidence_factors) / 6.0,  # Максимум 6 факторов в базовом анализе
        'color_diversity': color_diversity,
        'contrast': contrast,
        'saturation': saturation,
        'brown_ratio': brown_ratio * 100,  # в процентах
        'edge_intensity': edge_intensity,
        'brightness': brightness,
        'analysis_factors': confidence_factors,
        'recommendation': recommendation
    }
    
    return status, emoji, dirt_score, detailed_metrics, recommendation

def enhanced_damage_prediction(model, image_tensor, device, original_image):
    """Улучшенное предсказание повреждений с дополнительными проверками"""
    class_names = ['no_damage', 'minor_damage', 'major_damage']
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Получаем основное предсказание
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        probs = probabilities.cpu().numpy()[0]
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Дополнительный анализ изображения для коррекции предсказания
        img_array = np.array(original_image)
        
        if CV_AVAILABLE:
            # Анализ резкости (размытые изображения могут давать ложные срабатывания)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Анализ контрастности 
            contrast = gray.std()
            
            # Поиск четких линий (края кузова, детали)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
        else:
            # Fallback для случая без OpenCV
            gray = original_image.convert('L')
            gray_array = np.array(gray)
            laplacian_var = np.var(gray_array)  # Упрощенная оценка резкости
            contrast = gray_array.std()
            edge_density = 0.05  # Средняя оценка
        
        # Коррекция confidence на основе качества изображения
        quality_factor = 1.0
        
        if laplacian_var < 100:  # Размытое изображение
            quality_factor *= 0.8
            
        if contrast < 30:  # Низкий контраст
            quality_factor *= 0.9
            
        if edge_density < 0.05:  # Мало четких краев
            quality_factor *= 0.85
        
        # Применяем коррекцию
        adjusted_confidence = confidence_score * quality_factor
        
        # Если confidence упало ниже порога, понижаем категорию
        if adjusted_confidence < 0.6 and predicted_class == 'major_damage':
            # Пересматриваем как minor_damage
            if probs[1] > 0.3:  # Если minor_damage тоже вероятно
                predicted_class = 'minor_damage'
                confidence_score = probs[1] * quality_factor
                
        elif adjusted_confidence < 0.5 and predicted_class == 'minor_damage':
            # Пересматриваем как no_damage
            if probs[0] > 0.25:
                predicted_class = 'no_damage'
                confidence_score = probs[0] * quality_factor
        
        # Дополнительные метрики качества
        quality_metrics = {
            'sharpness': laplacian_var,
            'contrast': contrast,
            'edge_density': edge_density * 100,  # в процентах
            'quality_factor': quality_factor,
            'original_confidence': confidence.item(),
            'adjusted_confidence': adjusted_confidence
        }
        
        return predicted_class, confidence_score, probs, class_names, quality_metrics

def determine_repairability_enhanced(predicted_class, confidence, major_damage_prob, quality_metrics, dirt_status, dirt_score):
    """
    Улучшенное определение пригодности для такси с учетом качества анализа
    """
    
    # Более строгие пороги с учетом качества изображения
    quality_adjustment = quality_metrics.get('quality_factor', 1.0)
    
    # Базовые пороги
    TAXI_BAN_THRESHOLD = 80.0 * quality_adjustment
    REPAIR_REQUIRED_THRESHOLD = 60.0 * quality_adjustment  
    CONDITIONAL_THRESHOLD = 35.0 * quality_adjustment
    MINOR_DAMAGE_TAXI_LIMIT = 45.0 * quality_adjustment
    
    # Учитываем влияние грязи на точность анализа
    dirt_uncertainty = min(0.2, dirt_score / 50)  # До 20% неопределенности от грязи
    effective_confidence = confidence * (1 - dirt_uncertainty)
    
    # Предупреждения о качестве анализа
    quality_warnings = []
    if quality_metrics.get('sharpness', 1000) < 100:
        quality_warnings.append("⚠️ Изображение размыто - точность снижена")
    if quality_metrics.get('contrast', 100) < 30:
        quality_warnings.append("⚠️ Низкий контраст - возможны ошибки")
    if dirt_score > 6:
        quality_warnings.append("⚠️ Сильное загрязнение затрудняет анализ")
    
    if predicted_class == 'major_damage':
        if effective_confidence > 0.85 and major_damage_prob > TAXI_BAN_THRESHOLD:
            return "taxi_banned", (
                "🚫 АВТОМОБИЛЬ ЗАПРЕЩЕН ДЛЯ РАБОТЫ В ТАКСИ!",
                f"   📊 Вероятность критических повреждений: {major_damage_prob:.1f}%",
                f"   🎯 Уверенность анализа: {effective_confidence*100:.1f}%",
                f"   🚨 Превышен предельный порог безопасности ({TAXI_BAN_THRESHOLD:.0f}%)",
                "   ⚠️ РИСКИ: Угроза безопасности пассажиров и водителя",
                "   📉 РЕПУТАЦИЯ: Серьезный ущерб имиджу таксопарка",
                "   ⚖️ ПРАВО: Нарушение требований к коммерческим перевозкам",
                "   🎯 РЕШЕНИЕ: Исключить из парка, продать или утилизировать",
                *quality_warnings
            ), "safety_violation"
            
        elif effective_confidence > 0.65 or major_damage_prob > REPAIR_REQUIRED_THRESHOLD:
            return "repair_required", (
                "🔧 ОБЯЗАТЕЛЬНЫЙ РЕМОНТ ПЕРЕД ДОПУСКОМ К РАБОТЕ",
                f"   📊 Вероятность серьезных повреждений: {major_damage_prob:.1f}%",
                f"   🎯 Уверенность анализа: {effective_confidence*100:.1f}%",
                f"   ⚖️ Превышен порог допуска к перевозкам ({REPAIR_REQUIRED_THRESHOLD:.0f}%)",
                "   ❌ СТАТУС: ВРЕМЕННО ИСКЛЮЧЕН из эксплуатации",
                "   🔧 ТРЕБОВАНИЯ: Капитальный ремонт + техосмотр",
                "   💰 Ожидаемые затраты: 150-500 тыс. руб.",
                "   📋 Обязательна сертификация после ремонта",
                "   ⏱️ Время простоя: 2-4 недели",
                *quality_warnings
            ), "mandatory_repair"
            
        elif major_damage_prob > CONDITIONAL_THRESHOLD:
            return "conditional_taxi", (
                "⚠️ УСЛОВНО ДОПУСТИМ С СЕРЬЕЗНЫМИ ОГРАНИЧЕНИЯМИ",
                f"   📊 Вероятность серьезных повреждений: {major_damage_prob:.1f}%",
                f"   🎯 Уверенность анализа: {effective_confidence*100:.1f}%",
                f"   🔶 В пограничной зоне ({CONDITIONAL_THRESHOLD:.0f}-{REPAIR_REQUIRED_THRESHOLD:.0f}%)",
                "   🚫 ОГРАНИЧЕНИЯ: Только внутригородские поездки до 50 км",
                "   ❌ ЗАПРЕТ: Междугородние рейсы и VIP-клиенты",
                "   🔍 КОНТРОЛЬ: Ежедневные техосмотры обязательны",
                "   💼 СТРАХОВАНИЕ: Значительно повышенные тарифы",
                "   ⏰ ПЛАН: Срочный ремонт в течение 2 недель",
                "   🩺 ДИАГНОСТИКА: Обязательна экспертиза в автосервисе",
                *quality_warnings
            ), "high_risk_operation"
        else:
            return "conditional_taxi", (
                "🔧 КОСМЕТИЧЕСКИЙ РЕМОНТ НАСТОЯТЕЛЬНО РЕКОМЕНДОВАН",
                f"   📊 Вероятность повреждений: {major_damage_prob:.1f}%",
                f"   🎯 Уверенность анализа: {effective_confidence*100:.1f}%",
                "   ✅ Допустимо для осторожной эксплуатации",
                "   🎨 ИМИДЖ: Желательно устранить все видимые дефекты",
                "   💰 Затраты: 50-150 тыс. руб. на косметику",
                "   ⭐ РЕЙТИНГ: Поможет поддержать высокие оценки",
                *quality_warnings
            ), "cosmetic_repair"
    
    elif predicted_class == 'minor_damage':
        minor_damage_prob = 100 - major_damage_prob
        if effective_confidence > 0.7 and minor_damage_prob > MINOR_DAMAGE_TAXI_LIMIT:
            return "conditional_taxi", (
                "🔧 КОСМЕТИЧЕСКИЙ РЕМОНТ ЖЕЛАТЕЛЕН ДЛЯ ТАКСИ",
                f"   🎨 Заметные косметические дефекты: {minor_damage_prob:.1f}%",
                f"   🎯 Уверенность анализа: {effective_confidence*100:.1f}%",
                "   ✅ БЕЗОПАСНОСТЬ: Не влияет на безопасность движения",
                "   📉 ИМИДЖ: Может снижать рейтинг и привлекательность для клиентов",
                "   💰 Затраты на устранение: 30-100 тыс. руб.",
                "   📱 ОТЗЫВЫ: Возможны негативные комментарии о внешнем виде",
                "   🎯 РЕКОМЕНДАЦИЯ: Плановый косметический ремонт в течение месяца",
                *quality_warnings
            ), "image_improvement"
        else:
            return "taxi_ready", (
                "✅ ПРИГОДЕН ДЛЯ РАБОТЫ В ТАКСИ",
                f"   🎯 Уверенность анализа: {effective_confidence*100:.1f}%",
                "   🔧 Минимальные косметические дефекты",
                "   🚗 Полностью пригоден для коммерческих перевозок",
                "   💰 Затраты: 10-50 тыс. руб. на мелкий ремонт",
                "   ⏱️ Время ремонта: 1-3 дня",
                "   🏆 Сохранение хорошего рейтинга сервиса",
                *quality_warnings
            ), "minor_maintenance"
    
    else:  # no_damage
        return "taxi_ready", (
            "🏆 ИДЕАЛЕН ДЛЯ ПРЕМИУМ ТАКСИ-СЕРВИСА",
            f"   🎯 Уверенность анализа: {effective_confidence*100:.1f}%",
            "   ✨ Автомобиль в отличном состоянии",
            "   💎 КЛАСС: Подходит для VIP и бизнес-клиентов",
            "   📈 РЕЙТИНГ: Обеспечит максимальные оценки пассажиров",
            "   💰 ТАРИФЫ: Возможность работы в премиум-сегменте",
            "   🎯 СТАТУС: Эталон качества таксопарка",
            *quality_warnings
        ), "premium_ready"

def preprocess_image_enhanced(image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    """Улучшенная предобработка изображения"""
    # Дополнительная обработка для улучшения качества
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)  # Небольшое увеличение резкости
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)  # Небольшое увеличение контраста
    
    image_tensor = _tf(image).unsqueeze(0)
    
    return image_tensor, image

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

def analyze_image(img_bytes: bytes) -> Dict:
    """
    Главный вход: улучшенный анализ с экспертными рекомендациями v2.0
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)

    # --- продвинутый анализ грязи ---
    dirt_status, dirt_emoji, dirt_score, dirt_metrics, dirt_recommendation = advanced_dirt_analysis(img)

    # --- damage через улучшенную модель ---
    model = get_model()
    if model is None:
        # если модели нет — отдаём только грязь
        return {
            "dirty": dirt_score >= 5.0,  # Обновлено для новой шкалы 0-10
            "dirty_prob": round(dirt_score / 10.0, 4),  # Нормализуем к [0,1]
            "damaged": False,
            "damaged_prob": 0.0,
            "predicted_class": "unknown",
            "confidence": 0.0,
            "probabilities": {
                "no_damage": 0.0,
                "minor_damage": 0.0,
                "major_damage": 0.0
            },
            "dirt_status": dirt_status,
            "dirt_emoji": dirt_emoji,
            "dirt_score": dirt_score,
            "dirt_metrics": dirt_metrics,
            "dirt_recommendation": dirt_recommendation,
            "model_available": False,
            "expert_assessment": ["❌ Модель ИИ недоступна - только базовый анализ загрязнения"],
            "taxi_status": "unknown",
            "taxi_recommendations": ["Требуется загрузка модели для полного анализа"]
        }

    # Улучшенная предобработка
    image_tensor, processed_image = preprocess_image_enhanced(img)
    device = torch.device(DEVICE)
    
    # Улучшенное предсказание повреждений
    predicted_class, confidence, probabilities, class_names, quality_metrics = enhanced_damage_prediction(
        model, image_tensor, device, processed_image
    )
    
    # Основные метрики повреждений
    no_damage_prob = probabilities[0] * 100
    minor_damage_prob = probabilities[1] * 100  
    major_damage_prob = probabilities[2] * 100
    damaged_prob = 1.0 - probabilities[0]  # 1 - no_damage
    
    # Расчет целостности автомобиля (обратная к поврежденности)
    integrity_score = no_damage_prob  # Целостность = вероятность отсутствия повреждений
    
    # 🎯 НОВАЯ ЛОГИКА: Коррекция чистоты в зависимости от целостности
    original_dirt_score = dirt_score
    original_dirt_status = dirt_status
    
    if integrity_score < 20:  # Если целостность ниже 20%
        # Принудительно снижаем чистоту до ~15% (dirt_score ~8.5/10)
        corrected_dirt_score = max(dirt_score, 8.5)  # Минимум 8.5 из 10 (очень грязная)
        
        # Обновляем статус загрязнения
        if corrected_dirt_score >= 8.5:
            dirt_status = "критически грязная"
            dirt_emoji = "🟫" 
            dirt_recommendation = "СРОЧНАЯ профессиональная мойка + детейлинг (скрывает повреждения)"
        
        # Обновляем dirt_score
        dirt_score = corrected_dirt_score
        
        # Добавляем предупреждение в метрики
        dirt_metrics['integrity_correction'] = True
        dirt_metrics['original_dirt_score'] = original_dirt_score
        dirt_metrics['corrected_reason'] = f"Низкая целостность ({integrity_score:.1f}%) - грязь может маскировать серьезные повреждения"
        
        print(f"⚠️ Коррекция чистоты: {original_dirt_score:.1f} → {dirt_score:.1f} (целостность: {integrity_score:.1f}%)")
    else:
        # Отмечаем, что коррекция не применялась
        dirt_metrics['integrity_correction'] = False
    
    # Улучшенная оценка для такси
    taxi_status, taxi_msgs, economic_status = determine_repairability_enhanced(
        predicted_class, confidence, major_damage_prob, quality_metrics, dirt_status, dirt_score
    )
    
    # Экспертная оценка (краткая версия для API)
    expert_assessment = []
    expert_assessment.append(f"🤖 ИИ-ДИАГНОСТИКА v2.0: {predicted_class.replace('_', ' ').upper()}")
    expert_assessment.append(f"🎯 Уверенность: {confidence*100:.1f}%")
    expert_assessment.append(f"🔧 Целостность: {integrity_score:.1f}%")
    expert_assessment.append(f"🧼 Загрязнение: {dirt_status} ({dirt_score:.1f}/10)")
    
    # Добавляем информацию о коррекции, если она была применена
    if dirt_metrics.get('integrity_correction', False):
        expert_assessment.append(f"⚠️ КОРРЕКЦИЯ: Чистота скорректирована с {original_dirt_score:.1f} до {dirt_score:.1f}")
        expert_assessment.append(f"📊 Причина: Низкая целостность может маскировать повреждения грязью")
    
    expert_assessment.append(f"🚕 Статус для такси: {taxi_status.replace('_', ' ').upper()}")
    
    # Качество анализа
    if quality_metrics.get('sharpness', 1000) < 100:
        expert_assessment.append("⚠️ Низкое качество изображения влияет на точность")
    
    # Экономическая классификация
    economic_info = {
        "status": economic_status,
        "taxi_ready": taxi_status in ["taxi_ready"],
        "needs_repair": taxi_status in ["repair_required", "conditional_taxi"],
        "banned": taxi_status == "taxi_banned"
    }

    return {
        # Совместимость с существующим API
        "dirty": dirt_score >= 5.0,
        "dirty_prob": round(dirt_score / 10.0, 4),
        "damaged": damaged_prob >= 0.5,
        "damaged_prob": round(damaged_prob, 4),
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            "no_damage": round(probabilities[0], 4),
            "minor_damage": round(probabilities[1], 4),
            "major_damage": round(probabilities[2], 4)
        },
        
        # Новые продвинутые поля
        "integrity_score": round(integrity_score, 2),  # Новое поле - целостность автомобиля
        "dirt_status": dirt_status,
        "dirt_emoji": dirt_emoji,
        "dirt_score": round(dirt_score, 2),
        "dirt_metrics": dirt_metrics,
        "dirt_recommendation": dirt_recommendation,
        
        # Качество анализа
        "quality_metrics": quality_metrics,
        
        # Экспертная оценка
        "expert_assessment": expert_assessment,
        
        # Оценка для такси
        "taxi_status": taxi_status,
        "taxi_recommendations": list(taxi_msgs),
        "economic_info": economic_info,
        
        # Технические метрики
        "model_available": True,
        "analysis_version": "2.0",
        "cv_available": CV_AVAILABLE
    }

# удобный локальный тест
def debug_analyze_image(img_bytes: bytes) -> Dict:
    """
    Расширенная версия analyze_image с дополнительной отладочной информацией v2.0
    """
    result = analyze_image(img_bytes)
    
    # Добавляем дополнительную отладочную информацию
    result["debug"] = {
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "model_loaded": result.get("model_available", False),
        "opencv_available": CV_AVAILABLE,
        "analysis_version": "2.0",
        "features": {
            "advanced_dirt_analysis": CV_AVAILABLE,
            "enhanced_damage_prediction": True,
            "taxi_economic_assessment": True,
            "quality_metrics": True
        }
    }
    
    return result

# Функция для совместимости со старым API
def analyze_dirt_level(image: Image.Image) -> Tuple[float, Dict[str, float]]:
    """
    Обертка для совместимости со старым API
    Возвращает (dirt_prob [0..1], метрики).
    """
    dirt_status, dirt_emoji, dirt_score, dirt_metrics, dirt_recommendation = advanced_dirt_analysis(image)
    
    # Нормализуем dirt_score к диапазону [0,1] для совместимости
    dirt_prob = dirt_score / 10.0
    
    # Преобразуем метрики к старому формату
    legacy_metrics = {
        'dirt_score': dirt_score,
        'color_diversity': dirt_metrics.get('saturation', 100),
        'contrast': dirt_metrics.get('lightness', 100),
        'saturation': dirt_metrics.get('saturation', 100),
        'brown_ratio': dirt_metrics.get('total_dirt_area', 0) / 100,
        'edge_intensity': 50.0,  # Средняя оценка
        'brightness': dirt_metrics.get('lightness', 100)
    }
    
    return dirt_prob, legacy_metrics
