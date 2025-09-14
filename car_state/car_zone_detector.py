"""
Система детекции зон автомобиля для анализа повреждений по частям кузова
================================================================
Поддерживает:
1. Простую геометрическую сегментацию 
2. Интеграцию с ML-моделями для детекции зон
3. Анализ повреждений по зонам
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json

@dataclass
class ZoneAnalysis:
    """Результат анализа одной зоны"""
    zone_name: str
    damage_probability: float
    damage_class: str  # 'no_damage', 'minor_damage', 'major_damage'
    confidence: float
    integrity_score: float  # Процент целостности (100 - damage_probability)
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

@dataclass
class CarAnalysisReport:
    """Полный отчёт анализа автомобиля"""
    overall_integrity: float
    overall_grade: str
    zones: List[ZoneAnalysis]
    total_zones: int
    damaged_zones: int
    original_image_path: str
    processed_image_path: Optional[str] = None

class CarZoneDetector:
    """Детектор зон автомобиля"""
    
    # Определяем зоны автомобиля (простая геометрическая сегментация)
    ZONE_TEMPLATES = {
        'front': {'position': (0.2, 0.3, 0.8, 0.7), 'color': (255, 0, 0)},      # Передняя часть
        'rear': {'position': (0.2, 0.7, 0.8, 1.0), 'color': (0, 255, 0)},       # Задняя часть  
        'roof': {'position': (0.25, 0.0, 0.75, 0.3), 'color': (0, 0, 255)},     # Крыша
        'left_side': {'position': (0.0, 0.2, 0.4, 0.8), 'color': (255, 255, 0)}, # Левая сторона
        'right_side': {'position': (0.6, 0.2, 1.0, 0.8), 'color': (255, 0, 255)}, # Правая сторона
        'hood': {'position': (0.3, 0.15, 0.7, 0.45), 'color': (0, 255, 255)},   # Капот
        'trunk': {'position': (0.3, 0.55, 0.7, 0.85), 'color': (128, 128, 128)} # Багажник
    }
    
    def __init__(self, detection_method='geometric'):
        """
        Args:
            detection_method: 'geometric' или 'ml' (для будущих ML-детекторов)
        """
        self.detection_method = detection_method
        
    def detect_zones(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Детектирует зоны на изображении автомобиля
        
        Args:
            image: Изображение автомобиля (numpy array)
            
        Returns:
            Словарь {zone_name: (x1, y1, x2, y2)}
        """
        if self.detection_method == 'geometric':
            return self._detect_zones_geometric(image)
        else:
            raise NotImplementedError(f"Метод {self.detection_method} пока не реализован")
    
    def _detect_zones_geometric(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """Простая геометрическая сегментация"""
        h, w = image.shape[:2]
        zones = {}
        
        for zone_name, zone_info in self.ZONE_TEMPLATES.items():
            x1_rel, y1_rel, x2_rel, y2_rel = zone_info['position']
            
            x1 = int(x1_rel * w)
            y1 = int(y1_rel * h)
            x2 = int(x2_rel * w)
            y2 = int(y2_rel * h)
            
            zones[zone_name] = (x1, y1, x2, y2)
            
        return zones
    
    def extract_zone_images(self, image: np.ndarray, zones: Dict[str, Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
        """Извлекает изображения зон"""
        zone_images = {}
        
        for zone_name, (x1, y1, x2, y2) in zones.items():
            # Убеждаемся, что координаты в пределах изображения
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:
                zone_image = image[y1:y2, x1:x2]
                zone_images[zone_name] = zone_image
                
        return zone_images

class CarDamageAnalyzer:
    """Анализатор повреждений автомобиля по зонам"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
            model_path: Путь к обученной модели
            device: Устройство для инференса
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.zone_detector = CarZoneDetector()
        
        # Преобразования для модели
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _load_model(self, model_path: str):
        """Загружает обученную модель"""
        try:
            from multiclass_damage_model import MulticlassDamageModel
            
            model = MulticlassDamageModel(num_classes=3)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            model.to(self.device)
            print(f"✅ Модель загружена из {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return None
    
    def analyze_zone(self, zone_image: np.ndarray) -> Tuple[str, float, float]:
        """
        Анализирует повреждения в одной зоне
        
        Returns:
            (damage_class, damage_probability, confidence)
        """
        if self.model is None:
            # Заглушка для тестирования
            damage_prob = np.random.uniform(0, 1)
            if damage_prob < 0.6:
                return 'no_damage', damage_prob, 0.8
            elif damage_prob < 0.8:
                return 'minor_damage', damage_prob, 0.7
            else:
                return 'major_damage', damage_prob, 0.9
        
        # Преобразуем изображение для модели
        if len(zone_image.shape) == 3 and zone_image.shape[2] == 3:
            # RGB изображение
            input_tensor = self.transform(zone_image).unsqueeze(0).to(self.device)
        else:
            print("⚠️ Неподдерживаемый формат изображения")
            return 'no_damage', 0.0, 0.0
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Получаем предсказания
            class_probs = probabilities[0].cpu().numpy()
            predicted_class = np.argmax(class_probs)
            confidence = float(class_probs[predicted_class])
            
            # Mapping классов
            class_names = ['no_damage', 'minor_damage', 'major_damage']
            damage_class = class_names[predicted_class]
            
            # Общая вероятность повреждения (minor + major)
            damage_probability = float(class_probs[1] + class_probs[2])
            
            return damage_class, damage_probability, confidence
    
    def analyze_car(self, image_path: str) -> CarAnalysisReport:
        """
        Полный анализ автомобиля
        
        Args:
            image_path: Путь к изображению автомобиля
            
        Returns:
            Отчёт анализа
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Конвертируем в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детектируем зоны
        zones = self.zone_detector.detect_zones(image_rgb)
        zone_images = self.zone_detector.extract_zone_images(image_rgb, zones)
        
        # Анализируем каждую зону
        zone_analyses = []
        total_damage_prob = 0
        damaged_zones = 0
        
        for zone_name, zone_image in zone_images.items():
            if zone_name not in zones:
                continue
                
            # Анализируем зону
            damage_class, damage_prob, confidence = self.analyze_zone(zone_image)
            
            # Создаём анализ зоны
            zone_analysis = ZoneAnalysis(
                zone_name=zone_name,
                damage_probability=damage_prob,
                damage_class=damage_class,
                confidence=confidence,
                integrity_score=100 - (damage_prob * 100),
                bbox=zones[zone_name]
            )
            
            zone_analyses.append(zone_analysis)
            total_damage_prob += damage_prob
            
            if damage_class != 'no_damage':
                damaged_zones += 1
        
        # Вычисляем общие метрики
        avg_damage_prob = total_damage_prob / len(zone_analyses) if zone_analyses else 0
        overall_integrity = 100 - (avg_damage_prob * 100)
        
        # Определяем общую оценку
        if overall_integrity >= 90:
            overall_grade = "ОТЛИЧНОЕ"
        elif overall_integrity >= 75:
            overall_grade = "ХОРОШЕЕ"
        elif overall_integrity >= 60:
            overall_grade = "УДОВЛЕТВОРИТЕЛЬНОЕ"
        elif overall_integrity >= 40:
            overall_grade = "ПЛОХОЕ"
        else:
            overall_grade = "КРИТИЧЕСКОЕ"
        
        return CarAnalysisReport(
            overall_integrity=overall_integrity,
            overall_grade=overall_grade,
            zones=zone_analyses,
            total_zones=len(zone_analyses),
            damaged_zones=damaged_zones,
            original_image_path=image_path
        )

class CarReportGenerator:
    """Генератор детального отчёта"""
    
    def __init__(self):
        # Цвета для разных типов повреждений
        self.damage_colors = {
            'no_damage': (0, 255, 0),      # Зелёный
            'minor_damage': (255, 165, 0),  # Оранжевый
            'major_damage': (255, 0, 0)     # Красный
        }
    
    def create_visual_report(self, report: CarAnalysisReport, output_path: str) -> str:
        """
        Создаёт визуальный отчёт с разметкой зон
        
        Returns:
            Путь к созданному изображению отчёта
        """
        # Загружаем оригинальное изображение
        image = cv2.imread(report.original_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Создаём копию для разметки
        annotated_image = image_rgb.copy()
        
        # Рисуем зоны и подписи
        for zone in report.zones:
            x1, y1, x2, y2 = zone.bbox
            color = self.damage_colors[zone.damage_class]
            
            # Рисуем прямоугольник зоны
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Подпись с информацией о зоне
            label = f"{zone.zone_name}: {zone.integrity_score:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Фон для текста
            cv2.rectangle(annotated_image, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Текст
            cv2.putText(annotated_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Сохраняем аннотированное изображение
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_bgr)
        
        return output_path
    
    def generate_text_report(self, report: CarAnalysisReport) -> str:
        """Генерирует текстовый отчёт"""
        
        report_text = f"""
🚗 ДЕТАЛЬНЫЙ ОТЧЁТ АНАЛИЗА АВТОМОБИЛЯ
{'=' * 50}

📊 ОБЩАЯ ОЦЕНКА:
• Целостность: {report.overall_integrity:.1f}%
• Состояние: {report.overall_grade}
• Повреждённых зон: {report.damaged_zones}/{report.total_zones}

🔍 АНАЛИЗ ПО ЗОНАМ:
{'─' * 30}
"""
        
        for zone in sorted(report.zones, key=lambda x: x.integrity_score):
            status_emoji = "✅" if zone.damage_class == 'no_damage' else ("⚠️" if zone.damage_class == 'minor_damage' else "❌")
            
            report_text += f"""
{status_emoji} {zone.zone_name.upper().replace('_', ' ')}:
   • Целостность: {zone.integrity_score:.1f}%
   • Тип повреждения: {zone.damage_class.replace('_', ' ').title()}
   • Уверенность: {zone.confidence:.1f}%
"""
        
        return report_text
    
    def save_json_report(self, report: CarAnalysisReport, output_path: str):
        """Сохраняет отчёт в JSON формате"""
        
        report_dict = {
            'overall_integrity': report.overall_integrity,
            'overall_grade': report.overall_grade,
            'total_zones': report.total_zones,
            'damaged_zones': report.damaged_zones,
            'original_image': report.original_image_path,
            'zones': [
                {
                    'name': zone.zone_name,
                    'integrity_score': zone.integrity_score,
                    'damage_class': zone.damage_class,
                    'damage_probability': zone.damage_probability,
                    'confidence': zone.confidence,
                    'bbox': zone.bbox
                }
                for zone in report.zones
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

# Функция для быстрого тестирования
def test_car_analysis(image_path: str, model_path: str = None):
    """
    Тестовая функция для анализа автомобиля
    
    Args:
        image_path: Путь к изображению автомобиля
        model_path: Путь к модели (опционально)
    """
    print("🚗 Запуск тестового анализа автомобиля...")
    
    # Создаём анализатор (с моделью или без)
    if model_path:
        analyzer = CarDamageAnalyzer(model_path)
    else:
        print("⚠️ Модель не указана, используется тестовый режим")
        analyzer = CarDamageAnalyzer("dummy_path")  # Будет использовать заглушку
    
    # Анализируем автомобиль
    try:
        report = analyzer.analyze_car(image_path)
        
        # Создаём генератор отчётов
        report_generator = CarReportGenerator()
        
        # Генерируем текстовый отчёт
        text_report = report_generator.generate_text_report(report)
        print(text_report)
        
        # Создаём визуальный отчёт
        output_image = "test_car_analysis_result.jpg"
        visual_report_path = report_generator.create_visual_report(report, output_image)
        print(f"\n📸 Визуальный отчёт сохранён: {visual_report_path}")
        
        # Сохраняем JSON отчёт
        json_output = "test_car_analysis_result.json"
        report_generator.save_json_report(report, json_output)
        print(f"📄 JSON отчёт сохранён: {json_output}")
        
        return report
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        return None

if __name__ == "__main__":
    # Пример использования
    print("🔧 Тестирование системы зонального анализа автомобиля")
    
    # Создаём тестовое изображение (заглушку)
    test_image = np.ones((600, 800, 3), dtype=np.uint8) * 128
    cv2.imwrite("test_car.jpg", test_image)
    
    # Запускаем тест
    test_car_analysis("test_car.jpg")