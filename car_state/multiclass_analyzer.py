"""
Многоклассовый анализатор автомобилей с человеко-понятным выводом
Объединяет анализ повреждений (3 класса) и чистоты (эвристика)
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
import sys
import os

# Импорт наших модулей с правильными путями
try:
    from .multiclass_damage_model import MulticlassDamageModel, DamageClassifier
    from .dirt_analyzer import DirtAnalyzer, create_human_readable_cleanliness_report
except ImportError:
    # Fallback для прямого запуска
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from multiclass_damage_model import MulticlassDamageModel, DamageClassifier
    from dirt_analyzer import DirtAnalyzer, create_human_readable_cleanliness_report

class ComprehensiveCarAnalyzer:
    """
    Комплексный анализатор автомобилей
    Анализирует повреждения (3 класса) + чистоту (эвристика)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        
        # Инициализируем анализатор повреждений
        self.damage_classifier = DamageClassifier(model_path, device)
        
        # Инициализируем анализатор чистоты
        self.dirt_analyzer = DirtAnalyzer()
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Анализатор инициализирован на устройстве: {device}")
    
    def load_model(self, model_path: str):
        """Загрузка обученной модели"""
        self.damage_classifier.load_model(model_path)
        print(f"✅ Модель загружена: {model_path}")
    
    def analyze_image(self, image_path: str, car_name: str = "Автомобиль") -> Dict[str, Any]:
        """
        Полный анализ изображения автомобиля
        
        Args:
            image_path: Путь к изображению
            car_name: Название/модель автомобиля для отчета
            
        Returns:
            dict: Полный результат анализа
        """
        # Загружаем изображение
        image = Image.open(image_path).convert('RGB')
        
        # Анализ повреждений
        damage_result = self._analyze_damage(image)
        
        # Анализ чистоты
        cleanliness_result = self._analyze_cleanliness(image)
        
        # Создаем общий отчет
        comprehensive_report = self._create_comprehensive_report(
            car_name, damage_result, cleanliness_result
        )
        
        # Формируем финальный результат
        result = {
            'car_name': car_name,
            'damage_analysis': damage_result,
            'cleanliness_analysis': cleanliness_result,
            'comprehensive_report': comprehensive_report,
            'recommendations': self._generate_recommendations(damage_result, cleanliness_result),
            'overall_status': self._determine_overall_status(damage_result, cleanliness_result),
            'technical_summary': {
                'damage_confidence': damage_result['confidence'],
                'damage_probabilities': damage_result['class_probabilities'],
                'dirt_score': cleanliness_result['dirt_score'],
                'analysis_method': 'multiclass_damage + cv_cleanliness'
            }
        }
        
        return result
    
    def _analyze_damage(self, image: Image.Image) -> Dict[str, Any]:
        """Анализ повреждений"""
        # Применяем трансформации
        image_tensor = self.transform(image)
        
        # Получаем предсказание
        probabilities, predicted_class, confidence = self.damage_classifier.predict_single(image_tensor)
        
        # Классифицируем уровень повреждений
        damage_level = self.damage_classifier.classify_damage_level(probabilities)
        
        return damage_level
    
    def _analyze_cleanliness(self, image: Image.Image) -> Dict[str, Any]:
        """Анализ чистоты"""
        return self.dirt_analyzer.analyze_cleanliness(image)
    
    def _create_comprehensive_report(self, car_name: str, damage_result: Dict, cleanliness_result: Dict) -> str:
        """Создание комплексного человеко-понятного отчета"""
        
        # Заголовок
        report = f"🚗 {car_name} - Комплексный анализ состояния\n"
        report += "=" * 50 + "\n\n"
        
        # Анализ повреждений
        damage_icons = {
            'no_damage': '✅',
            'minor_damage': '🟡',
            'major_damage': '🔴',
            'suspected_damage': '🟠',
            'requires_inspection': '❓'
        }
        
        severity_map = {
            'none': 'отсутствуют',
            'low': 'незначительная',
            'high': 'существенная',
            'uncertain': 'требует уточнения'
        }
        
        damage_icon = damage_icons.get(damage_result['status'], '❓')
        severity_text = severity_map.get(damage_result['severity'], 'неопределенная')
        
        report += f"{damage_icon} ПОВРЕЖДЕНИЯ:\n"
        report += f"   Статус: {damage_result['description']}\n"
        report += f"   Степень: {severity_text}\n"
        
        if damage_result['status'] != 'no_damage':
            report += f"   Детали:\n"
            probs = damage_result['class_probabilities']
            if probs['minor_damage'] > 0.1:
                report += f"     • Вероятность незначительных: {probs['minor_damage']:.1%}\n"
            if probs['major_damage'] > 0.1:
                report += f"     • Вероятность существенных: {probs['major_damage']:.1%}\n"
        
        report += "\n"
        
        # Анализ чистоты
        cleanliness_icons = {
            'clean': '🧽',
            'slightly_dirty': '🟡',
            'dirty': '🟤'
        }
        
        clean_icon = cleanliness_icons.get(cleanliness_result['status'], '❓')
        
        report += f"{clean_icon} ЧИСТОТА:\n"
        report += f"   {cleanliness_result['description']}\n"
        report += f"   Рекомендация: {cleanliness_result['recommendation']}\n\n"
        
        return report
    
    def _generate_recommendations(self, damage_result: Dict, cleanliness_result: Dict) -> list:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Рекомендации по повреждениям
        if damage_result['status'] == 'major_damage':
            recommendations.extend([
                "🔧 Срочно обратитесь к специалисту для ремонта",
                "📋 Сделайте детальную фотофиксацию повреждений",
                "☎️ Свяжитесь со страховой компанией"
            ])
        elif damage_result['status'] == 'minor_damage':
            recommendations.extend([
                "🔍 Рекомендуется осмотр специалистом",
                "📝 Зафиксируйте повреждения для истории",
                "⚠️ Следите за развитием повреждений"
            ])
        elif damage_result['status'] == 'requires_inspection':
            recommendations.extend([
                "👨‍🔧 Требуется дополнительный осмотр экспертом",
                "📷 Сделайте фото с разных ракурсов",
                "🔄 Повторите анализ при лучшем освещении"
            ])
        elif damage_result['status'] == 'no_damage':
            recommendations.append("✅ Автомобиль в отличном состоянии")
        
        # Рекомендации по чистоте
        if cleanliness_result['status'] == 'dirty':
            recommendations.extend([
                "🚿 Рекомендуется мойка автомобиля",
                "🧽 Уделите внимание особо загрязненным участкам"
            ])
        elif cleanliness_result['status'] == 'slightly_dirty':
            recommendations.append("🧼 При желании можно освежить мойкой")
        
        # Общие рекомендации
        recommendations.extend([
            "🔄 Регулярно проводите визуальный осмотр",
            "📱 Сохраните результат анализа для истории"
        ])
        
        return recommendations
    
    def _determine_overall_status(self, damage_result: Dict, cleanliness_result: Dict) -> Dict[str, Any]:
        """Определение общего статуса автомобиля"""
        
        # Приоритет для статуса: повреждения важнее чистоты
        if damage_result['status'] == 'major_damage':
            overall = 'critical'
            description = "Требует немедленного внимания"
            color = 'red'
        elif damage_result['status'] == 'minor_damage':
            overall = 'attention_needed'
            description = "Требует внимания"
            color = 'yellow'
        elif damage_result['status'] == 'requires_inspection':
            overall = 'uncertain'
            description = "Необходима дополнительная проверка"
            color = 'orange'
        elif cleanliness_result['status'] == 'dirty':
            overall = 'maintenance_needed'
            description = "Нуждается в обслуживании"
            color = 'brown'
        else:
            overall = 'good'
            description = "В хорошем состоянии"
            color = 'green'
        
        return {
            'status': overall,
            'description': description,
            'color': color,
            'priority': self._get_priority_level(damage_result, cleanliness_result)
        }
    
    def _get_priority_level(self, damage_result: Dict, cleanliness_result: Dict) -> int:
        """Определение уровня приоритета (1-5, где 5 - критический)"""
        if damage_result['status'] == 'major_damage':
            return 5
        elif damage_result['status'] == 'minor_damage':
            return 3
        elif damage_result['status'] == 'requires_inspection':
            return 2
        elif cleanliness_result['status'] == 'dirty':
            return 1
        else:
            return 0
    
    def analyze_and_save_report(self, image_path: str, car_name: str = "Автомобиль", 
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Анализ с сохранением отчета в файл
        
        Args:
            image_path: Путь к изображению
            car_name: Название автомобиля
            save_path: Путь для сохранения отчета (опционально)
            
        Returns:
            dict: Результат анализа
        """
        result = self.analyze_image(image_path, car_name)
        
        if save_path:
            # Сохраняем детальный JSON отчет
            json_path = save_path.replace('.txt', '.json') if save_path.endswith('.txt') else f"{save_path}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            # Сохраняем человеко-читаемый отчет
            txt_path = save_path if save_path.endswith('.txt') else f"{save_path}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result['comprehensive_report'])
                f.write("\n📋 РЕКОМЕНДАЦИИ:\n")
                for i, rec in enumerate(result['recommendations'], 1):
                    f.write(f"{i}. {rec}\n")
            
            print(f"💾 Отчеты сохранены: {json_path}, {txt_path}")
        
        return result

def quick_analyze(image_path: str, model_path: str = "artifacts/multiclass_damage_model.pth", 
                 car_name: str = "Автомобиль") -> str:
    """
    Быстрый анализ изображения с выводом в консоль
    
    Args:
        image_path: Путь к изображению
        model_path: Путь к обученной модели
        car_name: Название автомобиля
        
    Returns:
        str: Человеко-читаемый отчет
    """
    try:
        # Создаем анализатор
        analyzer = ComprehensiveCarAnalyzer(model_path)
        
        # Анализируем
        result = analyzer.analyze_image(image_path, car_name)
        
        # Выводим отчет
        print(result['comprehensive_report'])
        print("📋 РЕКОМЕНДАЦИИ:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
        
        return result['comprehensive_report']
        
    except Exception as e:
        error_msg = f"❌ Ошибка анализа: {e}"
        print(error_msg)
        return error_msg

# Тестирование
if __name__ == "__main__":
    print("🧪 Тестирование многоклассового анализатора")
    
    # Создаем анализатор без модели (для тестирования структуры)
    analyzer = ComprehensiveCarAnalyzer()
    
    # Создаем тестовое изображение
    test_image = Image.new('RGB', (224, 224), (100, 150, 200))
    
    # Тестируем анализ чистоты
    cleanliness_result = analyzer._analyze_cleanliness(test_image)
    print(f"Результат анализа чистоты: {cleanliness_result['description']}")
    
    # Если есть обученная модель, можно протестировать полный анализ
    model_path = "artifacts/multiclass_damage_model.pth"
    if Path(model_path).exists():
        print(f"\n🚀 Тестирование с обученной моделью: {model_path}")
        # result = quick_analyze("path/to/test/image.jpg", model_path, "Test Car")
    else:
        print(f"\n⚠️ Модель не найдена: {model_path}")
        print("Запустите сначала train_multiclass_damage.py для обучения модели")
    
    print("\n✅ Структура анализатора протестирована успешно!")