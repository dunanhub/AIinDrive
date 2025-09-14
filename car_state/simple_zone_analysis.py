"""
Упрощённая система зонального анализа с использованием существующей модели
========================================================================
Использует готовую модель без изменения архитектуры
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from car_zone_detector import CarDamageAnalyzer, test_car_analysis
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleZoneAnalyzer:
    """Простой анализатор зон без дообучения"""
    
    def __init__(self, model_path="training_results/best_model.pth"):
        self.analyzer = CarDamageAnalyzer(model_path)
        logger.info("✅ Простой зональный анализатор инициализирован")
    
    def analyze_car_with_real_model(self, image_path):
        """Анализ автомобиля с реальной моделью"""
        try:
            report = self.analyzer.analyze_car(image_path)
            return report
        except Exception as e:
            logger.error(f"❌ Ошибка анализа: {e}")
            return None
    
    def create_demo_report(self, save_files=True):
        """Создаёт демо-отчёт для демонстрации возможностей"""
        
        # Создаём демо-изображение если его нет
        if not Path("demo_car.jpg").exists():
            logger.info("Создаём демо-изображение...")
            demo_image = np.ones((600, 800, 3), dtype=np.uint8) * 220
            
            # Рисуем простую схему автомобиля
            cv2.rectangle(demo_image, (200, 150), (600, 450), (100, 100, 100), -1)
            cv2.rectangle(demo_image, (250, 50), (550, 200), (80, 80, 80), -1)
            cv2.rectangle(demo_image, (250, 200), (550, 300), (120, 120, 120), -1)
            cv2.rectangle(demo_image, (250, 350), (550, 450), (90, 90, 90), -1)
            
            # Добавляем "повреждения"
            cv2.line(demo_image, (150, 200), (180, 280), (255, 0, 0), 3)
            cv2.circle(demo_image, (400, 250), 30, (60, 60, 60), -1)
            cv2.rectangle(demo_image, (520, 380), (580, 420), (200, 0, 0), -1)
            
            cv2.imwrite("demo_car.jpg", demo_image)
            logger.info("✅ Демо-изображение создано")
        
        # Анализируем
        logger.info("🔍 Выполняем анализ с реальной моделью...")
        report = self.analyze_car_with_real_model("demo_car.jpg")
        
        if report:
            # Выводим результат
            self.print_detailed_report(report)
            
            if save_files:
                self.save_enhanced_report(report)
            
            return report
        else:
            logger.error("❌ Не удалось создать отчёт")
            return None
    
    def print_detailed_report(self, report):
        """Выводит детальный отчёт в консоль"""
        
        print("\n" + "="*60)
        print("🚗 ДЕТАЛЬНЫЙ ОТЧЁТ АНАЛИЗА АВТОМОБИЛЯ С РЕАЛЬНОЙ МОДЕЛЬЮ")
        print("="*60)
        
        print(f"\n📊 ОБЩАЯ ОЦЕНКА:")
        print(f"• Целостность кузова: {report.overall_integrity:.1f}%")
        print(f"• Общее состояние: {report.overall_grade}")
        print(f"• Повреждённых зон: {report.damaged_zones} из {report.total_zones}")
        
        # Определяем рекомендации
        if report.overall_integrity >= 90:
            recommendation = "✅ Автомобиль в отличном состоянии"
        elif report.overall_integrity >= 75:
            recommendation = "🟡 Незначительные повреждения, требует внимания"
        elif report.overall_integrity >= 50:
            recommendation = "🟠 Значительные повреждения, требует ремонта"
        else:
            recommendation = "🔴 Критические повреждения, требует срочного ремонта"
        
        print(f"• Рекомендация: {recommendation}")
        
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПО ЗОНАМ:")
        print("-" * 50)
        
        # Сортируем зоны по степени повреждения
        sorted_zones = sorted(report.zones, key=lambda x: x.integrity_score)
        
        for zone in sorted_zones:
            # Эмодзи для статуса
            if zone.damage_class == "no_damage":
                status = "✅"
                color = "ЗЕЛЁНЫЙ"
            elif zone.damage_class == "minor_damage":
                status = "⚠️"
                color = "ЖЁЛТЫЙ"
            else:
                status = "❌" 
                color = "КРАСНЫЙ"
            
            # Название зоны
            zone_name = zone.zone_name.replace('_', ' ').title()
            
            print(f"\n{status} {zone_name.upper()}:")
            print(f"   • Целостность: {zone.integrity_score:.1f}%")
            print(f"   • Статус: {zone.damage_class.replace('_', ' ').title()}")
            print(f"   • Уверенность модели: {zone.confidence*100:.1f}%")
            print(f"   • Цветовая зона на схеме: {color}")
        
        # Дополнительная аналитика
        print(f"\n📈 ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА:")
        print("-" * 30)
        
        # Подсчёт по типам повреждений
        no_damage_count = sum(1 for z in report.zones if z.damage_class == "no_damage")
        minor_damage_count = sum(1 for z in report.zones if z.damage_class == "minor_damage")
        major_damage_count = sum(1 for z in report.zones if z.damage_class == "major_damage")
        
        print(f"• Неповреждённые зоны: {no_damage_count}")
        print(f"• Зоны с незначительными повреждениями: {minor_damage_count}")
        print(f"• Зоны с серьёзными повреждениями: {major_damage_count}")
        
        # Средняя уверенность модели
        avg_confidence = np.mean([z.confidence for z in report.zones])
        print(f"• Средняя уверенность модели: {avg_confidence*100:.1f}%")
        
        # Самая и наименее повреждённые зоны
        best_zone = max(report.zones, key=lambda x: x.integrity_score)
        worst_zone = min(report.zones, key=lambda x: x.integrity_score)
        
        print(f"• Лучшая зона: {best_zone.zone_name.replace('_', ' ').title()} ({best_zone.integrity_score:.1f}%)")
        print(f"• Худшая зона: {worst_zone.zone_name.replace('_', ' ').title()} ({worst_zone.integrity_score:.1f}%)")
    
    def save_enhanced_report(self, report):
        """Сохраняет расширенный отчёт"""
        
        # Создаём расширенный JSON отчёт
        enhanced_report = {
            "analysis_summary": {
                "overall_integrity": report.overall_integrity,
                "overall_grade": report.overall_grade,
                "total_zones": report.total_zones,
                "damaged_zones": report.damaged_zones,
                "analysis_timestamp": "2025-09-14T00:00:00Z"
            },
            "zone_details": [],
            "recommendations": [],
            "model_info": {
                "model_type": "MulticlassDamageModel",
                "classes": ["no_damage", "minor_damage", "major_damage"],
                "confidence_threshold": 0.5
            }
        }
        
        # Добавляем детали по зонам
        for zone in report.zones:
            zone_detail = {
                "zone_name": zone.zone_name,
                "display_name": zone.zone_name.replace('_', ' ').title(),
                "integrity_score": round(zone.integrity_score, 2),
                "damage_class": zone.damage_class,
                "damage_probability": round(zone.damage_probability * 100, 2),
                "model_confidence": round(zone.confidence * 100, 2),
                "bbox": zone.bbox,
                "color_code": self._get_color_code(zone.damage_class)
            }
            enhanced_report["zone_details"].append(zone_detail)
        
        # Добавляем рекомендации
        if report.overall_integrity >= 90:
            enhanced_report["recommendations"].append("Автомобиль в отличном состоянии, регулярное ТО")
        elif report.overall_integrity >= 75:
            enhanced_report["recommendations"].append("Проверить зоны с повреждениями")
        elif report.overall_integrity >= 50:
            enhanced_report["recommendations"].append("Рекомендуется ремонт повреждённых зон")
        else:
            enhanced_report["recommendations"].append("Требуется срочный комплексный ремонт")
        
        # Специфичные рекомендации по зонам
        for zone in report.zones:
            if zone.damage_class == "major_damage":
                enhanced_report["recommendations"].append(f"Срочный ремонт зоны: {zone.zone_name.replace('_', ' ')}")
        
        # Сохраняем
        with open("enhanced_car_analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(enhanced_report, f, ensure_ascii=False, indent=2)
        
        logger.info("💾 Расширенный отчёт сохранён: enhanced_car_analysis_report.json")
    
    def _get_color_code(self, damage_class):
        """Возвращает цветовой код для типа повреждения"""
        color_map = {
            "no_damage": "#00FF00",      # Зелёный
            "minor_damage": "#FFA500",   # Оранжевый  
            "major_damage": "#FF0000"    # Красный
        }
        return color_map.get(damage_class, "#808080")

def run_enhanced_demo():
    """Запускает улучшенную демонстрацию"""
    
    print("🚗 ДЕМОНСТРАЦИЯ СИСТЕМЫ ЗОНАЛЬНОГО АНАЛИЗА С РЕАЛЬНОЙ МОДЕЛЬЮ")
    print("=" * 65)
    
    try:
        # Создаём анализатор
        analyzer = SimpleZoneAnalyzer()
        
        # Создаём и выводим отчёт
        report = analyzer.create_demo_report()
        
        if report:
            print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
            print("\n📁 Созданные файлы:")
            print("   • demo_car.jpg - демо-изображение автомобиля")
            print("   • test_car_analysis_result.jpg - визуальный отчёт с разметкой")
            print("   • enhanced_car_analysis_report.json - детальный JSON отчёт")
            print("   • test_car_analysis_result.json - базовый JSON отчёт")
            
            print("\n🔧 Как использовать с реальными изображениями:")
            print("   analyzer = SimpleZoneAnalyzer()")
            print("   report = analyzer.analyze_car_with_real_model('path/to/car/image.jpg')")
            print("   analyzer.print_detailed_report(report)")
            
            return True
        else:
            print("❌ Демонстрация не удалась")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Запускаем улучшенную демонстрацию
    run_enhanced_demo()