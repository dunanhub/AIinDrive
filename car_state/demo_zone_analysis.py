"""
Тестовый скрипт для демонстрации системы зонального анализа автомобиля
========================================================================
Показывает как работает детекция зон и анализ повреждений
"""

import sys
import os
import cv2
import numpy as np
from car_zone_detector import CarDamageAnalyzer, test_car_analysis
import matplotlib.pyplot as plt

def create_demo_car_image(filename="demo_car.jpg"):
    """Создаёт демонстрационное изображение автомобиля"""
    
    # Создаём изображение 800x600
    img = np.ones((600, 800, 3), dtype=np.uint8) * 220  # Светло-серый фон
    
    # Рисуем простую схему автомобиля
    # Основной кузов
    cv2.rectangle(img, (200, 150), (600, 450), (100, 100, 100), -1)  # Основной кузов
    
    # Крыша
    cv2.rectangle(img, (250, 50), (550, 200), (80, 80, 80), -1)
    
    # Капот
    cv2.rectangle(img, (250, 200), (550, 300), (120, 120, 120), -1)
    
    # Багажник  
    cv2.rectangle(img, (250, 350), (550, 450), (90, 90, 90), -1)
    
    # Колёса
    cv2.circle(img, (250, 450), 40, (50, 50, 50), -1)  # Переднее левое
    cv2.circle(img, (550, 450), 40, (50, 50, 50), -1)  # Переднее правое
    cv2.circle(img, (250, 150), 40, (50, 50, 50), -1)  # Заднее левое
    cv2.circle(img, (550, 150), 40, (50, 50, 50), -1)  # Заднее правое
    
    # Добавляем "повреждения" для демонстрации
    # Царапина на левой стороне
    cv2.line(img, (150, 200), (180, 280), (255, 0, 0), 3)
    cv2.line(img, (160, 190), (190, 270), (255, 0, 0), 2)
    
    # Вмятина на капоте (тёмное пятно)
    cv2.circle(img, (400, 250), 30, (60, 60, 60), -1)
    
    # Повреждение на задней части
    cv2.rectangle(img, (520, 380), (580, 420), (200, 0, 0), -1)
    
    # Сохраняем изображение
    cv2.imwrite(filename, img)
    print(f"✅ Демо-изображение создано: {filename}")
    return filename

def run_demo_analysis():
    """Запускает демонстрацию анализа"""
    
    print("🚗 ДЕМОНСТРАЦИЯ СИСТЕМЫ ЗОНАЛЬНОГО АНАЛИЗА АВТОМОБИЛЯ")
    print("=" * 60)
    
    # 1. Создаём демо-изображение
    demo_image = create_demo_car_image()
    
    # 2. Проверяем наличие обученной модели
    model_path = "training_results/best_model.pth"
    
    if os.path.exists(model_path):
        print(f"✅ Найдена обученная модель: {model_path}")
        use_real_model = True
    else:
        print("⚠️ Обученная модель не найдена, используется тестовый режим")
        model_path = None
        use_real_model = False
    
    # 3. Запускаем анализ
    print("\n🔍 Запуск анализа...")
    try:
        report = test_car_analysis(demo_image, model_path)
        
        if report:
            print("\n🎉 Анализ завершён успешно!")
            print(f"📊 Общая целостность: {report.overall_integrity:.1f}%")
            print(f"🏆 Оценка: {report.overall_grade}")
            print(f"🔧 Повреждённых зон: {report.damaged_zones}/{report.total_zones}")
            
            return True
        else:
            print("❌ Анализ не удался")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        return False

def demo_zone_detection_only():
    """Демонстрация только детекции зон (без анализа повреждений)"""
    
    print("\n🎯 ДЕМОНСТРАЦИЯ ДЕТЕКЦИИ ЗОН")
    print("-" * 40)
    
    from car_zone_detector import CarZoneDetector
    
    # Создаём детектор
    detector = CarZoneDetector()
    
    # Загружаем демо-изображение
    demo_image = "demo_car.jpg"
    if not os.path.exists(demo_image):
        create_demo_car_image(demo_image)
    
    image = cv2.imread(demo_image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Детектируем зоны
    zones = detector.detect_zones(image_rgb)
    
    print(f"🔍 Обнаружено зон: {len(zones)}")
    
    # Визуализируем зоны
    annotated = image_rgb.copy()
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255), (128, 128, 128)]
    
    for i, (zone_name, (x1, y1, x2, y2)) in enumerate(zones.items()):
        color = colors[i % len(colors)]
        
        # Рисуем прямоугольник
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Подписываем зону
        cv2.putText(annotated, zone_name, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        print(f"  📍 {zone_name}: ({x1}, {y1}) -> ({x2}, {y2})")
    
    # Сохраняем результат
    output_path = "zones_detection_demo.jpg"
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, annotated_bgr)
    print(f"💾 Результат сохранён: {output_path}")

def show_comparison_with_without_model():
    """Показывает разницу между анализом с моделью и без неё"""
    
    print("\n📊 СРАВНЕНИЕ: С МОДЕЛЬЮ vs БЕЗ МОДЕЛИ")
    print("-" * 45)
    
    demo_image = "demo_car.jpg"
    if not os.path.exists(demo_image):
        create_demo_car_image(demo_image)
    
    # Анализ без модели (тестовый режим)
    print("\n1️⃣ Анализ БЕЗ модели (случайные данные):")
    from car_zone_detector import CarDamageAnalyzer
    
    analyzer_dummy = CarDamageAnalyzer("dummy_path")
    report_dummy = analyzer_dummy.analyze_car(demo_image)
    
    print(f"   Целостность: {report_dummy.overall_integrity:.1f}%")
    print(f"   Оценка: {report_dummy.overall_grade}")
    
    # Анализ с моделью (если доступна)
    model_path = "training_results/best_model.pth"
    if os.path.exists(model_path):
        print("\n2️⃣ Анализ С моделью:")
        try:
            analyzer_real = CarDamageAnalyzer(model_path)
            report_real = analyzer_real.analyze_car(demo_image)
            
            print(f"   Целостность: {report_real.overall_integrity:.1f}%")
            print(f"   Оценка: {report_real.overall_grade}")
            
            # Сравнение по зонам
            print("\n🔍 Сравнение по зонам:")
            for dummy_zone, real_zone in zip(report_dummy.zones, report_real.zones):
                print(f"   {dummy_zone.zone_name}:")
                print(f"     Без модели: {dummy_zone.integrity_score:.1f}% ({dummy_zone.damage_class})")
                print(f"     С моделью:  {real_zone.integrity_score:.1f}% ({real_zone.damage_class})")
                
        except Exception as e:
            print(f"   ❌ Ошибка загрузки модели: {e}")
    else:
        print("\n2️⃣ Модель не найдена, пропускаем сравнение")

if __name__ == "__main__":
    print("🔧 СИСТЕМА ЗОНАЛЬНОГО АНАЛИЗА АВТОМОБИЛЯ - ДЕМО")
    print("=" * 55)
    
    # Запускаем демонстрации
    try:
        # 1. Демо детекции зон
        demo_zone_detection_only()
        
        # 2. Полный анализ  
        run_demo_analysis()
        
        # 3. Сравнение с моделью и без
        show_comparison_with_without_model()
        
        print("\n✅ Демонстрация завершена!")
        print("📁 Проверьте созданные файлы:")
        print("   • demo_car.jpg - демо-изображение")
        print("   • zones_detection_demo.jpg - визуализация зон")
        print("   • test_car_analysis_result.jpg - результат анализа")
        print("   • test_car_analysis_result.json - JSON отчёт")
        
    except Exception as e:
        print(f"❌ Ошибка в демонстрации: {e}")
        import traceback
        traceback.print_exc()