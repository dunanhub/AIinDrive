#!/usr/bin/env python3
"""
Тестирование обновленной системы анализа v2.0
"""
import os
import sys
from pathlib import Path
from inference import analyze_image, debug_analyze_image

def test_with_sample_image():
    """Тест с примером изображения из публичной папки"""
    # Путь к тестовому изображению из фронтенда
    test_image_path = "../FrontEnd/car-condition-frontend/public/car.jpg"
    
    if not os.path.exists(test_image_path):
        print("⚠️ Тестовое изображение не найдено. Проверяем альтернативные пути...")
        
        # Проверяем другие возможные пути
        alternative_paths = [
            "../FrontEnd/car-condition-frontend/public/car.png",
            "test_image.jpg",
            "../test_car.jpg"
        ]
        
        test_image_path = None
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                test_image_path = alt_path
                break
        
        if not test_image_path:
            print("❌ Не найдено ни одного тестового изображения")
            print("📝 Создайте тестовое изображение или поместите car.jpg в public папку")
            return False
    
    print(f"🧪 Тестирование системы анализа v2.0")
    print(f"📂 Используется изображение: {test_image_path}")
    print("="*60)
    
    try:
        # Читаем изображение
        with open(test_image_path, 'rb') as f:
            img_bytes = f.read()
        
        print("🔍 Запуск продвинутого анализа...")
        result = debug_analyze_image(img_bytes)
        
        print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("="*60)
        
        # Основные результаты
        print(f"🧼 Загрязнение: {result.get('dirty', 'N/A')}")
        print(f"   📊 Вероятность: {result.get('dirty_prob', 0)*100:.1f}%")
        print(f"   🏷️ Статус: {result.get('dirt_status', 'N/A')}")
        print(f"   {result.get('dirt_emoji', '🤔')} Оценка: {result.get('dirt_score', 0):.1f}/10")
        print(f"   💡 Рекомендация: {result.get('dirt_recommendation', 'N/A')}")
        
        print(f"\n🔧 Повреждения: {result.get('damaged', 'N/A')}")
        print(f"   📊 Вероятность: {result.get('damaged_prob', 0)*100:.1f}%")
        print(f"   🎯 Класс: {result.get('predicted_class', 'N/A')}")
        print(f"   🔍 Уверенность: {result.get('confidence', 0)*100:.1f}%")
        
        # Анализ для такси
        print(f"\n🚕 ОЦЕНКА ДЛЯ ТАКСИ:")
        print(f"   📋 Статус: {result.get('taxi_status', 'N/A')}")
        taxi_recs = result.get('taxi_recommendations', [])
        if taxi_recs:
            for i, rec in enumerate(taxi_recs[:3], 1):  # Показываем первые 3
                print(f"   {i}. {rec}")
        
        # Экспертная оценка
        print(f"\n🤖 ЭКСПЕРТНАЯ ОЦЕНКА:")
        expert_assess = result.get('expert_assessment', [])
        if expert_assess:
            for assessment in expert_assess:
                print(f"   • {assessment}")
        
        # Технические детали
        print(f"\n🔧 ТЕХНИЧЕСКИЕ ДЕТАЛИ:")
        print(f"   📁 Модель загружена: {result.get('model_available', False)}")
        print(f"   🖼️ OpenCV доступен: {result.get('cv_available', False)}")
        print(f"   📋 Версия анализа: {result.get('analysis_version', 'N/A')}")
        
        debug_info = result.get('debug', {})
        if debug_info:
            print(f"   📊 Debug: {debug_info}")
        
        print("\n✅ Тест завершен успешно!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Проверка зависимостей системы"""
    print("🔍 Проверка зависимостей системы v2.0...")
    
    missing_deps = []
    
    # Основные зависимости
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
        print("❌ PyTorch не найден")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
        print("❌ OpenCV не найден")
    
    try:
        import scipy
        print(f"✅ SciPy: {scipy.__version__}")
    except ImportError:
        missing_deps.append("scipy")
        print("❌ SciPy не найден")
    
    try:
        import skimage
        print(f"✅ scikit-image: {skimage.__version__}")
    except ImportError:
        missing_deps.append("scikit-image")
        print("❌ scikit-image не найден")
    
    if missing_deps:
        print(f"\n⚠️ Отсутствующие зависимости: {', '.join(missing_deps)}")
        print("📥 Установите их командой:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    else:
        print("\n✅ Все зависимости установлены!")
        return True

def main():
    """Главная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ СИСТЕМЫ ИИ-ДИАГНОСТИКИ v2.0")
    print("="*60)
    
    # 1. Проверка зависимостей
    if not check_dependencies():
        print("\n❌ Некоторые зависимости отсутствуют. Продолжаем с ограничениями...")
    
    print("\n" + "="*60)
    
    # 2. Тестирование модели
    if not test_with_sample_image():
        print("\n❌ Тестирование не удалось")
        return False
    
    print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
    print("🚀 Система ИИ-диагностики v2.0 готова к работе!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)