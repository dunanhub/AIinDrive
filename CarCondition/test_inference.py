#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы обновленного inference.py
"""
import os
import sys
from pathlib import Path

# Добавляем директорию CarCondition в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

def test_model_loading():
    """Тестируем загрузку модели"""
    print("🔄 Тестирование загрузки модели...")
    
    try:
        from CarCondition.inference_old import get_model
        model = get_model()
        
        if model is not None:
            print("✅ Модель успешно загружена!")
            print(f"   Тип модели: {type(model)}")
            print(f"   Параметры: {sum(p.numel() for p in model.parameters()):,}")
            return True
        else:
            print("❌ Модель не загружена")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_function():
    """Тестируем функцию inference с тестовыми данными"""
    print("\n🔄 Тестирование функции inference...")
    
    try:
        from CarCondition.inference_old import analyze_image
        from PIL import Image
        import io
        import numpy as np
        
        # Создаем тестовое изображение 224x224
        test_img = Image.new('RGB', (224, 224), color=(100, 150, 200))
        
        # Преобразуем в bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Вызываем analyze_image
        result = analyze_image(img_bytes)
        
        print("✅ Функция analyze_image работает!")
        print("📊 Результат анализа:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for subkey, subvalue in value.items():
                    print(f"     {subkey}: {subvalue}")
            else:
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в функции inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_debug_function():
    """Тестируем debug функцию"""
    print("\n🔄 Тестирование debug функции...")
    
    try:
        from CarCondition.inference_old import debug_analyze_image
        from PIL import Image
        import io
        
        # Создаем тестовое изображение
        test_img = Image.new('RGB', (224, 224), color=(150, 100, 50))
        
        # Преобразуем в bytes
        img_bytes = io.BytesIO()
        test_img.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Вызываем debug_analyze_image
        result = debug_analyze_image(img_bytes)
        
        print("✅ Функция debug_analyze_image работает!")
        print("🔍 Debug информация:")
        if "debug" in result:
            for key, value in result["debug"].items():
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка в debug функции: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ ОБНОВЛЕННОГО INFERENCE.PY")
    print("=" * 50)
    
    # Проверяем окружение
    print(f"📂 Рабочая директория: {os.getcwd()}")
    print(f"🐍 Python версия: {sys.version}")
    
    # Запускаем тесты
    tests_passed = 0
    total_tests = 3
    
    if test_model_loading():
        tests_passed += 1
    
    if test_inference_function():
        tests_passed += 1
    
    if test_debug_function():
        tests_passed += 1
    
    # Итоги
    print("\n" + "=" * 50)
    print(f"📊 РЕЗУЛЬТАТ: {tests_passed}/{total_tests} тестов прошли успешно")
    
    if tests_passed == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Система готова к работе.")
    else:
        print("⚠️ Некоторые тесты не прошли. Проверьте логи выше.")
        
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)