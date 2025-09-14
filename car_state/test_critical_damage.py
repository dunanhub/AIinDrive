import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
from pathlib import Path

def create_correct_model_architecture():
    """Создает правильную архитектуру модели из finetune_existing_model.py"""
    
    class ImprovedDamageClassifier(nn.Module):
        def __init__(self, num_classes=3, dropout_rate=0.3):
            super(ImprovedDamageClassifier, self).__init__()
            
            # Backbone: ResNet50
            self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Identity()  # Remove the final layer
            
            # Enhanced classifier with more layers and regularization
            self.classifier = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(dropout_rate),
                
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout_rate * 0.7),
                
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    return ImprovedDamageClassifier()

def load_both_models():
    """Загружает обе модели для сравнения"""
    
    print("🔍 ЗАГРУЗКА МОДЕЛЕЙ ДЛЯ СРАВНЕНИЯ:")
    print("=" * 50)
    
    models_info = {}
    
    # 1. Базовая модель (F1=0.7383)
    base_path = "training_results/best_model.pth"
    if os.path.exists(base_path):
        try:
            base_checkpoint = torch.load(base_path, map_location='cpu', weights_only=False)
            base_model = create_correct_model_architecture()
            
            # Пробуем загрузить state_dict
            if 'model_state_dict' in base_checkpoint:
                try:
                    base_model.load_state_dict(base_checkpoint['model_state_dict'])
                    base_model.eval()
                    models_info['base'] = {
                        'model': base_model,
                        'f1': base_checkpoint.get('val_f1', 0.7383),
                        'name': 'Базовая модель'
                    }
                    print(f"✅ Базовая модель загружена: F1={base_checkpoint.get('val_f1', 0.7383):.4f}")
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки базовой модели: {e}")
        except Exception as e:
            print(f"❌ Ошибка с базовой моделью: {e}")
    
    # 2. Fine-tuned модель (F1=0.944)
    finetuned_path = "training_results/finetuned_best_model.pth"
    if os.path.exists(finetuned_path):
        try:
            finetuned_checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)
            finetuned_model = create_correct_model_architecture()
            
            if 'model_state_dict' in finetuned_checkpoint:
                try:
                    finetuned_model.load_state_dict(finetuned_checkpoint['model_state_dict'])
                    finetuned_model.eval()
                    models_info['finetuned'] = {
                        'model': finetuned_model,
                        'f1': finetuned_checkpoint.get('best_f1', 0.944),
                        'name': 'Fine-tuned модель'
                    }
                    print(f"✅ Fine-tuned модель загружена: F1={finetuned_checkpoint.get('best_f1', 0.944):.4f}")
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки fine-tuned модели: {e}")
        except Exception as e:
            print(f"❌ Ошибка с fine-tuned моделью: {e}")
    
    return models_info

def find_severely_damaged_cars():
    """Ищет изображения сильно поврежденных машин в датасетах"""
    
    print("\n🔍 ПОИСК СИЛЬНО ПОВРЕЖДЕННЫХ АВТОМОБИЛЕЙ:")
    print("=" * 50)
    
    # Поиск в различных директориях
    search_dirs = [
        "data/RoadAccident.v2i.multiclass/train/major_damage",
        "data/RoadAccident.v2i.multiclass/valid/major_damage", 
        "data/Car damages.v3i.multiclass/train",
        "data/Car damages.v3i.multiclass/valid",
        "data/integrated_multiclass_dataset/train/major_damage",
        "data/integrated_multiclass_dataset/valid/major_damage"
    ]
    
    severely_damaged_images = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"📁 Поиск в: {search_dir}")
            
            for file in os.listdir(search_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Ищем файлы с ключевыми словами, указывающими на серьезные повреждения
                    keywords = ['accident', 'crash', 'destroyed', 'wreck', 'severe', 'major', 'total', 'damage']
                    if any(keyword in file.lower() for keyword in keywords):
                        severely_damaged_images.append(os.path.join(search_dir, file))
            
            # Также добавляем случайные изображения из папок major_damage
            if 'major_damage' in search_dir:
                images = [f for f in os.listdir(search_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    # Берем несколько случайных
                    selected = random.sample(images, min(5, len(images)))
                    for img in selected:
                        img_path = os.path.join(search_dir, img)
                        if img_path not in severely_damaged_images:
                            severely_damaged_images.append(img_path)
    
    print(f"🎯 Найдено {len(severely_damaged_images)} изображений с потенциально серьезными повреждениями")
    return severely_damaged_images

def test_on_severely_damaged_cars(models_info, test_images):
    """Тестирует модели на сильно поврежденных автомобилях"""
    
    print("\n💥 ТЕСТИРОВАНИЕ НА КРИТИЧЕСКИ ПОВРЕЖДЕННЫХ АВТОМОБИЛЯХ:")
    print("=" * 60)
    
    # Трансформации
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class_names = ['No Damage', 'Minor Damage', 'Major Damage']
    results = {}
    
    # Инициализация результатов для каждой модели
    for model_key, model_info in models_info.items():
        results[model_key] = {
            'correct_major': 0,
            'predicted_no_damage': 0,
            'predicted_minor': 0,
            'predicted_major': 0,
            'total_tested': 0,
            'predictions': []
        }
    
    print(f"🎯 Тестируем {len(test_images)} изображений...")
    
    for i, img_path in enumerate(test_images[:20]):  # Ограничиваем для быстроты
        try:
            # Загрузка изображения
            image = Image.open(img_path).convert('RGB')
            input_tensor = test_transform(image).unsqueeze(0)
            
            print(f"\n📸 Изображение {i+1}: {os.path.basename(img_path)}")
            
            # Тестирование каждой модели
            for model_key, model_info in models_info.items():
                model = model_info['model']
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(outputs, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                # Сохранение результатов
                results[model_key]['total_tested'] += 1
                results[model_key]['predictions'].append({
                    'image': os.path.basename(img_path),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities[0].tolist()
                })
                
                if predicted_class == 0:  # No Damage - ПЛОХО для разрушенной машины!
                    results[model_key]['predicted_no_damage'] += 1
                    status = "❌ КРИТИЧЕСКАЯ ОШИБКА"
                elif predicted_class == 1:  # Minor Damage - недооценка
                    results[model_key]['predicted_minor'] += 1
                    status = "⚠️ НЕДООЦЕНКА"
                elif predicted_class == 2:  # Major Damage - правильно!
                    results[model_key]['predicted_major'] += 1
                    results[model_key]['correct_major'] += 1
                    status = "✅ ПРАВИЛЬНО"
                
                print(f"  {model_info['name']}: {class_names[predicted_class]} ({confidence:.3f}) {status}")
                
        except Exception as e:
            print(f"❌ Ошибка с {img_path}: {e}")
    
    return results

def analyze_critical_damage_detection(results, models_info):
    """Анализирует способность моделей обнаруживать критические повреждения"""
    
    print("\n📊 АНАЛИЗ ОБНАРУЖЕНИЯ КРИТИЧЕСКИХ ПОВРЕЖДЕНИЙ:")
    print("=" * 60)
    
    for model_key, model_info in models_info.items():
        model_results = results[model_key]
        total = model_results['total_tested']
        
        if total == 0:
            continue
            
        print(f"\n🔹 {model_info['name']} (F1={model_info['f1']:.4f}):")
        print(f"   📊 Всего протестировано: {total}")
        
        # Основные метрики
        correct_major = model_results['correct_major']
        no_damage_errors = model_results['predicted_no_damage']
        minor_errors = model_results['predicted_minor']
        
        major_detection_rate = correct_major / total
        critical_error_rate = no_damage_errors / total  # Самые опасные ошибки
        underestimation_rate = minor_errors / total
        
        print(f"   ✅ Правильно определено как Major Damage: {correct_major}/{total} ({major_detection_rate:.1%})")
        print(f"   ❌ КРИТИЧЕСКИЕ ОШИБКИ (No Damage): {no_damage_errors}/{total} ({critical_error_rate:.1%})")
        print(f"   ⚠️ Недооценка (Minor Damage): {minor_errors}/{total} ({underestimation_rate:.1%})")
        
        # Оценка качества
        if critical_error_rate == 0:
            print("   🏆 ОТЛИЧНО: Нет критических ошибок!")
        elif critical_error_rate < 0.1:
            print("   ✅ ХОРОШО: Мало критических ошибок")
        elif critical_error_rate < 0.3:
            print("   ⚠️ УДОВЛЕТВОРИТЕЛЬНО: Есть критические ошибки")
        else:
            print("   ❌ ПЛОХО: Много критических ошибок!")
        
        if major_detection_rate > 0.8:
            print("   🎯 ОТЛИЧНО: Высокая точность обнаружения серьезных повреждений")
        elif major_detection_rate > 0.6:
            print("   📈 ХОРОШО: Приемлемая точность")
        else:
            print("   📉 ТРЕБУЕТ УЛУЧШЕНИЯ: Низкая точность")

def create_comparison_visualization(results, models_info):
    """Создает визуализацию сравнения моделей"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    model_names = []
    major_detection_rates = []
    critical_error_rates = []
    
    for model_key, model_info in models_info.items():
        model_results = results[model_key]
        total = model_results['total_tested']
        
        if total > 0:
            model_names.append(model_info['name'])
            major_detection_rates.append(model_results['correct_major'] / total)
            critical_error_rates.append(model_results['predicted_no_damage'] / total)
    
    # График 1: Точность обнаружения Major Damage
    axes[0].bar(model_names, major_detection_rates, color=['#3498db', '#e74c3c'])
    axes[0].set_title('Точность обнаружения серьезных повреждений', fontweight='bold')
    axes[0].set_ylabel('Доля правильных предсказаний')
    axes[0].set_ylim(0, 1)
    
    for i, rate in enumerate(major_detection_rates):
        axes[0].text(i, rate + 0.02, f'{rate:.1%}', ha='center', fontweight='bold')
    
    # График 2: Критические ошибки (No Damage для разрушенных машин)
    axes[1].bar(model_names, critical_error_rates, color=['#3498db', '#e74c3c'])
    axes[1].set_title('Критические ошибки\n(разрушенная машина → "нет повреждений")', fontweight='bold')
    axes[1].set_ylabel('Доля критических ошибок')
    axes[1].set_ylim(0, max(critical_error_rates) + 0.1 if critical_error_rates else 0.1)
    
    for i, rate in enumerate(critical_error_rates):
        axes[1].text(i, rate + 0.01, f'{rate:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_results/critical_damage_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 График сравнения сохранен: training_results/critical_damage_comparison.png")
    
    return fig

def main():
    """Основная функция тестирования"""
    
    print("🚗💥 ТЕСТ НА КРИТИЧЕСКИЕ ПОВРЕЖДЕНИЯ АВТОМОБИЛЕЙ")
    print("=" * 70)
    print("Проверяем, может ли модель определять действительно разрушенные машины")
    print("(в отличие от предыдущей модели, которая давала 100% целостности)")
    print()
    
    # Загрузка моделей
    models_info = load_both_models()
    
    if not models_info:
        print("❌ Не удалось загрузить модели для сравнения")
        return
    
    # Поиск сильно поврежденных автомобилей
    severely_damaged_images = find_severely_damaged_cars()
    
    if not severely_damaged_images:
        print("❌ Не найдено изображений сильно поврежденных автомобилей")
        return
    
    # Тестирование моделей
    results = test_on_severely_damaged_cars(models_info, severely_damaged_images)
    
    # Анализ результатов
    analyze_critical_damage_detection(results, models_info)
    
    # Создание визуализации
    fig = create_comparison_visualization(results, models_info)
    plt.show()
    
    # Итоговые выводы
    print("\n🎯 ИТОГОВЫЕ ВЫВОДЫ:")
    print("=" * 40)
    
    if 'finetuned' in results and results['finetuned']['total_tested'] > 0:
        finetuned_results = results['finetuned']
        critical_errors = finetuned_results['predicted_no_damage']
        total = finetuned_results['total_tested']
        correct_major = finetuned_results['correct_major']
        
        print(f"📊 Fine-tuned модель (F1=0.944):")
        print(f"   ✅ Правильно определила серьезные повреждения: {correct_major}/{total} ({correct_major/total:.1%})")
        print(f"   ❌ Критические ошибки (убитая машина → 'целая'): {critical_errors}/{total} ({critical_errors/total:.1%})")
        
        if critical_errors == 0:
            print("\n🏆 ПРЕВОСХОДНО! Новая модель НЕ делает критических ошибок!")
            print("   Теперь она корректно определяет разрушенные автомобили!")
        elif critical_errors/total < 0.1:
            print("\n✅ ОТЛИЧНО! Значительное улучшение по сравнению с предыдущей версией!")
        else:
            print("\n⚠️ Есть улучшения, но еще требуется работа над критическими случаями")
    
    print("\n✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()