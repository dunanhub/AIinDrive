import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import os
import random
from pathlib import Path

def validate_model_performance():
    """Комплексная валидация модели для проверки достоверности высоких метрик"""
    
    print("🔍 ВАЛИДАЦИЯ МОДЕЛИ: ПРОВЕРКА ДОСТОВЕРНОСТИ МЕТРИК")
    print("=" * 70)
    
    # Загрузка checkpoint
    checkpoint_path = 'training_results/finetuned_best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("📊 АНАЛИЗ СОХРАНЕННЫХ ДАННЫХ:")
    print(f"🎯 F1 Score: {checkpoint.get('best_f1', 'N/A'):.6f}")
    print(f"📅 Эпоха: {checkpoint.get('epoch', 'N/A')}")
    
    # 1. Анализ Confusion Matrix
    if 'all_labels' in checkpoint and 'all_preds_improved' in checkpoint:
        print("\n📊 АНАЛИЗ CONFUSION MATRIX:")
        
        y_true = checkpoint['all_labels']
        y_pred = checkpoint['all_preds_improved']
        
        print(f"📈 Размер тестовой выборки: {len(y_true)} образцов")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        class_names = ['No Damage', 'Minor Damage', 'Major Damage']
        
        print("\n🔍 Confusion Matrix:")
        print("     Predicted:")
        print("        0    1    2")
        for i, row in enumerate(cm):
            print(f"True {i}: {row}")
        
        # Детальный анализ по классам
        print("\n📊 АНАЛИЗ ПО КЛАССАМ:")
        total_samples = len(y_true)
        unique, counts = np.unique(y_true, return_counts=True)
        
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            percentage = (count / total_samples) * 100
            print(f"  {class_names[class_idx]}: {count} образцов ({percentage:.1f}%)")
        
        # Проверка на переобучение: анализ точности по классам
        print("\n🔍 ДЕТАЛЬНАЯ ТОЧНОСТЬ ПО КЛАССАМ:")
        for i in range(len(class_names)):
            true_positives = cm[i, i]
            total_true = cm[i, :].sum()
            total_predicted = cm[:, i].sum()
            
            precision = true_positives / total_predicted if total_predicted > 0 else 0
            recall = true_positives / total_true if total_true > 0 else 0
            
            print(f"  {class_names[i]}:")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    Истинно положительные: {true_positives}/{total_true}")
        
        # Визуализация Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Fine-tuned Model', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('training_results/confusion_matrix_validation.png', dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix сохранена: training_results/confusion_matrix_validation.png")
        
        # 2. Проверка на дисбаланс классов
        print("\n⚖️ ПРОВЕРКА ДИСБАЛАНСА КЛАССОВ:")
        class_distribution = np.bincount(y_true)
        max_class = np.max(class_distribution)
        min_class = np.min(class_distribution)
        imbalance_ratio = max_class / min_class
        
        print(f"📊 Соотношение классов: {class_distribution}")
        print(f"⚖️ Коэффициент дисбаланса: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Сильный дисбаланс классов!")
            print("   Высокие метрики могут быть обманчивыми")
        elif imbalance_ratio > 3:
            print("⚡ Умеренный дисбаланс классов")
        else:
            print("✅ Хороший баланс классов")
    
    # 3. Анализ кривой обучения
    if 'val_f1_scores' in checkpoint:
        print("\n📈 АНАЛИЗ КРИВОЙ ОБУЧЕНИЯ:")
        val_f1_scores = checkpoint['val_f1_scores']
        
        print(f"📊 Эпох обучения: {len(val_f1_scores)}")
        print(f"🎯 Начальный F1: {val_f1_scores[0]:.4f}")
        print(f"🏆 Финальный F1: {val_f1_scores[-1]:.4f}")
        print(f"📈 Улучшение: {val_f1_scores[-1] - val_f1_scores[0]:.4f}")
        
        # Проверка на переобучение
        if len(val_f1_scores) > 5:
            last_5 = val_f1_scores[-5:]
            if all(score >= 0.9 for score in last_5):
                print("⚠️ ВОЗМОЖНОЕ ПЕРЕОБУЧЕНИЕ: F1 > 0.9 последние 5 эпох")
            elif np.std(last_5) < 0.01:
                print("✅ Стабильная сходимость")
            else:
                print("📊 Нормальная вариативность")
        
        # Визуализация кривой обучения
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(val_f1_scores) + 1), val_f1_scores, 'b-o', linewidth=2, markersize=6)
        plt.title('F1 Score во время обучения', fontsize=14, fontweight='bold')
        plt.xlabel('Эпоха')
        plt.ylabel('F1 Score')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Добавляем горизонтальные линии для справки
        plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Отличный результат (0.9)')
        plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Хороший результат (0.8)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_results/learning_curve_validation.png', dpi=300, bbox_inches='tight')
        print(f"📈 Кривая обучения сохранена: training_results/learning_curve_validation.png")
    
    # 4. Проверка параметров inference
    if 'inference_params' in checkpoint:
        print("\n🔧 ПАРАМЕТРЫ INFERENCE:")
        params = checkpoint['inference_params']
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    return checkpoint

def create_model_architecture():
    """Воссоздает архитектуру модели для загрузки весов"""
    
    # Создание модели с правильной архитектурой
    class DamageClassifier(nn.Module):
        def __init__(self, num_classes=3):
            super(DamageClassifier, self).__init__()
            self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Identity()  # Убираем последний слой
            
            # Добавляем классификатор
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features)
    
    return DamageClassifier()

def test_model_on_sample_images():
    """Тестирует модель на образцах изображений для практической проверки"""
    
    print("\n🖼️ ТЕСТИРОВАНИЕ НА ОБРАЗЦАХ ИЗОБРАЖЕНИЙ:")
    
    try:
        # Загрузка модели
        model = create_model_architecture()
        checkpoint = torch.load('training_results/finetuned_best_model.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Трансформации для тестирования
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Поиск тестовых изображений
        test_dirs = [
            "data/integrated_multiclass_dataset/valid",
            "data/car_damage.v8i.multiclass/valid", 
            "data/Car damages.v3i.multiclass/valid"
        ]
        
        class_names = ['No Damage', 'Minor Damage', 'Major Damage']
        test_results = []
        
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                print(f"\n📁 Тестирование в: {test_dir}")
                
                for class_name in class_names:
                    class_dir = os.path.join(test_dir, class_name.lower().replace(' ', '_'))
                    if os.path.exists(class_dir):
                        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                        
                        if len(images) > 0:
                            # Тестируем несколько случайных изображений
                            test_images = random.sample(images, min(3, len(images)))
                            
                            for img_name in test_images:
                                img_path = os.path.join(class_dir, img_name)
                                
                                try:
                                    # Загрузка и предобработка изображения
                                    image = Image.open(img_path).convert('RGB')
                                    input_tensor = test_transform(image).unsqueeze(0)
                                    
                                    # Предсказание
                                    with torch.no_grad():
                                        outputs = model(input_tensor)
                                        probabilities = torch.softmax(outputs, dim=1)
                                        predicted_class = torch.argmax(outputs, dim=1).item()
                                        confidence = probabilities[0][predicted_class].item()
                                    
                                    true_class = class_names.index(class_name)
                                    is_correct = predicted_class == true_class
                                    
                                    result = {
                                        'image': img_name,
                                        'true_class': class_name,
                                        'predicted_class': class_names[predicted_class],
                                        'confidence': confidence,
                                        'correct': is_correct
                                    }
                                    test_results.append(result)
                                    
                                    status = "✅" if is_correct else "❌"
                                    print(f"  {status} {img_name}: {class_name} → {class_names[predicted_class]} ({confidence:.3f})")
                                    
                                except Exception as e:
                                    print(f"  ⚠️ Ошибка с {img_name}: {e}")
                break  # Используем только первый найденный директорий
        
        # Анализ результатов тестирования
        if test_results:
            correct_predictions = sum(1 for r in test_results if r['correct'])
            total_predictions = len(test_results)
            accuracy = correct_predictions / total_predictions
            
            print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
            print(f"🎯 Точность на образцах: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
            print(f"📈 Средняя уверенность: {np.mean([r['confidence'] for r in test_results]):.4f}")
            
            # Анализ по классам
            for class_name in class_names:
                class_results = [r for r in test_results if r['true_class'] == class_name]
                if class_results:
                    class_accuracy = sum(1 for r in class_results if r['correct']) / len(class_results)
                    print(f"  {class_name}: {class_accuracy:.4f} ({len(class_results)} образцов)")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
        return []

def generate_validation_report():
    """Генерирует финальный отчет о валидации"""
    
    print("\n📋 ФИНАЛЬНЫЙ ОТЧЕТ ВАЛИДАЦИИ:")
    print("=" * 50)
    
    # Загрузка данных
    checkpoint = torch.load('training_results/finetuned_best_model.pth', map_location='cpu', weights_only=False)
    
    # Основные выводы
    conclusions = []
    
    # Анализ F1 score
    f1_score = checkpoint.get('best_f1', 0)
    if f1_score > 0.95:
        conclusions.append("⚠️ F1 > 0.95: Возможно переобучение")
    elif f1_score > 0.9:
        conclusions.append("✅ F1 > 0.9: Отличный результат")
    elif f1_score > 0.8:
        conclusions.append("✅ F1 > 0.8: Хороший результат")
    else:
        conclusions.append("📊 F1 < 0.8: Требует улучшения")
    
    # Анализ эпох обучения
    epoch = checkpoint.get('epoch', 0)
    if epoch > 15:
        conclusions.append("📈 Обучение > 15 эпох: Модель хорошо сходится")
    
    # Анализ Damage Recall
    damage_recall = checkpoint.get('damage_recall_imp', 0)
    if damage_recall > 0.95:
        conclusions.append("🎯 Damage Recall > 95%: Отлично обнаруживает повреждения")
    
    # Анализ Major Recall  
    major_recall = checkpoint.get('major_recall_imp', 0)
    if major_recall > 0.85:
        conclusions.append("🔍 Major Recall > 85%: Хорошо классифицирует серьезные повреждения")
    
    print("\n🎯 ВЫВОДЫ:")
    for conclusion in conclusions:
        print(f"  {conclusion}")
    
    print(f"\n📊 РЕКОМЕНДАЦИИ:")
    if f1_score > 0.92:
        print("  ✅ Модель готова к продакшену")
        print("  📋 Рекомендуется дополнительное тестирование на новых данных")
        print("  🔍 Проверьте работу на edge cases")
    else:
        print("  📈 Модель показывает хорошие результаты")
        print("  🔄 Возможно дальнейшее улучшение через больше данных")

if __name__ == "__main__":
    try:
        # Валидация модели
        checkpoint = validate_model_performance()
        
        # Тестирование на образцах
        test_results = test_model_on_sample_images()
        
        # Генерация отчета
        generate_validation_report()
        
        plt.show()
        
        print("\n✅ ВАЛИДАЦИЯ ЗАВЕРШЕНА!")
        print("📊 Графики и анализ сохранены в training_results/")
        
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
        import traceback
        traceback.print_exc()