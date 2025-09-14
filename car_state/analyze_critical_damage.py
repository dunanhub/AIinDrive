import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

def analyze_critical_damage_from_checkpoint():
    """Анализирует способность fine-tuned модели обнаруживать критические повреждения"""
    
    print("🚗💥 АНАЛИЗ ОБНАРУЖЕНИЯ КРИТИЧЕСКИХ ПОВРЕЖДЕНИЙ")
    print("=" * 60)
    print("Анализ на основе сохраненных результатов валидации")
    print()
    
    # Загрузка данных из checkpoint
    finetuned_path = "training_results/finetuned_best_model.pth"
    if not os.path.exists(finetuned_path):
        print("❌ Fine-tuned модель не найдена")
        return
    
    checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)
    
    # Извлечение данных валидации
    y_true = checkpoint.get('all_labels', [])
    y_pred_standard = checkpoint.get('all_preds_standard', [])
    y_pred_improved = checkpoint.get('all_preds_improved', [])
    y_probs = checkpoint.get('all_probs', [])
    
    if len(y_true) == 0:
        print("❌ Данные валидации не найдены в checkpoint")
        return
    
    print(f"📊 Данные валидации: {len(y_true)} образцов")
    
    class_names = ['No Damage', 'Minor Damage', 'Major Damage']
    
    # Анализ для каждого типа предсказаний
    predictions = {
        'Standard': y_pred_standard,
        'Improved': y_pred_improved
    }
    
    f1_scores = {
        'Standard': checkpoint.get('f1_standard', 0),
        'Improved': checkpoint.get('f1_improved', 0)
    }
    
    print("\n🔍 АНАЛИЗ КРИТИЧЕСКИХ ОШИБОК:")
    print("=" * 50)
    
    for pred_type, y_pred in predictions.items():
        if len(y_pred) == 0:
            continue
            
        print(f"\n🔹 {pred_type} предсказания (F1={f1_scores[pred_type]:.4f}):")
        
        # Фокусируемся на случаях, где истинный класс = Major Damage (класс 2)
        major_damage_indices = np.array(y_true) == 2
        major_damage_true = np.array(y_true)[major_damage_indices]
        major_damage_pred = np.array(y_pred)[major_damage_indices]
        
        total_major = len(major_damage_true)
        print(f"   📊 Всего случаев Major Damage: {total_major}")
        
        if total_major == 0:
            print("   ⚠️ Нет случаев Major Damage в валидационной выборке")
            continue
        
        # Анализ предсказаний для Major Damage
        correct_major = np.sum(major_damage_pred == 2)  # Правильно определили как Major
        predicted_minor = np.sum(major_damage_pred == 1)  # Недооценили как Minor
        predicted_no_damage = np.sum(major_damage_pred == 0)  # КРИТИЧЕСКАЯ ОШИБКА!
        
        print(f"   ✅ Правильно (Major → Major): {correct_major}/{total_major} ({correct_major/total_major:.1%})")
        print(f"   ⚠️ Недооценка (Major → Minor): {predicted_minor}/{total_major} ({predicted_minor/total_major:.1%})")
        print(f"   ❌ КРИТИЧЕСКАЯ ОШИБКА (Major → No Damage): {predicted_no_damage}/{total_major} ({predicted_no_damage/total_major:.1%})")
        
        # Оценка качества обнаружения критических повреждений
        critical_error_rate = predicted_no_damage / total_major
        correct_detection_rate = correct_major / total_major
        
        print(f"\n   📈 Точность обнаружения серьезных повреждений: {correct_detection_rate:.1%}")
        print(f"   📉 Уровень критических ошибок: {critical_error_rate:.1%}")
        
        # Вердикт
        if critical_error_rate == 0:
            print("   🏆 ПРЕВОСХОДНО! Нет критических ошибок!")
        elif critical_error_rate < 0.05:
            print("   ✅ ОТЛИЧНО! Очень мало критических ошибок")
        elif critical_error_rate < 0.15:
            print("   📈 ХОРОШО! Приемлемый уровень ошибок")
        else:
            print("   ⚠️ ТРЕБУЕТ УЛУЧШЕНИЯ! Много критических ошибок")
        
        if correct_detection_rate > 0.85:
            print("   🎯 ОТЛИЧНО! Высокая точность обнаружения")
        elif correct_detection_rate > 0.70:
            print("   📈 ХОРОШО! Приемлемая точность")
        else:
            print("   📉 ТРЕБУЕТ УЛУЧШЕНИЯ! Низкая точность")

def analyze_confidence_for_major_damage():
    """Анализирует уверенность модели при предсказании major damage"""
    
    print("\n🎯 АНАЛИЗ УВЕРЕННОСТИ МОДЕЛИ:")
    print("=" * 40)
    
    finetuned_path = "training_results/finetuned_best_model.pth"
    checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)
    
    y_true = np.array(checkpoint.get('all_labels', []))
    y_probs = np.array(checkpoint.get('all_probs', []))
    y_pred_improved = np.array(checkpoint.get('all_preds_improved', []))
    
    if len(y_true) == 0 or len(y_probs) == 0:
        print("❌ Данные о вероятностях не найдены")
        return
    
    # Анализируем случаи Major Damage
    major_indices = y_true == 2
    major_probs = y_probs[major_indices]
    major_preds = y_pred_improved[major_indices]
    
    if len(major_probs) == 0:
        print("❌ Нет данных о Major Damage")
        return
    
    print(f"📊 Анализ {len(major_probs)} случаев Major Damage:")
    
    # Вероятности для класса Major Damage (индекс 2)
    major_class_probs = major_probs[:, 2]
    
    # Статистика уверенности
    mean_confidence = np.mean(major_class_probs)
    median_confidence = np.median(major_class_probs)
    min_confidence = np.min(major_class_probs)
    max_confidence = np.max(major_class_probs)
    
    print(f"   📈 Средняя уверенность: {mean_confidence:.3f}")
    print(f"   📊 Медианная уверенность: {median_confidence:.3f}")
    print(f"   📉 Минимальная уверенность: {min_confidence:.3f}")
    print(f"   📈 Максимальная уверенность: {max_confidence:.3f}")
    
    # Анализ случаев с низкой уверенностью
    low_confidence_threshold = 0.5
    low_confidence_cases = major_class_probs < low_confidence_threshold
    num_low_confidence = np.sum(low_confidence_cases)
    
    print(f"\n⚠️ Случаи с низкой уверенностью (<{low_confidence_threshold}): {num_low_confidence}/{len(major_probs)} ({num_low_confidence/len(major_probs):.1%})")
    
    # Правильность предсказаний для Major Damage
    correct_major_preds = major_preds == 2
    accuracy_major = np.mean(correct_major_preds)
    
    print(f"✅ Точность на Major Damage: {accuracy_major:.1%}")
    
    # Анализ ошибок
    wrong_preds = major_preds != 2
    if np.any(wrong_preds):
        wrong_as_no_damage = np.sum(major_preds == 0)
        wrong_as_minor = np.sum(major_preds == 1)
        
        print(f"❌ Ошибки:")
        print(f"   Major → No Damage: {wrong_as_no_damage}")
        print(f"   Major → Minor Damage: {wrong_as_minor}")

def create_critical_damage_visualization():
    """Создает визуализацию результатов анализа критических повреждений"""
    
    finetuned_path = "training_results/finetuned_best_model.pth"
    checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)
    
    y_true = np.array(checkpoint.get('all_labels', []))
    y_pred_improved = np.array(checkpoint.get('all_preds_improved', []))
    
    if len(y_true) == 0:
        return
    
    # Создаем детальную confusion matrix
    cm = confusion_matrix(y_true, y_pred_improved)
    class_names = ['No Damage', 'Minor Damage', 'Major Damage']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix\nFine-tuned Model (F1=0.944)', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Выделяем критические ошибки
    axes[0].add_patch(plt.Rectangle((0, 2), 1, 1, fill=False, edgecolor='red', lw=3))
    axes[0].text(0.5, 2.5, 'КРИТИЧЕСКАЯ\nОШИБКА!', ha='center', va='center', 
                color='red', fontweight='bold', fontsize=10)
    
    # 2. Анализ ошибок для Major Damage
    major_indices = y_true == 2
    major_preds = y_pred_improved[major_indices]
    
    if len(major_preds) > 0:
        pred_counts = np.bincount(major_preds, minlength=3)
        pred_percentages = pred_counts / len(major_preds) * 100
        
        colors = ['red', 'orange', 'green']
        labels = ['→ No Damage\n(КРИТИЧЕСКАЯ ОШИБКА)', '→ Minor Damage\n(Недооценка)', '→ Major Damage\n(Правильно)']
        
        bars = axes[1].bar(range(3), pred_percentages, color=colors, alpha=0.7)
        axes[1].set_title('Предсказания для истинных\nMajor Damage случаев', fontweight='bold')
        axes[1].set_ylabel('Процент случаев')
        axes[1].set_xticks(range(3))
        axes[1].set_xticklabels(labels, rotation=0, fontsize=9)
        
        # Добавляем значения на столбцы
        for i, (bar, count, percent) in enumerate(zip(bars, pred_counts, pred_percentages)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{count}\n({percent:.1f}%)', ha='center', va='bottom', 
                        fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('training_results/critical_damage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Анализ критических повреждений сохранен: training_results/critical_damage_analysis.png")
    
    return fig

def compare_with_baseline():
    """Сравнивает с базовой моделью"""
    
    print("\n📊 СРАВНЕНИЕ С БАЗОВОЙ МОДЕЛЬЮ:")
    print("=" * 40)
    
    # Загрузка базовой модели
    base_path = "training_results/best_model.pth"
    if os.path.exists(base_path):
        base_checkpoint = torch.load(base_path, map_location='cpu', weights_only=False)
        base_f1 = base_checkpoint.get('val_f1', 0.7383)
        print(f"📊 Базовая модель F1: {base_f1:.4f}")
    else:
        base_f1 = 0.7383
        print(f"📊 Базовая модель F1: {base_f1:.4f} (из документации)")
    
    # Fine-tuned модель
    finetuned_path = "training_results/finetuned_best_model.pth"
    finetuned_checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)
    finetuned_f1 = finetuned_checkpoint.get('best_f1', 0.944)
    
    print(f"🚀 Fine-tuned модель F1: {finetuned_f1:.4f}")
    
    improvement = finetuned_f1 - base_f1
    improvement_pct = (improvement / base_f1) * 100
    
    print(f"📈 Улучшение: +{improvement:.4f} (+{improvement_pct:.1f}%)")
    
    # Специфический анализ Major Damage Recall
    major_recall_std = finetuned_checkpoint.get('major_recall_std', 0)
    major_recall_imp = finetuned_checkpoint.get('major_recall_imp', 0)
    
    print(f"\n🔍 Major Damage Recall:")
    print(f"   Standard: {major_recall_std:.1%}")
    print(f"   Improved: {major_recall_imp:.1%}")
    
    if major_recall_imp > 0.8:
        print("   🏆 ПРЕВОСХОДНО! Отлично обнаруживает серьезные повреждения")
    elif major_recall_imp > 0.7:
        print("   ✅ ХОРОШО! Приемлемое обнаружение серьезных повреждений")
    else:
        print("   ⚠️ ТРЕБУЕТ УЛУЧШЕНИЯ")

def main():
    """Основная функция анализа"""
    
    print("🔍 АНАЛИЗ СПОСОБНОСТИ МОДЕЛИ ОБНАРУЖИВАТЬ КРИТИЧЕСКИЕ ПОВРЕЖДЕНИЯ")
    print("=" * 80)
    print("Проверяем, решена ли проблема с определением 'убитых' машин как целых")
    print()
    
    # Основной анализ
    analyze_critical_damage_from_checkpoint()
    
    # Анализ уверенности
    analyze_confidence_for_major_damage()
    
    # Сравнение с базовой моделью
    compare_with_baseline()
    
    # Создание визуализации
    fig = create_critical_damage_visualization()
    plt.show()
    
    print("\n🎯 ЗАКЛЮЧЕНИЕ:")
    print("=" * 30)
    
    # Загружаем данные для финального вывода
    finetuned_path = "training_results/finetuned_best_model.pth"
    checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)
    
    y_true = np.array(checkpoint.get('all_labels', []))
    y_pred_improved = np.array(checkpoint.get('all_preds_improved', []))
    
    if len(y_true) > 0:
        major_indices = y_true == 2
        major_preds = y_pred_improved[major_indices]
        
        if len(major_preds) > 0:
            critical_errors = np.sum(major_preds == 0)  # Major → No Damage
            total_major = len(major_preds)
            correct_major = np.sum(major_preds == 2)
            
            print(f"📊 Итоги для Major Damage ({total_major} случаев):")
            print(f"   ✅ Правильно определено: {correct_major}/{total_major} ({correct_major/total_major:.1%})")
            print(f"   ❌ Критические ошибки: {critical_errors}/{total_major} ({critical_errors/total_major:.1%})")
            
            if critical_errors == 0:
                print("\n🏆 РЕШЕНИЕ НАЙДЕНО! Модель больше НЕ называет разрушенные машины целыми!")
                print("   Проблема с определением 'убитых' машин как 100% целых РЕШЕНА!")
            elif critical_errors/total_major < 0.05:
                print("\n✅ ЗНАЧИТЕЛЬНОЕ УЛУЧШЕНИЕ! Критических ошибок стало очень мало!")
            elif critical_errors/total_major < 0.15:
                print("\n📈 ЕСТЬ УЛУЧШЕНИЯ! Ситуация лучше, но можно еще улучшить")
            else:
                print("\n⚠️ ПРОБЛЕМА ЧАСТИЧНО РЕШЕНА. Требуется дополнительная работа")
    
    print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    main()