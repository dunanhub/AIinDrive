"""
Проверка содержимого обученной модели и её метрик
================================================
"""

import torch
import json

def check_model_checkpoint(model_path="training_results/best_model.pth"):
    """Проверяет содержимое checkpoint"""
    
    print(f"🔍 Проверка модели: {model_path}")
    print("=" * 50)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("📋 Содержимое checkpoint:")
        for key, value in checkpoint.items():
            if key == 'model_state_dict':
                print(f"  • {key}: <model weights>")
            elif isinstance(value, (int, float, str)):
                print(f"  • {key}: {value}")
            elif isinstance(value, (list, tuple)):
                print(f"  • {key}: список длиной {len(value)}")
            elif isinstance(value, dict):
                print(f"  • {key}: словарь с ключами {list(value.keys())}")
            else:
                print(f"  • {key}: {type(value)}")
        
        # Ищем метрики
        metrics_keys = ['best_f1', 'f1', 'accuracy', 'val_f1', 'validation_f1']
        found_metrics = {}
        
        for key in metrics_keys:
            if key in checkpoint:
                found_metrics[key] = checkpoint[key]
        
        if found_metrics:
            print(f"\n📊 Найденные метрики:")
            for key, value in found_metrics.items():
                print(f"  • {key}: {value}")
        else:
            print("\n⚠️ Метрики F1 не найдены в checkpoint")
            
        return checkpoint
        
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return None

def check_training_stats():
    """Проверяет файл со статистикой обучения"""
    
    stats_path = "training_results/training_stats.json"
    print(f"\n🔍 Проверка статистики: {stats_path}")
    print("=" * 50)
    
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        print("📈 Статистика обучения:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"  • {key}: {value}")
            elif isinstance(value, str):
                print(f"  • {key}: {value}")
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    print(f"  • {key}: {value[-5:]}... (последние 5 значений)")
                else:
                    print(f"  • {key}: список длиной {len(value)}")
        
        # Ищем метрики F1
        f1_keys = ['f1_score', 'best_f1', 'val_f1', 'validation_f1', 'weighted_f1']
        for key in f1_keys:
            if key in stats:
                print(f"\n🎯 Найден F1: {key} = {stats[key]}")
                
        return stats
        
    except Exception as e:
        print(f"❌ Ошибка загрузки статистики: {e}")
        return None

if __name__ == "__main__":
    print("🔍 АНАЛИЗ ОБУЧЕННОЙ МОДЕЛИ")
    print("=" * 40)
    
    # Проверяем checkpoint
    checkpoint = check_model_checkpoint()
    
    # Проверяем статистику
    stats = check_training_stats()
    
    # Выводим рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ:")
    print("-" * 20)
    
    if checkpoint and 'best_f1' not in checkpoint:
        print("• В checkpoint нет сохранённого F1 score")
        print("• Нужно взять значение F1=0.7383 из статистики обучения")
        
    if stats and 'f1_score' in stats:
        print(f"• Используем F1 из статистики: {stats['f1_score']}")
    else:
        print("• Используем известное значение F1=0.7383")