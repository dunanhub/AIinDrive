import torch
import json
import os

def check_finetuned_model():
    """Проверяем результаты fine-tuning модели"""
    
    # Путь к файлам
    base_model_path = "training_results/best_model.pth"
    finetuned_model_path = "training_results/finetuned_best_model.pth"
    
    print("🔍 ПРОВЕРКА РЕЗУЛЬТАТОВ FINE-TUNING")
    print("=" * 50)
    
    # Проверяем базовую модель
    if os.path.exists(base_model_path):
        base_checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
        if isinstance(base_checkpoint, dict) and 'val_f1' in base_checkpoint:
            base_f1 = base_checkpoint['val_f1']
            print(f"📊 Базовая модель F1-score: {base_f1:.4f}")
        else:
            print("⚠️  Базовая модель: метрики недоступны")
    
    # Проверяем fine-tuned модель
    if os.path.exists(finetuned_model_path):
        print(f"✅ Fine-tuned модель найдена")
        
        # Получаем размер файла
        size_mb = os.path.getsize(finetuned_model_path) / (1024 * 1024)
        print(f"📁 Размер файла: {size_mb:.2f} MB")
        
        try:
            finetuned_checkpoint = torch.load(finetuned_model_path, map_location='cpu', weights_only=False)
            
            if isinstance(finetuned_checkpoint, dict):
                print("📦 Содержимое checkpoint:")
                for key in finetuned_checkpoint.keys():
                    print(f"   • {key}")
                
                # Проверяем метрики
                if 'val_f1' in finetuned_checkpoint:
                    finetuned_f1 = finetuned_checkpoint['val_f1']
                    print(f"🎯 Fine-tuned модель F1-score: {finetuned_f1:.4f}")
                    
                    if 'val_f1' in locals() and 'base_f1' in locals():
                        improvement = finetuned_f1 - base_f1
                        improvement_pct = (improvement / base_f1) * 100
                        print(f"📈 Улучшение: {improvement:+.4f} ({improvement_pct:+.2f}%)")
                
                if 'val_metrics' in finetuned_checkpoint:
                    metrics = finetuned_checkpoint['val_metrics']
                    print(f"📊 Детальные метрики:")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            print(f"   • {metric}: {value:.4f}")
                
                if 'epoch' in finetuned_checkpoint:
                    print(f"🔢 Эпоха: {finetuned_checkpoint['epoch']}")
                    
        except Exception as e:
            print(f"❌ Ошибка загрузки fine-tuned модели: {e}")
    else:
        print("❌ Fine-tuned модель не найдена")
    
    # Проверяем логи fine-tuning
    log_files = [
        "finetuning_results.json",
        "finetuning_log.txt", 
        "training_results/finetuning_stats.json"
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n📄 Найден лог файл: {log_file}")
            try:
                if log_file.endswith('.json'):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        print(f"📊 Содержимое:")
                        for key, value in data.items():
                            print(f"   • {key}: {value}")
            except Exception as e:
                print(f"⚠️  Ошибка чтения {log_file}: {e}")

if __name__ == "__main__":
    check_finetuned_model()