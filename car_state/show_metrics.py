import torch

# Загрузка fine-tuned модели
checkpoint = torch.load('training_results/finetuned_best_model.pth', map_location='cpu', weights_only=False)

print("🏆 ТОЧНЫЕ МЕТРИКИ FINE-TUNED МОДЕЛИ:")
print("=" * 50)

# Извлечение всех метрик
metrics = {
    "Best F1": checkpoint.get("best_f1", "N/A"),
    "Standard F1": checkpoint.get("f1_standard", "N/A"),
    "Improved F1": checkpoint.get("f1_improved", "N/A"),
    "Damage Recall (Std)": checkpoint.get("damage_recall_std", "N/A"),
    "Damage Recall (Imp)": checkpoint.get("damage_recall_imp", "N/A"),
    "Major Recall (Std)": checkpoint.get("major_recall_std", "N/A"),
    "Major Recall (Imp)": checkpoint.get("major_recall_imp", "N/A"),
    "Composite (Std)": checkpoint.get("composite_std", "N/A"),
    "Composite (Imp)": checkpoint.get("composite_imp", "N/A"),
    "Epoch": checkpoint.get("epoch", "N/A")
}

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"📊 {key}: {value:.6f}")
    else:
        print(f"📊 {key}: {value}")

print("\n🎯 КРАТКОЕ РЕЗЮМЕ:")
print(f"🏆 Лучший F1 Score: {metrics['Best F1']:.6f}")
print(f"⚡ Улучшенный F1: {metrics['Improved F1']:.6f}") 
print(f"📈 Обнаружение повреждений: {metrics['Damage Recall (Imp)']:.4f}")
print(f"🔍 Обнаружение серьезных повреждений: {metrics['Major Recall (Imp)']:.4f}")