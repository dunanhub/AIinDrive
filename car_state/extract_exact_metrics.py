import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Установка стиля
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def extract_metrics_from_checkpoint():
    """Извлекает точные метрики из checkpoint файлов"""
    
    print("🔍 ИЗВЛЕЧЕНИЕ ТОЧНЫХ МЕТРИК ИЗ CHECKPOINT")
    print("=" * 60)
    
    # Пути к моделям
    models = {
        "Base Model": "training_results/best_model.pth",
        "Fine-tuned Model": "training_results/finetuned_best_model.pth"
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        if Path(model_path).exists():
            print(f"\n📦 Анализ: {model_name}")
            print(f"📁 Путь: {model_path}")
            
            try:
                # Загрузка checkpoint
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Информация о файле
                file_size = Path(model_path).stat().st_size / (1024 * 1024)
                print(f"📊 Размер файла: {file_size:.2f} MB")
                
                # Извлечение метрик
                model_metrics = {}
                
                if isinstance(checkpoint, dict):
                    print(f"📋 Содержимое checkpoint:")
                    for key in checkpoint.keys():
                        print(f"   • {key}")
                    
                    # Основные метрики
                    if 'val_f1' in checkpoint:
                        model_metrics['F1_Score'] = checkpoint['val_f1']
                        print(f"🎯 F1 Score: {checkpoint['val_f1']:.6f}")
                    
                    if 'epoch' in checkpoint:
                        model_metrics['Best_Epoch'] = checkpoint['epoch']
                        print(f"🔢 Лучшая эпоха: {checkpoint['epoch']}")
                    
                    # Детальные метрики
                    if 'val_metrics' in checkpoint:
                        val_metrics = checkpoint['val_metrics']
                        print(f"📊 Детальные метрики:")
                        
                        for metric_name, value in val_metrics.items():
                            model_metrics[metric_name] = value
                            if isinstance(value, (int, float)):
                                print(f"   • {metric_name}: {value:.4f}")
                            else:
                                print(f"   • {metric_name}: {value}")
                
                results[model_name] = model_metrics
                
            except Exception as e:
                print(f"❌ Ошибка загрузки {model_name}: {e}")
                results[model_name] = {"error": str(e)}
    
    return results

def create_metrics_visualization(results):
    """Создает визуализацию метрик"""
    
    # Определяем лучшую модель
    best_model = None
    best_f1 = 0
    
    for model_name, metrics in results.items():
        if 'F1_Score' in metrics and metrics['F1_Score'] > best_f1:
            best_f1 = metrics['F1_Score']
            best_model = model_name
    
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    print(f"🎯 ТОЧНАЯ F1 SCORE: {best_f1:.6f}")
    
    # Создание графиков
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Анализ производительности моделей\nЛучшая модель: {best_model} (F1={best_f1:.4f})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Сравнение F1 Score
    ax1 = axes[0, 0]
    model_names = []
    f1_scores = []
    colors = ['#3498db', '#e74c3c']
    
    for i, (model_name, metrics) in enumerate(results.items()):
        if 'F1_Score' in metrics:
            model_names.append(model_name)
            f1_scores.append(metrics['F1_Score'])
    
    if f1_scores:
        bars = ax1.bar(model_names, f1_scores, color=colors[:len(f1_scores)])
        ax1.set_title('F1 Score Comparison', fontweight='bold')
        ax1.set_ylabel('F1 Score')
        ax1.set_ylim(0, 1)
        
        # Добавляем значения на столбцы
        for bar, score in zip(bars, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Improvement Analysis
    ax2 = axes[0, 1]
    if len(f1_scores) == 2:
        improvement = f1_scores[1] - f1_scores[0]
        improvement_pct = (improvement / f1_scores[0]) * 100
        
        categories = ['Base F1', 'Fine-tuned F1', 'Improvement']
        values = [f1_scores[0], f1_scores[1], improvement]
        colors_imp = ['#3498db', '#e74c3c', '#2ecc71']
        
        bars = ax2.bar(categories, values, color=colors_imp)
        ax2.set_title('Performance Improvement', fontweight='bold')
        ax2.set_ylabel('F1 Score / Improvement')
        
        for bar, value in zip(bars, values):
            if value >= 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Добавляем процент улучшения
        ax2.text(1, f1_scores[1] + 0.05, f'+{improvement_pct:.1f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='green')
    
    # 3. Метрики лучшей модели
    ax3 = axes[1, 0]
    ax3.axis('off')
    
    if best_model and best_model in results:
        metrics_text = f"📊 ДЕТАЛЬНЫЕ МЕТРИКИ: {best_model}\n\n"
        
        best_metrics = results[best_model]
        for key, value in best_metrics.items():
            if isinstance(value, (int, float)):
                if 'f1' in key.lower() or 'F1' in key:
                    metrics_text += f"🎯 {key}: {value:.6f}\n"
                else:
                    metrics_text += f"📈 {key}: {value:.4f}\n"
            else:
                metrics_text += f"📝 {key}: {value}\n"
        
        ax3.text(0.1, 0.9, metrics_text, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                transform=ax3.transAxes)
    
    # 4. Progress Timeline
    ax4 = axes[1, 1]
    if len(results) >= 2:
        epochs = []
        f1_progression = []
        
        for model_name, metrics in results.items():
            if 'Best_Epoch' in metrics and 'F1_Score' in metrics:
                epochs.append(metrics['Best_Epoch'])
                f1_progression.append(metrics['F1_Score'])
        
        if len(epochs) >= 2:
            ax4.plot(epochs, f1_progression, 'o-', linewidth=3, markersize=8, 
                    color='#e74c3c', label='F1 Score Progress')
            ax4.set_title('Training Progress', fontweight='bold')
            ax4.set_xlabel('Best Epoch')
            ax4.set_ylabel('F1 Score')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            # Добавляем аннотации
            for i, (epoch, f1) in enumerate(zip(epochs, f1_progression)):
                model_name = list(results.keys())[i]
                ax4.annotate(f'{model_name}\nF1={f1:.4f}', 
                           (epoch, f1), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Сохранение графика
    save_path = 'training_results/best_model_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Графики сохранены: {save_path}")
    
    return fig

def save_detailed_report(results):
    """Сохраняет детальный отчет в JSON"""
    
    # Определяем лучшую модель
    best_model = None
    best_f1 = 0
    
    for model_name, metrics in results.items():
        if 'F1_Score' in metrics and metrics['F1_Score'] > best_f1:
            best_f1 = metrics['F1_Score']
            best_model = model_name
    
    report = {
        "analysis_summary": {
            "best_model": best_model,
            "best_f1_score": best_f1,
            "improvement_analysis": {}
        },
        "detailed_metrics": results
    }
    
    # Анализ улучшения
    if len(results) >= 2:
        models_list = list(results.items())
        if all('F1_Score' in metrics for _, metrics in models_list):
            base_f1 = models_list[0][1]['F1_Score']
            fine_f1 = models_list[1][1]['F1_Score']
            improvement = fine_f1 - base_f1
            improvement_pct = (improvement / base_f1) * 100
            
            report["analysis_summary"]["improvement_analysis"] = {
                "base_f1": base_f1,
                "finetuned_f1": fine_f1,
                "absolute_improvement": improvement,
                "percentage_improvement": improvement_pct
            }
    
    # Сохранение отчета
    report_path = 'training_results/detailed_model_analysis.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Детальный отчет сохранен: {report_path}")
    return report

if __name__ == "__main__":
    try:
        # Извлечение метрик
        results = extract_metrics_from_checkpoint()
        
        # Создание визуализации
        if results:
            fig = create_metrics_visualization(results)
            plt.show()
            
            # Сохранение отчета
            report = save_detailed_report(results)
            
            print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")
            print("📊 Графики и отчеты созданы")
            
        else:
            print("❌ Не удалось извлечь метрики из checkpoint файлов")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()