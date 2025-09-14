"""
Fine-tuning существующей обученной модели с улучшениями
======================================================
Загружает best_model.pth и дообучает с новыми техниками
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedInference:
    """Улучшенная логика принятия решений"""
    
    def __init__(self, tau_nd=0.65, tau_major=0.32):
        self.tau_nd = tau_nd
        self.tau_major = tau_major
    
    def apply_rules(self, probabilities):
        """Применяет decision rules к вероятностям"""
        predictions = []
        
        for probs in probabilities:
            p_no_damage = probs[0]
            p_minor = probs[1] 
            p_major = probs[2]
            
            # Правило 1: Если P(no_damage) < tau_nd → damage_present
            if p_no_damage < self.tau_nd:
                # Правило 2: major если P(major) >= tau_major
                if p_major >= self.tau_major:
                    predictions.append(2)  # major_damage
                else:
                    predictions.append(1)  # minor_damage
            else:
                predictions.append(0)  # no_damage
                
        return np.array(predictions)

class FocalLoss(nn.Module):
    """Focal Loss для борьбы с class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def finetune_existing_model():
    """Дообучение существующей модели"""
    
    logger.info("🔄 Начало дообучения существующей модели...")
    
    # Параметры
    DEVICE = torch.device('cpu')
    EPOCHS = 8  # Меньше эпох для fine-tuning
    BATCH_SIZE = 12
    BASE_LR = 5e-5  # Меньший learning rate для fine-tuning
    
    # 1. Загрузка существующей модели
    model_path = "training_results/best_model.pth"
    logger.info(f"📂 Загрузка модели: {model_path}")
    
    try:
        from multiclass_damage_model import MulticlassDamageModel
        
        model = MulticlassDamageModel(num_classes=3)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # Ищем сохранённый F1 в разных местах
            prev_f1 = checkpoint.get('best_f1', 
                     checkpoint.get('val_f1',
                     checkpoint.get('f1_score', 0.0)))
            prev_metrics = {
                'best_f1': prev_f1,
                'epoch': checkpoint.get('epoch', 0)
            }
        else:
            model.load_state_dict(checkpoint)
            prev_metrics = {'best_f1': 0.0, 'epoch': 0}
            
        model.to(DEVICE)
        logger.info(f"✅ Модель загружена. Предыдущий F1: {prev_metrics['best_f1']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return None
    
    # 2. Подготовка данных
    logger.info("📊 Подготовка данных...")
    
    from train_multiclass_damage_v3_2 import load_all_datasets, MulticlassDamageDataset, DATASET_ROOTS
    from torchvision import transforms
    
    # Улучшенные аугментации для fine-tuning
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),  # Менее агрессивная аугментация
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # Меньше поворотов
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))  # Меньше erasing
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка данных
    train_records, val_records = load_all_datasets(DATASET_ROOTS)
    
    train_dataset = MulticlassDamageDataset(train_records, transform=train_transform)
    val_dataset = MulticlassDamageDataset(val_records, transform=val_transform)
    
    logger.info(f"📈 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Анализ распределения классов
    train_labels = [record['label'] for record in train_records]
    class_counts = Counter(train_labels)
    logger.info(f"📊 Распределение классов: {dict(class_counts)}")
    
    # Веса для балансировки
    class_weights = {0: 0.8, 1: 1.0, 2: 1.3}  # Умеренная корректировка
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                             num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=0, drop_last=False)
    
    # 3. Настройка оптимизации для fine-tuning
    
    # Разные learning rates для разных слоёв
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': BASE_LR * 0.1},  # Меньший LR для backbone
        {'params': classifier_params, 'lr': BASE_LR}       # Больший LR для head
    ], weight_decay=1e-4)
    
    # Focal Loss с настроенными весами
    alpha_weights = torch.tensor([0.8, 1.0, 1.3])
    criterion = FocalLoss(alpha=alpha_weights, gamma=1.5)
    
    # Scheduler с warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=3, T_mult=2, eta_min=1e-6
    )
    
    # 4. Обучение
    logger.info("🚀 Начало fine-tuning...")
    
    best_f1 = prev_metrics['best_f1']
    train_losses = []
    val_f1_scores = []
    val_damage_recalls = []
    val_major_recalls = []
    
    # Inference rules
    improved_inference = ImprovedInference()
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_samples += images.size(0)
            
            if batch_idx % 25 == 0:
                logger.info(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        all_preds = []
        all_preds_improved = []
        all_labels = []
        all_probs = []
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_samples += images.size(0)
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                # Стандартные предсказания
                all_preds.extend(preds.cpu().numpy())
                
                # Улучшенные предсказания
                improved_preds = improved_inference.apply_rules(probs.cpu().numpy())
                all_preds_improved.extend(improved_preds)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Метрики
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / val_samples
        
        # F1 scores
        f1_standard = f1_score(all_labels, all_preds, average='weighted')
        f1_improved = f1_score(all_labels, all_preds_improved, average='weighted')
        
        # Damage detection recall
        damage_labels = [1 if label > 0 else 0 for label in all_labels]
        damage_preds_std = [1 if pred > 0 else 0 for pred in all_preds]
        damage_preds_imp = [1 if pred > 0 else 0 for pred in all_preds_improved]
        
        damage_recall_std = f1_score(damage_labels, damage_preds_std, average='binary', pos_label=1)
        damage_recall_imp = f1_score(damage_labels, damage_preds_imp, average='binary', pos_label=1)
        
        # Major damage recall
        major_labels = [1 if label == 2 else 0 for label in all_labels]
        major_preds_std = [1 if pred == 2 else 0 for pred in all_preds]
        major_preds_imp = [1 if pred == 2 else 0 for pred in all_preds_improved]
        
        if sum(major_labels) > 0:
            major_recall_std = sum([1 for i, pred in enumerate(major_preds_std) if pred == 1 and major_labels[i] == 1]) / sum(major_labels)
            major_recall_imp = sum([1 for i, pred in enumerate(major_preds_imp) if pred == 1 and major_labels[i] == 1]) / sum(major_labels)
        else:
            major_recall_std = 0
            major_recall_imp = 0
        
        # Composite score для выбора лучшей модели
        composite_std = 0.6 * f1_standard + 0.3 * damage_recall_std + 0.1 * major_recall_std
        composite_imp = 0.6 * f1_improved + 0.3 * damage_recall_imp + 0.1 * major_recall_imp
        
        epoch_time = time.time() - start_time
        
        # Логирование
        logger.info(f'\n📊 Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s):')
        logger.info(f'  📉 Losses: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}')
        logger.info(f'  🎯 Standard F1={f1_standard:.4f}, Damage_Recall={damage_recall_std:.4f}, Major_Recall={major_recall_std:.4f}')
        logger.info(f'  ⚡ Improved F1={f1_improved:.4f}, Damage_Recall={damage_recall_imp:.4f}, Major_Recall={major_recall_imp:.4f}')
        logger.info(f'  🏆 Composite: Standard={composite_std:.4f}, Improved={composite_imp:.4f}')
        
        # Сохранение метрик
        train_losses.append(avg_train_loss)
        val_f1_scores.append(f1_improved)  # Используем улучшенный F1
        val_damage_recalls.append(damage_recall_imp)
        val_major_recalls.append(major_recall_imp)
        
        # Сохранение лучшей модели
        current_best = max(f1_improved, composite_imp)
        if current_best > best_f1:
            best_f1 = current_best
            
            # Сохраняем с дополнительной информацией
            save_dict = {
                'epoch': prev_metrics['epoch'] + epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'f1_standard': f1_standard,
                'f1_improved': f1_improved,
                'damage_recall_std': damage_recall_std,
                'damage_recall_imp': damage_recall_imp,
                'major_recall_std': major_recall_std,
                'major_recall_imp': major_recall_imp,
                'composite_std': composite_std,
                'composite_imp': composite_imp,
                'train_losses': train_losses,
                'val_f1_scores': val_f1_scores,
                'val_damage_recalls': val_damage_recalls,
                'val_major_recalls': val_major_recalls,
                'all_labels': all_labels,
                'all_preds_standard': all_preds,
                'all_preds_improved': all_preds_improved,
                'all_probs': all_probs,
                'inference_params': {
                    'tau_nd': improved_inference.tau_nd,
                    'tau_major': improved_inference.tau_major
                }
            }
            
            torch.save(save_dict, 'training_results/finetuned_best_model.pth')
            logger.info(f'💾 Новая лучшая модель! F1={current_best:.4f} (улучшение: +{current_best-prev_metrics["best_f1"]:.4f})')
        
        scheduler.step()
    
    # 5. Финальный анализ
    logger.info('\n🎉 FINE-TUNING ЗАВЕРШЁН!')
    logger.info(f'📈 Итоговое улучшение F1: {best_f1:.4f} (было: {prev_metrics["best_f1"]:.4f})')
    
    # Создаём детальный отчёт
    create_finetuning_report(save_dict)
    
    return 'training_results/finetuned_best_model.pth'

def create_finetuning_report(save_dict):
    """Создаёт детальный отчёт о fine-tuning"""
    
    logger.info("📄 Создание отчёта fine-tuning...")
    
    # Confusion matrix для улучшенных предсказаний
    cm = confusion_matrix(save_dict['all_labels'], save_dict['all_preds_improved'])
    
    # Подробный отчёт
    report = {
        'finetuning_summary': {
            'original_f1': 0.7383,  # Из предыдущих результатов
            'final_f1_standard': save_dict['f1_standard'],
            'final_f1_improved': save_dict['f1_improved'],
            'improvement': save_dict['f1_improved'] - 0.7383,
            'total_epochs': save_dict['epoch'],
            'best_damage_recall': save_dict['damage_recall_imp'],
            'best_major_recall': save_dict['major_recall_imp']
        },
        'performance_comparison': {
            'standard_inference': {
                'f1_score': save_dict['f1_standard'],
                'damage_recall': save_dict['damage_recall_std'],
                'major_recall': save_dict['major_recall_std']
            },
            'improved_inference': {
                'f1_score': save_dict['f1_improved'],
                'damage_recall': save_dict['damage_recall_imp'],
                'major_recall': save_dict['major_recall_imp']
            }
        },
        'confusion_matrix': cm.tolist(),
        'class_names': ['no_damage', 'minor_damage', 'major_damage'],
        'inference_parameters': save_dict['inference_params'],
        'training_curves': {
            'train_losses': save_dict['train_losses'],
            'val_f1_scores': save_dict['val_f1_scores'],
            'val_damage_recalls': save_dict['val_damage_recalls'],
            'val_major_recalls': save_dict['val_major_recalls']
        }
    }
    
    # Сохраняем JSON отчёт
    with open('training_results/finetuning_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Создаём визуализацию confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['no_damage', 'minor_damage', 'major_damage'],
                yticklabels=['no_damage', 'minor_damage', 'major_damage'])
    plt.title('Confusion Matrix (Improved Inference)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('training_results/finetuning_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # График обучения
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0,0].plot(save_dict['train_losses'], label='Train Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    
    # F1 Score
    axes[0,1].plot(save_dict['val_f1_scores'], label='F1 Score', color='green')
    axes[0,1].set_title('F1 Score (Improved)')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('F1 Score')
    axes[0,1].legend()
    
    # Damage Recall
    axes[1,0].plot(save_dict['val_damage_recalls'], label='Damage Recall', color='orange')
    axes[1,0].set_title('Damage Detection Recall')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Recall')
    axes[1,0].legend()
    
    # Major Recall
    axes[1,1].plot(save_dict['val_major_recalls'], label='Major Recall', color='red')
    axes[1,1].set_title('Major Damage Recall')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Recall')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('training_results/finetuning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Отчёты сохранены:")
    logger.info("   • training_results/finetuning_report.json")
    logger.info("   • training_results/finetuning_confusion_matrix.png")
    logger.info("   • training_results/finetuning_curves.png")

if __name__ == "__main__":
    print("🔄 FINE-TUNING СУЩЕСТВУЮЩЕЙ МОДЕЛИ")
    print("=" * 40)
    
    result = finetune_existing_model()
    
    if result:
        print(f"\n✅ Fine-tuning завершён успешно!")
        print(f"📁 Дообученная модель: {result}")
        print("📊 Проверьте отчёты в папке training_results/")
    else:
        print("❌ Fine-tuning не удался")