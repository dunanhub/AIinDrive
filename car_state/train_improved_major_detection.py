"""
Улучшенная версия обучения с фокусом на major damage detection
================================================================
Основано на анализе: модель хорошо детектирует damage vs no-damage,
но плохо различает major vs minor damage
"""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging

# Импорт модели
from multiclass_damage_model import MulticlassDamageModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedInferenceModel:
    """Модель с улучшенной логикой принятия решений"""
    
    def __init__(self, model, tau_nd=0.65, tau_major=0.32, delta=0.12):
        self.model = model
        self.tau_nd = tau_nd          # Порог для no_damage
        self.tau_major = tau_major    # Порог для major_damage
        self.delta = delta            # Дельта для сравнения major vs minor
        
    def predict(self, x):
        """Предсказание с улучшенной логикой"""
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            
            # Применяем decision rules
            predictions = []
            for prob in probs:
                p_no_damage = prob[0].item()
                p_minor = prob[1].item() 
                p_major = prob[2].item()
                
                # Правило 1: Если P(no_damage) < tau_nd → damage_present
                if p_no_damage < self.tau_nd:
                    # Правило 2: major если P(major) >= tau_major ИЛИ P(major) >= P(minor) - delta
                    if p_major >= self.tau_major or p_major >= (p_minor - self.delta):
                        predictions.append(2)  # major_damage
                    else:
                        predictions.append(1)  # minor_damage
                else:
                    predictions.append(0)  # no_damage
                    
            return torch.tensor(predictions)
    
    def get_probabilities(self, x):
        """Получить вероятности классов"""
        with torch.no_grad():
            logits = self.model(x)
            return F.softmax(logits, dim=1)

class FocalLoss(nn.Module):
    """Focal Loss с настраиваемыми альфа-параметрами"""
    
    def __init__(self, alpha=None, gamma=1.5, reduction='mean'):
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

def create_balanced_weights(y_train):
    """Создание сбалансированных весов для классов"""
    class_counts = Counter(y_train)
    total_samples = len(y_train)
    
    # Умеренные веса (не экстремальные)
    weights = {}
    weights[0] = 0.7  # no_damage - немного снижаем
    weights[1] = 1.0  # minor_damage - базовый
    weights[2] = 1.5  # major_damage - увеличиваем, но не кардинально
    
    logger.info(f"Class weights: {weights}")
    return weights

def train_improved_model():
    """Основная функция обучения с улучшениями"""
    
    # Настройки
    DEVICE = torch.device('cpu')
    EPOCHS = 10  # Увеличиваем количество эпох
    BATCH_SIZE = 16
    
    # Learning rates (layer-wise)
    HEAD_LR = 1e-4      # Для головы классификатора
    BACKBONE_LR = 3e-5  # Для верхних блоков backbone
    
    # Загрузка данных
    logger.info("Загрузка данных...")
    
    # Загружаем данные используя существующую логику
    from train_multiclass_damage_v3_2 import load_all_datasets, MulticlassDamageDataset, DATASET_ROOTS
    
    train_records, val_records = load_all_datasets(DATASET_ROOTS)
    
    logger.info(f"Train samples: {len(train_records)}")
    logger.info(f"Val samples: {len(val_records)}")
    
    # Проверяем распределение классов
    train_labels = [record['label'] for record in train_records]
    train_class_counts = Counter(train_labels)
    logger.info(f"Train class distribution: {dict(train_class_counts)}")
    
    # Улучшенные трансформации (умеренные)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        # Убираем сильный RandomErasing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создаем датасеты используя существующую логику    
    train_dataset = MulticlassDamageDataset(train_records, transform=train_transform)
    val_dataset = MulticlassDamageDataset(val_records, transform=val_transform)
    
    # Создаем веса для классов
    class_weights = create_balanced_weights(train_labels)
    
    # Создаем сэмплер с умеренным балансированием
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                             num_workers=0, drop_last=True)
    # Валидация БЕЗ сэмплера и drop_last=False
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=0, drop_last=False)
    
    # Модель
    logger.info("Инициализация модели...")
    model = MulticlassDamageModel(num_classes=3)
    model.to(DEVICE)
    
    # Разморозка backbone раньше
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    # Layer-wise learning rates
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': HEAD_LR}
    ], weight_decay=1e-4)
    
    # Focal Loss с настроенными альфа
    alpha_weights = torch.tensor([0.7, 1.0, 1.5])  # [no_damage, minor, major]
    criterion = FocalLoss(alpha=alpha_weights, gamma=1.5)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Обучение
    logger.info("Начало обучения...")
    
    best_composite_score = 0
    train_losses = []
    val_f1_scores = []
    val_damage_recalls = []
    val_major_recalls = []
    
    for epoch in range(EPOCHS):
        # Тренировка
        model.train()
        train_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                logger.info(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Валидация
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Метрики
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Damage detection recall (minor + major vs no_damage)
        damage_labels = [1 if label > 0 else 0 for label in all_labels]
        damage_preds = [1 if pred > 0 else 0 for pred in all_preds]
        damage_recall = f1_score(damage_labels, damage_preds, average='binary', pos_label=1)
        
        # Major damage recall
        major_labels = [1 if label == 2 else 0 for label in all_labels]
        major_preds = [1 if pred == 2 else 0 for pred in all_preds]
        if sum(major_labels) > 0:
            major_recall = sum([1 for i, pred in enumerate(major_preds) if pred == 1 and major_labels[i] == 1]) / sum(major_labels)
        else:
            major_recall = 0
        
        # Композитная метрика: 0.7 * damage_recall + 0.3 * major_recall
        composite_score = 0.7 * damage_recall + 0.3 * major_recall
        
        train_losses.append(train_loss / len(train_loader))
        val_f1_scores.append(f1)
        val_damage_recalls.append(damage_recall)
        val_major_recalls.append(major_recall)
        
        logger.info(f'Epoch {epoch+1}: F1={f1:.4f}, Damage_Recall={damage_recall:.4f}, '
                   f'Major_Recall={major_recall:.4f}, Composite={composite_score:.4f}')
        
        # Сохранение лучшей модели по композитной метрике
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_composite_score': best_composite_score,
                'best_f1': f1,
                'damage_recall': damage_recall,
                'major_recall': major_recall,
                'all_labels': all_labels,
                'all_preds': all_preds,
                'all_probs': all_probs
            }, 'training_results/best_improved_model.pth')
            logger.info(f'Новая лучшая модель сохранена! Composite score: {composite_score:.4f}')
        
        scheduler.step()
    
    # Тестирование improved inference
    logger.info("\nТестирование улучшенной логики принятия решений...")
    
    # Загружаем лучшую модель
    checkpoint = torch.load('training_results/best_improved_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Создаем модель с улучшенной логикой
    improved_model = ImprovedInferenceModel(model)
    
    # Тестируем на валидационных данных
    model.eval()
    improved_preds = []
    original_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            
            # Оригинальные предсказания
            outputs = model(images)
            orig_preds = torch.argmax(outputs, dim=1)
            
            # Улучшенные предсказания
            impr_preds = improved_model.predict(images)
            
            improved_preds.extend(impr_preds.cpu().numpy())
            original_preds.extend(orig_preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Сравнение результатов
    logger.info("\n=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
    
    # Оригинальная модель
    orig_f1 = f1_score(all_labels, original_preds, average='weighted')
    orig_damage_labels = [1 if label > 0 else 0 for label in all_labels]
    orig_damage_preds = [1 if pred > 0 else 0 for pred in original_preds]
    orig_damage_recall = f1_score(orig_damage_labels, orig_damage_preds, average='binary', pos_label=1)
    
    orig_major_labels = [1 if label == 2 else 0 for label in all_labels]
    orig_major_preds = [1 if pred == 2 else 0 for pred in original_preds]
    orig_major_recall = sum([1 for i, pred in enumerate(orig_major_preds) if pred == 1 and orig_major_labels[i] == 1]) / sum(orig_major_labels) if sum(orig_major_labels) > 0 else 0
    
    # Улучшенная модель
    impr_f1 = f1_score(all_labels, improved_preds, average='weighted')
    impr_damage_preds = [1 if pred > 0 else 0 for pred in improved_preds]
    impr_damage_recall = f1_score(orig_damage_labels, impr_damage_preds, average='binary', pos_label=1)
    
    impr_major_preds = [1 if pred == 2 else 0 for pred in improved_preds]
    impr_major_recall = sum([1 for i, pred in enumerate(impr_major_preds) if pred == 1 and orig_major_labels[i] == 1]) / sum(orig_major_labels) if sum(orig_major_labels) > 0 else 0
    
    logger.info(f"Оригинальная модель:")
    logger.info(f"  F1: {orig_f1:.4f}")
    logger.info(f"  Damage Recall: {orig_damage_recall:.4f}")
    logger.info(f"  Major Recall: {orig_major_recall:.4f}")
    
    logger.info(f"Улучшенная модель:")
    logger.info(f"  F1: {impr_f1:.4f}")
    logger.info(f"  Damage Recall: {impr_damage_recall:.4f}")
    logger.info(f"  Major Recall: {impr_major_recall:.4f}")
    
    # Сохраняем результаты
    results = {
        'original_model': {
            'f1': orig_f1,
            'damage_recall': orig_damage_recall,
            'major_recall': orig_major_recall
        },
        'improved_model': {
            'f1': impr_f1,
            'damage_recall': impr_damage_recall,
            'major_recall': impr_major_recall,
            'decision_thresholds': {
                'tau_nd': improved_model.tau_nd,
                'tau_major': improved_model.tau_major,
                'delta': improved_model.delta
            }
        }
    }
    
    with open('training_results/improved_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Результаты сохранены в training_results/improved_results.json")
    logger.info("Обучение завершено!")

if __name__ == "__main__":
    train_improved_model()