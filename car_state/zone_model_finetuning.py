"""
Скрипт для дообучения существующей модели под задачу зонального анализа
=====================================================================
Возможности:
1. Fine-tuning существующей модели на новых данных
2. Добавление специальных слоёв для зонального анализа  
3. Обучение с учётом особенностей разных зон автомобиля
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ZoneTrainingConfig:
    """Конфигурация для обучения зональной модели"""
    base_model_path: str
    zones_data_path: str
    output_model_path: str
    epochs: int = 15
    learning_rate: float = 1e-4
    batch_size: int = 8
    freeze_backbone: bool = True  # Заморозить backbone на первых эпохах
    zone_weights: Dict[str, float] = None  # Веса для разных зон

class ZoneSpecificModel(nn.Module):
    """Модель с учётом специфики зон"""
    
    def __init__(self, base_model, num_zones=7, num_classes=3):
        super(ZoneSpecificModel, self).__init__()
        
        self.base_model = base_model
        self.num_zones = num_zones
        self.num_classes = num_classes
        
        # Получаем размер выходного слоя базовой модели
        # Сначала попробуем получить реальный размер через dummy forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        try:
            with torch.no_grad():
                dummy_features = base_model.backbone(dummy_input)
                # Применяем GlobalAveragePooling если есть
                if hasattr(base_model, 'avgpool'):
                    dummy_features = base_model.avgpool(dummy_features)
                # Flatten
                dummy_features = dummy_features.view(dummy_features.size(0), -1)
                base_features = dummy_features.size(1)
                logger.info(f"Определён размер features: {base_features}")
        except Exception as e:
            logger.warning(f"Не удалось определить размер через forward pass: {e}")
            base_features = 2048  # Fallback для ResNet50
        
        # Заменяем классификатор базовой модели на identity
        # но сохраняем backbone + avgpool
        self.backbone = base_model.backbone
        if hasattr(base_model, 'avgpool'):
            self.avgpool = base_model.avgpool
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Добавляем зонально-специфичные слои
        self.zone_embedding = nn.Embedding(num_zones, 64)
        
        # Объединённый классификатор
        self.combined_classifier = nn.Sequential(
            nn.Linear(base_features + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Mapping зон к ID
        self.zone_to_id = {
            'front': 0, 'rear': 1, 'roof': 2, 'left_side': 3, 
            'right_side': 4, 'hood': 5, 'trunk': 6
        }
    
    def forward(self, x, zone_ids):
        """
        Forward pass с учётом зоны
        
        Args:
            x: Изображения (batch_size, 3, 224, 224)
            zone_ids: ID зон (batch_size,)
        """
        # Извлекаем features из backbone
        features = self.backbone(x)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Получаем zone embeddings
        zone_emb = self.zone_embedding(zone_ids)  # (batch_size, 64)
        
        # Объединяем features
        combined = torch.cat([features, zone_emb], dim=1)
        
        # Финальная классификация
        output = self.combined_classifier(combined)
        
        return output

class ZoneDataset(Dataset):
    """Dataset для зонального обучения"""
    
    def __init__(self, zones_data_path: str, transform=None):
        """
        Args:
            zones_data_path: Путь к JSON файлу с данными зон
            transform: Трансформации для изображений
        """
        self.transform = transform
        
        # Загружаем данные зон
        with open(zones_data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.samples = []
        
        # Подготавливаем сэмплы
        for item in self.data:
            if 'zones' in item:
                base_image_path = item['image_path']
                
                for zone_info in item['zones']:
                    sample = {
                        'image_path': base_image_path,
                        'zone_name': zone_info['name'],
                        'zone_bbox': zone_info['bbox'],
                        'damage_class': zone_info.get('damage_class', 0),
                        'zone_id': self._zone_name_to_id(zone_info['name'])
                    }
                    self.samples.append(sample)
        
        logger.info(f"Загружено {len(self.samples)} зональных сэмплов")
    
    def _zone_name_to_id(self, zone_name: str) -> int:
        """Конвертирует имя зоны в ID"""
        zone_mapping = {
            'front': 0, 'rear': 1, 'roof': 2, 'left_side': 3, 
            'right_side': 4, 'hood': 5, 'trunk': 6
        }
        return zone_mapping.get(zone_name, 0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Загружаем основное изображение
        image = cv2.imread(sample['image_path'])
        if image is None:
            # Если изображение не найдено, создаём заглушку
            image = np.ones((224, 224, 3), dtype=np.uint8) * 128
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Извлекаем зону по bbox
        x1, y1, x2, y2 = sample['zone_bbox']
        zone_image = image[y1:y2, x1:x2]
        
        # Убеждаемся, что зона не пустая
        if zone_image.size == 0:
            zone_image = np.ones((64, 64, 3), dtype=np.uint8) * 128
        
        # Применяем трансформации
        if self.transform:
            try:
                from PIL import Image
                zone_pil = Image.fromarray(zone_image)
                zone_image = self.transform(zone_pil)
            except Exception as e:
                logger.warning(f"Ошибка трансформации: {e}")
                # Fallback: простая нормализация
                zone_image = cv2.resize(zone_image, (224, 224))
                zone_image = torch.from_numpy(zone_image.transpose(2, 0, 1)).float() / 255.0
        
        return zone_image, torch.tensor(sample['zone_id']), torch.tensor(sample['damage_class'])

def create_zone_training_data(base_images_dir: str, output_json: str):
    """
    Создаёт тренировочные данные для зонального анализа
    
    Args:
        base_images_dir: Директория с изображениями автомобилей
        output_json: Файл для сохранения данных
    """
    from car_zone_detector import CarZoneDetector
    
    logger.info("Создание тренировочных данных для зональной модели...")
    
    detector = CarZoneDetector()
    training_data = []
    
    # Обрабатываем все изображения в директории
    image_paths = list(Path(base_images_dir).glob("*.jpg")) + list(Path(base_images_dir).glob("*.png"))
    
    for img_path in image_paths[:20]:  # Ограничиваем для демо
        try:
            # Загружаем изображение
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Детектируем зоны
            zones = detector.detect_zones(image_rgb)
            
            # Создаём случайные метки для зон (в реальности нужна разметка)
            zones_info = []
            for zone_name, bbox in zones.items():
                # Случайное определение класса повреждения (для демо)
                damage_class = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
                
                zones_info.append({
                    'name': zone_name,
                    'bbox': list(bbox),
                    'damage_class': int(damage_class)
                })
            
            training_data.append({
                'image_path': str(img_path),
                'zones': zones_info
            })
            
        except Exception as e:
            logger.warning(f"Ошибка обработки {img_path}: {e}")
    
    # Сохраняем данные
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Создано {len(training_data)} записей в {output_json}")
    return output_json

def finetune_model_for_zones(config: ZoneTrainingConfig):
    """
    Дообучает модель для зонального анализа
    
    Args:
        config: Конфигурация обучения
    """
    logger.info("Начало дообучения модели для зонального анализа...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    # 1. Загружаем базовую модель
    logger.info(f"Загрузка базовой модели: {config.base_model_path}")
    
    try:
        from multiclass_damage_model import MulticlassDamageModel
        base_model = MulticlassDamageModel(num_classes=3)
        
        checkpoint = torch.load(config.base_model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            base_model.load_state_dict(checkpoint)
            
        logger.info("✅ Базовая модель загружена")
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return None
    
    # 2. Создаём зональную модель
    zone_model = ZoneSpecificModel(base_model, num_zones=7, num_classes=3)
    zone_model.to(device)
    
    # Заморозка backbone если требуется
    if config.freeze_backbone:
        logger.info("🧊 Замораживаем backbone")
        for param in zone_model.backbone.parameters():
            param.requires_grad = False
    
    # 4. Подготовка данных
    logger.info("Подготовка тренировочных данных...")
    
    # Создаём тренировочные данные если файл не существует
    if not Path(config.zones_data_path).exists():
        logger.info("Создаём тренировочные данные...")
        # Используем демо-данные
        demo_data = [
            {
                'image_path': 'demo_car.jpg',
                'zones': [
                    {'name': 'front', 'bbox': [160, 180, 640, 420], 'damage_class': 1},
                    {'name': 'rear', 'bbox': [160, 420, 640, 600], 'damage_class': 0},
                    {'name': 'roof', 'bbox': [200, 0, 600, 180], 'damage_class': 0},
                    {'name': 'hood', 'bbox': [240, 90, 560, 270], 'damage_class': 2},
                ]
            }
        ]
        
        with open(config.zones_data_path, 'w', encoding='utf-8') as f:
            json.dump(demo_data, f, ensure_ascii=False, indent=2)
    
    # Трансформации
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создаём dataset
    try:
        dataset = ZoneDataset(config.zones_data_path, transform=transform)
        
        # Split на train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания dataset: {e}")
        return None
    
    # 5. Настройка обучения
    criterion = nn.CrossEntropyLoss()
    
    # Разные learning rates для разных частей модели
    params_to_update = []
    
    # Параметры новых слоёв
    for param in zone_model.zone_embedding.parameters():
        params_to_update.append({'params': param, 'lr': config.learning_rate})
    
    for param in zone_model.combined_classifier.parameters():
        params_to_update.append({'params': param, 'lr': config.learning_rate})
    
    # Параметры базовой модели (если не заморожены)
    if not config.freeze_backbone:
        for param in zone_model.backbone.parameters():
            params_to_update.append({'params': param, 'lr': config.learning_rate * 0.1})
    
    optimizer = torch.optim.AdamW(params_to_update, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # 6. Обучение
    logger.info("🚀 Начало обучения зональной модели...")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config.epochs):
        # Training
        zone_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, zone_ids, labels) in enumerate(train_loader):
            images = images.to(device)
            zone_ids = zone_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = zone_model(images, zone_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if batch_idx % 5 == 0:
                logger.info(f'Epoch {epoch+1}/{config.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        zone_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, zone_ids, labels in val_loader:
                images = images.to(device)
                zone_ids = zone_ids.to(device)
                labels = labels.to(device)
                
                outputs = zone_model(images, zone_ids)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Метрики эпохи
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}/{config.epochs}:')
        logger.info(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': zone_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': config.__dict__
            }, config.output_model_path)
            logger.info(f'💾 Лучшая модель сохранена: {config.output_model_path}')
        
        scheduler.step()
        
        # Разморозка backbone на половине обучения
        if config.freeze_backbone and epoch == config.epochs // 2:
            logger.info("🔓 Размораживаем backbone")
            for param in zone_model.backbone.parameters():
                param.requires_grad = True
            
            # Обновляем оптимизатор
            params_to_update = []
            for param in zone_model.parameters():
                if param.requires_grad:
                    params_to_update.append({'params': param, 'lr': config.learning_rate * 0.1})
            
            optimizer = torch.optim.AdamW(params_to_update, weight_decay=1e-4)
    
    logger.info("✅ Обучение зональной модели завершено!")
    return config.output_model_path

def test_zone_model():
    """Тестирование зональной модели"""
    
    logger.info("🧪 Тестирование зональной модели...")
    
    # Конфигурация
    config = ZoneTrainingConfig(
        base_model_path="training_results/best_model.pth",
        zones_data_path="zone_training_data.json",
        output_model_path="training_results/zone_specific_model.pth",
        epochs=5,  # Мало эпох для быстрого теста
        learning_rate=1e-4,
        batch_size=4
    )
    
    # Запускаем дообучение
    result = finetune_model_for_zones(config)
    
    if result:
        logger.info(f"✅ Зональная модель сохранена: {result}")
        return True
    else:
        logger.error("❌ Не удалось обучить зональную модель")
        return False

if __name__ == "__main__":
    print("🔧 ДООБУЧЕНИЕ МОДЕЛИ ДЛЯ ЗОНАЛЬНОГО АНАЛИЗА")
    print("=" * 50)
    
    # Запускаем тест
    success = test_zone_model()
    
    if success:
        print("\n🎉 Дообучение завершено успешно!")
        print("📁 Файлы:")
        print("   • zone_training_data.json - данные для обучения")
        print("   • training_results/zone_specific_model.pth - дообученная модель")
    else:
        print("\n❌ Дообучение не удалось")