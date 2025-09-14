"""
Исправленная версия обучения многоклассовой модели
Фиксит проблемы с синтаксисом и добавляет поддержку интегрированного датасета
"""
import os
import re
import json
import random
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import f1_score, roc_curve, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Импорт функций трансформаций из основного модуля
try:
    from multiclass_damage_model import MulticlassDamageModel, FocalLoss, create_training_transforms, create_validation_transforms
except ImportError:
    print("⚠️ Не удалось импортировать функции трансформаций, создаем локальные версии")
    
    def create_training_transforms():
        """Создает трансформации для обучения"""
        return transforms.Compose([
            # --- PIL stage (до ToTensor) ---
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            
            # --- Tensor stage (после ToTensor) ---
            transforms.ToTensor(),  # ОБЯЗАТЕЛЬНО перед Normalize/RandomErasing
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), value='random'),
        ])
    
    def create_validation_transforms():
        """Создает трансформации для валидации"""
        return transforms.Compose([
            # --- PIL stage ---
            transforms.Resize((224, 224)),
            
            # --- Tensor stage ---
            transforms.ToTensor(),  # ОБЯЗАТЕЛЬНО перед Normalize
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Простая версия модели, если импорт не удался
    class MulticlassDamageModel(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.15),
                nn.Linear(512, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)

# Импорт наших модулей
try:
    from .multiclass_damage_model import MulticlassDamageModel, FocalLoss, create_training_transforms, create_validation_transforms
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from multiclass_damage_model import MulticlassDamageModel, FocalLoss, create_training_transforms, create_validation_transforms

# Настройки
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Пути к датасетам (обновленные для интегрированного датасета)
DATASET_ROOTS = [
    r"C:\Users\Димаш\Desktop\python\hackaton\data\Rust and Scrach.v1i.multiclass\train",
    r"C:\Users\Димаш\Desktop\python\hackaton\data\Car Scratch and Dent.v5i.multiclass\train",
    r"C:\Users\Димаш\Desktop\python\hackaton\data\Dent_Detection.v1i.multiclass\train",
    r"C:\Users\Димаш\Desktop\python\hackaton\data\integrated_multiclass_dataset",  # Новый интегрированный датасет
    r"C:\Users\Димаш\Desktop\python\hackaton\data\car  scratch.v2i.multiclass\train",  # Дополнительный датасет для улучшения производительности
    r"C:\Users\Димаш\Desktop\python\hackaton\data\Car damages.v3i.multiclass\train",  # Расширенный датасет повреждений с большим количеством примеров
]

def set_seeds(seed=RANDOM_SEED):
    """Устанавливает seeds для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def normalize_columns(df):
    """Нормализует названия колонок в CSV"""
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Алиасы для распространенных вариантов
    column_aliases = {
        'filename': 'image_path',
        'file': 'image_path', 
        'image': 'image_path',
        'path': 'image_path'
    }
    
    for old_name, new_name in column_aliases.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df

def convert_multilabel_to_multiclass(row, dataset_type):
    """Конвертирует Multi-Label аннотации в Multi-Class"""
    
    if dataset_type == "dent_detection":
        # Простая бинарная логика: dent=1 → minor_damage, dent=0 → no_damage
        dent_val = row.get('dent', 0)
        if dent_val == 1:
            return 1  # minor_damage
        else:
            return 0  # no_damage
    
    elif dataset_type == "rust_scratch":
        # Rust and Scratch датасет
        rust = row.get('rust', 0)
        scratch = row.get('scratch', 0) 
        dent = row.get('dent', 0)
        
        # Приоритетная логика
        if rust == 1:  # Ржавчина = серьезное повреждение
            return 2  # major_damage
        elif dent == 1:  # Вмятина = серьезное повреждение  
            return 2  # major_damage
        elif scratch == 1:  # Царапина = незначительное
            return 1  # minor_damage
        else:
            return 0  # no_damage
            
    elif dataset_type == "car_scratch_dent":
        # Car Scratch and Dent датасет
        dent = row.get('dent', 0)
        scratch = row.get('scratch', 0)
        dirt = row.get('dirt', 0)
        
        # Приоритетная логика
        if dent == 1:  # Вмятина = серьезное
            return 2  # major_damage
        elif scratch == 1:  # Царапина = незначительное
            return 1  # minor_damage
        elif dirt == 1:  # Грязь = незначительное (можно почистить)
            return 1  # minor_damage  
        else:
            return 0  # no_damage
    
    elif dataset_type == "car_scratch_v2":
        # Car scratch.v2i.multiclass датасет (новый)
        # Колонки: '0', 'scratch', 'car-scratch'
        zero_class = row.get('0', 0)  # Класс "0"  
        scratch = row.get('scratch', 0)  # Царапина
        car_scratch = row.get('car-scratch', 0)  # Автомобильная царапина
        
        # Приоритетная логика: если несколько меток активны, выбираем более серьезную
        if car_scratch == 1:  # car-scratch = серьезное повреждение
            return 2  # major_damage
        elif scratch == 1:  # scratch = незначительное повреждение  
            return 1  # minor_damage
        elif zero_class == 1:  # класс "0" = без повреждений
            return 0  # no_damage
        else:
            return 0  # по умолчанию - без повреждений
    
    elif dataset_type == "car_scratch_v2":
        # Car scratch.v2i.multiclass датасет (новый)
        # Колонки: '0', 'scratch', 'car-scratch'
        zero_class = row.get('0', 0)  # Класс "0"  
        scratch = row.get('scratch', 0)  # Царапина
        car_scratch = row.get('car-scratch', 0)  # Автомобильная царапина
        
        # Приоритетная логика
        if car_scratch == 1:  # car-scratch = серьезное повреждение
            return 2  # major_damage
        elif scratch == 1:  # scratch = незначительное повреждение
            return 1  # minor_damage  
        elif zero_class == 1:  # класс "0" = нет повреждений
            return 0  # no_damage
        else:
            return 0  # no_damage (по умолчанию)
    
    # По умолчанию
    return 0

def load_integrated_dataset_split(root_path):
    """
    Загружает интегрированный датасет с уже готовым split'ом train/test/valid
    """
    train_records = []
    val_records = []
    
    class_mapping = {
        "no_damage": 0,
        "minor_damage": 1, 
        "major_damage": 2
    }
    
    # Обрабатываем train split
    train_path = root_path / "train"
    if train_path.exists():
        print("   📂 Загружается train split")
        for class_name, class_id in class_mapping.items():
            class_path = train_path / class_name
            if not class_path.exists():
                continue
                
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(class_path.glob(ext))
            
            for img_path in image_files:
                record = {
                    'path': str(img_path),
                    'label': int(class_id),
                    'source': 'integrated_dataset',
                    'dataset_type': 'integrated'
                }
                train_records.append(record)
            
            print(f"      {class_name}: {len(image_files)} изображений")
    
    # Обрабатываем valid split
    valid_path = root_path / "valid"
    if valid_path.exists():
        print("   📂 Загружается valid split")
        for class_name, class_id in class_mapping.items():
            class_path = valid_path / class_name
            if not class_path.exists():
                continue
                
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(class_path.glob(ext))
            
            for img_path in image_files:
                record = {
                    'path': str(img_path),
                    'label': int(class_id),
                    'source': 'integrated_dataset',
                    'dataset_type': 'integrated'
                }
                val_records.append(record)
            
            print(f"      {class_name}: {len(image_files)} изображений")
    
    print(f"   ✅ Train: {len(train_records)}, Valid: {len(val_records)}")
    return train_records, val_records

def load_csv_dataset(root_path, csv_file):
    """Загружает данные из CSV-based датасета"""
    items = []
    class_distribution = Counter()
    
    try:
        df = pd.read_csv(csv_file)
        df = normalize_columns(df)
        
        # Определяем тип датасета
        columns = set(df.columns)
        if {'image_path', 'dent', 'dirt', 'scratch'}.issubset(columns):
            dataset_type = "car_scratch_dent"
        elif {'image_path', 'car', 'dent', 'rust', 'scratch'}.issubset(columns):
            dataset_type = "rust_scratch"  
        elif {'image_path', 'dent'}.issubset(columns) and len(columns) <= 3:
            dataset_type = "dent_detection"
        elif {'image_path', '0', 'scratch', 'car-scratch'}.issubset(columns):
            dataset_type = "car_scratch_v2"  # Новый тип для car scratch.v2i.multiclass
        else:
            print(f"   ❌ Неизвестный формат CSV. Колонки: {columns}")
            return items, class_distribution
        
        # Обрабатываем изображения
        for idx, row in df.iterrows():
            image_name = row['image_path']
            image_path = root_path / image_name
            
            if not image_path.exists():
                continue
            
            class_id = convert_multilabel_to_multiclass(row, dataset_type)
            items.append((str(image_path), class_id))
            class_distribution[class_id] += 1
        
        print(f"   ✅ Обработано: {len(items)} изображений")
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
    
    return items, class_distribution

def load_all_datasets(roots):
    """Загружает все датасеты"""
    # Устанавливаем seed для воспроизводимости
    random.seed(RANDOM_SEED)
    
    all_train_records = []
    all_val_records = []
    
    print("🔍 ЗАГРУЗКА ВСЕХ ДАТАСЕТОВ")
    
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"❌ Путь не найден: {root_path}")
            continue
            
        print(f"\n📁 Обрабатываем: {root_path.name}")
        
        # Специальная обработка интегрированного датасета
        if "integrated_multiclass_dataset" in str(root_path):
            print("   🆕 Интегрированный датасет")
            train_records, val_records = load_integrated_dataset_split(root_path)
            all_train_records.extend(train_records)
            all_val_records.extend(val_records)
            continue
        
        # Обычные CSV датасеты
        csv_files = list(root_path.glob("*.csv"))
        if not csv_files:
            print(f"   ❌ CSV не найден")
            continue
            
        csv_file = csv_files[0]
        items, dist = load_csv_dataset(root_path, csv_file)
        
        # Конвертируем в records формат
        dataset_name = root_path.parent.name.lower()
        for path, label in items:
            record = {
                'path': path,
                'label': label,
                'source': dataset_name,
                'dataset_type': 'csv'
            }
            # Для Dent_Detection - только в train
            if "dent_detection" in dataset_name:
                all_train_records.append(record)
            else:
                # Для остальных делаем split
                if random.random() < 0.7:  # 70% в train
                    all_train_records.append(record)
                else:
                    all_val_records.append(record)
    
    return all_train_records, all_val_records

class MulticlassDamageDataset(Dataset):
    """Dataset для многоклассовой модели"""
    
    def __init__(self, records, transform=None):
        self.records = records
        self.transform = transform
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        record = self.records[idx]
        
        # Загружаем изображение
        try:
            image = Image.open(record['path']).convert('RGB')  # ОБЯЗАТЕЛЬНО RGB
        except Exception as e:
            print(f"Ошибка загрузки {record['path']}: {e}")
            # Возвращаем черное изображение в случае ошибки
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # ОБЯЗАТЕЛЬНО применяем трансформации
        if self.transform is not None:
            image = self.transform(image)  # PIL → Tensor
        
        label = record['label']
        return image, label

def create_weighted_sampler(records):
    """Создает взвешенный сэмплер для балансировки классов"""
    labels = [r['label'] for r in records]
    label_counts = Counter(labels)
    
    # Рассчитываем веса
    total = len(labels)
    weights = []
    
    for label in labels:
        weight = total / (len(label_counts) * label_counts[label])
        weights.append(weight)
    
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Обучение одной эпохи"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1

def validate_epoch(model, dataloader, criterion, device):
    """Валидация одной эпохи"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, f1

def main():
    """Основная функция обучения"""
    print("🤖 ОБУЧЕНИЕ МНОГОКЛАССОВОЙ МОДЕЛИ v3.2")
    print("="*60)
    
    set_seeds(RANDOM_SEED)
    
    # Загружаем данные
    print("\n1️⃣ Загрузка данных")
    train_records, val_records = load_all_datasets(DATASET_ROOTS)
    
    if len(train_records) == 0:
        print("❌ Нет данных для обучения!")
        return
    
    print(f"\n📊 СТАТИСТИКА ДАННЫХ:")
    print(f"   Train: {len(train_records)} изображений")
    print(f"   Valid: {len(val_records)} изображений")
    
    # Показываем распределение классов
    train_dist = Counter([r['label'] for r in train_records])
    val_dist = Counter([r['label'] for r in val_records])
    
    class_names = ["no_damage", "minor_damage", "major_damage"]
    
    print(f"\n📈 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
    print(f"{'Class':<15} {'Train':<10} {'Valid':<10}")
    print("-" * 35)
    for i, name in enumerate(class_names):
        print(f"{name:<15} {train_dist[i]:<10} {val_dist[i]:<10}")
    
    # Создаем трансформации с отладкой
    print("\n2️⃣ Подготовка трансформаций")
    
    train_transforms = create_training_transforms()
    val_transforms = create_validation_transforms()
    
    print(f"   ✅ Train transforms: {len(train_transforms.transforms)} этапов")
    print(f"   ✅ Val transforms: {len(val_transforms.transforms)} этапов")
    
    # Проверяем порядок
    for i, transform in enumerate(train_transforms.transforms):
        print(f"      {i+1}. {type(transform).__name__}")
    
    # Создаем датасеты с ОБЯЗАТЕЛЬНОЙ передачей трансформаций
    train_dataset = MulticlassDamageDataset(train_records, transform=train_transforms)
    val_dataset = MulticlassDamageDataset(val_records, transform=val_transforms)
    
    print(f"   ✅ Train dataset: {len(train_dataset)} записей")
    print(f"   ✅ Val dataset: {len(val_dataset)} записей")
    
    # Создаем сэмплер
    print("\n3️⃣ Создание взвешенного сэмплера")
    sampler = create_weighted_sampler(train_records)
    
    # DataLoader'ы с исправленными настройками для отладки
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=0,  # Для отладки отключаем многопоточность
        pin_memory=False  # На CPU нет смысла
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,  # Для отладки отключаем многопоточность
        pin_memory=False  # На CPU нет смысла
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Valid batches: {len(val_loader)}")
    
    # Безопасная проверка первого батча для отладки
    print("\n🔍 Проверка первого батча:")
    try:
        imgs, labels = next(iter(train_loader))
        
        # Проверяем тип данных
        print(f"   Тип изображений: {type(imgs)}")
        print(f"   Атрибут shape: {getattr(imgs, 'shape', 'НЕТ АТРИБУТА SHAPE!')}")
        
        # Если это тензор - показываем детали
        if isinstance(imgs, torch.Tensor):
            assert imgs.ndim == 4 and imgs.size(1) == 3 and imgs.size(2) == 224
            print(f"   ✅ Batch shape: {imgs.shape}")
            print(f"   ✅ Batch dtype: {imgs.dtype}")
            print(f"   ✅ Value range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
            print(f"   ✅ Labels: {labels[:5].tolist()}")
            print("   ✅ Батч загружен успешно!")
        else:
            print(f"   ❌ ОШИБКА: Ожидали torch.Tensor, получили {type(imgs)}")
            print("   🔧 Проблема в трансформациях - ToTensor() не применился!")
            return
            
    except Exception as e:
        print(f"   ❌ Ошибка загрузки батча: {e}")
        print(f"   🔧 Трансформации работают неправильно!")
        return
    
    # Создаем модель
    print("\n4️⃣ Создание модели")
    model = MulticlassDamageModel(num_classes=3).to(DEVICE)
    
    # 🧮 ПЕРЕСЧИТЫВАЕМ веса для FocalLoss на основе нового распределения
    # Новое распределение: [1473, 424, 292] → инвертируем для баланса
    total_samples = 1473 + 424 + 292  # 2189
    no_damage_weight = total_samples / (3 * 1473)      # 0.496
    minor_damage_weight = total_samples / (3 * 424)    # 1.721  
    major_damage_weight = total_samples / (3 * 292)    # 2.499
    
    print(f"📊 НОВЫЕ ВЕСА КЛАССОВ:")
    print(f"   • no_damage: {no_damage_weight:.3f}")
    print(f"   • minor_damage: {minor_damage_weight:.3f}")
    print(f"   • major_damage: {major_damage_weight:.3f}")
    
    # Criterion и optimizer с ПЕРЕСЧИТАННЫМИ весами для классов
    criterion = FocalLoss(alpha=[no_damage_weight, minor_damage_weight, major_damage_weight], gamma=2.0, device=DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0001,  # 🔧 СНИЖАЕМ с 0.001 до 0.0001 для fine-tuning
        weight_decay=5e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    print(f"   Device: {DEVICE}")
    print(f"   Параметры модели: {sum(p.numel() for p in model.parameters()):,}")
    
    # Проверяем есть ли сохраненная модель - ИСПРАВЛЕН ПУТЬ НА ХОРОШИЙ CHECKPOINT
    checkpoint_path = "training_results/best_model.pth"  # ХОРОШИЙ checkpoint с F1=0.6020 (относительно car_state)
    start_epoch = 0
    best_f1 = 0.0
    
    if os.path.exists(checkpoint_path):
        print(f"\n🔄 ЗАГРУЖАЕМ СОХРАНЕННУЮ МОДЕЛЬ: {checkpoint_path}")
        print(f"   📁 Размер файла: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            print(f"   📦 Тип checkpoint: {type(checkpoint)}")
            
            # Проверяем формат checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Новый формат с метаданными
                print(f"   🆕 ОБНАРУЖЕН ПОЛНЫЙ CHECKPOINT С МЕТАДАННЫМИ")
                print(f"   📋 Доступные ключи: {list(checkpoint.keys())}")
                
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # НЕ загружаем optimizer state из-за несовместимости
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                start_epoch = checkpoint.get('epoch', 0)
                best_f1 = checkpoint.get('val_f1', checkpoint.get('best_f1', 0.0))
                
                print(f"   ✅ УСПЕШНО ЗАГРУЖЕНА МОДЕЛЬ:")
                print(f"      🎯 Эпоха: {start_epoch}")
                print(f"      🏆 Лучший F1-score: {best_f1:.4f}")
                print(f"      ⚠️ Optimizer state НЕ загружен (несовместимость)")
                
                # Показываем детальные метрики, если есть
                if 'val_metrics' in checkpoint:
                    val_metrics = checkpoint['val_metrics']
                    print(f"   📊 ДЕТАЛЬНЫЕ МЕТРИКИ МОДЕЛИ:")
                    print(f"      • Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"      • Macro F1: {val_metrics.get('macro_f1', 'N/A'):.4f}")
                    print(f"      • Weighted F1: {val_metrics.get('weighted_f1', 'N/A'):.4f}")
                    print(f"      • F1 no_damage: {val_metrics.get('f1_no_damage', 'N/A'):.4f}")
                    print(f"      • F1 minor_damage: {val_metrics.get('f1_minor_damage', 'N/A'):.4f}")
                    print(f"      • F1 major_damage: {val_metrics.get('f1_major_damage', 'N/A'):.4f}")
                
                # Показываем optimizer state
                if 'optimizer_state_dict' in checkpoint:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"      🔧 Learning Rate: {current_lr:.6f}")
                
                print(f"   🚀 ПРОДОЛЖАЕМ ОБУЧЕНИЕ С ПРЕДОБУЧЕННЫМИ ВЕСАМИ!")
                
            else:
                # Старый формат - только веса модели (OrderedDict)
                print(f"   📜 ОБНАРУЖЕН СТАРЫЙ ФОРМАТ - ТОЛЬКО ВЕСА МОДЕЛИ")
                print(f"   📋 Количество параметров в checkpoint: {len(checkpoint)}")
                
                model.load_state_dict(checkpoint)
                start_epoch = 0
                best_f1 = 0.0
                
                print(f"   ✅ УСПЕШНО ЗАГРУЖЕНЫ ВЕСА МОДЕЛИ:")
                print(f"      ⚠️ Только веса модели (без метаданных)")
                print(f"      ⚠️ Epoch и optimizer state сброшены")
                print(f"      🔄 Начинаем с эпохи 1, но с предобученными весами!")
                
        except Exception as e:
            print(f"   ❌ ОШИБКА ЗАГРУЗКИ CHECKPOINT:")
            print(f"      Причина: {e}")
            print(f"   🆕 Начинаем обучение С НУЛЯ (случайные веса)")
            start_epoch = 0
            best_f1 = 0.0
    else:
        print(f"\n🆕 СОХРАНЕННАЯ МОДЕЛЬ НЕ НАЙДЕНА")
        print(f"   📁 Путь поиска: {checkpoint_path}")
        print(f"   🎲 Начинаем обучение С НУЛЯ (случайные веса)")
    
    # Обучение
    print(f"\n🚀 НАЧАЛО ОБУЧЕНИЯ")
    print(f"   📊 Конфигурация:")
    print(f"      • Стартовая эпоха: {start_epoch + 1}")
    print(f"      • Общее количество эпох: 30")
    print(f"      • Текущий лучший F1: {best_f1:.4f}")
    print(f"      • Устройство: {DEVICE}")
    print(f"      • Batch size: {train_loader.batch_size}")
    
    # Безопасный вывод learning rates
    if len(optimizer.param_groups) == 1:
        print(f"      • Learning rate: {optimizer.param_groups[0]['lr']:.1e}")
    else:
        print(f"      • Learning rates: backbone={optimizer.param_groups[0]['lr']:.1e}, classifier={optimizer.param_groups[1]['lr']:.1e}")
    
    if start_epoch > 0:
        print(f"   🔄 РЕЖИМ: Продолжение обучения с предобученными весами")
        print(f"   🎯 ЦЕЛЬ: Улучшить F1-score с {best_f1:.4f}")
    else:
        if best_f1 > 0.4:  # Если F1 > 0.4, значит веса предобучены
            print(f"   🔄 РЕЖИМ: Дообучение с предобученными весами (F1={best_f1:.4f})")
            print(f"   🎯 ЦЕЛЬ: Превзойти предыдущий результат F1={best_f1:.4f}")
        else:
            print(f"   🆕 РЕЖИМ: Обучение с нуля (случайные веса)")
            print(f"   🎯 ЦЕЛЬ: Достичь F1-score > 0.6")
    
    print(f"   ⏰ Early stopping: {10} эпох без улучшения")
    print(f"="*60)
    
    patience = 10
    patience_counter = 0
    
    num_epochs = 30
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n📅 Эпоха {epoch+1}/{num_epochs}")
        if start_epoch > 0 and epoch == start_epoch:
            print(f"   🔄 Продолжаем с предобученной модели (F1={best_f1:.4f})")
        print("-" * 30)
        
        # Обучение
        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Валидация
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, DEVICE)
        
        # Scheduler step
        scheduler.step()
        
        # Результаты эпохи
        print(f"📊 РЕЗУЛЬТАТЫ ЭПОХИ {epoch+1}:")
        print(f"   🏋️ Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"   🎯 Valid Loss: {val_loss:.4f}, Valid F1: {val_f1:.4f}")
        print(f"   ⚙️ Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Специальная проверка для первой эпохи
        if epoch == start_epoch:
            print(f"\n🔍 АНАЛИЗ ПЕРВОЙ ЭПОХИ:")
            if val_f1 > 0.4:  # Если F1 > 0.4, то явно не случайные веса
                print(f"   ✅ F1={val_f1:.4f} > 0.4 → ПРЕДОБУЧЕННЫЕ ВЕСА ЗАГРУЖЕНЫ УСПЕШНО!")
                print(f"   🎉 Модель начинает с хорошего уровня, а не с нуля")
            elif val_f1 > 0.2:
                print(f"   ⚠️ F1={val_f1:.4f} > 0.2 → Возможно частично загружены веса")
                print(f"   🤔 Или просто удачная инициализация")
            else:
                print(f"   ❌ F1={val_f1:.4f} < 0.2 → ПОХОЖЕ НА СЛУЧАЙНЫЕ ВЕСА!")
                print(f"   🚨 Checkpoint мог не загрузиться или сброситься")
        
        # Сохранение лучшей модели
        if val_f1 > best_f1:
            improvement = val_f1 - best_f1
            best_f1 = val_f1
            
            # Сохраняем полный checkpoint с метаданными
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_f1,
                'val_metrics': {
                    'accuracy': train_f1,  # Здесь нужно добавить реальные метрики
                    'macro_f1': val_f1,
                    'train_f1': train_f1,
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
            }, 'training_results/best_model.pth')
            
            print(f"\n🏆 НОВАЯ ЛУЧШАЯ МОДЕЛЬ!")
            print(f"   📈 F1-score: {best_f1:.4f} (+{improvement:.4f})")
            print(f"   💾 Сохранено в: training_results/best_model.pth")
            patience_counter = 0
        else:
            decline = best_f1 - val_f1
            patience_counter += 1
            print(f"\n📉 Без улучшения: F1={val_f1:.4f} (лучший: {best_f1:.4f}, -{decline:.4f})")
            print(f"   ⏰ Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"\n⏰ EARLY STOPPING на эпохе {epoch+1}")
            print(f"   🛑 {patience} эпох без улучшения")
            print(f"   🏆 Финальный лучший F1: {best_f1:.4f}")
            break
    
    print(f"\n🎉 Обучение завершено!")
    print(f"   Лучший F1-score: {best_f1:.4f}")
    print(f"   Модель сохранена: ../best_multiclass_model_v3.2.pth")

if __name__ == "__main__":
    main()