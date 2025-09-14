"""
Система обучения многоклассовой модели повреждений
Объединяет два датасета и обучает модель с 3 классами
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

# Импорт наших модулей с правильными путями
try:
    from .multiclass_damage_model import MulticlassDamageModel, FocalLoss, create_training_transforms, create_validation_transforms
except ImportError:
    # Fallback для прямого запуска
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
]

def convert_multilabel_to_multiclass(row, dataset_type):
    """
    Конвертирует Multi-Label аннотации в Multi-Class
    
    Логика приоритизации:
    - major_damage (2): серьезные повреждения (ржавчина, вмятины)
    - minor_damage (1): легкие повреждения (царапины)
    - no_damage (0): нет повреждений
    """
    if dataset_type == "car_scratch_dent":
        # Car Scratch and Dent dataset: dent, dirt, scratch
        if row['dent'] == 1:
            return 2  # major_damage (вмятины - серьезно)
        elif row['scratch'] == 1:
            return 1  # minor_damage (царапины - незначительно)
        else:
            return 0  # no_damage (хотя таких почти нет в этом датасете)
            
    elif dataset_type == "rust_scratch":
        # Rust and Scratch dataset: car, dunt, rust, scracth
        if row['rust'] == 1 or row['dunt'] == 1:
            return 2  # major_damage (ржавчина/вмятины - серьезно)
        elif row['scracth'] == 1:  # Опечатка в названии столбца
            return 1  # minor_damage (царапины - незначительно)
        elif row['car'] == 1:
            return 0  # no_damage (чистая машина)
        else:
            return 1  # по умолчанию minor если что-то непонятно
    
    elif dataset_type == "dent_detection":
        # Dent Detection dataset: бинарная колонка "dent" (0/1)
        if row['dent'] == 1:
            return 1  # minor_damage (вмятины как незначительные)
        else:
            return 0  # no_damage (нет вмятин)
    
    return 1  # fallback

def normalize_columns(df):
    """
    Нормализует названия колонок CSV файлов для унификации
    """
    # Убираем пробелы и приводим к нижнему регистру
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    
    # Унифицируем ключевые имена (исправляем опечатки)
    rename_map = {
        "scracth": "scratch",  # Исправляем опечатку в Rust dataset
        "dunt": "dent",        # Исправляем опечатку в Rust dataset
        " dent": "dent",       # На всякий случай
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # Добавляем отсутствующие бинарные столбцы нулями
    for col in ["dent", "scratch", "rust", "clean", "car", "dirt"]:
        if col not in df.columns:
            df[col] = 0
    
    # Выравниваем имя файла
    if "image_path" not in df.columns and "filename" in df.columns:
        df = df.rename(columns={"filename": "image_path"})
    
    return df

def collect_images_from_roots(roots):
    """
    Собирает изображения из CSV файлов датасетов и новой интегрированной структуры
    
    Returns:
        tuple: (items, class_distribution)
            items: список кортежей (путь_к_файлу, класс_id)
            class_distribution: Counter с распределением классов
    """
    items = []
    class_distribution = Counter()
    
    print("🔍 ЗАГРУЗКА ДАННЫХ ИЗ ДАТАСЕТОВ")
    
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"❌ Путь не найден: {root}")
            continue
            
        print(f"\n📁 Обрабатываем: {root}")
        
        # Проверяем, это интегрированный датасет или CSV-based
        if "integrated_multiclass_dataset" in str(root):
            print("   🆕 Обнаружен интегрированный датасет с train/test/valid структурой")
            integrated_items, integrated_dist = load_integrated_dataset(root)
            items.extend(integrated_items)
            class_distribution.update(integrated_dist)
        else:
            # Обычная CSV логика
            csv_files = list(root.glob("*.csv"))
            if not csv_files:
                print(f"   ❌ Не найден CSV файл в {root}")
                continue
            
            csv_file = csv_files[0]
            print(f"   📄 Найден CSV: {csv_file.name}")
            
            csv_items, csv_dist = load_csv_dataset(root, csv_file)
            items.extend(csv_items)
            class_distribution.update(csv_dist)
    
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Всего изображений: {len(items)}")
    
    total = sum(class_distribution.values())
    for class_id in [0, 1, 2]:
        class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
        count = class_distribution[class_id]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"   {class_id} ({class_name}): {count} ({percentage:.1f}%)")
    
    return items, class_distribution

def load_integrated_dataset(root_path):
    """
    Загружает данные из интегрированного датасета с train/test/valid структурой
    """
    items = []
    class_distribution = Counter()
    
    class_mapping = {
        "no_damage": 0,
        "minor_damage": 1, 
        "major_damage": 2
    }
    
    # Обрабатываем все split'ы (train, test, valid)
    for split_name in ["train", "test", "valid"]:
        split_path = root_path / split_name
        if not split_path.exists():
            continue
            
        print(f"   📂 Split: {split_name}")
        
        for class_name, class_id in class_mapping.items():
            class_path = split_path / class_name
            if not class_path.exists():
                continue
                
            # Собираем все изображения из папки класса
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(class_path.glob(ext))
            
            for img_path in image_files:
                items.append((str(img_path), class_id))
                class_distribution[class_id] += 1
            
            print(f"      {class_name}: {len(image_files)} изображений")
    
    return items, class_distribution

def load_csv_dataset(root_path, csv_file):
    """
    Загружает данные из CSV-based датасета
    """
    items = []
    class_distribution = Counter()
    
    try:
        df = pd.read_csv(csv_file)
        print(f"   📊 Строк в CSV: {len(df)}")
        print(f"   📋 Исходные колонки: {list(df.columns)}")
        
        # Нормализуем названия колонок
        df = normalize_columns(df)
        print(f"   📋 Нормализованные колонки: {list(df.columns)}")
        
    except Exception as e:
        print(f"   ❌ Ошибка чтения CSV: {e}")
        return items, class_distribution
            
        csv_file = csv_files[0]
        print(f"   📄 Найден CSV: {csv_file.name}")
        
        # Загружаем CSV
        try:
            df = pd.read_csv(csv_file)
            print(f"   📊 Строк в CSV: {len(df)}")
            print(f"   📋 Исходные колонки: {list(df.columns)}")
            
            # Нормализуем названия колонок
            df = normalize_columns(df)
            print(f"   📋 Нормализованные колонки: {list(df.columns)}")
            
        except Exception as e:
            print(f"   ❌ Ошибка чтения CSV: {e}")
            return items, class_distribution
        
        # Определяем тип датасета по столбцам
        columns = set(df.columns)
        if {'image_path', 'dent', 'dirt', 'scratch'}.issubset(columns):
            dataset_type = "car_scratch_dent"
            print(f"   🏷️ Тип датасета: Car Scratch and Dent")
        elif {'image_path', 'car', 'dent', 'rust', 'scratch'}.issubset(columns):
            dataset_type = "rust_scratch"  
            print(f"   🏷️ Тип датасета: Rust and Scratch")
        elif {'image_path', 'dent'}.issubset(columns) and len(columns) <= 3:
            dataset_type = "dent_detection"
            print(f"   🏷️ Тип датасета: Dent Detection (binary)")
        else:
            print(f"   ❌ Неизвестный формат CSV. Колонки: {columns}")
            print(f"   ❌ Ожидаемые форматы:")
            print(f"      - Car Scratch: ['image_path', 'dent', 'dirt', 'scratch']")
            print(f"      - Rust Scratch: ['image_path', 'car', 'dent', 'rust', 'scratch']") 
            print(f"      - Dent Detection: ['image_path', 'dent']")
            return items, class_distribution
        
        # Обрабатываем каждую строку
        processed_count = 0
        debug_mapping = {"dent_detection": {"minor": 0, "no_damage": 0}}
        
        for idx, row in df.iterrows():
            image_name = row['image_path']
            image_path = root / image_name
            
            # Проверяем существование файла
            if not image_path.exists():
                continue  # Пропускаем отсутствующие файлы молча
            
            # Конвертируем в multi-class
            class_id = convert_multilabel_to_multiclass(row, dataset_type)
            
            # Отладка для Dent Detection
            if dataset_type == "dent_detection":
                dent_val = row.get('dent', 0)
                if dent_val == 1:
                    debug_mapping["dent_detection"]["minor"] += 1
                else:
                    debug_mapping["dent_detection"]["no_damage"] += 1
                if processed_count < 5:  # Первые 5 строк
                    print(f"      DEBUG: row {idx}: dent={dent_val} → class_id={class_id}")
            
            items.append((str(image_path), class_id))
            class_distribution[class_id] += 1
            processed_count += 1
        
        # Отладочная информация для Dent Detection
        if dataset_type == "dent_detection":
            print(f"   🔍 ОТЛАДКА Dent Detection:")
            print(f"      dent=1 → minor_damage: {debug_mapping['dent_detection']['minor']}")
            print(f"      dent=0 → no_damage: {debug_mapping['dent_detection']['no_damage']}")
        
        print(f"   ✅ Обработано изображений: {processed_count}")
        
        # Показываем распределение по этому датасету
        local_dist = Counter()
        for _, class_id in items[-processed_count:]:
            local_dist[class_id] += 1
        
        print(f"   📈 Локальное распределение:")
        for class_id in [0, 1, 2]:
            class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
            count = local_dist[class_id]
            print(f"      {class_id} ({class_name}): {count}")
    
    # Перемешиваем данные
    random.Random(RANDOM_SEED).shuffle(items)
    
    # Общая статистика
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"   Всего изображений: {len(items)}")
    total_images = len(items)
    for class_id in [0, 1, 2]:
        class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
        count = class_distribution[class_id]
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"   {class_id} ({class_name}): {count} ({percentage:.1f}%)")
    
    return items, class_distribution

class MulticlassDamageDataset(Dataset):
    """Датасет для многоклассовой классификации повреждений"""
    
    def __init__(self, items, transforms=None):
        self.items = items
        self.transforms = transforms
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, class_id = self.items[idx]
        
        try:
            # Загружаем изображение
            image = Image.open(img_path).convert('RGB')
            
            # Применяем трансформации
            if self.transforms:
                image = self.transforms(image)
            
            return image, class_id
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {img_path}: {e}")
            # Возвращаем черное изображение как fallback
            if self.transforms:
                dummy = self.transforms(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                dummy = torch.zeros(3, 224, 224)
            return dummy, class_id

def split_train_validation(items, val_ratio=0.3, seed=RANDOM_SEED):
    """Разделение на train/validation с сохранением пропорций классов (увеличена доля валидации)"""
    random.Random(seed).shuffle(items)
    
    # Группируем по классам
    class_items = {0: [], 1: [], 2: []}
    for item in items:
        class_items[item[1]].append(item)
    
    train_items = []
    val_items = []
    
    # Для каждого класса делаем split
    for class_id, class_data in class_items.items():
        n_val = int(len(class_data) * val_ratio)
        val_items.extend(class_data[:n_val])
        train_items.extend(class_data[n_val:])
    
    # Перемешиваем
    random.Random(seed).shuffle(train_items)
    random.Random(seed).shuffle(val_items)
    
    print(f"📊 Разделение данных (validation {val_ratio*100:.0f}%):")
    print(f"   Train: {len(train_items)} изображений")
    print(f"   Validation: {len(val_items)} изображений")
    
    # Проверяем распределение по классам
    train_dist = Counter([item[1] for item in train_items])
    val_dist = Counter([item[1] for item in val_items])
    
    print(f"   📈 Распределение train:")
    for class_id in [0, 1, 2]:
        class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
        count = train_dist[class_id]
        print(f"      {class_id} ({class_name}): {count}")
    
    print(f"   📈 Распределение validation:")
    for class_id in [0, 1, 2]:
        class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
        count = val_dist[class_id]
        print(f"      {class_id} ({class_name}): {count}")
    
    return train_items, val_items

def load_integrated_dataset_split(root_path):
    """
    Загружает интегрированный датасет с уже готовым split'ом train/test/valid
    Возвращает записи для обучения и валидации
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
    
    # Обрабатываем valid split (используем как валидацию)
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

def proper_dataset_split(roots, val_ratio=0.3, seed=42):
    """
    Правильное разделение по датасетам:
    - Dent_Detection → только train
    - Остальные → stratified train/val split
    - Сохраняем source для каждого сэмпла
    """
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    print("🔄 ПРАВИЛЬНОЕ РАЗДЕЛЕНИЕ ПО ДАТАСЕТАМ")
    print("   Dent_Detection → ТОЛЬКО train")
    print("   Остальные → честный stratified split")
    
    train_records = []
    val_records = []
    
    for root in roots:
        root_path = Path(root)
        
        print(f"\n📁 Обрабатываем: {root_path.name}")
        
        # Специальная обработка интегрированного датасета
        if "integrated_multiclass_dataset" in str(root_path):
            print("   🆕 Обнаружен интегрированный датасет")
            integrated_train, integrated_val = load_integrated_dataset_split(root_path)
            train_records.extend(integrated_train)
            val_records.extend(integrated_val)
            continue
        
        # Ищем CSV файл для обычных датасетов
        csv_files = list(root_path.glob("*.csv"))
        if not csv_files:
            print(f"   ❌ Не найден CSV файл в {root_path}")
            continue
            
        csv_file = csv_files[0]
        
        # Загружаем и нормализуем CSV
        try:
            df = pd.read_csv(csv_file)
            df = normalize_columns(df)
            
            print(f"   📄 CSV файл: {csv_file.name}")
            print(f"   📊 Строк в CSV: {len(df)}")
            
        except Exception as e:
            print(f"   ❌ Ошибка чтения CSV: {e}")
            continue
        
        # Определяем правильное имя датасета (родительская папка, а не "train")
        dataset_name = root_path.parent.name.lower()  # Получаем имя датасета 
        source_name = dataset_name  # Используем правильное имя источника
        
        print(f"   🏷️ Источник данных: {dataset_name}")
        
        columns = set(df.columns)
        
        if {'image_path', 'dent', 'dirt', 'scratch'}.issubset(columns):
            dataset_type = "car_scratch_dent"
            print(f"   🏷️ Тип: Car Scratch and Dent")
        elif {'image_path', 'car', 'dent', 'rust', 'scratch'}.issubset(columns):
            dataset_type = "rust_scratch"  
            print(f"   🏷️ Тип: Rust and Scratch")
        elif {'image_path', 'dent'}.issubset(columns) and len(columns) <= 3:
            dataset_type = "dent_detection"
            print(f"   🏷️ Тип: Dent Detection (binary)")
        else:
            print(f"   ❌ Неизвестный формат CSV")
            continue
        
        # Создаем записи с полной информацией
        records = []
        processed_count = 0
        
        for idx, row in df.iterrows():
            image_name = row['image_path']
            image_path = root_path / image_name
            
            # Проверяем существование файла
            if not image_path.exists():
                continue
            
            # Конвертируем в multi-class
            class_id = convert_multilabel_to_multiclass(row, dataset_type)
            
            # Создаем полную запись
            record = {
                'path': str(image_path),
                'label': int(class_id),
                'source': source_name,
                'dataset_type': dataset_type
            }
            
            records.append(record)
            processed_count += 1
        
        print(f"   ✅ Обработано изображений: {processed_count}")
        
        # Показываем локальное распределение
        local_dist = Counter([rec['label'] for rec in records])
        print(f"   📈 Локальное распределение:")
        for class_id in [0, 1, 2]:
            class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
            count = local_dist[class_id]
            percentage = (count / processed_count * 100) if processed_count > 0 else 0
            print(f"      {class_id} ({class_name}): {count} ({percentage:.1f}%)")
        
        # Разделяем по стратегии на основе ПРАВИЛЬНОГО имени датасета
        if "dent_detection" in dataset_name:
            # ❗ Dent_Detection полностью в train
            train_records.extend(records)
            print(f"   🟡 Добавлено в TRAIN (только): {len(records)} записей")
        else:
            # Честный stratified split для остальных
            if len(records) > 0:
                labels = np.array([rec['label'] for rec in records])
                indices = np.arange(len(records))
                
                # Проверяем наличие всех классов для stratify
                unique_labels = np.unique(labels)
                if len(unique_labels) > 1:
                    try:
                        train_idx, val_idx = train_test_split(
                            indices,
                            test_size=val_ratio,
                            random_state=seed,
                            stratify=labels
                        )
                    except ValueError as e:
                        print(f"   ⚠️ Stratify failed: {e}, using random split")
                        # Если stratify не удается, делаем обычный split
                        train_idx, val_idx = train_test_split(
                            indices,
                            test_size=val_ratio,
                            random_state=seed
                        )
                else:
                    # Если только один класс, делаем простое разделение
                    split_point = int(len(records) * (1 - val_ratio))
                    train_idx = indices[:split_point]
                    val_idx = indices[split_point:]
                
                train_subset = [records[i] for i in train_idx]
                val_subset = [records[i] for i in val_idx]
                
                train_records.extend(train_subset)
                val_records.extend(val_subset)
                
                print(f"   🟢 Добавлено в TRAIN: {len(train_subset)} записей")
                print(f"   🟢 Добавлено в VAL: {len(val_subset)} записей")
    
    # Жёсткая проверка что Dent_Detection не попал в validation
    dent_in_val = [rec for rec in val_records if "dent_detection" in rec['source']]
    if len(dent_in_val) > 0:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: Dent_Detection попал в validation: {len(dent_in_val)} записей!")
        print(f"   Проблемные источники в val: {set(rec['source'] for rec in dent_in_val)}")
        raise AssertionError("Dent_Detection должен быть ТОЛЬКО в train!")
    
    print(f"✅ Проверка пройдена: Dent_Detection только в train")
    
    # Перемешиваем
    random.Random(seed).shuffle(train_records)
    random.Random(seed).shuffle(val_records)
    
    # Подробная статистика по источникам
    def summarize_records(records, title):
        from collections import Counter
        by_src = Counter(rec['source'] for rec in records)
        by_cls = Counter(rec['label'] for rec in records)
        
        print(f"\n📈 {title}:")
        print(f"   📊 По источникам:")
        for source, count in sorted(by_src.items()):
            percentage = (count / len(records) * 100) if len(records) > 0 else 0
            print(f"      • {source}: {count} ({percentage:.1f}%)")
        
        print(f"   📊 По классам:")
        for class_id in [0, 1, 2]:
            class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
            count = by_cls[class_id]
            percentage = (count / len(records) * 100) if len(records) > 0 else 0
            print(f"      • {class_id} ({class_name}): {count} ({percentage:.1f}%)")
    
    # Подробный анализ
    summarize_records(train_records, "TRAIN sources/classes")
    summarize_records(val_records, "VAL sources/classes")
    
    # Финальная статистика
    print(f"\n📊 ФИНАЛЬНАЯ СТАТИСТИКА:")
    print(f"   Train: {len(train_records)} записей")
    print(f"   Validation: {len(val_records)} записей")
    
    # Подробный анализ
    summarize_records(train_records, "TRAIN sources/classes")
    summarize_records(val_records, "VAL sources/classes")
    
    # Конвертируем в старый формат (path, label) для совместимости
    train_items = [(rec['path'], rec['label']) for rec in train_records]
    val_items = [(rec['path'], rec['label']) for rec in val_records]
    
    # Сохраняем записи для использования в sampler
    return train_items, val_items, train_records, val_records

def split_data_stratified_internal(items, val_ratio=0.3, seed=42):
    """
    Стратифицированное разделение данных с сохранением пропорций классов
    """
    # Группируем по классам
    class_items = {}
    for item in items:
        class_id = item[1]
        if class_id not in class_items:
            class_items[class_id] = []
        class_items[class_id].append(item)
    
    # Перемешиваем внутри каждого класса
    for class_id in class_items:
        random.Random(seed).shuffle(class_items[class_id])
    
    train_items = []
    val_items = []
    
    # Для каждого класса делаем split
    for class_id, class_data in class_items.items():
        n_val = int(len(class_data) * val_ratio)
        val_items.extend(class_data[:n_val])
        train_items.extend(class_data[n_val:])
    
    return train_items, val_items

def create_enhanced_sampler_weights(train_records, boost_no_damage=1.4):
    """
    Создание весов для сэмплера с учетом источников данных и приглушением Dent_Detection
    """
    print("\n🎯 СОЗДАНИЕ УЛУЧШЕННЫХ ВЕСОВ ДЛЯ СЭМПЛЕРА")
    
    # Веса по датасетам (приглушаем Dent_Detection)
    dataset_weights = {
        "rust and scrach.v1i.multiclass": 1.0,
        "car scratch and dent.v5i.multiclass": 1.0,
        "dent_detection.v1i.multiclass": 0.6,  # Приглушаем перекос
    }
    
    # Получаем только train labels для расчета класс весов
    train_labels = [rec['label'] for rec in train_records]
    
    # Effective Number weights
    class_counts = Counter(train_labels)
    cls_counts = np.array([class_counts[0], class_counts[1], class_counts[2]], dtype=np.float32)
    
    print(f"   📊 Распределение классов в train:")
    total_train = len(train_labels)
    for i, count in enumerate(cls_counts):
        class_name = ["no_damage", "minor_damage", "major_damage"][i]
        percentage = (count / total_train * 100)
        print(f"      {i} ({class_name}): {count:.0f} ({percentage:.1f}%)")
    
    # Effective Number calculation
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    
    # Буст для no_damage класса
    weights[0] *= boost_no_damage
    
    # Нормализуем веса
    weights = weights / weights.sum() * len(weights)
    
    # Ограничиваем разброс весов (max/min ≤ 12×)
    max_ratio = 12.0
    weight_ratio = weights.max() / weights.min()
    if weight_ratio > max_ratio:
        print(f"   ⚠️  Ограничиваем разброс весов: {weight_ratio:.2f}× → {max_ratio:.2f}×")
        weights = np.clip(weights, weights.min(), weights.min() * max_ratio)
        weights = weights / weights.sum() * len(weights)
    
    print(f"   🎯 Effective Number веса классов:")
    for i, weight in enumerate(weights):
        class_name = ["no_damage", "minor_damage", "major_damage"][i]
        print(f"      {i} ({class_name}): {weight:.2f}")
    
    print(f"   🎯 Веса по источникам:")
    for source, weight in dataset_weights.items():
        print(f"      {source}: {weight:.1f}")
    
    # Создаем sample weights
    sample_weights = []
    
    for rec in train_records:
        class_weight = weights[rec['label']]
        
        # Находим подходящий dataset weight
        dataset_weight = 1.0
        for source_key, dw in dataset_weights.items():
            if source_key in rec['source']:
                dataset_weight = dw
                break
        
        # Комбинированный вес
        combined_weight = class_weight * dataset_weight
        sample_weights.append(combined_weight)
    
    print(f"   ✅ Создано {len(sample_weights)} sample weights")
    
    # Статистика весов
    sample_weights = np.array(sample_weights)
    print(f"   📈 Статистика sample weights:")
    print(f"      Min: {sample_weights.min():.3f}")
    print(f"      Max: {sample_weights.max():.3f}")
    print(f"      Mean: {sample_weights.mean():.3f}")
    print(f"      Ratio: {sample_weights.max()/sample_weights.min():.2f}×")
    
    return sample_weights, weights

def create_effective_number_weights(labels, boost_no_damage=1.3):
    """
    Создание весов классов с использованием Effective Number of Samples
    + дополнительный буст для no_damage класса
    """
    class_counts = Counter(labels)
    cls_counts = np.array([class_counts[0], class_counts[1], class_counts[2]], dtype=np.float32)
    
    print(f"📊 Effective Number Weights расчет:")
    print(f"   Изначальное распределение: {cls_counts}")
    
    # Effective Number (β близко к 1 для больших дисбалансов)
    beta = 0.9999
    eff_num = (1 - np.power(beta, cls_counts)) / (1 - beta)
    weights = eff_num.sum() / eff_num
    
    print(f"   Effective Numbers: {eff_num}")
    print(f"   Веса до бустинга: {weights}")
    
    # Дополнительно бустим класс no_damage (класс 0)
    weights[0] *= boost_no_damage
    
    print(f"   Буст для no_damage: ×{boost_no_damage}")
    print(f"   Финальные веса: {weights}")
    
    return torch.tensor(weights, dtype=torch.float32)

def create_weighted_sampler(labels):
    """Создание сэмплера для балансировки классов (обновленный)"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Вычисляем веса классов (обратно пропорционально частоте)
    class_weights = {}
    for class_id, count in class_counts.items():
        class_weights[class_id] = total_samples / (len(class_counts) * count)
    
    # Создаем веса для каждого семпла
    sample_weights = [class_weights[label] for label in labels]
    
    print(f"📊 Веса классов для сэмплера:")
    for class_id in sorted(class_weights.keys()):
        class_name = ["no_damage", "minor_damage", "major_damage"][class_id]
        weight = class_weights[class_id]
        count = class_counts[class_id]
        print(f"   {class_id} ({class_name}): {weight:.3f} (семплов: {count})")
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def calculate_metrics(y_true, y_pred, y_scores=None):
    """Вычисление метрик качества"""
    metrics = {}
    
    # Основные метрики
    metrics['accuracy'] = (y_true == y_pred).mean()
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Метрики по классам
    f1_per_class = f1_score(y_true, y_pred, average=None)
    class_names = ["no_damage", "minor_damage", "major_damage"]
    
    for i, class_name in enumerate(class_names):
        metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics

def save_confusion_matrix(y_true, y_pred, save_path):
    """Сохранение матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["no_damage", "minor_damage", "major_damage"]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad_norm=1.0):
    """Тренировка одной эпохи с градиентным клиппингом"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Градиентное клиппирование (критически важно с FocalLoss + class weights)
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Обновляем прогресс
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss / len(dataloader), correct / total

def save_confusion_matrix(y_true, y_pred, save_path):
    """Сохраняет confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['no_damage', 'minor_damage', 'major_damage'],
                yticklabels=['no_damage', 'minor_damage', 'major_damage'])
    plt.title('Confusion Matrix (Best Model)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def validate_epoch(model, dataloader, criterion, device):
    """Валидация одной эпохи"""
    model.eval()
    total_loss = 0
    y_true = []
    y_pred = []
    y_scores = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # Сохраняем для метрик
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(F.softmax(outputs, dim=1).cpu().numpy())
    
    # Вычисляем метрики
    metrics = calculate_metrics(np.array(y_true), np.array(y_pred), np.array(y_scores))
    
    return total_loss / len(dataloader), metrics, y_true, y_pred

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, save_dir, patience=7, freeze_backbone_epochs=0):
    """Основной цикл обучения с улучшенной ранней остановкой"""
    
    best_val_f1 = 0
    bad_epochs = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_f1s = []
    
    print(f"\n🚀 Начинаем обучение на {num_epochs} эпох")
    print(f"🔧 Устройство: {device}")
    print(f"⏰ Early stopping patience: {patience}")
    if freeze_backbone_epochs > 0:
        print(f"🧊 Заморозка backbone на {freeze_backbone_epochs} эпох")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Эпоха {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Опциональная заморозка backbone в начале
        if epoch < freeze_backbone_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("🧊 Backbone заморожен - обучается только classifier")
        elif epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            for param in model.backbone.parameters():
                param.requires_grad = True
            print("🔥 Backbone разморожен - обучается вся модель")
        
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, val_metrics, y_true, y_pred = validate_epoch(model, val_loader, criterion, device)
        
        # Обновляем learning rate (важно: по macro F1, не по loss!)
        scheduler.step(val_metrics['macro_f1'])
        
        # Сохраняем статистику
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_f1s.append(val_metrics['macro_f1'])
        
        # Выводим результаты
        print(f"\n📊 Результаты эпохи {epoch+1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val F1 (macro): {val_metrics['macro_f1']:.4f}")
        print(f"   Val F1 (weighted): {val_metrics['weighted_f1']:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # F1-score по классам (КРИТИЧЕСКИ ВАЖНО для imbalanced data!)
        print(f"   F1 по классам:")
        for class_name in ["no_damage", "minor_damage", "major_damage"]:
            f1_val = val_metrics[f'f1_{class_name}']
            print(f"     {class_name}: {f1_val:.4f}")
        
        # Проверяем улучшение
        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            bad_epochs = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'val_metrics': val_metrics
            }, save_dir / 'best_model.pth')
            
            # Сохраняем confusion matrix для лучшей модели
            save_confusion_matrix(y_true, y_pred, save_dir / 'confusion_matrix.png')
            
            print(f"   ✅ Новая лучшая модель! F1: {best_val_f1:.4f}")
        else:
            bad_epochs += 1
            print(f"   📉 Без улучшения: {bad_epochs}/{patience}")
        
        # Ранняя остановка по macro F1 (НЕ по loss!)
        if bad_epochs >= patience:
            print(f"\n⏰ Ранняя остановка на эпохе {epoch+1}")
            print(f"   Лучший macro F1: {best_val_f1:.4f}")
            break
        
        # Мониторинг overfitting
        if len(train_losses) > 1:
            train_val_gap = train_acc - val_metrics['accuracy']
            if train_val_gap > 0.2:  # 20% gap
                print(f"   ⚠️ Признаки overfitting: gap = {train_val_gap:.3f}")
    
    # Сохраняем графики обучения
    save_training_plots(train_losses, val_losses, train_accs, val_f1s, save_dir)
    
    return best_val_f1

def save_training_plots(train_losses, val_losses, train_accs, val_f1s, save_dir):
    """Сохранение графиков обучения"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy vs F1
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_f1s, label='Validation F1')
    ax2.set_title('Training Accuracy vs Validation F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True)
    
    # Learning curves
    ax3.plot(range(len(train_losses)), train_losses, 'b-', alpha=0.7, label='Train')
    ax3.plot(range(len(val_losses)), val_losses, 'r-', alpha=0.7, label='Validation')
    ax3.set_title('Learning Curves')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True)
    
    # F1 progression
    ax4.plot(val_f1s, 'g-', linewidth=2)
    ax4.set_title('Validation F1 Score Progress')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1 Score')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_plots.png', dpi=150)
    plt.close()

def main():
    """Основная функция обучения"""
    print("🤖 Система обучения многоклассовой модели повреждений")
    print("="*60)
    
    # Создаем директорию для результатов
    save_dir = Path("training_results")
    save_dir.mkdir(exist_ok=True)
    
    # Загружаем данные с правильной стратегией разделения
    print("\n1️⃣ Загрузка данных с улучшенной стратегией")
    train_items, val_items, train_records, val_records = proper_dataset_split(DATASET_ROOTS, val_ratio=0.3)
    # Проверяем что данные загружены
    if len(train_items) == 0:
        print("❌ Не найдено ни одного тренировочного изображения!")
        return
    
    # Создаем трансформации
    print("\n2️⃣ Подготовка трансформаций")
    train_transforms = create_training_transforms()
    val_transforms = create_validation_transforms()
    
    # Создаем датасеты
    train_dataset = MulticlassDamageDataset(train_items, train_transforms)
    val_dataset = MulticlassDamageDataset(val_items, val_transforms)
    
    # Создаем улучшенный сэмплер с учетом источников
    print("\n3️⃣ Создание сэмплера с учетом источников")
    sample_weights, class_weights = create_enhanced_sampler_weights(train_records, boost_no_damage=1.4)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Создаем DataLoader'ы с критически важными настройками
    batch_size = 16  # Оптимально для CPU
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,  # Для CPU не больше
        pin_memory=False,  # Не нужно без GPU
        drop_last=True  # КРИТИЧНО: избегаем BN ошибку на batch_size=1
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,  # Для валидации можно больше
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Создаем модель
    print("\n4️⃣ Инициализация модели")
    model = MulticlassDamageModel(num_classes=3, dropout=0.5)
    model = model.to(DEVICE)
    
    # Подсчитываем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Общее количество параметров: {total_params:,}")
    print(f"   Обучаемые параметры: {trainable_params:,}")
    
    # Создаем FocalLoss с обновленными весами
    print("\n4️⃣ Настройка loss функции")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    # FocalLoss с gamma=2.0 (можно попробовать 2.5 если no_damage F1 низкий)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    
    print(f"   Финальные веса для FocalLoss: {class_weights}")
    print(f"   Gamma: 2.0 (оптимально для дисбаланса)")
    print(f"   ✅ Loss учитывает источники данных и класс дисбаланс")
    
    # ДИФФЕРЕНЦИРОВАННЫЙ LEARNING RATE (критически важно!)
    print("\n5️⃣ Настройка оптимизатора")
    backbone_lr = 1e-5  # Осторожно с pretrained weights
    classifier_lr = 1e-4  # Новые слои быстрее
    
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.classifier.parameters(), "lr": classifier_lr},
    ], weight_decay=5e-4)  # Увеличено с 1e-4 до 5e-4 против overfitting
    
    print(f"   Backbone LR: {backbone_lr} (осторожно с pretrained)")
    print(f"   Classifier LR: {classifier_lr} (новые слои)")
    print(f"   Weight decay: 5e-4 (усилена регуляризация против overfitting)")
    
    # Scheduler по macro F1 (НЕ по loss!) с совместимостью версий
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7, verbose=True
        )
    except TypeError:
        # Старые версии PyTorch не поддерживают verbose
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7
        )
    
    print(f"   Scheduler: ReduceLROnPlateau по macro-F1")
    print(f"   Factor: 0.5, Patience: 3, Min LR: 1e-7")
    
    # Обучение с улучшенными настройками
    print("\n6️⃣ Начинаем полное обучение с тремя датасетами")
    num_epochs = 30  # Полное обучение на 30 эпох
    freeze_backbone_epochs = 3  # Заморозка backbone на 3 эпохи
    
    best_f1 = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=DEVICE,
        save_dir=save_dir,
        patience=10,  # Early stopping после 10 эпох без улучшения
        freeze_backbone_epochs=freeze_backbone_epochs
    )
    
    print(f"\n🎉 Обучение завершено!")
    print(f"   Лучший macro F1-score: {best_f1:.4f}")
    print(f"   Результаты сохранены в: {save_dir}")
    
    # Сохраняем финальную статистику
    total_images = len(train_items) + len(val_items)
    train_labels = [item[1] for item in train_items]
    val_labels = [item[1] for item in val_items]
    combined_labels = train_labels + val_labels
    
    final_stats = {
        'dataset_info': {
            'total_images': total_images,
            'train_images': len(train_items),
            'val_images': len(val_items),
            'class_distribution': dict(Counter(combined_labels))
        },
        'training_info': {
            'device': DEVICE,
            'batch_size': batch_size,
            'backbone_lr': backbone_lr,
            'classifier_lr': classifier_lr,
            'num_epochs': num_epochs,
            'best_f1': float(best_f1),
            'gamma': 2.0,
            'boost_no_damage': 1.3
        }
    }
    
    with open(save_dir / 'training_stats.json', 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Воспроизводимость
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    main()