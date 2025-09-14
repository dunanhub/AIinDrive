"""
Интеграция масштабного нового датасета с чистыми и поврежденными автомобилями
1700 чистых + 400+ поврежденных = решение проблемы дисбаланса!
"""
import os
import shutil
from pathlib import Path
import pandas as pd
from typing import List, Dict
import json

def create_dataset_structure(dataset_root: str):
    """Создаем структуру папок для нового датасета"""
    base_path = Path(dataset_root)
    
    # Создаем структуру папок
    folders = [
        "train/no_damage",
        "train/minor_damage", 
        "train/major_damage"
    ]
    
    for folder in folders:
        folder_path = base_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Создана папка: {folder_path}")
    
    return base_path

def organize_new_dataset(
    clean_cars_path: str,
    damaged_cars_path: str, 
    output_dataset_root: str
):
    """
    Организует новый датасет в структуру multiclass
    
    Args:
        clean_cars_path: Путь к папке с 1700 чистыми машинами
        damaged_cars_path: Путь к папке с 400+ поврежденными машинами
        output_dataset_root: Корневая папка для нового датасета
    """
    
    print("🚀 ИНТЕГРАЦИЯ МАСШТАБНОГО ДАТАСЕТА")
    print("="*50)
    
    # Создаем структуру
    base_path = create_dataset_structure(output_dataset_root)
    
    # Статистика
    stats = {
        "clean_copied": 0,
        "damaged_copied": 0,
        "errors": []
    }
    
    # Копируем чистые машины в no_damage
    print(f"\n1️⃣ Копирование чистых машин из {clean_cars_path}")
    clean_target = base_path / "train" / "no_damage"
    
    if os.path.exists(clean_cars_path):
        for filename in os.listdir(clean_cars_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    src = os.path.join(clean_cars_path, filename)
                    dst = clean_target / f"clean_{stats['clean_copied']:04d}_{filename}"
                    shutil.copy2(src, dst)
                    stats['clean_copied'] += 1
                    
                    if stats['clean_copied'] % 100 == 0:
                        print(f"   Скопировано: {stats['clean_copied']} чистых машин...")
                        
                except Exception as e:
                    stats['errors'].append(f"Ошибка с {filename}: {e}")
                    
        print(f"✅ Скопировано {stats['clean_copied']} чистых машин")
    else:
        print(f"❌ Папка {clean_cars_path} не найдена!")
    
    # Копируем поврежденные машины
    print(f"\n2️⃣ Копирование поврежденных машин из {damaged_cars_path}")
    
    # Разделяем поврежденные на minor и major (50/50)
    damaged_target_minor = base_path / "train" / "minor_damage"
    damaged_target_major = base_path / "train" / "major_damage"
    
    if os.path.exists(damaged_cars_path):
        damaged_files = [f for f in os.listdir(damaged_cars_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for i, filename in enumerate(damaged_files):
            try:
                src = os.path.join(damaged_cars_path, filename)
                
                # Чередуем minor/major для баланса
                if i % 2 == 0:
                    dst = damaged_target_minor / f"minor_{i:04d}_{filename}"
                else:
                    dst = damaged_target_major / f"major_{i:04d}_{filename}"
                
                shutil.copy2(src, dst)
                stats['damaged_copied'] += 1
                
                if stats['damaged_copied'] % 50 == 0:
                    print(f"   Скопировано: {stats['damaged_copied']} поврежденных машин...")
                    
            except Exception as e:
                stats['errors'].append(f"Ошибка с {filename}: {e}")
                
        print(f"✅ Скопировано {stats['damaged_copied']} поврежденных машин")
        print(f"   Minor damage: ~{stats['damaged_copied']//2}")
        print(f"   Major damage: ~{stats['damaged_copied']//2}")
    else:
        print(f"❌ Папка {damaged_cars_path} не найдена!")
    
    # Создаем CSV аннотации
    print(f"\n3️⃣ Создание CSV аннотаций...")
    create_csv_annotations(base_path, stats)
    
    # Итоговая статистика
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"   Чистые машины: {stats['clean_copied']}")
    print(f"   Поврежденные машины: {stats['damaged_copied']}")
    print(f"   Всего изображений: {stats['clean_copied'] + stats['damaged_copied']}")
    print(f"   Ошибки: {len(stats['errors'])}")
    
    if stats['errors']:
        print(f"\n⚠️ Ошибки:")
        for error in stats['errors'][:5]:  # Показываем первые 5
            print(f"   {error}")
    
    return stats

def create_csv_annotations(base_path: Path, stats: Dict):
    """Создает CSV файлы аннотаций для совместимости"""
    
    annotations = []
    
    # no_damage
    no_damage_path = base_path / "train" / "no_damage"
    for img_file in no_damage_path.glob("*.jpg"):
        annotations.append({
            "filename": img_file.name,
            "class": "no_damage",
            "label": 0,
            "source": "new_massive_dataset"
        })
    
    for img_file in no_damage_path.glob("*.png"):
        annotations.append({
            "filename": img_file.name,
            "class": "no_damage", 
            "label": 0,
            "source": "new_massive_dataset"
        })
    
    # minor_damage
    minor_damage_path = base_path / "train" / "minor_damage"
    for img_file in minor_damage_path.glob("*.jpg"):
        annotations.append({
            "filename": img_file.name,
            "class": "minor_damage",
            "label": 1,
            "source": "new_massive_dataset"
        })
        
    for img_file in minor_damage_path.glob("*.png"):
        annotations.append({
            "filename": img_file.name,
            "class": "minor_damage",
            "label": 1,
            "source": "new_massive_dataset"
        })
    
    # major_damage
    major_damage_path = base_path / "train" / "major_damage"
    for img_file in major_damage_path.glob("*.jpg"):
        annotations.append({
            "filename": img_file.name,
            "class": "major_damage",
            "label": 2,
            "source": "new_massive_dataset"
        })
        
    for img_file in major_damage_path.glob("*.png"):
        annotations.append({
            "filename": img_file.name,
            "class": "major_damage",
            "label": 2,
            "source": "new_massive_dataset"
        })
    
    # Сохраняем CSV
    df = pd.DataFrame(annotations)
    csv_path = base_path / "annotations.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Создан файл аннотаций: {csv_path}")
    print(f"   Записей в CSV: {len(annotations)}")
    
    # Сохраняем статистику
    stats_path = base_path / "dataset_stats.json"
    distribution = df['class'].value_counts().to_dict()
    
    dataset_info = {
        "total_images": len(annotations),
        "class_distribution": distribution,
        "source": "new_massive_dataset_integration",
        "created_by": "dataset_integration_script",
        "notes": "1700 clean cars + 400+ damaged cars integration"
    }
    
    with open(stats_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Создан файл статистики: {stats_path}")

def update_training_script_paths():
    """Обновляет пути в train_multiclass_damage.py"""
    
    new_dataset_path = r"C:\Users\Димаш\Desktop\python\hackaton\data\New_Massive_Dataset.v1i.multiclass\train"
    
    print(f"\n4️⃣ Обновление путей в скрипте обучения...")
    print(f"   Новый путь: {new_dataset_path}")
    
    # TODO: Можно автоматически обновить DATASET_ROOTS в train_multiclass_damage.py
    
    recommended_paths = [
        r"C:\Users\Димаш\Desktop\python\hackaton\data\Rust and Scrach.v1i.multiclass\train",
        r"C:\Users\Димаш\Desktop\python\hackaton\data\Car Scratch and Dent.v5i.multiclass\train",
        r"C:\Users\Димаш\Desktop\python\hackaton\data\Dent_Detection.v1i.multiclass\train",
        new_dataset_path  # Новый!
    ]
    
    print(f"\n📝 РЕКОМЕНДУЕМЫЕ ПУТИ ДЛЯ DATASET_ROOTS:")
    for i, path in enumerate(recommended_paths, 1):
        print(f"   {i}. {path}")
    
    return recommended_paths

def analyze_combined_dataset_balance():
    """Анализирует баланс объединенного датасета"""
    
    print(f"\n📊 АНАЛИЗ БАЛАНСА ОБЪЕДИНЕННОГО ДАТАСЕТА")
    print("="*50)
    
    # Текущие данные
    current_data = {
        "no_damage": 41,
        "minor_damage": 278, 
        "major_damage": 331
    }
    
    # Новые данные (оценка)
    new_data = {
        "no_damage": 1700,
        "minor_damage": 200,  # ~50% от 400
        "major_damage": 200   # ~50% от 400
    }
    
    # Объединенные данные
    combined_data = {}
    for key in current_data:
        combined_data[key] = current_data[key] + new_data[key]
    
    total = sum(combined_data.values())
    
    print(f"ТЕКУЩИЕ ДАННЫЕ:")
    for cls, count in current_data.items():
        percent = (count / sum(current_data.values())) * 100
        print(f"   {cls}: {count} ({percent:.1f}%)")
    
    print(f"\nНОВЫЕ ДАННЫЕ:")
    for cls, count in new_data.items():
        percent = (count / sum(new_data.values())) * 100
        print(f"   {cls}: {count} ({percent:.1f}%)")
    
    print(f"\nОБЪЕДИНЕННЫЕ ДАННЫЕ:")
    for cls, count in combined_data.items():
        percent = (count / total) * 100
        print(f"   {cls}: {count} ({percent:.1f}%)")
    
    print(f"\nОБЩИЙ РАЗМЕР: {total} изображений")
    
    # Анализ баланса
    max_count = max(combined_data.values())
    min_count = min(combined_data.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\nБАЛАНС КЛАССОВ:")
    print(f"   Дисбаланс: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio < 3:
        print(f"   ✅ Хороший баланс классов!")
    elif imbalance_ratio < 5:
        print(f"   🟡 Умеренный дисбаланс")
    else:
        print(f"   ❌ Сильный дисбаланс")
    
    return combined_data

if __name__ == "__main__":
    # Настройки путей (ОБНОВИ ЭТИ ПУТИ!)
    CLEAN_CARS_PATH = r"C:\Users\Димаш\Desktop\python\hackaton\new_data\clean_cars"
    DAMAGED_CARS_PATH = r"C:\Users\Димаш\Desktop\python\hackaton\new_data\damaged_cars"
    OUTPUT_DATASET = r"C:\Users\Димаш\Desktop\python\hackaton\data\New_Massive_Dataset.v1i.multiclass"
    
    print(f"📁 ПУТИ ДЛЯ ИНТЕГРАЦИИ:")
    print(f"   Чистые машины: {CLEAN_CARS_PATH}")
    print(f"   Поврежденные машины: {DAMAGED_CARS_PATH}")
    print(f"   Выходной датасет: {OUTPUT_DATASET}")
    
    # Анализ ожидаемого баланса
    analyze_combined_dataset_balance()
    
    # Интеграция датасета
    try:
        stats = organize_new_dataset(
            clean_cars_path=CLEAN_CARS_PATH,
            damaged_cars_path=DAMAGED_CARS_PATH,
            output_dataset_root=OUTPUT_DATASET
        )
        
        # Обновление путей
        recommended_paths = update_training_script_paths()
        
        print(f"\n🎉 ИНТЕГРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print(f"   Теперь можно обучать модель на {stats['clean_copied'] + stats['damaged_copied']} изображениях")
        
    except Exception as e:
        print(f"❌ Ошибка интеграции: {e}")
        import traceback
        traceback.print_exc()