"""
Специализированный интегратор для car.v2i.multiclass и Car damages.v3i.multiclass
Обрабатывает CSV аннотации и создает правильную multiclass структуру
"""
import os
import pandas as pd
import shutil
from pathlib import Path
import json
from collections import defaultdict
from PIL import Image
import numpy as np

class NewDatasetIntegrator:
    def __init__(self):
        self.dataset_paths = {
            "car_v2": r"C:\Users\Димаш\Desktop\python\hackaton\data\car.v2i.multiclass",
            "car_damages_v3": r"C:\Users\Димаш\Desktop\python\hackaton\data\Car damages.v3i.multiclass"
        }
        
        self.output_base = r"C:\Users\Димаш\Desktop\python\hackaton\data"
        
        # Стратегии маппинга для каждого датасета
        self.mapping_strategies = {
            "car_v2": {
                "csv_columns": ["filename", "bus", "car", "repair-car", "truck"],
                "class_mapping": {
                    # Основываемся на one-hot encoding в CSV
                    "car": 0,           # no_damage (чистые машины)
                    "repair-car": 1,    # minor_damage (машины в ремонте)
                    "bus": 1,           # minor_damage (автобусы считаем поврежденными)
                    "truck": 1          # minor_damage (грузовики считаем поврежденными)
                },
                "description": "Датасет транспорта с one-hot кодированием"
            },
            
            "car_damages_v3": {
                "csv_columns": ["filename", "dent", "good_condition", "scratch", "severe damage"],
                "class_mapping": {
                    "good_condition": 0,    # no_damage
                    "scratch": 1,           # minor_damage  
                    "dent": 1,              # minor_damage
                    "severe damage": 2      # major_damage
                },
                "description": "Датасет повреждений автомобилей"
            }
        }
        
        self.class_names = ["no_damage", "minor_damage", "major_damage"]
        
    def analyze_csv_annotations(self, dataset_key: str):
        """Анализирует CSV аннотации для конкретного датасета"""
        
        dataset_path = self.dataset_paths[dataset_key]
        strategy = self.mapping_strategies[dataset_key]
        
        print(f"\n🔍 АНАЛИЗ CSV АННОТАЦИЙ: {dataset_key}")
        print("="*50)
        
        analysis = {
            "total_samples": 0,
            "splits": {},
            "class_distribution": defaultdict(int),
            "samples_per_split": {}
        }
        
        # Проходим по всем split'ам (train/test/valid)
        for split in ["train", "test", "valid"]:
            split_path = os.path.join(dataset_path, split)
            csv_path = os.path.join(split_path, "_classes.csv")
            
            if not os.path.exists(csv_path):
                continue
                
            print(f"\n📊 Анализ {split}:")
            
            df = pd.read_csv(csv_path)
            print(f"   Загружено записей: {len(df)}")
            print(f"   Колонки: {list(df.columns)}")
            
            # Анализируем one-hot кодирование
            split_distribution = defaultdict(int)
            
            for idx, row in df.iterrows():
                filename = row['filename']
                
                # Определяем класс на основе one-hot encoding
                predicted_class = self.predict_class_from_csv(row, strategy)
                
                if predicted_class is not None:
                    split_distribution[predicted_class] += 1
                    analysis["class_distribution"][predicted_class] += 1
                    analysis["total_samples"] += 1
            
            analysis["splits"][split] = dict(split_distribution)
            analysis["samples_per_split"][split] = len(df)
            
            print(f"   Распределение классов:")
            for class_idx, count in split_distribution.items():
                class_name = self.class_names[class_idx]
                percentage = (count / len(df)) * 100 if len(df) > 0 else 0
                print(f"     {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n📈 ОБЩАЯ СТАТИСТИКА {dataset_key}:")
        print(f"   Всего образцов: {analysis['total_samples']}")
        
        if analysis["total_samples"] > 0:
            for class_idx, count in analysis["class_distribution"].items():
                class_name = self.class_names[class_idx]
                percentage = (count / analysis["total_samples"]) * 100
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return analysis
    
    def predict_class_from_csv(self, row, strategy):
        """Предсказывает класс на основе one-hot кодирования в CSV"""
        
        class_mapping = strategy["class_mapping"]
        
        # Ищем активированные колонки (значение 1)
        for col_name, class_idx in class_mapping.items():
            if col_name in row and row[col_name] == 1:
                return class_idx
        
        # Если ничего не найдено, возвращаем None
        return None
    
    def create_integrated_dataset(self):
        """Создает объединенный датасет из всех источников"""
        
        print(f"\n🚀 СОЗДАНИЕ ИНТЕГРИРОВАННОГО ДАТАСЕТА")
        print("="*60)
        
        output_path = os.path.join(self.output_base, "integrated_multiclass_dataset")
        
        # Создаем структуру папок
        for split in ["train", "test", "valid"]:
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(output_path, split, class_name)
                os.makedirs(class_dir, exist_ok=True)
        
        total_stats = {
            "datasets_processed": 0,
            "total_images": 0,
            "class_distribution": defaultdict(int),
            "split_distribution": defaultdict(int),
            "errors": []
        }
        
        # Обрабатываем каждый датасет
        for dataset_key in self.dataset_paths.keys():
            print(f"\n📁 ОБРАБОТКА ДАТАСЕТА: {dataset_key}")
            
            try:
                dataset_stats = self.process_single_dataset(dataset_key, output_path)
                
                # Объединяем статистику
                total_stats["datasets_processed"] += 1
                total_stats["total_images"] += dataset_stats["images_processed"]
                
                for class_idx, count in dataset_stats["class_distribution"].items():
                    total_stats["class_distribution"][class_idx] += count
                
                for split, count in dataset_stats["split_distribution"].items():
                    total_stats["split_distribution"][split] += count
                    
            except Exception as e:
                error_msg = f"Ошибка обработки {dataset_key}: {e}"
                total_stats["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Выводим итоговую статистику
        self.print_integration_summary(total_stats, output_path)
        
        # Создаем CSV с мета-информацией
        self.create_integration_metadata(total_stats, output_path)
        
        return output_path, total_stats
    
    def process_single_dataset(self, dataset_key: str, output_path: str):
        """Обрабатывает один датасет"""
        
        dataset_path = self.dataset_paths[dataset_key]
        strategy = self.mapping_strategies[dataset_key]
        
        stats = {
            "images_processed": 0,
            "images_skipped": 0,
            "class_distribution": defaultdict(int),
            "split_distribution": defaultdict(int)
        }
        
        # Обрабатываем каждый split
        for split in ["train", "test", "valid"]:
            split_path = os.path.join(dataset_path, split)
            csv_path = os.path.join(split_path, "_classes.csv")
            
            if not os.path.exists(csv_path):
                continue
            
            print(f"   📂 Обработка {split}...")
            
            df = pd.read_csv(csv_path)
            
            for idx, row in df.iterrows():
                filename = row['filename']
                source_image_path = os.path.join(split_path, filename)
                
                if not os.path.exists(source_image_path):
                    stats["images_skipped"] += 1
                    continue
                
                # Определяем класс
                predicted_class = self.predict_class_from_csv(row, strategy)
                
                if predicted_class is None:
                    stats["images_skipped"] += 1
                    continue
                
                # Копируем изображение в соответствующую папку
                class_name = self.class_names[predicted_class]
                target_dir = os.path.join(output_path, split, class_name)
                
                # Создаем уникальное имя файла
                base_name, ext = os.path.splitext(filename)
                unique_filename = f"{dataset_key}_{base_name}{ext}"
                target_path = os.path.join(target_dir, unique_filename)
                
                try:
                    shutil.copy2(source_image_path, target_path)
                    
                    stats["images_processed"] += 1
                    stats["class_distribution"][predicted_class] += 1
                    stats["split_distribution"][split] += 1
                    
                except Exception as e:
                    print(f"      ❌ Ошибка копирования {filename}: {e}")
                    stats["images_skipped"] += 1
            
            processed = stats["split_distribution"][split]
            print(f"      ✅ Обработано: {processed} изображений")
        
        return stats
    
    def print_integration_summary(self, stats, output_path):
        """Выводит итоговую статистику интеграции"""
        
        print(f"\n🎉 ИТОГОВАЯ СТАТИСТИКА ИНТЕГРАЦИИ")
        print("="*60)
        print(f"📁 Выходная папка: {output_path}")
        print(f"📊 Датасетов обработано: {stats['datasets_processed']}")
        print(f"🖼️ Всего изображений: {stats['total_images']}")
        
        if stats["errors"]:
            print(f"❌ Ошибки ({len(stats['errors'])}):")
            for error in stats["errors"]:
                print(f"   {error}")
        
        print(f"\n📈 РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ:")
        total = sum(stats["class_distribution"].values())
        
        for class_idx, count in stats["class_distribution"].items():
            class_name = self.class_names[class_idx]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        print(f"\n📋 РАСПРЕДЕЛЕНИЕ ПО SPLIT'АМ:")
        for split, count in stats["split_distribution"].items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"   {split}: {count} ({percentage:.1f}%)")
    
    def create_integration_metadata(self, stats, output_path):
        """Создает мета-данные интеграции"""
        
        metadata = {
            "integration_date": "2025-09-13",
            "source_datasets": self.dataset_paths,
            "mapping_strategies": self.mapping_strategies,
            "statistics": {
                "total_images": stats["total_images"],
                "datasets_processed": stats["datasets_processed"],
                "class_distribution": dict(stats["class_distribution"]),
                "split_distribution": dict(stats["split_distribution"]),
                "errors": stats["errors"]
            },
            "class_names": self.class_names
        }
        
        metadata_path = os.path.join(output_path, "integration_metadata.json")
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Метаданные сохранены: {metadata_path}")
    
    def validate_integrated_dataset(self, output_path):
        """Проверяет целостность интегрированного датасета"""
        
        print(f"\n🔍 ВАЛИДАЦИЯ ИНТЕГРИРОВАННОГО ДАТАСЕТА")
        print("="*60)
        
        validation_results = {
            "structure_valid": True,
            "image_counts": {},
            "corrupted_images": [],
            "missing_classes": []
        }
        
        # Проверяем структуру папок
        for split in ["train", "test", "valid"]:
            validation_results["image_counts"][split] = {}
            
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(output_path, split, class_name)
                
                if not os.path.exists(class_dir):
                    validation_results["missing_classes"].append(f"{split}/{class_name}")
                    validation_results["structure_valid"] = False
                    continue
                
                # Подсчитываем изображения
                image_files = [f for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                validation_results["image_counts"][split][class_name] = len(image_files)
                
                # Проверяем несколько изображений на целостность
                for img_file in image_files[:3]:  # Проверяем первые 3
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                    except Exception as e:
                        validation_results["corrupted_images"].append(f"{split}/{class_name}/{img_file}")
        
        # Выводим результаты валидации
        print(f"✅ Структура папок: {'OK' if validation_results['structure_valid'] else 'ОШИБКА'}")
        
        if validation_results["missing_classes"]:
            print(f"❌ Отсутствующие классы: {validation_results['missing_classes']}")
        
        if validation_results["corrupted_images"]:
            print(f"❌ Поврежденные изображения: {len(validation_results['corrupted_images'])}")
            for corrupted in validation_results["corrupted_images"][:5]:  # Показываем первые 5
                print(f"   {corrupted}")
        else:
            print(f"✅ Все проверенные изображения целые")
        
        # Выводим финальные счетчики
        print(f"\n📊 ФИНАЛЬНЫЕ СЧЕТЧИКИ:")
        total_per_split = {}
        
        for split in ["train", "test", "valid"]:
            split_total = sum(validation_results["image_counts"][split].values())
            total_per_split[split] = split_total
            print(f"\n{split} ({split_total} изображений):")
            
            for class_name, count in validation_results["image_counts"][split].items():
                percentage = (count / split_total) * 100 if split_total > 0 else 0
                print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        grand_total = sum(total_per_split.values())
        print(f"\n🎯 ИТОГО: {grand_total} изображений")
        
        return validation_results

def main():
    """Основная функция интеграции"""
    
    integrator = NewDatasetIntegrator()
    
    print("🚀 ИНТЕГРАЦИЯ НОВЫХ ДАТАСЕТОВ")
    print("="*80)
    
    # 1. Анализируем CSV аннотации
    print("\n📊 ШАГ 1: АНАЛИЗ CSV АННОТАЦИЙ")
    
    all_analysis = {}
    for dataset_key in integrator.dataset_paths.keys():
        analysis = integrator.analyze_csv_annotations(dataset_key)
        all_analysis[dataset_key] = analysis
    
    # 2. Создаем интегрированный датасет
    print("\n🔧 ШАГ 2: СОЗДАНИЕ ИНТЕГРИРОВАННОГО ДАТАСЕТА")
    
    output_path, integration_stats = integrator.create_integrated_dataset()
    
    # 3. Валидируем результат
    print("\n✅ ШАГ 3: ВАЛИДАЦИЯ РЕЗУЛЬТАТА")
    
    validation_results = integrator.validate_integrated_dataset(output_path)
    
    # 4. Финальный отчет
    print(f"\n🎉 ИНТЕГРАЦИЯ ЗАВЕРШЕНА!")
    print(f"📁 Результат: {output_path}")
    print(f"🖼️ Всего изображений: {integration_stats['total_images']}")
    print(f"✅ Структура: {'OK' if validation_results['structure_valid'] else 'ОШИБКА'}")
    
    return output_path, integration_stats, validation_results

if __name__ == "__main__":
    main()