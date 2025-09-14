"""
Предиктор для inDrive - определение состояния автомобиля
Классификация: чистый/грязный, целый/битый
"""
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
from typing import Dict, Tuple, Optional

from multiclass_damage_model import MulticlassDamageModel, create_validation_transforms

class InDriveCarPredictor:
    """Предиктор состояния автомобиля для inDrive"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        self.transforms = create_validation_transforms()
        
        # Интерпретация классов для inDrive
        self.class_mapping = {
            0: {
                "status": "ОТЛИЧНОЕ",
                "safety": "БЕЗОПАСНО", 
                "trust": "ВЫСОКОЕ",
                "description": "Автомобиль в отличном состоянии",
                "color": "green",
                "score": 100
            },
            1: {
                "status": "УДОВЛЕТВОРИТЕЛЬНОЕ",
                "safety": "БЕЗОПАСНО",
                "trust": "СРЕДНЕЕ", 
                "description": "Автомобиль с незначительными дефектами",
                "color": "yellow",
                "score": 75
            },
            2: {
                "status": "ПЛОХОЕ",
                "safety": "ТРЕБУЕТ ПРОВЕРКИ",
                "trust": "НИЗКОЕ",
                "description": "Автомобиль с серьезными повреждениями", 
                "color": "red",
                "score": 40
            }
        }
    
    def _load_model(self, model_path: str) -> MulticlassDamageModel:
        """Загрузка обученной модели"""
        model = MulticlassDamageModel(num_classes=3)
        
        # Пробуем загрузить checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Новый формат с метаданными
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"✅ Загружена модель с метаданными")
                    if 'val_f1' in checkpoint:
                        print(f"   F1-score: {checkpoint['val_f1']:.4f}")
                else:
                    # Старый формат - только веса
                    model.load_state_dict(checkpoint)
                    print(f"✅ Загружены веса модели")
            else:
                # Очень старый формат
                model.load_state_dict(checkpoint)
                print(f"✅ Загружена модель (старый формат)")
                
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise
            
        model.to(self.device)
        model.eval()
        return model
    
    def predict_image(self, image_path: str) -> Dict:
        """
        Предсказание состояния автомобиля
        
        Returns:
            Dict с результатами анализа для inDrive
        """
        try:
            # Загружаем и обрабатываем изображение
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)[0]
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[predicted_class].item()
            
            # Интерпретация для inDrive
            class_info = self.class_mapping[predicted_class]
            
            # Детальный анализ вероятностей
            prob_analysis = {
                "excellent_condition": float(probabilities[0]),  # no_damage
                "minor_issues": float(probabilities[1]),         # minor_damage  
                "serious_damage": float(probabilities[2])        # major_damage
            }
            
            # Рекомендации для водителя/пассажира
            recommendations = self._generate_recommendations(predicted_class, confidence)
            
            result = {
                "image_path": image_path,
                "predicted_class": predicted_class,
                "confidence": round(confidence * 100, 2),
                "status": class_info["status"],
                "safety_level": class_info["safety"],
                "trust_level": class_info["trust"],
                "description": class_info["description"],
                "color_indicator": class_info["color"],
                "quality_score": class_info["score"],
                "probability_breakdown": prob_analysis,
                "recommendations": recommendations,
                "timestamp": str(Path(image_path).stat().st_mtime)
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Ошибка анализа изображения: {str(e)}",
                "image_path": image_path
            }
    
    def _generate_recommendations(self, predicted_class: int, confidence: float) -> Dict:
        """Генерация рекомендаций для inDrive"""
        
        if predicted_class == 0:  # Отличное состояние
            return {
                "for_passenger": "Автомобиль в отличном состоянии. Поездка будет комфортной.",
                "for_driver": "Поддерживайте автомобиль в таком же состоянии!",
                "action": "proceed",
                "priority": "low"
            }
        elif predicted_class == 1:  # Удовлетворительное
            return {
                "for_passenger": "Автомобиль имеет незначительные дефекты, но безопасен для поездки.",
                "for_driver": "Рекомендуется устранить мелкие недостатки для улучшения рейтинга.",
                "action": "proceed_with_note",
                "priority": "medium"
            }
        else:  # Плохое состояние
            return {
                "for_passenger": "⚠️ Автомобиль имеет серьезные повреждения. Рекомендуется выбрать другой автомобиль.",
                "for_driver": "🚫 Необходим ремонт автомобиля перед выходом на линию.",
                "action": "review_required",
                "priority": "high"
            }
    
    def batch_predict(self, image_dir: str, output_file: str = None) -> Dict:
        """Анализ множества изображений"""
        image_dir = Path(image_dir)
        results = []
        
        # Поддерживаемые форматы
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(ext))
        
        print(f"🔍 Найдено {len(image_files)} изображений для анализа")
        
        for img_path in image_files:
            result = self.predict_image(str(img_path))
            results.append(result)
        
        # Статистика
        if results:
            stats = self._calculate_batch_stats(results)
        else:
            stats = {"error": "Нет результатов для анализа"}
        
        batch_result = {
            "total_images": len(image_files),
            "results": results,
            "statistics": stats,
            "analysis_date": str(Path().cwd())
        }
        
        # Сохраняем результаты
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, indent=2, ensure_ascii=False)
            print(f"💾 Результаты сохранены в {output_file}")
        
        return batch_result
    
    def _calculate_batch_stats(self, results: list) -> Dict:
        """Расчет статистики по партии изображений"""
        # Фильтруем успешные результаты
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {"error": "Нет валидных результатов"}
        
        # Подсчет по классам
        class_counts = {0: 0, 1: 0, 2: 0}
        confidence_sum = 0
        
        for result in valid_results:
            class_counts[result['predicted_class']] += 1
            confidence_sum += result['confidence']
        
        total = len(valid_results)
        
        return {
            "excellent_condition": {
                "count": class_counts[0],
                "percentage": round(class_counts[0] / total * 100, 1)
            },
            "minor_issues": {
                "count": class_counts[1], 
                "percentage": round(class_counts[1] / total * 100, 1)
            },
            "serious_damage": {
                "count": class_counts[2],
                "percentage": round(class_counts[2] / total * 100, 1)
            },
            "average_confidence": round(confidence_sum / total, 2),
            "safety_assessment": self._fleet_safety_assessment(class_counts, total)
        }
    
    def _fleet_safety_assessment(self, class_counts: Dict, total: int) -> str:
        """Оценка безопасности автопарка"""
        serious_damage_pct = class_counts[2] / total * 100
        
        if serious_damage_pct > 30:
            return "КРИТИЧНО: Слишком много автомобилей с серьезными повреждениями"
        elif serious_damage_pct > 15:
            return "ВНИМАНИЕ: Повышенный процент поврежденных автомобилей"
        elif serious_damage_pct > 5:
            return "НОРМАЛЬНО: Приемлемый уровень повреждений"
        else:
            return "ОТЛИЧНО: Автопарк в хорошем состоянии"

def main():
    """Пример использования"""
    # Путь к обученной модели
    model_path = "best_multiclass_model_v3.2.pth"
    
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        print("Сначала обучите модель с помощью train_multiclass_damage_v3_2.py")
        return
    
    # Инициализация предиктора
    predictor = InDriveCarPredictor(model_path, device='cpu')
    
    # Пример анализа одного изображения
    test_image = "../data/integrated_multiclass_dataset/test/no_damage"
    if Path(test_image).exists():
        print("\n🚗 ДЕМО: Анализ состояния автомобиля для inDrive")
        print("="*50)
        
        # Берем первое изображение из тестовой папки
        image_files = list(Path(test_image).glob("*.jpg"))
        if image_files:
            result = predictor.predict_image(str(image_files[0]))
            
            print(f"📁 Файл: {result.get('image_path', 'N/A')}")
            print(f"🎯 Статус: {result.get('status', 'N/A')}")
            print(f"🛡️  Безопасность: {result.get('safety_level', 'N/A')}")
            print(f"⭐ Доверие: {result.get('trust_level', 'N/A')}")
            print(f"📊 Уверенность: {result.get('confidence', 'N/A')}%")
            print(f"💬 Описание: {result.get('description', 'N/A')}")
            
            print(f"\n📋 Рекомендации:")
            recs = result.get('recommendations', {})
            print(f"   Для пассажира: {recs.get('for_passenger', 'N/A')}")
            print(f"   Для водителя: {recs.get('for_driver', 'N/A')}")
    
    print(f"\n✅ InDrive Car Predictor готов к использованию!")

if __name__ == "__main__":
    main()