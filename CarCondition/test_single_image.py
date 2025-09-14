import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os

# Импортируем правильную модель
from multiclass_damage_model import MulticlassDamageModel

def load_model(model_path):
    """Загрузка обученной модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создаем модель с правильной архитектурой
    model = MulticlassDamageModel(num_classes=3)
    
    # Загружаем checkpoint (решаем проблему с PyTorch 2.6)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Модель загружена. Эпоха: {checkpoint.get('epoch', 'неизвестно')}")
        if 'f1_score' in checkpoint:
            print(f"📊 F1-score модели: {checkpoint['f1_score']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Модель загружена (старый формат)")
    
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Предобработка изображения"""
    # Трансформации как при обучении
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загружаем и обрабатываем изображение
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
    
    return image_tensor, image

def predict_damage(model, image_tensor, device):
    """Предсказание повреждений"""
    class_names = ['no_damage', 'minor_damage', 'major_damage']
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Получаем предсказания
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Конвертируем в numpy для удобства
        probs = probabilities.cpu().numpy()[0]
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score, probs, class_names

def main():
    # Пути к файлам
    image_path = r"C:\Users\Димаш\Desktop\python\hackaton\data\WhatsApp Image 2025-09-14 at 14.53.25.jpeg"
    model_path = r"C:\Users\Димаш\Desktop\python\hackaton\car_state\training_results\finetuned_best_model.pth"
    
    print("🚗 Тестирование модели определения повреждений автомобиля")
    print("="*60)
    
    # Проверяем существование файлов
    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ Модель не найдена: {model_path}")
        return
    
    try:
        # Загружаем модель
        print("📥 Загрузка модели...")
        model, device = load_model(model_path)
        
        # Обрабатываем изображение
        print("🖼️  Обработка изображения...")
        image_tensor, original_image = preprocess_image(image_path)
        print(f"   Размер изображения: {original_image.size}")
        
        # Делаем предсказание
        print("🔍 Анализ повреждений...")
        predicted_class, confidence, probabilities, class_names = predict_damage(model, image_tensor, device)
        
        # Выводим результаты
        print("\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("="*60)
        
        print(f"🎯 Предсказанный класс: {predicted_class}")
        print(f"📈 Уверенность: {confidence:.1%}")
        
        print("\n📋 Детальные вероятности:")
        for name, prob in zip(class_names, probabilities):
            bar_length = int(prob * 30)  # Масштабируем для визуализации
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {name:15}: {prob:.1%} |{bar}|")
        
        print("\n" + "="*60)
        
        # Интерпретация результатов
        print("🔍 ИНТЕРПРЕТАЦИЯ:")
        if predicted_class == 'no_damage':
            if confidence > 0.8:
                print("✅ Автомобиль в отличном состоянии, видимых повреждений не обнаружено")
            else:
                print("⚠️  Автомобиль скорее всего без серьезных повреждений, но стоит проверить детально")
        elif predicted_class == 'minor_damage':
            print("🔧 Обнаружены незначительные повреждения (царапины, небольшие вмятины)")
        else:  # major_damage
            print("🚨 ОБНАРУЖЕНЫ СЕРЬЕЗНЫЕ ПОВРЕЖДЕНИЯ!")
            print("   Автомобиль требует значительного ремонта")
            
        print("="*60)
        
    except Exception as e:
        print(f"❌ Ошибка при анализе: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()