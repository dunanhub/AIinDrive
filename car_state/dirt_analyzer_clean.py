import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFilter, ImageStat
import torch.nn.functional as F
import os
import sys
import numpy as np

class MulticlassDamageModel(nn.Module):
    def __init__(self, num_classes=3, dropout=0.6):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.25),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

def analyze_dirt_level_detailed(image):
    """Детальный анализ загрязненности с подробными метриками"""
    
    img_array = np.array(image)
    
    print("\\n" + "="*60)
    print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ЗАГРЯЗНЕННОСТИ:")
    print("="*60)
    
    # 1. Цветовое разнообразие
    unique_colors_r = len(np.unique(img_array[:,:,0]))
    unique_colors_g = len(np.unique(img_array[:,:,1])) 
    unique_colors_b = len(np.unique(img_array[:,:,2]))
    color_diversity = (unique_colors_r + unique_colors_g + unique_colors_b) / 3
    
    print(f"🎨 ЦВЕТОВОЕ РАЗНООБРАЗИЕ:")
    print(f"   • Красный канал: {unique_colors_r} уникальных цветов")
    print(f"   • Зеленый канал: {unique_colors_g} уникальных цветов")
    print(f"   • Синий канал: {unique_colors_b} уникальных цветов")
    print(f"   • Среднее разнообразие: {color_diversity:.1f}")
    
    # 2. Контраст
    gray = image.convert('L')
    contrast = ImageStat.Stat(gray).stddev[0]
    
    print(f"\\n📊 КОНТРАСТ:")
    print(f"   • Стандартное отклонение яркости: {contrast:.1f}")
    print(f"   • Интерпретация: {'Высокий' if contrast > 40 else 'Средний' if contrast > 25 else 'Низкий'}")
    
    # 3. Насыщенность
    hsv = image.convert('HSV')
    hsv_array = np.array(hsv)
    saturation = np.mean(hsv_array[:,:,1])
    saturation_std = np.std(hsv_array[:,:,1])
    
    print(f"\\n🌈 НАСЫЩЕННОСТЬ:")
    print(f"   • Средняя насыщенность: {saturation:.1f}")
    print(f"   • Разброс насыщенности: {saturation_std:.1f}")
    print(f"   • Интерпретация: {'Высокая' if saturation > 100 else 'Средняя' if saturation > 60 else 'Низкая'}")
    
    # 4. Коричневые оттенки
    brown_mask = (
        (img_array[:,:,0] > img_array[:,:,2]) &
        (img_array[:,:,1] > img_array[:,:,2]) &
        (img_array[:,:,0] < 150) &
        (img_array[:,:,1] < 120)
    )
    brown_ratio = np.sum(brown_mask) / (img_array.shape[0] * img_array.shape[1])
    
    print(f"\\n🟤 КОРИЧНЕВЫЕ ОТТЕНКИ:")
    print(f"   • Процент коричневых пикселей: {brown_ratio:.1%}")
    print(f"   • Количество пикселей: {np.sum(brown_mask):,} из {img_array.shape[0] * img_array.shape[1]:,}")
    print(f"   • Интерпретация: {'Много грязи' if brown_ratio > 0.15 else 'Умеренно' if brown_ratio > 0.08 else 'Чисто'}")
    
    # 5. Четкость краев
    edge_image = gray.filter(ImageFilter.FIND_EDGES)
    edge_intensity = np.mean(np.array(edge_image))
    edge_std = np.std(np.array(edge_image))
    
    print(f"\\n🔍 ЧЕТКОСТЬ КРАЕВ:")
    print(f"   • Средняя интенсивность краев: {edge_intensity:.1f}")
    print(f"   • Разброс интенсивности: {edge_std:.1f}")
    print(f"   • Интерпретация: {'Четкие' if edge_intensity > 25 else 'Средние' if edge_intensity > 15 else 'Размытые'}")
    
    # 6. Яркость
    brightness = np.mean(img_array)
    brightness_std = np.std(img_array)
    
    print(f"\\n💡 ЯРКОСТЬ:")
    print(f"   • Средняя яркость: {brightness:.1f}")
    print(f"   • Разброс яркости: {brightness_std:.1f}")
    print(f"   • Интерпретация: {'Высокая' if brightness > 110 else 'Средняя' if brightness > 90 else 'Низкая'}")
    
    # Подсчет итогового индекса грязи
    print(f"\\n📊 РАСЧЕТ ИНДЕКСА ГРЯЗИ:")
    dirt_score = 0
    
    # Цветовое разнообразие
    if color_diversity < 80:
        score_add = 2
        dirt_score += score_add
        print(f"   🔴 Очень низкое цветовое разнообразие: +{score_add}")
    elif color_diversity < 120:
        score_add = 1
        dirt_score += score_add
        print(f"   🟡 Низкое цветовое разнообразие: +{score_add}")
    else:
        print(f"   🟢 Хорошее цветовое разнообразие: +0")
    
    # Контраст
    if contrast < 25:
        score_add = 2
        dirt_score += score_add
        print(f"   🔴 Очень низкий контраст: +{score_add}")
    elif contrast < 40:
        score_add = 1
        dirt_score += score_add
        print(f"   🟡 Низкий контраст: +{score_add}")
    else:
        print(f"   🟢 Хороший контраст: +0")
    
    # Насыщенность
    if saturation < 60:
        score_add = 1.5
        dirt_score += score_add
        print(f"   🔴 Очень низкая насыщенность: +{score_add}")
    elif saturation < 100:
        score_add = 0.5
        dirt_score += score_add
        print(f"   🟡 Низкая насыщенность: +{score_add}")
    else:
        print(f"   🟢 Хорошая насыщенность: +0")
    
    # Коричневые оттенки
    if brown_ratio > 0.15:
        score_add = 2
        dirt_score += score_add
        print(f"   🔴 Много коричневых оттенков: +{score_add}")
    elif brown_ratio > 0.08:
        score_add = 1
        dirt_score += score_add
        print(f"   🟡 Умеренно коричневых оттенков: +{score_add}")
    else:
        print(f"   🟢 Мало коричневых оттенков: +0")
    
    # Четкость краев
    if edge_intensity < 15:
        score_add = 1.5
        dirt_score += score_add
        print(f"   🔴 Очень размытые края: +{score_add}")
    elif edge_intensity < 25:
        score_add = 0.5
        dirt_score += score_add
        print(f"   🟡 Размытые края: +{score_add}")
    else:
        print(f"   🟢 Четкие края: +0")
    
    # Яркость
    if brightness < 90:
        score_add = 1
        dirt_score += score_add
        print(f"   🔴 Низкая яркость: +{score_add}")
    elif brightness < 110:
        score_add = 0.5
        dirt_score += score_add
        print(f"   🟡 Умеренная яркость: +{score_add}")
    else:
        print(f"   🟢 Хорошая яркость: +0")
    
    # Итоговая оценка
    print(f"\\n🏆 ИТОГОВЫЙ ИНДЕКС ГРЯЗИ: {dirt_score:.1f}")
    
    if dirt_score >= 6:
        status = "очень грязная"
        emoji = "🟤"
        explanation = "Критическое загрязнение - требуется профессиональная мойка"
    elif dirt_score >= 4:
        status = "грязная"
        emoji = "🟫"
        explanation = "Сильное загрязнение - рекомендуется тщательная мойка"
    elif dirt_score >= 2:
        status = "слегка грязная"
        emoji = "🟨"
        explanation = "Легкое загрязнение - обычная мойка"
    elif dirt_score >= 1:
        status = "достаточно чистая"
        emoji = "🟩"
        explanation = "Хорошее состояние - легкая очистка"
    else:
        status = "очень чистая"
        emoji = "✨"
        explanation = "Отличное состояние чистоты"
    
    print(f"📋 ЗАКЛЮЧЕНИЕ: {emoji} {status.upper()}")
    print(f"💬 {explanation}")
    print("="*60)
    
    return status, emoji, dirt_score

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    model = MulticlassDamageModel(num_classes=3)
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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image

def predict_damage(model, image_tensor, device):
    class_names = ['no_damage', 'minor_damage', 'major_damage']
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        probs = probabilities.cpu().numpy()[0]
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score, probs, class_names

def analyze_image_with_dirt_details(image_filename):
    data_folder = r"C:\\Users\\Димаш\\Desktop\\python\\hackaton\\data"
    model_path = r"C:\\Users\\Димаш\\Desktop\\python\\hackaton\\car_state\\training_results\\finetuned_best_model.pth"
    
    image_path = os.path.join(data_folder, image_filename)
    
    print("🚗 Детальный анализатор повреждений и загрязнений")
    print("="*60)
    print(f"📂 Папка данных: {data_folder}")
    print(f"🖼️  Анализируемое изображение: {image_filename}")
    
    if not os.path.exists(image_path):
        print(f"❌ Изображение не найдено: {image_path}")
        return
    
    try:
        print("\\n📥 Загрузка модели...")
        model, device = load_model(model_path)
        
        print("🖼️  Обработка изображения...")
        image_tensor, original_image = preprocess_image(image_path)
        print(f"   Размер изображения: {original_image.size}")
        
        # Детальный анализ грязи
        dirt_status, dirt_emoji, dirt_score = analyze_dirt_level_detailed(original_image)
        
        # Анализ повреждений
        print("\\n🔍 Анализ повреждений...")
        predicted_class, confidence, probabilities, class_names = predict_damage(model, image_tensor, device)
        
        # Выводим результаты повреждений
        print("\\n" + "="*60)
        print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА ПОВРЕЖДЕНИЙ:")
        print("="*60)
        
        print(f"🎯 Предсказанный класс: {predicted_class}")
        print(f"📈 Уверенность: {confidence:.1%}")
        print(f"🧼 Чистота: {dirt_emoji} {dirt_status} (индекс: {dirt_score:.1f})")
        
        print("\\n📋 Детальные вероятности:")
        for name, prob in zip(class_names, probabilities):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            if name == 'no_damage':
                emoji = "✅"
            elif name == 'minor_damage':
                emoji = "🔧"
            else:
                emoji = "🚨"
                
            print(f"   {emoji} {name:15}: {prob:.1%} |{bar}|")
        
        print("="*60)
        
    except Exception as e:
        print(f"❌ Ошибка при анализе: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    if len(sys.argv) > 1:
        image_filename = sys.argv[1]
    else:
        image_filename = input("Введите имя файла изображения: ").strip()
        
        if not image_filename:
            print("❌ Имя файла не указано!")
            return
    
    analyze_image_with_dirt_details(image_filename)

if __name__ == "__main__":
    main()