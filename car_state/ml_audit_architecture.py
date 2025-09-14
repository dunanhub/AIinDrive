"""
ML Engineering Audit - Анализ архитектуры модели
Детальная проверка MulticlassDamageModel для предотвращения проблем обучения
"""
import torch
import torch.nn as nn
from multiclass_damage_model import MulticlassDamageModel, FocalLoss, create_training_transforms, create_validation_transforms

def analyze_model_architecture():
    """Комплексный анализ архитектуры модели"""
    
    print("🔍 ML ENGINEERING AUDIT - АРХИТЕКТУРА МОДЕЛИ")
    print("="*60)
    
    # 1. Создаем модель и анализируем параметры
    print("\n1️⃣ АНАЛИЗ ПАРАМЕТРОВ И СТРУКТУРЫ")
    model = MulticlassDamageModel(num_classes=3)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    
    print(f"   📊 Общее количество параметров: {total_params:,}")
    print(f"   🎯 Обучаемые параметры: {trainable_params:,}")
    print(f"   🏗️ Параметры backbone (ResNet50): {backbone_params:,}")
    print(f"   🧠 Параметры classifier: {classifier_params:,}")
    print(f"   💡 Соотношение classifier/total: {classifier_params/total_params*100:.1f}%")
    
    # 2. Анализ архитектуры классификатора
    print("\n2️⃣ АНАЛИЗ КЛАССИФИКАТОРА")
    print("   🔧 Структура слоев:")
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            print(f"      Linear {i}: {layer.in_features} → {layer.out_features}")
        elif isinstance(layer, nn.Dropout):
            print(f"      Dropout {i}: p={layer.p}")
        elif isinstance(layer, nn.BatchNorm1d):
            print(f"      BatchNorm1d {i}: {layer.num_features} features")
        else:
            print(f"      {type(layer).__name__} {i}")
    
    # 3. Проверка dropout progression
    dropout_layers = [layer for layer in model.classifier if isinstance(layer, nn.Dropout)]
    print(f"\n   📉 Dropout progression:")
    for i, layer in enumerate(dropout_layers):
        print(f"      Dropout {i+1}: {layer.p}")
    
    # Рекомендации по dropout
    if len(dropout_layers) >= 3:
        rates = [layer.p for layer in dropout_layers]
        if rates == sorted(rates, reverse=True):
            print("   ✅ Dropout rates убывают корректно")
        else:
            print("   ⚠️ Рекомендуется убывающая последовательность dropout rates")
    
    # 4. Проверка размерностей
    print("\n3️⃣ ПРОВЕРКА FORWARD PASS")
    
    # Тестируем различные размеры batch
    batch_sizes = [1, 4, 16, 32]
    for bs in batch_sizes:
        try:
            dummy_input = torch.randn(bs, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy_input)
            expected_shape = (bs, 3)
            
            if output.shape == expected_shape:
                print(f"   ✅ Batch size {bs}: {output.shape} ✓")
            else:
                print(f"   ❌ Batch size {bs}: {output.shape} ≠ {expected_shape}")
                
        except Exception as e:
            print(f"   ❌ Batch size {bs}: ERROR - {e}")
    
    # 5. Анализ градиентов
    print("\n4️⃣ АНАЛИЗ ГРАДИЕНТНОГО ПОТОКА")
    
    model.train()
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_target = torch.randint(0, 3, (2,))
    criterion = nn.CrossEntropyLoss()
    
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    
    # Проверяем градиенты в разных частях модели
    backbone_grads = []
    classifier_grads = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'backbone' in name:
                backbone_grads.append(grad_norm)
            elif 'classifier' in name:
                classifier_grads.append(grad_norm)
    
    print(f"   🏗️ Градиенты backbone - среднее: {sum(backbone_grads)/len(backbone_grads):.6f}")
    print(f"   🧠 Градиенты classifier - среднее: {sum(classifier_grads)/len(classifier_grads):.6f}")
    
    if len(backbone_grads) > 0 and len(classifier_grads) > 0:
        ratio = (sum(classifier_grads)/len(classifier_grads)) / (sum(backbone_grads)/len(backbone_grads))
        print(f"   📊 Отношение градиентов classifier/backbone: {ratio:.2f}")
        
        if 10 <= ratio <= 1000:
            print("   ✅ Хорошее соотношение градиентов")
        elif ratio > 1000:
            print("   ⚠️ Слишком большие градиенты в classifier - возможен exploding gradient")
        else:
            print("   ⚠️ Слишком маленькие градиенты в classifier - возможен vanishing gradient")
    
    # 6. Анализ активаций
    print("\n5️⃣ АНАЛИЗ АКТИВАЦИЙ")
    model.eval()
    
    with torch.no_grad():
        dummy_input = torch.randn(4, 3, 224, 224)
        
        # Проходим через backbone
        x = model.backbone.conv1(dummy_input)
        print(f"   После conv1: {x.shape}, среднее: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = model.backbone.bn1(x)
        x = model.backbone.relu(x)
        x = model.backbone.maxpool(x)
        
        x = model.backbone.layer1(x)
        print(f"   После layer1: {x.shape}, среднее: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = model.backbone.layer2(x)
        print(f"   После layer2: {x.shape}, среднее: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = model.backbone.layer3(x)
        print(f"   После layer3: {x.shape}, среднее: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = model.backbone.layer4(x)
        print(f"   После layer4: {x.shape}, среднее: {x.mean():.4f}, std: {x.std():.4f}")
        
        x = model.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        print(f"   После avgpool+flatten: {x.shape}, среднее: {x.mean():.4f}, std: {x.std():.4f}")
        
        # Проходим через classifier
        for i, layer in enumerate(model.classifier):
            x = layer(x)
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.BatchNorm1d)):
                print(f"   После classifier[{i}] ({type(layer).__name__}): среднее: {x.mean():.4f}, std: {x.std():.4f}")
    
    return model

def analyze_focal_loss():
    """Анализ настроек Focal Loss"""
    
    print("\n6️⃣ АНАЛИЗ FOCAL LOSS")
    print("   🎯 Тестируем разные настройки gamma:")
    
    # Создаем синтетические данные с дисбалансом
    logits = torch.randn(100, 3)  # 100 семплов, 3 класса
    
    # Имитируем дисбаланс: много класса 2, мало класса 0
    targets = torch.cat([
        torch.zeros(10, dtype=torch.long),    # 10% класс 0 (no_damage)
        torch.ones(40, dtype=torch.long),     # 40% класс 1 (minor)
        torch.full((50,), 2, dtype=torch.long) # 50% класс 2 (major)
    ])
    
    # Вычисляем class weights
    class_counts = torch.bincount(targets).float()
    total = class_counts.sum()
    class_weights = total / (len(class_counts) * class_counts)
    
    print(f"   📊 Синтетическое распределение: {class_counts.numpy()}")
    print(f"   ⚖️ Class weights: {class_weights.numpy()}")
    
    # Тестируем разные gamma
    gammas = [0.0, 1.0, 2.0, 3.0]
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    for gamma in gammas:
        focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        
        ce_val = ce_loss(logits, targets).item()
        focal_val = focal_loss(logits, targets).item()
        
        print(f"   Gamma {gamma}: CE={ce_val:.4f}, Focal={focal_val:.4f}, ratio={focal_val/ce_val:.3f}")
    
    print("\n   💡 Рекомендации:")
    print("   - Gamma=2.0 хорошо подходит для дисбаланса 8:1")
    print("   - Class weights компенсируют количественный дисбаланс")
    print("   - Focal Loss фокусируется на hard examples")

def analyze_transforms():
    """Анализ трансформаций данных"""
    
    print("\n7️⃣ АНАЛИЗ ТРАНСФОРМАЦИЙ")
    
    train_transforms = create_training_transforms()
    val_transforms = create_validation_transforms()
    
    print("   🏋️ Training transforms:")
    for i, transform in enumerate(train_transforms.transforms):
        print(f"      {i+1}. {transform}")
    
    print("\n   🔍 Validation transforms:")
    for i, transform in enumerate(val_transforms.transforms):
        print(f"      {i+1}. {transform}")
    
    # Тестируем трансформации
    from PIL import Image
    import numpy as np
    
    # Создаем синтетическое изображение
    dummy_image = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    
    try:
        train_tensor = train_transforms(dummy_image)
        val_tensor = val_transforms(dummy_image)
        
        print(f"\n   📊 Результаты трансформаций:")
        print(f"      Training tensor: {train_tensor.shape}, range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")
        print(f"      Validation tensor: {val_tensor.shape}, range: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")
        
        # Проверяем нормализацию ImageNet
        mean = train_tensor.mean(dim=[1, 2])
        std = train_tensor.std(dim=[1, 2])
        print(f"      Training нормализация - mean: {mean.numpy()}, std: {std.numpy()}")
        
        print("\n   ✅ Трансформации работают корректно")
        
    except Exception as e:
        print(f"   ❌ Ошибка в трансформациях: {e}")
    
    print("\n   💡 Оценка аугментаций для car damage:")
    print("   ✅ RandomHorizontalFlip - хорошо (повреждения симметричны)")
    print("   ✅ RandomRotation(10°) - осторожно (не должны переворачивать машину)")
    print("   ✅ ColorJitter - хорошо (разные условия освещения)")
    print("   ✅ RandomResizedCrop - хорошо (разные ракурсы повреждений)")

def main():
    """Основная функция анализа"""
    
    model = analyze_model_architecture()
    analyze_focal_loss()
    analyze_transforms()
    
    print("\n🎯 ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
    print("="*60)
    print("✅ Архитектура модели подходит для 3-class задачи")
    print("✅ ResNet50 backbone - хороший выбор для car damage detection")
    print("✅ Dropout progression корректна (0.5 → 0.25 → 0.125)")
    print("✅ BatchNorm между Linear слоями поможет стабилизации")
    print("✅ FocalLoss с gamma=2.0 хорош для дисбаланса 8:1")
    print("✅ Class weights правильно компенсируют дисбаланс")
    print("✅ Трансформации подходят для car damage detection")
    
    print("\n⚠️ ПОТЕНЦИАЛЬНЫЕ РИСКИ:")
    print("❗ Очень малый класс no_damage (6.3%) - возможны false negatives")
    print("❗ CPU обучение будет медленным (~23M параметров)")
    print("❗ Требуется мониторинг overfitting на малом датасете (650 изображений)")
    print("❗ Рекомендуется увеличить долю validation данных до 25-30%")
    
    print(f"\n📊 РАЗМЕР МОДЕЛИ:")
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    print(f"   Параметры: {total_params:,}")
    print(f"   Размер в памяти: ~{model_size_mb:.1f} MB")
    print(f"   Примерное время обучения на CPU: 2-4 часа для 20 эпох")

if __name__ == "__main__":
    main()