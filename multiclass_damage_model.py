"""
Архитектура модели для классификации повреждений автомобилей
============================================================
"""

import torch
import torch.nn as nn
import torchvision.models as models

class MulticlassDamageModel(nn.Module):
    """
    Модель для многоклассовой классификации повреждений автомобилей
    
    Архитектура:
    - Backbone: ResNet50 (pretrained)
    - Output: 3 класса (no_damage, minor_damage, major_damage)
    """
    
    def __init__(self, num_classes=3, pretrained=True):
        """
        Args:
            num_classes (int): Количество классов (по умолчанию 3)
            pretrained (bool): Использовать предобученные веса ImageNet
        """
        super(MulticlassDamageModel, self).__init__()
        
        # Загрузка ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Замена последнего слоя для нашей задачи
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Прямой проход
        
        Args:
            x (torch.Tensor): Входное изображение [B, 3, 224, 224]
            
        Returns:
            torch.Tensor: Логиты для каждого класса [B, num_classes]
        """
        # Извлечение признаков через ResNet (без последнего слоя)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Global Average Pooling
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout перед классификатором
        x = self.dropout(x)
        
        # Финальная классификация
        x = self.resnet.fc(x)
        
        return x
    
    def get_features(self, x):
        """
        Извлечение признаков (без классификации)
        
        Args:
            x (torch.Tensor): Входное изображение
            
        Returns:
            torch.Tensor: Вектор признаков
        """
        # Все слои кроме последнего
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def predict_proba(self, x):
        """
        Предсказание с вероятностями
        
        Args:
            x (torch.Tensor): Входное изображение
            
        Returns:
            torch.Tensor: Вероятности для каждого класса
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x):
        """
        Предсказание класса
        
        Args:
            x (torch.Tensor): Входное изображение
            
        Returns:
            torch.Tensor: Индексы предсказанных классов
        """
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
    
    def get_model_info(self):
        """
        Информация о модели
        
        Returns:
            dict: Словарь с информацией о модели
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "MulticlassDamageModel",
            "backbone": "ResNet50",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": "[B, 3, 224, 224]",
            "output_size": "[B, 3]",
            "classes": ["no_damage", "minor_damage", "major_damage"]
        }

def create_model(num_classes=3, pretrained=True):
    """
    Создание экземпляра модели
    
    Args:
        num_classes (int): Количество классов
        pretrained (bool): Использовать предобученные веса
        
    Returns:
        MulticlassDamageModel: Инициализированная модель
    """
    return MulticlassDamageModel(num_classes=num_classes, pretrained=pretrained)

if __name__ == "__main__":
    # Тестирование модели
    model = create_model()
    print("Информация о модели:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Тест прямого прохода
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nТест прямого прохода:")
    print(f"  Вход: {dummy_input.shape}")
    print(f"  Выход: {output.shape}")
    print(f"  Логиты: {output.squeeze().detach().numpy()}")
    
    # Тест предсказания
    probabilities = model.predict_proba(dummy_input)
    predictions = model.predict(dummy_input)
    print(f"\nТест предсказания:")
    print(f"  Вероятности: {probabilities.squeeze().detach().numpy()}")
    print(f"  Предсказанный класс: {predictions.item()}")