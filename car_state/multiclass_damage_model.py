"""
Многоклассовая модель для анализа повреждений автомобилей
3 класса: no_damage (0), minor_damage (1), major_damage (2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
from typing import Tuple, Dict, Any

class MulticlassDamageModel(nn.Module):
    """
    Улучшенная модель для классификации повреждений с 3 классами
    """
    def __init__(self, num_classes=3, dropout=0.6):  # Увеличено с 0.5 до 0.6
        super().__init__()
        
        # Загружаем предобученный ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Убираем последний fc слой
        self.backbone.fc = nn.Identity()
        
        # Создаем улучшенный классификатор с усиленной регуляризацией
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),  # 0.6
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout * 0.5),  # 0.3 (было 0.25)
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.25),  # 0.15 (было 0.125)
            nn.Linear(512, num_classes)
        )
        
        # Для Grad-CAM (опционально)
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        """Хук для сохранения градиентов для Grad-CAM"""
        self.gradients = grad
        
    def forward(self, x):
        # Извлекаем features через ResNet50
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Сохраняем активации для Grad-CAM
        if x.requires_grad:
            self.activations = x
            h = x.register_hook(self.activations_hook)
        
        # Global Average Pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Классификатор
        x = self.classifier(x)
        
        return x

class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
        super().__init__()
        
        # Если alpha задан как число - преобразуем в тензор для 3 классов
        if alpha is None:
            # Веса по умолчанию: больше вес для редких классов
            # no_damage (много), minor_damage (средне), major_damage (мало)
            self.alpha = torch.tensor([0.5, 1.0, 2.0]).to(device)
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha).to(device)
        elif isinstance(alpha, (int, float)):
            # Если передали одно число - используем как базовый вес
            self.alpha = torch.tensor([alpha, alpha * 1.5, alpha * 2.0]).to(device)
        else:
            self.alpha = alpha
        
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DamageClassifier:
    """
    Обертка для удобного использования модели
    """
    
    # Маппинг классов
    CLASS_NAMES = {
        0: "no_damage",
        1: "minor_damage", 
        2: "major_damage"
    }
    
    CLASS_DESCRIPTIONS = {
        0: "Нет повреждений",
        1: "Незначительные повреждения",
        2: "Существенные повреждения"
    }
    
    def __init__(self, model_path=None, device='cpu'):
        self.device = device
        self.model = MulticlassDamageModel(num_classes=3)
        self.model.to(device)
        self.model.eval()
        
        # Трансформации для валидации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Калиброванные пороги (будут загружены из checkpoint)
        self.thresholds = {
            'damage_threshold': 0.5,
            'major_threshold': 0.55,
            'confidence_threshold': 0.7
        }
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Загрузка обученной модели"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Загружаем калиброванные пороги если есть
        if 'best' in checkpoint:
            best = checkpoint['best']
            self.thresholds.update({
                'damage_threshold': best.get('threshold_f1', 0.5),
                'major_threshold': max(best.get('threshold_f1', 0.5), 0.55)
            })
        
        self.model.eval()
        print(f"✅ Модель загружена с порогами: {self.thresholds}")
    
    def predict_single(self, image_tensor) -> Tuple[np.ndarray, int, float]:
        """
        Предсказание для одного изображения
        
        Returns:
            probabilities: массив вероятностей [p0, p1, p2]
            predicted_class: предсказанный класс (0, 1, 2)
            confidence: уверенность (максимальная вероятность)
        """
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            probs_np = probabilities.cpu().numpy()[0]
            predicted_class = int(np.argmax(probs_np))
            confidence = float(np.max(probs_np))
            
            return probs_np, predicted_class, confidence
    
    def classify_damage_level(self, probabilities) -> Dict[str, Any]:
        """
        Определение уровня повреждений с учетом калиброванных порогов
        """
        p0, p1, p2 = probabilities
        confidence = float(max(probabilities))
        
        # Комбинированная вероятность повреждений
        damage_prob = p1 + p2
        
        # Применяем пороги и логику
        if confidence < self.thresholds['confidence_threshold']:
            status = "requires_inspection"
            description = "Требуется дополнительный осмотр"
            severity = "uncertain"
            
        elif p2 >= self.thresholds['major_threshold']:
            status = "major_damage"
            description = f"Существенные повреждения ({int(p2*100)}% уверенности)"
            severity = "high"
            
        elif p1 >= 0.45:  # Заниженный порог для ловли мелких повреждений
            status = "minor_damage"
            description = f"Незначительные повреждения ({int(p1*100)}% уверенности)"
            severity = "low"
            
        elif damage_prob >= self.thresholds['damage_threshold']:
            status = "suspected_damage"
            description = f"Подозрение на повреждения ({int(damage_prob*100)}% уверенности)"
            severity = "low"
            
        else:
            status = "no_damage"
            description = f"Без видимых повреждений ({int(p0*100)}% уверенности)"
            severity = "none"
        
        return {
            'status': status,
            'description': description,
            'severity': severity,
            'confidence': confidence,
            'damage_probability': damage_prob,
            'class_probabilities': {
                'no_damage': float(p0),
                'minor_damage': float(p1),
                'major_damage': float(p2)
            }
        }

def create_training_transforms():
    """Трансформации для обучения с усиленной аугментацией против overfitting"""
    return transforms.Compose([
        # --- PIL stage (до ToTensor) ---
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),  # Более агрессивный crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Увеличено с 10 до 15
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Усилено с 0.2 до 0.3
        
        # --- Tensor stage (после ToTensor) ---
        transforms.ToTensor(),  # ОБЯЗАТЕЛЬНО ДО RandomErasing и Normalize
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # После ToTensor!
    ])

def create_validation_transforms():
    """Трансформации для валидации без аугментации"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Тестирование
if __name__ == "__main__":
    # Создаем модель
    model = MulticlassDamageModel(num_classes=3)
    print(f"Создана модель с {sum(p.numel() for p in model.parameters())} параметрами")
    
    # Тестируем forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Выход модели: {output.shape}")
    
    # Тестируем Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    dummy_target = torch.randint(0, 3, (2,))
    loss = focal_loss(output, dummy_target)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Тестируем классификатор
    classifier = DamageClassifier()
    probs, pred_class, confidence = classifier.predict_single(dummy_input[0])
    result = classifier.classify_damage_level(probs)
    
    print(f"Предсказание: {result}")