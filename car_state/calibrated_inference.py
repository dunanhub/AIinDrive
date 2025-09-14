"""
Калиброванные пороги для inference - критически важно для car damage detection
Не используем "сырые" softmax вероятности!
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from multiclass_damage_model import MulticlassDamageModel, DamageClassifier

class CalibratedDamageClassifier:
    """
    Улучшенный классификатор с калиброванными порогами
    Решает проблему: лучше перестраховаться, чем пропустить повреждение
    """
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = MulticlassDamageModel(num_classes=3)
        self.model.to(device)
        self.model.eval()
        
        # Калиброванные пороги (подбираются на валидации)
        self.thresholds = {
            'confidence_min': 0.7,      # Минимальная уверенность
            'damage_threshold': 0.5,     # Порог "есть повреждения" (minor+major vs no)
            'major_threshold': 0.55,     # Жесткий порог для major damage
            'minor_threshold': 0.45,     # Мягкий порог для minor damage (чтобы не пропустить)
        }
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Загрузка модели с порогами"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Загружаем калиброванные пороги если есть
        if 'calibrated_thresholds' in checkpoint:
            self.thresholds.update(checkpoint['calibrated_thresholds'])
            
        self.model.eval()
        print(f"✅ Модель загружена с порогами: {self.thresholds}")
    
    def predict_with_calibrated_thresholds(self, image_tensor):
        """
        Предсказание с калиброванными порогами
        
        Логика:
        1. Проверяем общую уверенность
        2. Применяем специфические пороги для каждого класса
        3. Если неуверенно -> requires_inspection
        """
        with torch.no_grad():
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            
            probs = probabilities.cpu().numpy()[0]  # [p_no, p_minor, p_major]
            p_no, p_minor, p_major = probs
            
            # Максимальная уверенность
            max_confidence = float(np.max(probs))
            
            # Общая вероятность повреждений
            damage_prob = p_minor + p_major
            
            # КАЛИБРОВАННАЯ ЛОГИКА ПРИНЯТИЯ РЕШЕНИЙ
            
            # 1. Слишком низкая уверенность -> осмотр
            if max_confidence < self.thresholds['confidence_min']:
                return {
                    'status': 'requires_inspection',
                    'description': f'Низкая уверенность ({max_confidence:.2f}), требуется осмотр',
                    'confidence': max_confidence,
                    'probabilities': probs,
                    'reason': 'low_confidence'
                }
            
            # 2. Четкое отсутствие повреждений
            if p_no >= 0.6 and damage_prob < self.thresholds['damage_threshold']:
                return {
                    'status': 'no_damage',
                    'description': f'Без видимых повреждений ({p_no:.2f} уверенности)',
                    'confidence': max_confidence,
                    'probabilities': probs,
                    'reason': 'clear_no_damage'
                }
            
            # 3. Жесткий порог для major damage (строгий, чтобы не false positive)
            if p_major >= self.thresholds['major_threshold']:
                return {
                    'status': 'major_damage',
                    'description': f'Существенные повреждения ({p_major:.2f} уверенности)',
                    'confidence': max_confidence,
                    'probabilities': probs,
                    'reason': 'clear_major_damage'
                }
            
            # 4. Мягкий порог для minor damage (чтобы не пропустить мелкие царапины)
            if p_minor >= self.thresholds['minor_threshold']:
                return {
                    'status': 'minor_damage',
                    'description': f'Незначительные повреждения ({p_minor:.2f} уверенности)',
                    'confidence': max_confidence,
                    'probabilities': probs,
                    'reason': 'detected_minor_damage'
                }
            
            # 5. Есть подозрение на повреждения, но неясно какие
            if damage_prob >= self.thresholds['damage_threshold']:
                return {
                    'status': 'requires_inspection',
                    'description': f'Подозрение на повреждения ({damage_prob:.2f}), требуется осмотр',
                    'confidence': max_confidence,
                    'probabilities': probs,
                    'reason': 'suspected_damage'
                }
            
            # 6. По умолчанию - нет повреждений (но с меньшей уверенностью)
            return {
                'status': 'no_damage',
                'description': f'Вероятно без повреждений ({p_no:.2f} уверенности)',
                'confidence': max_confidence,
                'probabilities': probs,
                'reason': 'probable_no_damage'
            }

def calibrate_thresholds_on_validation(model, val_loader, device='cpu'):
    """
    Калибровка порогов на валидационном наборе
    Находим оптимальные пороги для максимизации F1-score каждого класса
    """
    print("🎯 Калибровка порогов на валидационных данных...")
    
    model.eval()
    all_probs = []
    all_labels = []
    
    # Собираем предсказания на валидации
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    print(f"   📊 Собрано предсказаний: {len(all_labels)}")
    
    # Находим оптимальные пороги
    calibrated_thresholds = {}
    
    # 1. Порог для "damage vs no_damage"
    damage_probs = all_probs[:, 1] + all_probs[:, 2]  # minor + major
    damage_labels = (all_labels > 0).astype(int)  # 0 если no_damage, 1 если есть повреждения
    
    # Перебираем пороги и ищем лучший F1
    best_f1 = 0
    best_damage_threshold = 0.5
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        pred_damage = (damage_probs >= threshold).astype(int)
        f1 = f1_score(damage_labels, pred_damage)
        
        if f1 > best_f1:
            best_f1 = f1
            best_damage_threshold = threshold
    
    calibrated_thresholds['damage_threshold'] = best_damage_threshold
    print(f"   ✅ Оптимальный порог damage: {best_damage_threshold:.3f} (F1: {best_f1:.3f})")
    
    # 2. Порог для major damage (более консервативный)
    major_labels = (all_labels == 2).astype(int)
    major_probs = all_probs[:, 2]
    
    best_f1 = 0
    best_major_threshold = 0.55
    
    for threshold in np.arange(0.4, 0.8, 0.05):
        pred_major = (major_probs >= threshold).astype(int)
        f1 = f1_score(major_labels, pred_major)
        
        if f1 > best_f1:
            best_f1 = f1
            best_major_threshold = threshold
    
    # Делаем порог чуть жестче для консерватизма
    calibrated_thresholds['major_threshold'] = max(best_major_threshold, 0.55)
    print(f"   ✅ Оптимальный порог major: {calibrated_thresholds['major_threshold']:.3f}")
    
    # 3. Порог для minor damage (более мягкий, чтобы не пропустить)
    minor_labels = (all_labels == 1).astype(int)
    minor_probs = all_probs[:, 1]
    
    best_f1 = 0
    best_minor_threshold = 0.45
    
    for threshold in np.arange(0.3, 0.7, 0.05):
        pred_minor = (minor_probs >= threshold).astype(int)
        f1 = f1_score(minor_labels, pred_minor)
        
        if f1 > best_f1:
            best_f1 = f1
            best_minor_threshold = threshold
    
    # Делаем порог мягче для высокого recall
    calibrated_thresholds['minor_threshold'] = min(best_minor_threshold, 0.45)
    print(f"   ✅ Оптимальный порог minor: {calibrated_thresholds['minor_threshold']:.3f}")
    
    # 4. Порог уверенности (на основе максимальной вероятности)
    max_probs = np.max(all_probs, axis=1)
    confidence_threshold = np.percentile(max_probs, 30)  # 30% самых неуверенных -> inspection
    calibrated_thresholds['confidence_min'] = max(confidence_threshold, 0.6)
    
    print(f"   ✅ Порог уверенности: {calibrated_thresholds['confidence_min']:.3f}")
    
    return calibrated_thresholds

def test_calibrated_classifier():
    """Тестирование калиброванного классификатора"""
    
    print("🧪 Тестирование калиброванного классификатора")
    
    # Создаем тестовые вероятности
    test_cases = [
        ([0.8, 0.15, 0.05], "Четкое no_damage"),
        ([0.3, 0.6, 0.1], "Четкое minor_damage"), 
        ([0.2, 0.3, 0.5], "Четкое major_damage"),
        ([0.4, 0.35, 0.25], "Неуверенный случай"),
        ([0.6, 0.25, 0.15], "Подозрение на damage"),
    ]
    
    classifier = CalibratedDamageClassifier(model_path=None)
    
    for probs, description in test_cases:
        # Создаем fake tensor
        fake_tensor = torch.zeros(1, 3)
        fake_tensor[0] = torch.tensor(probs)
        
        # Обходим модель и используем прямо вероятности
        with torch.no_grad():
            p_no, p_minor, p_major = probs
            max_confidence = max(probs)
            damage_prob = p_minor + p_major
            
            # Применяем нашу логику
            if max_confidence < classifier.thresholds['confidence_min']:
                status = 'requires_inspection'
                reason = 'low_confidence'
            elif p_no >= 0.6 and damage_prob < classifier.thresholds['damage_threshold']:
                status = 'no_damage'
                reason = 'clear_no_damage'
            elif p_major >= classifier.thresholds['major_threshold']:
                status = 'major_damage'
                reason = 'clear_major_damage'
            elif p_minor >= classifier.thresholds['minor_threshold']:
                status = 'minor_damage'
                reason = 'detected_minor_damage'
            elif damage_prob >= classifier.thresholds['damage_threshold']:
                status = 'requires_inspection'
                reason = 'suspected_damage'
            else:
                status = 'no_damage'
                reason = 'probable_no_damage'
        
        print(f"   {description}:")
        print(f"      Probs: {probs}")
        print(f"      Status: {status}")
        print(f"      Reason: {reason}")
        print()

if __name__ == "__main__":
    test_calibrated_classifier()