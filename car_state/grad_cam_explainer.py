"""
Система Grad-CAM для объяснения предсказаний модели
Показывает, на какие области изображения смотрит модель
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

class GradCAM:
    """Реализация Grad-CAM для объяснения предсказаний"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Если слой не указан, используем последний conv слой
        if target_layer is None:
            self.target_layer = self.model.backbone.layer4[-1].conv3
        else:
            self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Регистрируем хуки
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Сохраняем активации"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Сохраняем градиенты"""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, target_class=None):
        """Генерируем карту внимания"""
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()
        
        # Получаем градиенты и активации
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Вычисляем веса как средние градиенты
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Взвешенная комбинация активаций
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # ReLU
        cam = F.relu(cam)
        
        # Нормализация
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, original_image, cam, alpha=0.4):
        """Визуализация карты внимания поверх исходного изображения"""
        # Преобразуем изображение в numpy если нужно
        if isinstance(original_image, torch.Tensor):
            if original_image.dim() == 4:
                original_image = original_image[0]
            original_image = original_image.permute(1, 2, 0).cpu().numpy()
            # Денормализация
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_image = original_image * std + mean
            original_image = np.clip(original_image, 0, 1)
        
        # Изменяем размер карты под исходное изображение
        h, w = original_image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Применяем цветовую карту
        heatmap = cm.jet(cam_resized)[:, :, :3]
        
        # Накладываем тепловую карту на изображение
        result = (1 - alpha) * original_image + alpha * heatmap
        result = np.clip(result, 0, 1)
        
        return result, heatmap

class ExplainableAnalyzer:
    """Анализатор с объяснениями через Grad-CAM"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        
        # Загружаем модель
        from improved_training import ImprovedDamageModel
        self.model = ImprovedDamageModel(num_classes=2)
        
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Получаем оптимальный порог
        self.threshold = checkpoint.get('threshold', 0.5)
        
        # Создаем Grad-CAM
        self.grad_cam = GradCAM(self.model)
        
        # Трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def analyze_with_explanation(self, image_path):
        """Анализ с объяснением через Grad-CAM"""
        # Загружаем и подготавливаем изображение
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Получаем предсказание
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities.max().item()
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # Определяем статус с учетом порога
        damage_prob = probabilities[0, 1].item()
        
        if confidence < 0.7:  # Низкая уверенность
            status = "требует_осмотра"
            confidence_level = "низкая"
        elif damage_prob > self.threshold:
            status = "повреждения_обнаружены"
            confidence_level = "высокая" if confidence > 0.85 else "средняя"
        else:
            status = "повреждения_не_обнаружены"
            confidence_level = "высокая" if confidence > 0.85 else "средняя"
        
        # Генерируем Grad-CAM
        cam = self.grad_cam.generate_cam(input_tensor, target_class=predicted_class)
        
        # Создаем визуализацию
        explained_image, heatmap = self.grad_cam.visualize_cam(
            original_image, cam, alpha=0.4
        )
        
        return {
            'status': status,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'damage_probability': damage_prob,
            'threshold': self.threshold,
            'explanation': {
                'original_image': original_image,
                'heatmap': heatmap,
                'explained_image': explained_image,
                'attention_map': cam
            },
            'human_readable': self.format_explanation(
                status, confidence, damage_prob, confidence_level
            )
        }
    
    def format_explanation(self, status, confidence, damage_prob, confidence_level):
        """Форматирование объяснения для пользователя"""
        status_map = {
            'повреждения_обнаружены': 'Обнаружены повреждения',
            'повреждения_не_обнаружены': 'Повреждения не обнаружены',
            'требует_осмотра': 'Требуется дополнительный осмотр'
        }
        
        confidence_map = {
            'высокая': 'высокой',
            'средняя': 'средней', 
            'низкая': 'низкой'
        }
        
        result = f"🔍 {status_map[status]} с {confidence_map[confidence_level]} уверенностью ({confidence:.1%})"
        
        if status == 'повреждения_обнаружены':
            result += f"\n🚨 Вероятность повреждений: {damage_prob:.1%}"
            result += f"\n💡 Рекомендация: Обратитесь к специалисту для оценки ущерба"
        elif status == 'требует_осмотра':
            result += f"\n⚠️ Модель не уверена в результате"
            result += f"\n💡 Рекомендация: Требуется осмотр экспертом"
        else:
            result += f"\n✅ Автомобиль в хорошем состоянии"
        
        return result
    
    def save_explanation(self, analysis_result, save_path):
        """Сохранение объяснения в файл"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Исходное изображение
        axes[0].imshow(analysis_result['explanation']['original_image'])
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        # Тепловая карта
        axes[1].imshow(analysis_result['explanation']['heatmap'])
        axes[1].set_title('Карта внимания модели')
        axes[1].axis('off')
        
        # Объясненное изображение
        axes[2].imshow(analysis_result['explanation']['explained_image'])
        axes[2].set_title('Области внимания')
        axes[2].axis('off')
        
        # Добавляем текстовое объяснение
        fig.suptitle(analysis_result['human_readable'], fontsize=12, y=0.02)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Объяснение сохранено: {save_path}")

def test_explainable_analyzer():
    """Тестирование объяснимого анализатора"""
    try:
        analyzer = ExplainableAnalyzer('improved_model.pth')
        
        # Ищем тестовое изображение
        test_image = None
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images = list(Path('.').glob(ext))
            if images:
                test_image = images[0]
                break
        
        if test_image:
            print(f"🧪 Тестируем на изображении: {test_image}")
            
            result = analyzer.analyze_with_explanation(test_image)
            
            print(f"\n{result['human_readable']}")
            print(f"\n📊 Детальная информация:")
            print(f"   Статус: {result['status']}")
            print(f"   Уверенность: {result['confidence']:.3f}")
            print(f"   Вероятность повреждений: {result['damage_probability']:.3f}")
            print(f"   Порог: {result['threshold']:.3f}")
            
            # Сохраняем объяснение
            analyzer.save_explanation(result, 'explanation_result.png')
            
        else:
            print("❌ Не найдено изображений для тестирования")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    test_explainable_analyzer()