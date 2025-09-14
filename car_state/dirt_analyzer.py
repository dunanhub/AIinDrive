"""
Анализ чистоты автомобиля через CV эвристики
Определяет уровень загрязнения без машинного обучения
"""
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DirtAnalyzer:
    """
    Анализатор грязи на основе компьютерного зрения
    Использует эвристики для определения уровня загрязнения
    """
    
    def __init__(self):
        # Пороги для классификации чистоты
        self.thresholds = {
            'clean': 0.35,        # < 0.35 = чистая
            'slightly_dirty': 0.60  # 0.35-0.60 = слегка грязная, >0.60 = грязная
        }
        
        # Веса для компонентов грязи
        self.weights = {
            'contrast': 0.45,     # Низкий контраст = грязь
            'saturation': 0.35,   # Низкая насыщенность = грязь
            'noise': 0.20         # Высокий шум = грязь
        }
    
    def analyze_contrast(self, image_array: np.ndarray) -> float:
        """
        Анализ контраста изображения
        Низкий контраст может указывать на грязь/пыль
        """
        # Преобразуем в градации серого
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Вычисляем стандартное отклонение (мера контраста)
        contrast = float(np.std(gray))
        
        # Нормализуем (типичный диапазон 0-80)
        normalized_contrast = np.clip(contrast / 80.0, 0, 1)
        
        # Инвертируем: низкий контраст = высокий показатель грязи
        dirt_score = 1.0 - normalized_contrast
        
        return dirt_score
    
    def analyze_saturation(self, image_array: np.ndarray) -> float:
        """
        Анализ насыщенности цветов
        Низкая насыщенность может указывать на пыль/грязь
        """
        # Преобразуем в HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        
        # Средняя насыщенность
        mean_saturation = float(np.mean(saturation))
        
        # Инвертируем: низкая насыщенность = высокий показатель грязи
        dirt_score = 1.0 - mean_saturation
        
        return dirt_score
    
    def analyze_noise(self, image_array: np.ndarray) -> float:
        """
        Анализ шума/зернистости изображения
        Высокий шум может указывать на грязь/пятна
        """
        # Преобразуем в градации серого
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Применяем оператор Лапласа для детекции краев/шума
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_variance = float(laplacian.var())
        
        # Нормализуем (типичный диапазон 0-1000)
        normalized_noise = np.clip(noise_variance / 1000.0, 0, 1)
        
        return normalized_noise
    
    def analyze_brightness_uniformity(self, image_array: np.ndarray) -> float:
        """
        Анализ равномерности яркости
        Неравномерная яркость может указывать на пятна/грязь
        """
        # Преобразуем в HSV и берем канал яркости (V)
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        brightness = hsv[:, :, 2].astype(np.float32) / 255.0
        
        # Вычисляем стандартное отклонение яркости
        brightness_std = float(np.std(brightness))
        
        # Нормализуем
        uniformity_score = np.clip(brightness_std * 2.0, 0, 1)
        
        return uniformity_score
    
    def analyze_color_deviation(self, image_array: np.ndarray) -> float:
        """
        Анализ отклонения от типичных цветов автомобиля
        Коричневые/серые тона могут указывать на грязь
        """
        # Преобразуем в HSV
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        
        # Типичные оттенки грязи (коричневый, серый) в HSV
        # Коричневый: 10-20, Серый: 0-180 с низкой насыщенностью
        dirt_hue_mask = ((hue >= 10) & (hue <= 30))  # Коричневые тона
        
        dirt_pixels_ratio = float(np.sum(dirt_hue_mask)) / dirt_hue_mask.size
        
        return dirt_pixels_ratio
    
    def compute_dirt_score(self, pil_image: Image.Image) -> float:
        """
        Основная функция вычисления показателя грязи
        
        Args:
            pil_image: PIL изображение
            
        Returns:
            float: Показатель грязи от 0.0 (чистая) до 1.0 (грязная)
        """
        # Преобразуем PIL в numpy array
        image_array = np.array(pil_image.convert("RGB"))
        
        # Изменяем размер для ускорения вычислений
        if image_array.shape[0] > 512 or image_array.shape[1] > 512:
            pil_resized = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
            image_array = np.array(pil_resized)
        
        # Вычисляем компоненты
        contrast_score = self.analyze_contrast(image_array)
        saturation_score = self.analyze_saturation(image_array)
        noise_score = self.analyze_noise(image_array)
        uniformity_score = self.analyze_brightness_uniformity(image_array)
        color_deviation_score = self.analyze_color_deviation(image_array)
        
        # Взвешенная комбинация
        dirt_score = (
            self.weights['contrast'] * contrast_score +
            self.weights['saturation'] * saturation_score +
            self.weights['noise'] * noise_score +
            0.15 * uniformity_score +
            0.10 * color_deviation_score
        )
        
        # Ограничиваем диапазон [0, 1]
        dirt_score = np.clip(dirt_score, 0.0, 1.0)
        
        return float(dirt_score)
    
    def classify_cleanliness(self, dirt_score: float) -> Dict[str, Any]:
        """
        Классификация уровня чистоты на основе показателя грязи
        
        Args:
            dirt_score: Показатель грязи (0.0 - 1.0)
            
        Returns:
            dict: Результат классификации
        """
        if dirt_score < self.thresholds['clean']:
            status = "clean"
            description = f"Автомобиль чистый (показатель грязи: {int(dirt_score*100)}%)"
            recommendation = "Автомобиль в отличном состоянии чистоты"
            level = "excellent"
            
        elif dirt_score < self.thresholds['slightly_dirty']:
            status = "slightly_dirty"
            description = f"Автомобиль слегка грязный, в допустимых пределах (показатель: {int(dirt_score*100)}%)"
            recommendation = "Небольшие загрязнения, но в целом приемлемо"
            level = "acceptable"
            
        else:
            status = "dirty"
            description = f"Автомобиль грязный (показатель грязи: {int(dirt_score*100)}%)"
            recommendation = "Рекомендуется мойка автомобиля"
            level = "poor"
        
        return {
            'status': status,
            'level': level,
            'description': description,
            'recommendation': recommendation,
            'dirt_score': dirt_score,
            'score_percentage': int(dirt_score * 100),
            'thresholds_used': self.thresholds.copy()
        }
    
    def analyze_cleanliness(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Полный анализ чистоты изображения
        
        Args:
            pil_image: PIL изображение для анализа
            
        Returns:
            dict: Полный результат анализа чистоты
        """
        # Вычисляем показатель грязи
        dirt_score = self.compute_dirt_score(pil_image)
        
        # Классифицируем результат
        result = self.classify_cleanliness(dirt_score)
        
        # Добавляем технические детали
        image_array = np.array(pil_image.convert("RGB"))
        if image_array.shape[0] > 512 or image_array.shape[1] > 512:
            pil_resized = pil_image.resize((512, 512), Image.Resampling.LANCZOS)
            image_array = np.array(pil_resized)
        
        technical_details = {
            'contrast_component': self.analyze_contrast(image_array),
            'saturation_component': self.analyze_saturation(image_array),
            'noise_component': self.analyze_noise(image_array),
            'uniformity_component': self.analyze_brightness_uniformity(image_array),
            'color_deviation_component': self.analyze_color_deviation(image_array),
            'component_weights': self.weights.copy()
        }
        
        result['technical_details'] = technical_details
        
        return result

def create_human_readable_cleanliness_report(cleanliness_result: Dict[str, Any]) -> str:
    """
    Создание человекочитаемого отчета о чистоте
    
    Args:
        cleanliness_result: Результат анализа чистоты
        
    Returns:
        str: Человекочитаемый отчет
    """
    status_map = {
        'clean': '🧽 Чистый',
        'slightly_dirty': '🟡 Слегка грязный',
        'dirty': '🟤 Грязный'
    }
    
    level_map = {
        'excellent': 'отличное',
        'acceptable': 'приемлемое', 
        'poor': 'плохое'
    }
    
    status_icon = status_map.get(cleanliness_result['status'], '❓')
    level_text = level_map.get(cleanliness_result['level'], 'неопределенное')
    
    report = f"{status_icon} Состояние чистоты: {level_text}\n"
    report += f"📊 {cleanliness_result['description']}\n"
    report += f"💡 {cleanliness_result['recommendation']}"
    
    return report

# Тестирование
if __name__ == "__main__":
    # Создаем анализатор
    analyzer = DirtAnalyzer()
    
    # Создаем тестовое изображение
    test_image = Image.new('RGB', (224, 224), (128, 128, 128))
    
    # Тестируем анализ
    result = analyzer.analyze_cleanliness(test_image)
    report = create_human_readable_cleanliness_report(result)
    
    print("🧪 Тест анализатора чистоты:")
    print(f"Результат: {result}")
    print(f"\nОтчет:\n{report}")
    
    # Тестируем разные показатели
    test_scores = [0.2, 0.5, 0.8]
    for score in test_scores:
        classification = analyzer.classify_cleanliness(score)
        print(f"\nПоказатель грязи {score}: {classification['status']} - {classification['description']}")