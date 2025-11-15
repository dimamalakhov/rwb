"""
Метрики оценки качества геокодирования.
"""
from typing import List, Dict, Tuple
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from geopy.distance import geodesic


class GeocodingMetrics:
    """Класс для вычисления метрик качества геокодирования."""
    
    @staticmethod
    def levenshtein_distance_raw(predicted: str, true: str) -> int:
        """
        Вычисление "сырого" расстояния Левенштейна между двумя строками.
        
        Args:
            predicted: Предсказанный адрес
            true: Эталонный адрес
            
        Returns:
            Целое расстояние Левенштейна (0 - полное совпадение)
        """
        if predicted is None:
            predicted = ""
        if true is None:
            true = ""
        
        pred_norm = predicted.lower().strip()
        true_norm = true.lower().strip()
        
        return int(Levenshtein.distance(pred_norm, true_norm))
    
    @staticmethod
    def levenshtein_similarity(predicted: str, true: str) -> float:
        """
        Нормированная схожесть на основе расстояния Левенштейна.
        
        Формула из задания:
            score = 1 - L(A_pred, A_true) / max(len(A_pred), len(A_true))
        """
        if not predicted and not true:
            return 1.0
        if not predicted or not true:
            return 0.0
        
        pred_norm = predicted.lower().strip()
        true_norm = true.lower().strip()
        if not pred_norm and not true_norm:
            return 1.0
        
        dist = GeocodingMetrics.levenshtein_distance_raw(pred_norm, true_norm)
        max_len = max(len(pred_norm), len(true_norm))
        if max_len == 0:
            return 1.0
        
        score = 1.0 - dist / max_len
        # На всякий случай ограничим диапазон [0,1]
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def address_text_score(predicted_address: Dict, true_address: Dict) -> float:
        """
        Вычисление score по тексту адреса.
        
        Args:
            predicted_address: Предсказанный адрес (словарь с locality, street, number)
            true_address: Эталонный адрес
            
        Returns:
            Score от 0 до 1
        """
        # Формируем полные адреса
        pred_full = f"{predicted_address.get('locality', '')}, {predicted_address.get('street', '')}, {predicted_address.get('number', '')}"
        true_full = f"{true_address.get('locality', '')}, {true_address.get('street', '')}, {true_address.get('number', '')}"
        
        return GeocodingMetrics.levenshtein_similarity(pred_full, true_full)
    
    @staticmethod
    def coordinate_distance(pred_lat: float, pred_lon: float,
                           true_lat: float, true_lon: float) -> float:
        """
        Вычисление расстояния между координатами в метрах.
        
        Args:
            pred_lat: Предсказанная широта
            pred_lon: Предсказанная долгота
            true_lat: Эталонная широта
            true_lon: Эталонная долгота
            
        Returns:
            Расстояние в метрах
        """
        pred_point = (pred_lat, pred_lon)
        true_point = (true_lat, true_lon)
        
        return geodesic(pred_point, true_point).meters
    
    @staticmethod
    def coordinate_score(pred_lat: float, pred_lon: float,
                        true_lat: float, true_lon: float,
                        max_distance_m: float = 1000.0) -> float:
        """
        Вычисление score по координатам (0-1).
        
        Args:
            pred_lat: Предсказанная широта
            pred_lon: Предсказанная долгота
            true_lat: Эталонная широта
            true_lon: Эталонная долгота
            max_distance_m: Максимальное расстояние для score=0 (в метрах)
            
        Returns:
            Score от 0 до 1
        """
        distance = GeocodingMetrics.coordinate_distance(
            pred_lat, pred_lon, true_lat, true_lon
        )
        
        # Экспоненциальное затухание
        if distance >= max_distance_m:
            return 0.0
        
        # Score уменьшается с расстоянием
        score = 1.0 - (distance / max_distance_m)
        return max(0.0, score)
    
    @staticmethod
    def combined_score(predicted: Dict, true: Dict,
                      text_weight: float = 0.5,
                      coord_weight: float = 0.5) -> float:
        """
        Комбинированный score (текст + координаты).
        
        Args:
            predicted: Предсказанный результат
            true: Эталонный результат
            text_weight: Вес текстовой метрики
            coord_weight: Вес метрики координат
            
        Returns:
            Комбинированный score от 0 до 1
        """
        # Score по тексту
        text_score = GeocodingMetrics.address_text_score(
            {
                'locality': predicted.get('locality', ''),
                'street': predicted.get('street', ''),
                'number': predicted.get('number', '')
            },
            {
                'locality': true.get('locality', ''),
                'street': true.get('street', ''),
                'number': true.get('number', '')
            }
        )
        
        # Score по координатам
        coord_score = GeocodingMetrics.coordinate_score(
            predicted.get('lat', 0),
            predicted.get('lon', 0),
            true.get('lat', 0),
            true.get('lon', 0)
        )
        
        # Взвешенная сумма
        total_weight = text_weight + coord_weight
        combined = (text_weight * text_score + coord_weight * coord_score) / total_weight
        
        return combined
    
    @staticmethod
    def evaluate_batch(predictions: List[Dict], ground_truth: List[Dict],
                      text_weight: float = 0.5, coord_weight: float = 0.5) -> Dict:
        """
        Оценка качества на батче данных.
        
        Args:
            predictions: Список предсказаний
            ground_truth: Список эталонных данных
            text_weight: Вес текстовой метрики
            coord_weight: Вес метрики координат
            
        Returns:
            Словарь с метриками
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Количество предсказаний и эталонов должно совпадать")
        
        text_scores = []
        coord_scores = []
        combined_scores = []
        distances = []
        
        for pred, true in zip(predictions, ground_truth):
            # Текстовая метрика
            text_score = GeocodingMetrics.address_text_score(
                {
                    'locality': pred.get('locality', ''),
                    'street': pred.get('street', ''),
                    'number': pred.get('number', '')
                },
                {
                    'locality': true.get('locality', ''),
                    'street': true.get('street', ''),
                    'number': true.get('number', '')
                }
            )
            text_scores.append(text_score)
            
            # Метрика координат
            coord_score = GeocodingMetrics.coordinate_score(
                pred.get('lat', 0),
                pred.get('lon', 0),
                true.get('lat', 0),
                true.get('lon', 0)
            )
            coord_scores.append(coord_score)
            
            # Расстояние
            distance = GeocodingMetrics.coordinate_distance(
                pred.get('lat', 0),
                pred.get('lon', 0),
                true.get('lat', 0),
                true.get('lon', 0)
            )
            distances.append(distance)
            
            # Комбинированный score
            combined = GeocodingMetrics.combined_score(
                pred, true, text_weight, coord_weight
            )
            combined_scores.append(combined)
        
        # Вычисляем статистику
        import numpy as np
        
        return {
            'text_score_mean': float(np.mean(text_scores)),
            'text_score_median': float(np.median(text_scores)),
            'coord_score_mean': float(np.mean(coord_scores)),
            'coord_score_median': float(np.median(coord_scores)),
            'combined_score_mean': float(np.mean(combined_scores)),
            'combined_score_median': float(np.median(combined_scores)),
            'distance_mean_m': float(np.mean(distances)),
            'distance_median_m': float(np.median(distances)),
            'distance_p95_m': float(np.percentile(distances, 95))
        }

