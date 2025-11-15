"""
Базовый алгоритм геокодирования - точное совпадение компонентов адреса.
"""
from typing import List, Dict, Optional
import pandas as pd
import geopandas as gpd
from .address_normalizer import AddressNormalizer


class BaseGeocoder:
    """Базовый геокодер с точным совпадением."""
    
    def __init__(self, buildings: gpd.GeoDataFrame):
        """
        Инициализация геокодера.
        
        Args:
            buildings: GeoDataFrame с зданиями Москвы
        """
        self.buildings = buildings
        self.normalizer = AddressNormalizer()
        
        # Создаем простой индекс для быстрого поиска
        self._build_index()
    
    def _build_index(self):
        """Построение индекса для поиска."""
        # Нормализуем улицы в данных до канонического вида
        # (учитываем тип: ул/пл/ш и т.п., чтобы "Тверская площадь" и "пл. Тверская" совпадали)
        self.buildings['street_normalized'] = self.buildings['street'].apply(
            lambda x: self.normalizer.canonicalize_street(str(x)) if pd.notna(x) else ""
        )
        # Нормализация номера дома (учёт корпуса/строения)
        self.buildings['number_normalized'] = self.buildings['number'].apply(
            lambda x: self.normalizer.normalize_house_number(str(x)) if pd.notna(x) else ""
        )
        # Основной номер дома (без корпуса/строения) для более грубых совпадений
        def _main_number(val: str) -> str:
            if pd.isna(val):
                return ""
            norm = self.normalizer.normalize_house_number(str(val))
            info = self.normalizer.parse_house_number(norm)
            return info.get("main") or ""

        self.buildings['number_main'] = self.buildings['number'].apply(_main_number)
    
    def geocode(self, address: str, max_results: int = 10) -> List[Dict]:
        """
        Геокодирование адреса (базовый алгоритм).
        
        Args:
            address: Адресная строка
            max_results: Максимальное количество результатов
            
        Returns:
            Список найденных объектов
        """
        # Парсим входной адрес
        parsed = self.normalizer.parse_address(address)
        
        # Нормализуем компоненты
        street_query = self.normalizer.normalize(parsed['street'] or "")
        number_query = self.normalizer.normalize_house_number(parsed['number'] or "")
        number_main_query = ""
        if number_query:
            info_q = self.normalizer.parse_house_number(number_query)
            number_main_query = info_q.get("main") or ""
        
        # 1) Строгое совпадение улицы и полного номера
        matches = self.buildings[
            (self.buildings['street_normalized'] == street_query) &
            (self.buildings['number_normalized'] == number_query)
        ].copy()

        # 2) Если нет, совпадение улицы и основного номера (игнорируем корпус/строение)
        if len(matches) == 0 and street_query and number_main_query:
            matches = self.buildings[
                (self.buildings['street_normalized'] == street_query) &
                (self.buildings['number_main'] == number_main_query)
            ].copy()

        # 3) Если нет, частичное совпадение улицы + полный номер
        if len(matches) == 0 and street_query and number_query:
            matches = self.buildings[
                (self.buildings['street_normalized'].str.contains(street_query, na=False)) &
                (self.buildings['number_normalized'] == number_query)
            ].copy()

        # 4) Если нет, частичное совпадение улицы + основной номер
        if len(matches) == 0 and street_query and number_main_query:
            matches = self.buildings[
                (self.buildings['street_normalized'].str.contains(street_query, na=False)) &
                (self.buildings['number_main'] == number_main_query)
            ].copy()

        # 5) Если все еще нет совпадений, ищем только по улице (строгое или частичное совпадение)
        if len(matches) == 0 and street_query:
            matches = self.buildings[
                (self.buildings['street_normalized'] == street_query) |
                (self.buildings['street_normalized'].str.contains(street_query, na=False))
            ].copy()
        
        # Формируем результат
        results = []
        for idx, row in matches.head(max_results).iterrows():
            centroid = row.geometry.centroid
            locality_raw = str(row.get('locality', 'Москва'))
            street_raw = str(row.get('street', ''))
            number_raw = str(row.get('number', ''))
            components = self.normalizer.normalize_components_for_output(
                locality_raw, street_raw, number_raw
            )

            results.append({
                'locality': components['locality'],
                'street': components['street'],
                'number': components['number'],
                'lon': float(centroid.x),
                'lat': float(centroid.y),
                'score': 1.0,  # Базовый алгоритм возвращает score=1.0 для точных совпадений
            })
        
        return results

