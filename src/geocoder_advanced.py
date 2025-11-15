"""
Улучшенный алгоритм геокодирования с нечетким поиском и индексацией.
"""
from typing import List, Dict, Optional, Tuple
import geopandas as gpd
import pandas as pd
from rapidfuzz import fuzz, process
from rtree import index
from shapely.geometry import Point
from .address_normalizer import AddressNormalizer


class AdvancedGeocoder:
    """Улучшенный геокодер с нечетким поиском."""
    
    def __init__(self, buildings: gpd.GeoDataFrame):
        """
        Инициализация геокодера.
        
        Args:
            buildings: GeoDataFrame с зданиями Москвы
        """
        self.buildings = buildings
        self.normalizer = AddressNormalizer()
        
        # Строим индексы
        self._build_indexes()
    
    def _build_indexes(self):
        """Построение индексов для поиска."""
        # Нормализуем адреса и разбиваем улицу на тип и имя
        def _street_parts(x):
            if pd.isna(x):
                return pd.Series({"street_name_normalized": "", "street_type_normalized": ""})
            info = self.normalizer.parse_street(str(x))
            name_norm = self.normalizer.normalize(info.get("name", "") or "")
            type_norm = info.get("type") or ""
            return pd.Series({"street_name_normalized": name_norm, "street_type_normalized": type_norm})

        street_parts = self.buildings['street'].apply(_street_parts)
        self.buildings['street_normalized'] = street_parts['street_name_normalized']
        self.buildings['street_type_normalized'] = street_parts['street_type_normalized']
        self.buildings['number_normalized'] = self.buildings['number'].apply(
            lambda x: self.normalizer.normalize_house_number(str(x)) if pd.notna(x) else ""
        )
        
        # Создаем полный адрес для каждого здания
        self.buildings['full_address'] = self.buildings.apply(
            lambda row: self.normalizer.create_full_address(
                str(row.get('locality', 'Москва')),
                str(row.get('street_normalized', '')),
                str(row.get('number_normalized', ''))
            ),
            axis=1
        )
        
        # Создаем пространственный индекс (R-tree)
        self.spatial_index = index.Index()
        for idx, row in self.buildings.iterrows():
            centroid = row.geometry.centroid
            # R-tree использует (minx, miny, maxx, maxy)
            self.spatial_index.insert(
                idx,
                (centroid.x, centroid.y, centroid.x, centroid.y)
            )
        
        # Создаем словарь для быстрого доступа к адресам
        self.address_dict = {
            idx: {
                'street_name': row['street_normalized'],
                'street_type': row.get('street_type_normalized', ''),
                'number': row['number_normalized'],
                'full': row['full_address'],
            }
            for idx, row in self.buildings.iterrows()
        }

        # Индекс по улицам: street_name_normalized -> список индексов зданий
        self.street_index = {}
        for idx, row in self.buildings.iterrows():
            sname = row['street_normalized']
            if not sname:
                continue
            self.street_index.setdefault(sname, []).append(idx)
    
    def _fuzzy_match_street(self, query_street: str, threshold: int = 70) -> List[Tuple[int, float]]:
        """
        Нечеткий поиск улицы.
        
        Args:
            query_street: Запрос улицы
            threshold: Порог схожести (0-100)
            
        Returns:
            Список кортежей (индекс, score)
        """
        # Нормализуем улицу и выделяем тип/имя
        street_info = self.normalizer.parse_street(query_street)
        query_name = street_info.get("name") or ""
        query_type = street_info.get("type") or ""
        query_name_norm = self.normalizer.normalize(query_name)

        matches: List[Tuple[int, float]] = []

        if not query_name_norm:
            return matches

        # Динамический порог и ограничение на число улиц:
        # - для очень коротких названий улиц (<=4 символов без пробелов) требуем более высокое сходство,
        #   чтобы не ловить массу ложных совпадений;
        # - ограничиваемся top-N улицами по сходству.
        name_len = len(query_name_norm.replace(" ", ""))
        eff_threshold = threshold
        if name_len <= 4:
            eff_threshold = max(threshold, 80)

        max_streets = 20

        # Сначала считаем похожесть только для уникальных названий улиц
        street_candidates: List[Tuple[str, float]] = []
        for street_name, idx_list in self.street_index.items():
            if not street_name:
                continue
            similarity = fuzz.ratio(query_name_norm, street_name)
            if similarity < eff_threshold:
                continue
            street_candidates.append((street_name, similarity / 100.0))

        if not street_candidates:
            return matches

        # Берем только top-N самых похожих улиц
        street_candidates.sort(key=lambda x: x[1], reverse=True)
        street_candidates = street_candidates[:max_streets]

        # Разворачиваем выбранные улицы в здания и добавляем бонус/штраф за тип улицы
        for street_name, base_sim in street_candidates:
            idx_list = self.street_index.get(street_name, [])
            for idx in idx_list:
                addr_data = self.address_dict[idx]
                cand_type = addr_data.get('street_type', '')
                type_bonus = 0.0
                if query_type:
                    if cand_type == query_type:
                        type_bonus = 0.05  # совпал тип
                    elif cand_type:
                        type_bonus = -0.05  # типы различаются
                sim_score = max(0.0, min(1.0, base_sim + type_bonus))
                matches.append((idx, sim_score))

        # Сортируем по убыванию схожести
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def _calculate_score(self, row, parsed_query: Dict, street_similarity: float, query_full: str) -> float:
        """
        Вычисление итогового score для результата с помощью логистической модели.
        
        Args:
            row: Строка из GeoDataFrame
            parsed_query: Распарсенный запрос
            street_similarity: Схожесть улицы (0-1)
            query_full: Нормализованный полный адрес запроса
            
        Returns:
            Score от 0 до 1
        """
        import math

        # --- Фичи для логистической модели ---

        # 1. Схожесть номера дома (учет основного номера и корпуса/строения)
        number_query_raw = str(parsed_query.get('number', '') or "").strip()
        number_query_norm = self.normalizer.normalize_house_number(number_query_raw) if number_query_raw else ""
        number_data_norm = str(row.get('number_normalized', '') or "").strip()

        if number_query_norm and number_data_norm:
            q_info = self.normalizer.parse_house_number(number_query_norm)
            d_info = self.normalizer.parse_house_number(number_data_norm)

            if q_info["main"] and d_info["main"]:
                if q_info["main"] == d_info["main"]:
                    if (
                        q_info["extra_type"] == d_info["extra_type"]
                        and q_info["extra_number"] == d_info["extra_number"]
                        and q_info["extra_type"] is not None
                    ):
                        number_similarity = 1.0
                    else:
                        number_similarity = 0.8
                else:
                    number_similarity = 0.0
            else:
                number_similarity = 0.0
        else:
            number_similarity = 0.5 if not number_query_norm else 0.0

        # 2. Схожесть полного адреса
        full_addr = str(row.get('full_address', '') or '')
        if query_full and full_addr:
            full_similarity = self.normalizer.calculate_similarity(query_full, full_addr)
        else:
            full_similarity = street_similarity  # fallback

        # 3. Соответствие локальности
        query_locality = parsed_query.get('locality') or 'Москва'
        cand_locality = str(row.get('locality', '') or '')
        q = query_locality.lower()
        c = cand_locality.lower()
        q_is_moscow = 'москва' in q
        c_is_moscow = 'москва' in c
        if q_is_moscow and c_is_moscow:
            locality_match = 1.0
        elif (not q and c_is_moscow) or (q_is_moscow and not c):
            locality_match = 0.5
        elif not q and not c:
            locality_match = 0.5
        else:
            locality_match = 0.0

        # --- Логистическая модель ---
        # Веса получены из train_scoring_model.py (LogisticRegression):
        # coef_ = [-0.26517875  6.32422384  7.88170754 -2.11618385]
        # intercept_ = -6.790378454666675

        w_street = -0.26517875
        w_number = 6.32422384
        w_full = 7.88170754
        w_locality = -2.11618385
        bias = -6.790378454666675

        z = (
            bias
            + w_street * street_similarity
            + w_number * number_similarity
            + w_full * full_similarity
            + w_locality * locality_match
        )
        score = 1.0 / (1.0 + math.exp(-z))

        return float(score)
    
    def geocode(self, address: str, max_results: int = 10, 
                street_threshold: int = 60) -> List[Dict]:
        """
        Геокодирование адреса (улучшенный алгоритм).
        
        Args:
            address: Адресная строка
            max_results: Максимальное количество результатов
            street_threshold: Порог схожести для улицы (0-100)
            
        Returns:
            Список найденных объектов с score
        """
        # Парсим входной адрес
        parsed = self.normalizer.parse_address(address)
        
        query_street = parsed.get('street', '') or ''
        query_number = parsed.get('number', '') or ''

        # Нормализованные компоненты для полного адреса
        query_locality = parsed.get('locality') or 'Москва'
        query_street_norm = self.normalizer.normalize(query_street)
        query_number_norm = self.normalizer.normalize_house_number(query_number)
        query_full = self.normalizer.create_full_address(
            query_locality,
            query_street_norm,
            query_number_norm,
        )
        
        # Нечеткий поиск улиц
        street_matches = self._fuzzy_match_street(query_street, street_threshold)
        
        # Фильтруем по номеру дома если указан
        candidates = []
        for idx, street_score in street_matches:
            row = self.buildings.loc[idx]
            number_data_norm = str(row.get('number_normalized', '') or '').strip()
            
            # Если номер дома указан, проверяем совпадение
            if query_number_norm:
                q_info = self.normalizer.parse_house_number(query_number_norm)
                d_info = self.normalizer.parse_house_number(number_data_norm) if number_data_norm else {
                    "main": None, "extra_type": None, "extra_number": None
                }

                match = False
                if q_info["main"] and d_info["main"]:
                    # Совпадение основного номера
                    if q_info["main"] == d_info["main"]:
                        match = True

                if match:
                    score = self._calculate_score(row, parsed, street_score, query_full)
                    candidates.append((idx, score))
            else:
                # Номер не указан, используем только схожесть улицы
                score = self._calculate_score(row, parsed, street_score, query_full)
                candidates.append((idx, score))

        # Сортируем по score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Формируем результат
        results = []
        for idx, score in candidates[:max_results]:
            row = self.buildings.loc[idx]
            centroid = row.geometry.centroid
            results.append({
                'locality': str(row.get('locality', 'Москва')),
                'street': str(row.get('street', '')),
                'number': str(row.get('number', '')),
                'lon': float(centroid.x),
                'lat': float(centroid.y),
                'score': round(score, 4)
            })
        
        return results
    
    def geocode_with_location_hint(self, address: str, lat: Optional[float] = None,
                                   lon: Optional[float] = None, max_results: int = 10,
                                   distance_weight: float = 0.2) -> List[Dict]:
        """
        Геокодирование с учетом геолокации (дополнительная функция).
        
        Args:
            address: Адресная строка
            lat: Широта подсказки
            lon: Долгота подсказки
            max_results: Максимальное количество результатов
            distance_weight: Вес расстояния в финальном score (0-1)
            
        Returns:
            Список найденных объектов
        """
        # Сначала получаем результаты обычного геокодирования
        results = self.geocode(address, max_results=max_results * 2)
        
        if lat is None or lon is None:
            return results[:max_results]
        
        # Вычисляем расстояния и корректируем score
        from geopy.distance import geodesic
        
        hint_point = (lat, lon)
        
        for result in results:
            result_point = (result['lat'], result['lon'])
            distance_km = geodesic(hint_point, result_point).kilometers
            
            # Нормализуем расстояние (чем ближе, тем выше score)
            # Используем экспоненциальное затухание
            distance_score = 1.0 / (1.0 + distance_km / 5.0)  # 5 км - половина score
            
            # Комбинируем с исходным score
            original_score = result['score']
            result['score'] = (1 - distance_weight) * original_score + distance_weight * distance_score
        
        # Сортируем по новому score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:max_results]

