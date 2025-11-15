"""
Тренировка простой модели скоринга (логистическая регрессия) для advanced-геокодера.

Идея:
- Для N случайных зданий генерируем запросы.
- Для каждого запроса собираем кандидатов через текущий AdvancedGeocoder.
- Считаем фичи (street_sim, number_sim, full_sim, locality_match) между запросом и кандидатом.
- Отмечаем, какой кандидат является "истинным" (по совпадению координат с исходным зданием).
- Обучаем LogisticRegression предсказывать вероятность того, что кандидат - правильный дом.
- Считаем метрики top-1 для модели.

Важно: код не меняет поведение геокодера автоматически, а служит для экспериментов.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import geopandas as gpd
from geopy.distance import geodesic
from rapidfuzz import fuzz

from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.address_normalizer import AddressNormalizer
from src.geocoder_advanced import AdvancedGeocoder


def build_query(row: Dict, variant: int = 0) -> str:
    """
    Формирование текстового запроса по строке зданий.
    """
    locality = row.get("locality") or "Москва"
    street = row.get("street") or ""
    number = row.get("number") or ""

    if variant == 0:
        return f"{locality}, {street}, д. {number}"
    elif variant == 1:
        return f"{street}, {number}"
    elif variant == 2:
        return f"г {locality}, {street} {number}"
    else:
        return f"{locality}, {street}, {number}"


def compute_features_for_candidate(
    normalizer: AddressNormalizer,
    query: str,
    parsed_query: Dict,
    cand: Dict,
) -> Tuple[float, float, float, float]:
    """
    Вычисление фич для кандидата.

    Возвращает:
        street_sim, number_sim, full_sim, locality_match
    """
    # Разбираем запрос
    query_street = parsed_query.get("street") or ""
    query_number = parsed_query.get("number") or ""
    query_locality = parsed_query.get("locality") or "Москва"

    query_street_norm = normalizer.normalize(query_street)
    query_number_norm = normalizer.normalize_house_number(query_number)
    query_full = normalizer.create_full_address(
        query_locality,
        query_street_norm,
        query_number_norm,
    )

    # Кандидат
    cand_street = cand.get("street") or ""
    cand_number = cand.get("number") or ""
    cand_locality = cand.get("locality") or ""

    cand_street_norm = normalizer.normalize(cand_street)
    cand_number_norm = normalizer.normalize_house_number(cand_number)
    cand_full = normalizer.create_full_address(
        cand_locality,
        cand_street_norm,
        cand_number_norm,
    )

    # Схожесть улицы
    street_sim = fuzz.ratio(query_street_norm, cand_street_norm) / 100.0 if query_street_norm and cand_street_norm else 0.0

    # Схожесть номера
    if query_number_norm and cand_number_norm:
        if query_number_norm == cand_number_norm:
            number_sim = 1.0
        else:
            number_sim = fuzz.ratio(query_number_norm, cand_number_norm) / 100.0
    else:
        # Если номер не указан, средний сигнал
        number_sim = 0.5 if not query_number_norm else 0.0

    # Схожесть полного адреса
    full_sim = normalizer.calculate_similarity(query_full, cand_full) if query_full and cand_full else street_sim

    # Локальность
    q_loc = normalizer.normalize(query_locality)
    c_loc = normalizer.normalize(cand_locality)
    locality_match = 1.0 if ("моск" in q_loc and "моск" in c_loc) else 0.0

    return street_sim, number_sim, full_sim, locality_match


def main():
    data_path = "data/moscow_buildings.geojson"
    if not os.path.exists(data_path):
        print(f"Ошибка: файл данных не найден: {data_path}")
        return

    print("Загрузка данных...")
    buildings = gpd.read_file(data_path)
    print(f"Всего зданий: {len(buildings)}")

    # Оставляем только здания с валидной геометрией
    buildings = buildings[buildings.geometry.notna()].copy()

    normalizer = AddressNormalizer()

    print("\nИнициализация advanced-геокодера...")
    geocoder = AdvancedGeocoder(buildings)

    # Выборка для обучения
    n_samples = min(300, len(buildings))
    samples = buildings.sample(n=n_samples, random_state=123).reset_index()
    # index столбец 'index' содержит индекс в исходном GeoDataFrame

    X: List[List[float]] = []
    y: List[int] = []

    print(f"\nСбор обучающих данных на {n_samples} зданиях...")

    for _, sample_row in samples.iterrows():
        true_idx = sample_row["index"]
        true_geom = buildings.loc[true_idx].geometry
        true_centroid = true_geom.centroid
        true_point = (true_centroid.y, true_centroid.x)

        # Один-два варианта запроса
        for variant in [0, 1]:
            query = build_query(sample_row, variant=variant)
            parsed = normalizer.parse_address(query)

            # Запрос к текущему advanced-геокодеру (как генератор кандидатов)
            try:
                candidates = geocoder.geocode(query, max_results=20)
            except Exception as e:
                print(f"[!] Ошибка геокодера для '{query}': {e}")
                continue

            if not candidates:
                continue

            # Для каждого кандидата считаем фичи и метку
            for cand in candidates:
                # Фичи
                street_sim, number_sim, full_sim, locality_match = compute_features_for_candidate(
                    normalizer, query, parsed, cand
                )

                # Метка: 1, если кандидат достаточно близко к истинному дому
                cand_point = (cand["lat"], cand["lon"])
                dist_m = geodesic(true_point, cand_point).meters
                label = 1 if dist_m <= 30.0 else 0  # 30 метров как порог одного дома

                X.append([street_sim, number_sim, full_sim, locality_match])
                y.append(label)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    print(f"\nСобрано обучающих примеров: {len(y)}")
    if len(y) == 0:
        print("Недостаточно данных для обучения.")
        return

    # Немного статистики по классам
    pos = int(y.sum())
    neg = int(len(y) - pos)
    print(f"Положительных примеров (y=1): {pos}")
    print(f"Отрицательных примеров (y=0): {neg}")

    # Обучаем логистическую регрессию
    print("\nОбучение LogisticRegression на фичах [street_sim, number_sim, full_sim, locality_match]...")
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )
    model.fit(X, y)

    print("\nКоэффициенты модели:")
    print(f"  intercept: {model.intercept_[0]:.4f}")
    for name, coef in zip(
        ["street_sim", "number_sim", "full_sim", "locality_match"],
        model.coef_[0],
    ):
        print(f"  {name}: {coef:.4f}")

    # Оценим, насколько модель умеет находить правильный дом в top-1
    print("\nОценка top-1 accuracy модели на новой выборке (100 запросов)...")

    test_samples = buildings.sample(n=min(100, len(buildings)), random_state=321).reset_index()
    correct_top1 = 0
    total = 0
    distances = []

    for _, sample_row in test_samples.iterrows():
        true_idx = sample_row["index"]
        true_geom = buildings.loc[true_idx].geometry
        true_centroid = true_geom.centroid
        true_point = (true_centroid.y, true_centroid.x)

        query = build_query(sample_row, variant=0)
        parsed = normalizer.parse_address(query)

        try:
            candidates = geocoder.geocode(query, max_results=20)
        except Exception:
            continue

        if not candidates:
            continue

        feats = []
        for cand in candidates:
            feats.append(
                compute_features_for_candidate(normalizer, query, parsed, cand)
            )

        feats_arr = np.array(feats, dtype=float)
        probs = model.predict_proba(feats_arr)[:, 1]
        best_idx = int(np.argmax(probs))
        best_cand = candidates[best_idx]

        # Проверяем, насколько top-1 близок к истинному дому
        cand_point = (best_cand["lat"], best_cand["lon"])
        dist_m = geodesic(true_point, cand_point).meters
        distances.append(dist_m)

        if dist_m <= 50.0:
            correct_top1 += 1

        total += 1

    if total > 0:
        acc_top1 = correct_top1 / total
        print(f"  Количество запросов в тесте: {total}")
        print(f"  Accuracy@1 по координатам (≤50 м): {acc_top1:.3f}")
        print(f"  Средняя дистанция, м: {float(np.mean(distances)):.2f}")
        print(f"  Медианная дистанция, м: {float(np.median(distances)):.2f}")
    else:
        print("Недостаточно данных для оценки модели.")


if __name__ == "__main__":
    main()

"""
Обучение простой логистической модели скоринга для AdvancedGeocoder.

Идея:
- Используем только те фичи, которые доступны при реальном запросе:
  street_sim, number_sim, full_sim, locality_match.
- Для набора запросов (по "истинным" домам) собираем кандидатов,
  считаем фичи и маркируем правильный дом как y=1, остальные как y=0.
- Обучаем LogisticRegression и выводим веса, которые потом можно
  зашить в AdvancedGeocoder._calculate_score.

Этот скрипт нужен только для оффлайн-обучения, для работы геокодера
он не обязателен.
"""

import os
import sys
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.address_normalizer import AddressNormalizer
from src.geocoder_advanced import AdvancedGeocoder
from src.evaluate_geocoder import build_query


def compute_number_similarity(normalizer: AddressNormalizer, query_number: str, data_number: str) -> float:
    """
    Вычисление похожести номера дома для фичи number_sim (0-1).
    Использует ту же логику, что и в улучшенном алгоритме:
    - совпадает основной номер + корпус/строение -> 1.0
    - совпадает только основной номер -> 0.8
    - иначе 0.0 (если номера есть), 0.5 если номер в запросе отсутствует.
    """
    number_query_raw = (query_number or "").strip()
    number_query_norm = normalizer.normalize_house_number(number_query_raw) if number_query_raw else ""
    number_data_norm = (data_number or "").strip()

    if number_query_norm and number_data_norm:
        q_info = normalizer.parse_house_number(number_query_norm)
        d_info = normalizer.parse_house_number(number_data_norm)

        if q_info["main"] and d_info["main"]:
            if q_info["main"] == d_info["main"]:
                if (
                    q_info["extra_type"] == d_info["extra_type"]
                    and q_info["extra_number"] == d_info["extra_number"]
                    and q_info["extra_type"] is not None
                ):
                    return 1.0
                else:
                    return 0.8
            else:
                return 0.0
        else:
            # Если один из номеров не распознан, оцениваем грубо
            return 0.0
    else:
        # Если номер не указан в запросе, даем средний вес
        return 0.5 if not number_query_norm else 0.0


def compute_locality_match(query_locality: str, cand_locality: str) -> float:
    """
    Простая фича locality_match:
    - 1.0 если оба явно указывают на "Москва"
    - 0.5 если в одном из мест нет информации
    - 0.0 если явно разные.
    """
    q = (query_locality or "").lower()
    c = (cand_locality or "").lower()

    q_is_moscow = "москва" in q
    c_is_moscow = "москва" in c

    if q_is_moscow and c_is_moscow:
        return 1.0
    if not q and not c:
        return 0.5
    if (not q and c_is_moscow) or (q_is_moscow and not c):
        return 0.5
    return 0.0


def main():
    from sklearn.linear_model import LogisticRegression

    data_path = "data/moscow_buildings.geojson"

    if not os.path.exists(data_path):
        print(f"Ошибка: файл данных не найден: {data_path}")
        return

    print("Загрузка данных...")
    buildings = gpd.read_file(data_path)
    print(f"Всего зданий: {len(buildings)}")

    # Фильтруем только здания с непустыми адресами
    buildings = buildings[
        (buildings["street"].notna())
        & (buildings["street"] != "")
        & (buildings["number"].notna())
        & (buildings["number"] != "")
    ].copy()
    print(f"Зданий с адресами: {len(buildings)}")

    if len(buildings) == 0:
        print("Нет данных для обучения.")
        return

    # Инициализируем AdvancedGeocoder (он создаст street_normalized, number_normalized, full_address)
    print("\nИнициализация AdvancedGeocoder...")
    adv = AdvancedGeocoder(buildings)
    normalizer = adv.normalizer

    # Параметры обучения
    n_samples = min(500, len(buildings))
    max_candidates = 20
    street_threshold = 60

    print(f"\nБерем {n_samples} случайных зданий для обучения.")
    sample_indices = np.random.choice(len(buildings), size=n_samples, replace=False)

    X: List[List[float]] = []
    y: List[int] = []

    total_queries = 0
    used_queries = 0

    for idx_true in sample_indices:
        row_true = buildings.iloc[idx_true]

        # Несколько вариантов формулировки запроса
        for variant in (0, 1, 2):
            query = build_query(
                {
                    "locality": row_true.get("locality", "Москва"),
                    "street": row_true.get("street", ""),
                    "number": row_true.get("number", ""),
                },
                variant=variant,
            )

            total_queries += 1

            parsed = normalizer.parse_address(query)
            query_street = parsed.get("street", "") or ""
            query_number = parsed.get("number", "") or ""
            query_locality = parsed.get("locality") or "Москва"

            query_street_norm = normalizer.normalize(query_street)
            query_number_norm = normalizer.normalize_house_number(query_number)
            query_full = normalizer.create_full_address(
                query_locality, query_street_norm, query_number_norm
            )

            # Нечеткий поиск улиц
            street_matches: List[Tuple[int, float]] = adv._fuzzy_match_street(
                query_street, threshold=street_threshold
            )
            if not street_matches:
                continue

            # Проверяем, что истинный дом вообще попал в кандидаты
            candidate_indices = [i for i, _ in street_matches]
            if idx_true not in candidate_indices:
                continue

            used_queries += 1

            for idx_cand, street_sim in street_matches[:max_candidates]:
                row_cand = buildings.loc[idx_cand]

                # Фичи
                street_sim_val = street_sim  # уже в [0, 1]

                number_sim_val = compute_number_similarity(
                    normalizer,
                    query_number_norm,
                    str(row_cand.get("number_normalized", "") or ""),
                )

                full_sim_val = normalizer.calculate_similarity(
                    query_full, str(row_cand.get("full_address", "") or "")
                )

                locality_match_val = compute_locality_match(
                    query_locality,
                    str(row_cand.get("locality", "") or ""),
                )

                X.append(
                    [
                        street_sim_val,
                        number_sim_val,
                        full_sim_val,
                        locality_match_val,
                    ]
                )
                y.append(1 if idx_cand == idx_true else 0)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    print(f"\nСобрано кандидатов: {len(X)}")
    print(f"Из них положительных (y=1): {int(y.sum())}")
    print(f"Использовано запросов (где GT в топе): {used_queries} из {total_queries}")

    if len(X) == 0 or y.sum() == 0:
        print("Недостаточно данных для обучения (нет положительных примеров).")
        return

    print("\nОбучение LogisticRegression...")
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )
    model.fit(X, y)

    print("\nОбучение завершено.")
    print("Коэффициенты (street_sim, number_sim, full_sim, locality_match):")
    print(model.coef_[0])
    print("Смещение (bias):")
    print(model.intercept_[0])

    # Также можно оценить качество на обучении (чисто для информации)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = (preds == y).mean()
    print(f"\nAccuracy на обучающей выборке (порог 0.5): {acc:.4f}")


if __name__ == "__main__":
    main()


