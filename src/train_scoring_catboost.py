"""
Эксперимент: сравнение LogisticRegression и CatBoostClassifier для скоринга кандидатов.

Фичи:
- street_sim  (0-1)
- number_sim  (0-1)
- full_sim    (0-1)
- locality_match (0/0.5/1)

Подход:
- Берем N случайных зданий из moscow_buildings.geojson.
- Для каждого генерируем несколько вариантов запроса.
- Через AdvancedGeocoder._fuzzy_match_street собираем кандидатов.
- Строим датасет (X, y): y=1 если кандидат — правильный дом (тот же индекс), иначе 0.
- Делим на train/test, обучаем логрег и CatBoost.
- Для каждого запроса в тесте ранжируем кандидатов по вероятности и считаем
  Accuracy@1 по координатам (≤50 м) и расстояния.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import geopandas as gpd
from geopy.distance import geodesic

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.address_normalizer import AddressNormalizer
from src.geocoder_advanced import AdvancedGeocoder
from src.evaluate_geocoder import build_query


def compute_number_similarity(normalizer: AddressNormalizer, query_number: str, data_number: str) -> float:
    """
    Вычисление похожести номера дома (0-1) на основе normalize_house_number/parse_house_number.
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
            return 0.0
    else:
        return 0.5 if not number_query_norm else 0.0


def compute_locality_match(query_locality: str, cand_locality: str) -> float:
    """
    Фича locality_match:
    - 1.0 если оба явно "Москва"
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

    print("\nИнициализация AdvancedGeocoder...")
    adv = AdvancedGeocoder(buildings)
    normalizer = adv.normalizer

    # Параметры
    n_samples = min(500, len(buildings))
    max_candidates = 20
    street_threshold = 60

    print(f"\nБерем {n_samples} случайных зданий для обучения.")
    sample_indices = np.random.choice(len(buildings), size=n_samples, replace=False)

    X: List[List[float]] = []
    y: List[int] = []
    queries_info: List[Tuple[int, str]] = []  # для последующей оценки (true_idx, query)

    total_queries = 0
    used_queries = 0

    for idx_true in sample_indices:
        row_true = buildings.iloc[idx_true]

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

            street_matches = adv._fuzzy_match_street(query_street, threshold=street_threshold)
            if not street_matches:
                continue

            candidate_indices = [i for i, _ in street_matches]
            if idx_true not in candidate_indices:
                continue

            used_queries += 1
            queries_info.append((idx_true, query))

            for idx_cand, street_sim in street_matches[:max_candidates]:
                row_cand = buildings.loc[idx_cand]

                street_sim_val = street_sim  # уже 0-1
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
                    [street_sim_val, number_sim_val, full_sim_val, locality_match_val]
                )
                y.append(1 if idx_cand == idx_true else 0)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    print(f"\nСобрано кандидатов: {len(X)}")
    print(f"Из них положительных (y=1): {int(y.sum())}")
    print(f"Использовано запросов (где GT в топе): {used_queries} из {total_queries}")

    if len(X) == 0 or y.sum() == 0:
        print("Недостаточно данных для обучения.")
        return

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- Логистическая регрессия ---
    print("\nОбучение LogisticRegression...")
    logreg = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )
    logreg.fit(X_train, y_train)

    # --- CatBoost ---
    print("\nОбучение CatBoostClassifier...")
    cat = CatBoostClassifier(
        depth=4,
        learning_rate=0.1,
        iterations=300,
        loss_function="Logloss",
        verbose=False,
    )
    cat.fit(X_train, y_train)

    # Для оценки top-1 по координатам нужно работать на уровне запросов, а не отдельных кандидатов.
    # Поэтому повторно пройдемся по части buildings и сравним ранжирование logreg vs catboost.
    print("\nОценка top-1 accuracy по координатам (≤50 м) для LogReg и CatBoost...")

    test_buildings = buildings.sample(n=min(150, len(buildings)), random_state=777)
    normalizer = adv.normalizer

    def eval_model(model, name: str):
        correct_top1 = 0
        total_q = 0
        dists: List[float] = []

        for _, row_true in test_buildings.iterrows():
            true_geom = row_true.geometry
            if true_geom is None:
                continue
            true_centroid = true_geom.centroid
            true_point = (true_centroid.y, true_centroid.x)

            query = build_query(
                {
                    "locality": row_true.get("locality", "Москва"),
                    "street": row_true.get("street", ""),
                    "number": row_true.get("number", ""),
                },
                variant=0,
            )

            parsed = normalizer.parse_address(query)
            q_street = parsed.get("street", "") or ""
            q_number = parsed.get("number", "") or ""
            q_loc = parsed.get("locality") or "Москва"

            q_street_norm = normalizer.normalize(q_street)
            q_number_norm = normalizer.normalize_house_number(q_number)
            q_full = normalizer.create_full_address(q_loc, q_street_norm, q_number_norm)

            street_matches = adv._fuzzy_match_street(q_street, threshold=street_threshold)
            if not street_matches:
                continue

            feats = []
            cand_rows = []
            for idx_cand, street_sim in street_matches[:max_candidates]:
                row_cand = buildings.loc[idx_cand]
                street_sim_val = street_sim
                number_sim_val = compute_number_similarity(
                    normalizer,
                    q_number_norm,
                    str(row_cand.get("number_normalized", "") or ""),
                )
                full_sim_val = normalizer.calculate_similarity(
                    q_full, str(row_cand.get("full_address", "") or "")
                )
                locality_match_val = compute_locality_match(
                    q_loc,
                    str(row_cand.get("locality", "") or ""),
                )

                feats.append(
                    [street_sim_val, number_sim_val, full_sim_val, locality_match_val]
                )
                cand_rows.append(row_cand)

            if not feats:
                continue

            feats_arr = np.array(feats, dtype=float)
            probs = model.predict_proba(feats_arr)[:, 1]
            best_idx = int(np.argmax(probs))
            best_row = cand_rows[best_idx]

            best_centroid = best_row.geometry.centroid
            cand_point = (best_centroid.y, best_centroid.x)
            dist_m = geodesic(true_point, cand_point).meters

            dists.append(dist_m)
            if dist_m <= 50.0:
                correct_top1 += 1
            total_q += 1

        if total_q == 0:
            print(f"{name}: недостаточно запросов для оценки.")
            return

        acc = correct_top1 / total_q
        print(f"\n{name}:")
        print(f"  Кол-во запросов: {total_q}")
        print(f"  Accuracy@1 (≤50 м): {acc:.3f}")
        print(f"  Средняя дистанция, м: {float(np.mean(dists)):.2f}")
        print(f"  Медианная дистанция, м: {float(np.median(dists)):.2f}")

    eval_model(logreg, "LogisticRegression")
    eval_model(cat, "CatBoostClassifier")


if __name__ == "__main__":
    main()


