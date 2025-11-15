"""
Скрипт для оценки качества геокодеров на 100 адресах.

Подход:
- Берём случайные 100 зданий из data/moscow_buildings.geojson.
- Для каждого формируем запрос (несколько вариантов формата адреса).
- Прогоняем через базовый и улучшенный геокодеры.
- Считаем метрики:
  - текстовая (по адресу),
  - по координатам,
  - комбинированная,
  - accuracy@1 по тексту и по координатам.
"""
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geocoder_base import BaseGeocoder
from src.geocoder_advanced import AdvancedGeocoder
from src.metrics import GeocodingMetrics


def build_query(row: Dict, variant: int = 0) -> str:
    """
    Формирование текстового запроса по строке зданий.

    variant:
        0 – полный адрес с городом: "Москва, <улица>, д. <номер>"
        1 – без города: "<улица>, <номер>"
        2 – разговорный формат: "г Москва, <улица> <номер>"
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


def evaluate_algorithm(
    algo_name: str,
    geocoder,
    samples: gpd.GeoDataFrame,
    max_results: int = 5,
    text_threshold: float = 0.95,
    dist_threshold_m: float = 50.0,
) -> Dict:
    """
    Оценка одного алгоритма на выборке.
    """
    text_scores: List[float] = []
    coord_scores: List[float] = []
    combined_scores: List[float] = []
    distances: List[float] = []

    acc_text_top1 = 0
    acc_coord_top1 = 0

    results_per_query: List[Dict] = []

    print("\n" + "=" * 60)
    print(f"ОЦЕНКА АЛГОРИТМА: {algo_name}")
    print("=" * 60)

    for idx, row in samples.iterrows():
        true = {
            "locality": str(row.get("locality", "")),
            "street": str(row.get("street", "")),
            "number": str(row.get("number", "")),
            "lat": float(row.geometry.centroid.y),
            "lon": float(row.geometry.centroid.x),
        }

        # Вариант запроса (слегка рандомизируем формат)
        variant = random.choice([0, 1, 2])
        query = build_query(true, variant=variant)

        try:
            preds = geocoder.geocode(query, max_results=max_results)
        except Exception as e:
            print(f"\n[!] Ошибка геокодера для запроса '{query}': {e}")
            preds = []

        if preds:
            top = preds[0]
            pred = {
                "locality": top.get("locality", ""),
                "street": top.get("street", ""),
                "number": top.get("number", ""),
                "lat": top.get("lat", 0.0),
                "lon": top.get("lon", 0.0),
            }

            t_score = GeocodingMetrics.address_text_score(pred, true)
            c_score = GeocodingMetrics.coordinate_score(
                pred["lat"], pred["lon"], true["lat"], true["lon"]
            )
            dist = GeocodingMetrics.coordinate_distance(
                pred["lat"], pred["lon"], true["lat"], true["lon"]
            )
            comb = GeocodingMetrics.combined_score(pred, true)
        else:
            t_score = 0.0
            c_score = 0.0
            comb = 0.0
            # грубо считаем, что дистанция очень большая
            dist = 1e6

        text_scores.append(t_score)
        coord_scores.append(c_score)
        combined_scores.append(comb)
        distances.append(dist)

        if t_score >= text_threshold:
            acc_text_top1 += 1
        if dist <= dist_threshold_m:
            acc_coord_top1 += 1

        results_per_query.append(
            {
                "query": query,
                "true": true,
                "top_pred": preds[0] if preds else None,
                "text_score": t_score,
                "coord_score": c_score,
                "combined_score": comb,
                "distance_m": dist,
            }
        )

    n = len(samples)
    stats = {
        "algo": algo_name,
        "n": n,
        "text_score_mean": float(np.mean(text_scores)) if n > 0 else 0.0,
        "text_score_median": float(np.median(text_scores)) if n > 0 else 0.0,
        "coord_score_mean": float(np.mean(coord_scores)) if n > 0 else 0.0,
        "coord_score_median": float(np.median(coord_scores)) if n > 0 else 0.0,
        "combined_score_mean": float(np.mean(combined_scores)) if n > 0 else 0.0,
        "combined_score_median": float(np.median(combined_scores)) if n > 0 else 0.0,
        "distance_mean_m": float(np.mean(distances)) if n > 0 else 0.0,
        "distance_median_m": float(np.median(distances)) if n > 0 else 0.0,
        "text_top1_accuracy": acc_text_top1 / n if n > 0 else 0.0,
        "coord_top1_accuracy": acc_coord_top1 / n if n > 0 else 0.0,
    }

    print("\nРЕЗЮМЕ ДЛЯ АЛГОРИТМА:", algo_name)
    print("-" * 60)
    print(f"Количество запросов:        {n}")
    print(f"Текстовая метрика (mean):   {stats['text_score_mean']:.4f}")
    print(f"Текстовая метрика (median): {stats['text_score_median']:.4f}")
    print(f"Координатная метрика (mean):   {stats['coord_score_mean']:.4f}")
    print(f"Координатная метрика (median): {stats['coord_score_median']:.4f}")
    print(f"Комб. метрика (mean):       {stats['combined_score_mean']:.4f}")
    print(f"Комб. метрика (median):     {stats['combined_score_median']:.4f}")
    print(f"Средняя дистанция, м:       {stats['distance_mean_m']:.2f}")
    print(f"Медианная дистанция, м:     {stats['distance_median_m']:.2f}")
    print(f"Accuracy@1 по тексту (≥{text_threshold}):  {stats['text_top1_accuracy']:.3f}")
    print(f"Accuracy@1 по координатам (≤{dist_threshold_m} м): {stats['coord_top1_accuracy']:.3f}")

    return {"stats": stats, "details": results_per_query}


def main():
    data_path = "data/moscow_buildings.geojson"

    if not os.path.exists(data_path):
        print(f"Ошибка: файл данных не найден: {data_path}")
        print("Сначала подготовьте данные (см. README).")
        return

    print("Загрузка данных...")
    buildings = gpd.read_file(data_path)
    print(f"Всего зданий в датасете: {len(buildings)}")

    # Фильтрация на всякий случай
    buildings = buildings[
        (buildings["street"].notna())
        & (buildings["street"] != "")
        & (buildings["number"].notna())
        & (buildings["number"] != "")
    ].copy()

    print(f"Зданий с непустыми адресами: {len(buildings)}")

    n_samples = min(100, len(buildings))
    samples = buildings.sample(n=n_samples, random_state=42).reset_index(drop=True)
    print(f"\nВыбрано {n_samples} зданий для оценки.")

    # Инициализируем геокодеры на всём датасете
    print("\nИнициализация геокодеров...")
    base_geocoder = BaseGeocoder(buildings)
    advanced_geocoder = AdvancedGeocoder(buildings)

    # Оценка базового
    base_result = evaluate_algorithm("base", base_geocoder, samples)

    # Оценка улучшенного
    adv_result = evaluate_algorithm("advanced", advanced_geocoder, samples)

    print("\n" + "=" * 60)
    print("СВОДКА ПО ОБОИМ АЛГОРИТМАМ")
    print("=" * 60)

    for res in [base_result["stats"], adv_result["stats"]]:
        print(f"\nАлгоритм: {res['algo']}")
        print(f"  text_mean:      {res['text_score_mean']:.4f}")
        print(f"  coord_mean:     {res['coord_score_mean']:.4f}")
        print(f"  combined_mean:  {res['combined_score_mean']:.4f}")
        print(f"  dist_mean_m:    {res['distance_mean_m']:.2f}")
        print(f"  acc@1_text:     {res['text_top1_accuracy']:.3f}")
        print(f"  acc@1_coord:    {res['coord_top1_accuracy']:.3f}")


if __name__ == "__main__":
    main()


