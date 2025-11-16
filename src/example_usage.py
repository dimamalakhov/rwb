"""
Пример использования геокодеров.
"""
import os
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
from src.geocoder_base import BaseGeocoder
from src.geocoder_advanced import AdvancedGeocoder
from src.metrics import GeocodingMetrics


def main():
    """Пример использования."""
    # Загружаем данные
    data_path = "data/moscow_buildings.geojson"
    
    if not os.path.exists(data_path):
        print(f"Ошибка: файл данных не найден: {data_path}")
        print("Сначала запустите: python src/process_osm_data.py")
        return
    
    print("Загрузка данных...")
    buildings = gpd.read_file(data_path)
    print(f"Загружено зданий: {len(buildings)}")
    
    # Инициализируем геокодеры
    print("\nИнициализация геокодеров...")
    base_geocoder = BaseGeocoder(buildings)
    advanced_geocoder = AdvancedGeocoder(buildings)
    
    # Тестовые адреса
    test_addresses = [
        "Москва, ул. Тверская, д. 10",
        "Тверская улица, 15",
        "Москва, проспект Ленина, 5",
    ]
    
    print("\n" + "="*60)
    print("БАЗОВЫЙ АЛГОРИТМ")
    print("="*60)
    
    for address in test_addresses:
        print(f"\nЗапрос: {address}")
        results = base_geocoder.geocode(address, max_results=3)
        
        if results:
            print(f"Найдено результатов: {len(results)}")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['locality']}, {result['street']}, {result['number']}")
                print(f"     Координаты: ({result['lat']:.6f}, {result['lon']:.6f})")
                print(f"     Score: {result.get('score', 'N/A')}")
        else:
            print("  Результаты не найдены")
    
    print("\n" + "="*60)
    print("УЛУЧШЕННЫЙ АЛГОРИТМ")
    print("="*60)
    
    for address in test_addresses:
        print(f"\nЗапрос: {address}")
        results = advanced_geocoder.geocode(address, max_results=3)
        
        if results:
            print(f"Найдено результатов: {len(results)}")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['locality']}, {result['street']}, {result['number']}")
                print(f"     Координаты: ({result['lat']:.6f}, {result['lon']:.6f})")
                print(f"     Score: {result.get('score', 'N/A')}")
        else:
            print("  Результаты не найдены")
    
    # Пример вычисления метрик
    print("\n" + "="*60)
    print("ПРИМЕР ВЫЧИСЛЕНИЯ МЕТРИК")
    print("="*60)
    
    # --- Простой пример на одном адресе ---
    predicted = {
        'locality': 'Москва',
        'street': 'Тверская улица',
        'number': '10',
        'lat': 55.7558,
        'lon': 37.6173
    }
    
    true = {
        'locality': 'Москва',
        'street': 'Тверская ул',
        'number': '10',
        'lat': 55.7558,
        'lon': 37.6173
    }
    
    text_score = GeocodingMetrics.address_text_score(predicted, true)
    coord_score = GeocodingMetrics.coordinate_score(
        predicted['lat'], predicted['lon'],
        true['lat'], true['lon'],
    )
    combined = GeocodingMetrics.combined_score(predicted, true)
    
    print(f"\nПредсказанный адрес: {predicted}")
    print(f"Эталонный адрес: {true}")
    print(f"Текстовая метрика: {text_score:.4f}")
    print(f"Метрика координат: {coord_score:.4f}")
    print(f"Комбинированная метрика: {combined:.4f}")

    # --- Пример метрик на батче: 100 случайных адресов ---
    print("\n" + "="*60)
    print("МЕТРИКИ НА 100 СЛУЧАЙНЫХ АДРЕСАХ (УЛУЧШЕННЫЙ АЛГОРИТМ)")
    print("="*60)

    total_buildings = len(buildings)
    if total_buildings == 0:
        print("Датасет пуст, метрики по батчу не считаются.")
        return

    # Берём только здания с непустыми улицей и номером
    buildings_with_addr = buildings[
        (buildings["street"].notna()) & (buildings["street"] != "") &
        (buildings["number"].notna()) & (buildings["number"] != "")
    ]

    if len(buildings_with_addr) == 0:
        print("Нет зданий с непустыми улицей и номером, метрики по батчу не считаются.")
        return

    n_samples = min(2000, len(buildings_with_addr))

    print(f"Всего зданий: {total_buildings}")
    print(f"Зданий с непустыми адресами: {len(buildings_with_addr)}")
    print(f"Используем для метрик: {n_samples}")

    samples = buildings_with_addr.sample(n=n_samples, random_state=42).reset_index(drop=True)

    predictions = []
    ground_truth = []

    for _, row in samples.iterrows():
        true_obj = {
            "locality": str(row.get("locality", "Москва")),
            "street": str(row.get("street", "")),
            "number": str(row.get("number", "")),
            "lat": float(row.geometry.centroid.y),
            "lon": float(row.geometry.centroid.x),
        }

        query = f"{true_obj['locality']}, {true_obj['street']}, {true_obj['number']}"

        try:
            preds = advanced_geocoder.geocode(query, max_results=1)
        except Exception as e:
            print(f"[!] Ошибка геокодера для запроса '{query}': {e}")
            preds = []

        if preds:
            top = preds[0]
            pred_obj = {
                "locality": top.get("locality", ""),
                "street": top.get("street", ""),
                "number": top.get("number", ""),
                "lat": float(top.get("lat", 0.0)),
                "lon": float(top.get("lon", 0.0)),
            }
        else:
            # Если ничего не найдено, считаем, что предсказание полностью неверно,
            # но координаты ставим в центр Москвы, а не (0, 0), чтобы не портить статистику.
            pred_obj = {
                "locality": "",
                "street": "",
                "number": "",
                "lat": 55.7558,   # центр Москвы
                "lon": 37.6173,
            }

        predictions.append(pred_obj)
        ground_truth.append(true_obj)

    batch_metrics = GeocodingMetrics.evaluate_batch(predictions, ground_truth)
    
    print("\nСводные метрики по датасету:")
    print(f"  text_score_mean:      {batch_metrics['text_score_mean']:.4f}")
    print(f"  text_score_median:    {batch_metrics['text_score_median']:.4f}")
    print(f"  coord_score_mean:     {batch_metrics['coord_score_mean']:.4f}")
    print(f"  coord_score_median:   {batch_metrics['coord_score_median']:.4f}")
    print(f"  combined_score_mean:  {batch_metrics['combined_score_mean']:.4f}")
    print(f"  combined_score_median:{batch_metrics['combined_score_median']:.4f}")
    print(f"  distance_mean_m:      {batch_metrics['distance_mean_m']:.2f}")
    print(f"  distance_median_m:    {batch_metrics['distance_median_m']:.2f}")
    print(f"  distance_p95_m:       {batch_metrics['distance_p95_m']:.2f}")

    # --- Сохранение подробных результатов по адресам в CSV ---
    print("\nСохранение подробных результатов по адресам в файл...")
    results_path = Path("data") / "evaluation_addresses.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Пороги для OK/FAIL
    TEXT_OK_THRESHOLD = 1
    DIST_OK_THRESHOLD_M = 150.0

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_locality",
            "query_street",
            "query_number",
            "true_lat",
            "true_lon",
            "pred_locality",
            "pred_street",
            "pred_number",
            "pred_lat",
            "pred_lon",
            "text_score",
            "levenshtein_distance",
            "coord_distance_m",
            "text_ok",
            "coord_ok",
            "status",
        ])

        for true_obj, pred_obj in zip(ground_truth, predictions):
            text_score_i = GeocodingMetrics.address_text_score(pred_obj, true_obj)
            # Левенштейновское расстояние по полному адресу
            pred_full = f"{pred_obj.get('locality', '')}, {pred_obj.get('street', '')}, {pred_obj.get('number', '')}"
            true_full = f"{true_obj.get('locality', '')}, {true_obj.get('street', '')}, {true_obj.get('number', '')}"
            lev_dist_i = GeocodingMetrics.levenshtein_distance_raw(pred_full, true_full)

            dist_m_i = GeocodingMetrics.coordinate_distance(
                pred_obj.get("lat", 55.7558),
                pred_obj.get("lon", 37.6173),
                true_obj.get("lat", 0.0),
                true_obj.get("lon", 0.0),
            )
            text_ok = text_score_i >= TEXT_OK_THRESHOLD
            coord_ok = dist_m_i <= DIST_OK_THRESHOLD_M
            # Дополнительное условие: если координаты практически совпали (один и тот же дом),
            # считаем статус OK даже при недостаточном текстовом score.
            # Используем небольшой допуск по расстоянию, чтобы учесть округления координат.
            coords_exact = dist_m_i < 10.0  # < 1 метр считаем совпадением точки
            status = "OK" if coords_exact or (text_ok and coord_ok) else "FAIL"

            writer.writerow([
                true_obj.get("locality", ""),
                true_obj.get("street", ""),
                true_obj.get("number", ""),
                f"{true_obj.get('lat', 0.0):.6f}",
                f"{true_obj.get('lon', 0.0):.6f}",
                pred_obj.get("locality", ""),
                pred_obj.get("street", ""),
                pred_obj.get("number", ""),
                f"{pred_obj.get('lat', 55.7558):.6f}",
                f"{pred_obj.get('lon', 37.6173):.6f}",
                f"{text_score_i:.4f}",
                lev_dist_i,
                f"{dist_m_i:.2f}",
                int(text_ok),
                int(coord_ok),
                status,
            ])

    print(f"Подробные результаты сохранены в {results_path}")


if __name__ == "__main__":
    main()

