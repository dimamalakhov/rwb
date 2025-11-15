"""
Оптимизированная обработка большого GeoJSON файла по частям.
"""
import json
import os
from typing import Dict, List
import geopandas as gpd
from shapely.geometry import shape, Point
from shapely.prepared import prep


def process_geojson_chunked(input_file: str, output_file: str, 
                           chunk_size: int = 10000):
    """
    Обработка большого GeoJSON файла по частям для экономии памяти.
    
    Args:
        input_file: Путь к входному GeoJSON файлу
        output_file: Путь к выходному GeoJSON файлу
        chunk_size: Размер чанка для обработки
    """
    print("=" * 60)
    print("ОБРАБОТКА БОЛЬШОГО GEOJSON ФАЙЛА")
    print("=" * 60)
    
    moscow_bounds = {
        'min_lat': 55.5, 'max_lat': 55.9,
        'min_lon': 37.3, 'max_lon': 37.9
    }
    
    # Создаем bounding box для быстрой фильтрации
    moscow_bbox = shape({
        'type': 'Polygon',
        'coordinates': [[
            [moscow_bounds['min_lon'], moscow_bounds['min_lat']],
            [moscow_bounds['max_lon'], moscow_bounds['min_lat']],
            [moscow_bounds['max_lon'], moscow_bounds['max_lat']],
            [moscow_bounds['min_lon'], moscow_bounds['max_lat']],
            [moscow_bounds['min_lon'], moscow_bounds['min_lat']]
        ]]
    })
    moscow_prep = prep(moscow_bbox)
    
    print(f"\nВходной файл: {input_file}")
    print(f"Выходной файл: {output_file}")
    print(f"Размер чанка: {chunk_size}")
    print("\nНачало обработки...")
    
    # Открываем файл для чтения
    features = []
    total_processed = 0
    total_in_moscow = 0
    total_with_address = 0
    
    try:
        print("Загрузка GeoJSON файла...")
        print("⚠️  Внимание: файл очень большой, загрузка может занять время...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            # Читаем GeoJSON
            # Для очень больших файлов может потребоваться много памяти
            data = json.load(f)
            
            if 'features' not in data:
                print("Ошибка: файл не содержит features")
                return
            
            total_features = len(data['features'])
            print(f"Всего объектов в файле: {total_features}")
            print("\nОбработка по частям...")
            
            # Обрабатываем по частям
            for i in range(0, total_features, chunk_size):
                chunk = data['features'][i:i + chunk_size]
                chunk_num = (i // chunk_size) + 1
                total_chunks = (total_features + chunk_size - 1) // chunk_size
                
                print(f"\nОбработка чанка {chunk_num}/{total_chunks} "
                      f"(объекты {i+1}-{min(i+chunk_size, total_features)})...")
                
                chunk_features = []
                
                for feature in chunk:
                    total_processed += 1
                    
                    try:
                        geom = shape(feature['geometry'])
                        
                        # Быстрая проверка по bounding box
                        if not moscow_prep.intersects(geom):
                            continue
                        
                        # Точная проверка координат через центроид
                        centroid = geom.centroid
                        lon, lat = centroid.x, centroid.y
                        
                        if not (moscow_bounds['min_lat'] <= lat <= moscow_bounds['max_lat'] and
                                moscow_bounds['min_lon'] <= lon <= moscow_bounds['max_lon']):
                            continue
                        
                        total_in_moscow += 1
                        
                        # Извлекаем адресные атрибуты
                        props = feature.get('properties', {})
                        
                        locality = props.get('addr:city') or props.get('addr:place') or 'Москва'
                        street = props.get('addr:street') or props.get('street') or ''
                        number = props.get('addr:housenumber') or props.get('housenumber') or ''
                        
                        # Проверяем, что это здание
                        building_type = props.get('building') or props.get('type')
                        if not building_type:
                            # Пропускаем объекты без типа building
                            continue
                        
                        # Проверяем наличие адреса
                        if not street or not number:
                            continue
                        
                        total_with_address += 1
                        
                        # Создаем новую feature
                        new_feature = {
                            'type': 'Feature',
                            'geometry': feature['geometry'],
                            'properties': {
                                'locality': locality,
                                'street': street,
                                'number': number
                            }
                        }
                        
                        chunk_features.append(new_feature)
                        
                    except Exception as e:
                        # Пропускаем проблемные объекты
                        continue
                
                features.extend(chunk_features)
                print(f"  Найдено зданий с адресами в чанке: {len(chunk_features)}")
                
                # Периодически сохраняем промежуточные результаты
                if len(features) > 0 and len(features) % 50000 == 0:
                    print(f"  Промежуточное сохранение ({len(features)} объектов)...")
                    save_geojson(features, output_file + '.tmp')
        
        print(f"\n{'='*60}")
        print("СТАТИСТИКА ОБРАБОТКИ")
        print(f"{'='*60}")
        print(f"Всего обработано объектов: {total_processed}")
        print(f"Объектов в границах Москвы: {total_in_moscow}")
        print(f"Зданий с адресами: {total_with_address}")
        print(f"{'='*60}\n")
        
        # Сохраняем финальный результат
        print("Сохранение результата...")
        save_geojson(features, output_file)
        
        print(f"\n✓ Обработка завершена!")
        print(f"Результат сохранен: {output_file}")
        print(f"Всего зданий с адресами: {len(features)}")
        
    except MemoryError:
        print("\n⚠️  Нехватка памяти при загрузке файла.")
        print("Файл слишком большой для загрузки целиком.")
        print("\nРекомендуется использовать потоковую обработку или")
        print("предварительно отфильтровать данные через osmium-tool.")
        raise
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_geojson(features: List[Dict], output_file: str):
    """Сохранение features в GeoJSON файл."""
    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', 
                exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    
    # Если есть временный файл, удаляем его
    if os.path.exists(output_file + '.tmp'):
        os.remove(output_file + '.tmp')


if __name__ == "__main__":
    import sys
    
    input_file = "data/moscow_buildings_raw.geojson"
    output_file = "data/moscow_buildings.geojson"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Ошибка: файл не найден: {input_file}")
        sys.exit(1)
    
    process_geojson_chunked(input_file, output_file, chunk_size=10000)

