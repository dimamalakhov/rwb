"""
Скрипт для обработки OSM данных и извлечения зданий Москвы.
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.osm_parser import OSMParser


def main():
    """Основная функция обработки данных."""
    # Пути
    pbf_path = "central-fed-district-251113.osm.pbf"
    output_path = "data/moscow_buildings.geojson"
    
    # Проверяем наличие PBF файла
    if not os.path.exists(pbf_path):
        print(f"Ошибка: PBF файл не найден: {pbf_path}")
        print("\nАльтернативный вариант:")
        print("Используйте готовый GeoJSON файл или конвертируйте PBF в GeoJSON.")
        print("См. файл CONVERT_OSM.md для инструкций.")
        sys.exit(1)
    
    # Проверяем, есть ли уже готовый GeoJSON файл
    if os.path.exists(output_path):
        print(f"✓ Найден готовый файл: {output_path}")
        print("Используйте его напрямую. Для повторной обработки удалите файл.")
        return
    
    # Создаем директорию для данных
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    print("Начало обработки OSM данных...")
    print(f"Входной файл: {pbf_path}")
    print(f"Выходной файл: {output_path}")
    print("\n" + "="*60)
    
    try:
        # Создаем парсер
        parser = OSMParser(pbf_path)
        
        # Извлекаем здания Москвы
        buildings = parser.extract_moscow_buildings(output_path)
        
        if len(buildings) > 0:
            print(f"\n✓ Обработка завершена успешно!")
            print(f"Извлечено зданий: {len(buildings)}")
            print(f"Данные сохранены в: {output_path}")
            
            # Показываем примеры
            print("\nПримеры извлеченных адресов:")
            for idx in range(min(5, len(buildings))):
                row = buildings.iloc[idx]
                print(f"  {idx+1}. {row.get('locality', '')}, {row.get('street', '')}, {row.get('number', '')}")
        else:
            print("\n⚠️  Не удалось извлечь здания из PBF файла.")
            print("Рекомендуется использовать готовые инструменты для конвертации.")
            print("См. файл CONVERT_OSM.md для инструкций.")
        
    except Exception as e:
        print(f"\nОшибка при обработке данных: {e}")
        print("\nРекомендации:")
        print("1. Установите osmium-tool: brew install osmium-tool (macOS)")
        print("2. Используйте QGIS для конвертации PBF в GeoJSON")
        print("3. Используйте готовые предобработанные GeoJSON файлы")
        print("\nПодробные инструкции в файле CONVERT_OSM.md")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

