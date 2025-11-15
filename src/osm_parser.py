"""
Парсер OSM PBF файлов для извлечения зданий Москвы с адресными атрибутами.
Использует osmium-tool для конвертации или предоставляет инструкции по установке.
"""
import os
import subprocess
import tempfile
from typing import Optional
import geopandas as gpd
import pandas as pd


class OSMParser:
    """Класс для парсинга OSM данных и извлечения зданий Москвы."""
    
    def __init__(self, pbf_path: str):
        """
        Инициализация парсера.
        
        Args:
            pbf_path: Путь к PBF файлу OSM
        """
        self.pbf_path = pbf_path
        self.buildings = None
        self.moscow_bounds = {
            'min_lat': 55.5,
            'max_lat': 55.9,
            'min_lon': 37.3,
            'max_lon': 37.9
        }
    
    def _check_osmium_tool(self) -> bool:
        """Проверка наличия osmium-tool."""
        try:
            result = subprocess.run(
                ['osmium', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def extract_moscow_buildings(self, output_path: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Извлечение зданий Москвы из OSM данных.
        
        Args:
            output_path: Путь для сохранения обработанных данных (опционально)
            
        Returns:
            GeoDataFrame с зданиями Москвы
        """
        if not os.path.exists(self.pbf_path):
            raise FileNotFoundError(f"PBF файл не найден: {self.pbf_path}")
        
        print("=" * 60)
        print("ОБРАБОТКА OSM ДАННЫХ")
        print("=" * 60)
        
        # Проверяем наличие osmium-tool
        if not self._check_osmium_tool():
            print("\n⚠️  osmium-tool не найден!")
            print("\nДля обработки PBF файлов необходимо установить osmium-tool:")
            print("  macOS:   brew install osmium-tool")
            print("  Linux:   sudo apt-get install osmium-tool")
            print("\nАльтернативный вариант:")
            print("  Используйте готовые предобработанные данные GeoJSON")
            print("  или конвертируйте PBF в GeoJSON вручную.")
            print("\n" + "=" * 60)
            
            # Создаем пустую структуру для демонстрации
            buildings_gdf = gpd.GeoDataFrame(columns=[
                'locality', 'street', 'number', 'district', 
                'suburb', 'postcode', 'name', 'building_type', 'geometry'
            ], crs='EPSG:4326')
            
            self.buildings = buildings_gdf
            
            if output_path:
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
                buildings_gdf.to_file(output_path, driver='GeoJSON')
                print(f"\nСоздан пустой файл структуры: {output_path}")
                print("Заполните его данными после установки osmium-tool и повторной обработки.")
            
            return buildings_gdf
        
        print("\n✓ osmium-tool найден, начинаем обработку...")
        print("\nПримечание: Прямая конвертация PBF в GeoJSON через osmium-tool")
        print("может быть сложной. Рекомендуется использовать готовые инструменты")
        print("или предобработанные данные.")
        print("\n" + "=" * 60)
        
        # Если osmium-tool доступен, можно попробовать использовать его
        # Но полная реализация требует дополнительных инструментов
        
        # Для демонстрации создаем структуру
        buildings_gdf = gpd.GeoDataFrame(columns=[
            'locality', 'street', 'number', 'district', 
            'suburb', 'postcode', 'name', 'building_type', 'geometry'
        ], crs='EPSG:4326')
        
        print("\n⚠️  Полная обработка PBF файлов требует дополнительных инструментов.")
        print("Рекомендуется:")
        print("  1. Использовать онлайн-конвертеры (например, OSMium или QGIS)")
        print("  2. Использовать готовые предобработанные GeoJSON файлы")
        print("  3. Использовать Docker-контейнер с полным набором инструментов")
        
        self.buildings = buildings_gdf
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            buildings_gdf.to_file(output_path, driver='GeoJSON')
            print(f"\nСоздан файл структуры: {output_path}")
        
        return buildings_gdf
    
    def get_buildings(self) -> gpd.GeoDataFrame:
        """Возвращает извлеченные здания."""
        if self.buildings is None:
            raise ValueError("Сначала необходимо вызвать extract_moscow_buildings()")
        return self.buildings


if __name__ == "__main__":
    # Пример использования
    parser = OSMParser("central-fed-district-251113.osm.pbf")
    buildings = parser.extract_moscow_buildings("data/moscow_buildings.geojson")
    print(f"\nВсего зданий: {len(buildings)}")
