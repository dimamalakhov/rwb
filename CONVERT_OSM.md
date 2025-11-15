# Инструкция по конвертации OSM PBF в GeoJSON

Из-за проблем с зависимостями Python-библиотек для парсинга PBF файлов, рекомендуется использовать готовые инструменты для конвертации.

## Вариант 1: Использование osmium-tool (рекомендуется)

### Установка

**macOS:**
```bash
brew install osmium-tool
```

**Linux:**
```bash
sudo apt-get install osmium-tool
```

### Конвертация

```bash
# Извлечение зданий с адресами
osmium tags-filter central-fed-district-251113.osm.pbf nwr/building -o buildings.osm.pbf

# Конвертация в GeoJSON (требует дополнительных инструментов)
# Или используйте QGIS для конвертации
```

## Вариант 2: Использование QGIS

1. Откройте QGIS
2. Вектор → OpenStreetMap → Загрузить данные
3. Выберите PBF файл
4. Экспортируйте в GeoJSON

## Вариант 3: Использование онлайн-конвертеров

- [OSMium](https://osmium-tool.org/)
- [Overpass Turbo](https://overpass-turbo.eu/) - для извлечения данных по запросу

## Вариант 4: Использование Docker

Создайте Docker-контейнер с полным набором инструментов OSM:

```bash
docker run -it -v $(pwd):/data osrm/osrm-backend:latest bash
# В контейнере используйте osmium-tool для обработки
```

## После конвертации

После получения GeoJSON файла, поместите его в `data/moscow_buildings.geojson` и используйте напрямую:

```python
import geopandas as gpd
buildings = gpd.read_file("data/moscow_buildings.geojson")
```

