#!/bin/bash
# Оптимизированная конвертация с предварительной фильтрацией

echo "=== ОПТИМИЗИРОВАННАЯ КОНВЕРТАЦИЯ PBF В GEOJSON ==="
echo ""

# 1. Проверка osmium-tool
if ! command -v osmium &> /dev/null; then
    echo "Установка osmium-tool..."
    brew install osmium-tool
fi

echo "✓ osmium-tool готов"
echo ""

# 2. Предварительная фильтрация по границам Москвы
echo "Шаг 1: Фильтрация зданий в границах Москвы..."
echo "Это может занять несколько минут..."

# Используем osmium для фильтрации по bounding box Москвы
# Москва: 55.5-55.9°N, 37.3-37.9°E
INPUT_FILE="buildings.osm.pbf"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Ошибка: файл $INPUT_FILE не найден"
    exit 1
fi

# Фильтруем по bounding box и зданиям с адресами
osmium extract \
    --bbox 37.3,55.5,37.9,55.9 \
    "$INPUT_FILE" \
    -o buildings_moscow.osm.pbf

if [ $? -ne 0 ]; then
    echo "Ошибка при фильтрации"
    exit 1
fi

echo "✓ Фильтрация завершена"
echo ""

# 3. Конвертация в GeoJSON
echo "Шаг 2: Конвертация в GeoJSON..."
osmium export buildings_moscow.osm.pbf -o data/moscow_buildings_filtered.geojson --format geojson

if [ $? -ne 0 ]; then
    echo "Ошибка при конвертации"
    exit 1
fi

echo "✓ Конвертация завершена"
echo ""

# 4. Обработка Python скриптом (теперь файл намного меньше)
echo "Шаг 3: Обработка и фильтрация адресов..."
python3 src/process_large_geojson.py data/moscow_buildings_filtered.geojson data/moscow_buildings.geojson

if [ $? -eq 0 ]; then
    echo ""
    echo "=== КОНВЕРТАЦИЯ ЗАВЕРШЕНА УСПЕШНО! ==="
    echo "Результат: data/moscow_buildings.geojson"
    
    # Показываем статистику
    if [ -f "data/moscow_buildings.geojson" ]; then
        SIZE=$(ls -lh data/moscow_buildings.geojson | awk '{print $5}')
        echo "Размер файла: $SIZE"
    fi
else
    echo ""
    echo "⚠️  Ошибка при обработке. Попробуйте запустить скрипт вручную:"
    echo "python3 src/process_large_geojson.py data/moscow_buildings_filtered.geojson data/moscow_buildings.geojson"
fi


