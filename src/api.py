"""
REST API сервис для геокодирования адресов.
"""
import os
import pickle
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import geopandas as gpd

from .geocoder_base import BaseGeocoder
from .geocoder_advanced import AdvancedGeocoder


app = FastAPI(
    title="Геокодирование адресов Москвы",
    description="API для геокодирования адресов по данным OpenStreetMap",
    version="1.0.0"
)


# Глобальные переменные для геокодеров
base_geocoder: Optional[BaseGeocoder] = None
advanced_geocoder: Optional[AdvancedGeocoder] = None


class GeocodeRequest(BaseModel):
    """Модель запроса геокодирования."""
    address: str = Field(..., description="Адресная строка для геокодирования")
    algorithm: str = Field(default="advanced", description="Алгоритм: 'base' или 'advanced'")
    max_results: int = Field(default=10, ge=1, le=50, description="Максимальное количество результатов")


class GeocodeObject(BaseModel):
    """Модель объекта результата геокодирования."""
    locality: str
    street: str
    number: str
    lon: float
    lat: float
    score: Optional[float] = None


class GeocodeResponse(BaseModel):
    """Модель ответа геокодирования."""
    searched_address: str
    objects: List[GeocodeObject]


def load_geocoders():
    """Загрузка геокодеров при старте приложения."""
    global base_geocoder, advanced_geocoder
    
    # Путь к обработанным данным
    data_path = os.getenv("BUILDINGS_DATA_PATH", "data/moscow_buildings.geojson")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Файл с данными не найден: {data_path}. "
            "Сначала запустите обработку OSM данных."
        )
    
    # Загружаем здания
    buildings = gpd.read_file(data_path)
    
    # Инициализируем геокодеры
    base_geocoder = BaseGeocoder(buildings)
    advanced_geocoder = AdvancedGeocoder(buildings)
    
    print(f"Геокодеры загружены. Всего зданий: {len(buildings)}")


@app.on_event("startup")
async def startup_event():
    """Инициализация при старте приложения."""
    try:
        load_geocoders()
    except FileNotFoundError as e:
        print(f"Предупреждение: {e}")
        print("API будет работать после загрузки данных")


@app.get("/")
async def root():
    """Корневой эндпоинт."""
    return {
        "message": "Геокодирование адресов Москвы",
        "version": "1.0.0",
        "endpoints": {
            "geocode": "/geocode (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health():
    """Проверка здоровья сервиса."""
    geocoders_loaded = base_geocoder is not None and advanced_geocoder is not None
    return {
        "status": "healthy" if geocoders_loaded else "not_ready",
        "geocoders_loaded": geocoders_loaded
    }


@app.post("/geocode", response_model=GeocodeResponse)
async def geocode(request: GeocodeRequest):
    """
    Геокодирование адреса.
    
    Args:
        request: Запрос с адресом и параметрами
        
    Returns:
        Результаты геокодирования
    """
    if base_geocoder is None or advanced_geocoder is None:
        raise HTTPException(
            status_code=503,
            detail="Геокодеры не загружены. Убедитесь, что данные обработаны."
        )
    
    # Выбираем алгоритм
    if request.algorithm == "base":
        geocoder = base_geocoder
    elif request.algorithm == "advanced":
        geocoder = advanced_geocoder
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестный алгоритм: {request.algorithm}. Используйте 'base' или 'advanced'"
        )
    
    # Выполняем геокодирование
    try:
        results = geocoder.geocode(request.address, max_results=request.max_results)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при геокодировании: {str(e)}"
        )
    
    # Формируем ответ
    objects = [
        GeocodeObject(
            locality=obj.get('locality', ''),
            street=obj.get('street', ''),
            number=obj.get('number', ''),
            lon=obj.get('lon', 0.0),
            lat=obj.get('lat', 0.0),
            score=obj.get('score')
        )
        for obj in results
    ]
    
    return GeocodeResponse(
        searched_address=request.address,
        objects=objects
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

