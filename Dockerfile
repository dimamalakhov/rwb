FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей для работы с геоданными
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание директории для данных
RUN mkdir -p data

# Порт для API
EXPOSE 8000

# Команда запуска
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

