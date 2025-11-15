"""
Нормализация адресов с учетом классификатора FIAS.
"""
import re
from typing import Dict, Optional, Tuple


class AddressNormalizer:
    """Класс для нормализации адресов."""
    
    # Словарь сокращений согласно классификатору FIAS (и приближённый)
    # https://www.alta.ru/fias/socrname/
    ABBREVIATIONS = {
        # Городские объекты
        'город': 'г',
        'поселок': 'п',
        'поселок городского типа': 'пгт',
        'село': 'с',
        'деревня': 'д',
        'станица': 'ст',
        'хутор': 'х',
        
        # Улицы
        'улица': 'ул',
        'ул.': 'ул',
        'ул': 'ул',
        'проспект': 'пр-кт',
        'пр-т': 'пр-кт',
        'пр-т.': 'пр-кт',
        'пр-кт': 'пр-кт',
        'просп.': 'пр-кт',
        'переулок': 'пер',
        'пер.': 'пер',
        'пер': 'пер',
        'проезд': 'пр-д',
        'пр-д': 'пр-д',
        'бульвар': 'б-р',
        'б-р': 'б-р',
        'площадь': 'пл',
        'пл.': 'пл',
        'пл': 'пл',
        'набережная': 'наб',
        'наб.': 'наб',
        'наб': 'наб',
        'шоссе': 'ш',
        'ш.': 'ш',
        'ш': 'ш',
        'аллея': 'ал',
        'ал.': 'ал',
        'ал': 'ал',
        'тупик': 'туп',
        'линия': 'лин',
        'квартал': 'кв-л',
        
        # Дома
        'дом': 'д',
        'д.': 'д',
        'д': 'д',
        'строение': 'стр',
        'стр.': 'стр',
        'стр': 'стр',
        'корпус': 'корп',
        'корп.': 'корп',
        'корп': 'корп',
        'литера': 'лит',
        'лит.': 'лит',
        'лит': 'лит',
    }
    
    # Обратный словарь для расшифровки (используется при expand_abbreviations)
    EXPANSIONS = {v: k for k, v in ABBREVIATIONS.items()}
    
    def __init__(self):
        """Инициализация нормализатора."""
        pass
    
    def normalize(self, address: str) -> str:
        """
        Нормализация адреса.
        
        Args:
            address: Исходный адрес
            
        Returns:
            Нормализованный адрес
        """
        if not address:
            return ""
        
        # Приводим к нижнему регистру
        normalized = address.lower().strip()
        
        # Удаляем лишние пробелы
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Нормализуем запятые
        normalized = normalized.replace(',', ', ')
        normalized = re.sub(r',\s*,', ',', normalized)
        
        # Удаляем лишние запятые в начале/конце
        normalized = normalized.strip(', ')
        
        return normalized
    
    # --- Работа с типами улиц ---

    def parse_street(self, text: str) -> Dict[str, Optional[str]]:
        """
        Парсинг строки улицы на тип и основное имя.

        Примеры:
            "ул. Тверская" -> {"type": "ул", "name": "тверская"}
            "Тверская улица" -> {"type": "ул", "name": "тверская"}
            "пр-т Мира" -> {"type": "пр-кт", "name": "мира"}
        """
        result = {"type": None, "name": ""}
        if not text:
            return result

        s = self.normalize(text)
        # Убираем запятые, оставляем только текст
        s = s.replace(',', ' ')
        s = re.sub(r'\s+', ' ', s).strip()

        # Токенизируем
        tokens = s.split(' ')

        street_type = None
        # Ищем токены, которые соответствуют типу улицы (или их комбинации)
        # Просматриваем от начала и от конца
        for i, tok in enumerate(tokens):
            t = tok.strip('.').lower()
            if t in self.ABBREVIATIONS:
                canon = self.ABBREVIATIONS[t]
                # Учитываем только типы улиц (ул/пр-кт/ш/б-р/пл и т.п.)
                if canon in {'ул', 'пр-кт', 'пер', 'пр-д', 'б-р', 'пл', 'наб', 'ш', 'ал', 'туп', 'лин', 'кв-л'}:
                    street_type = canon
                    tokens[i] = ''
                    break

        if street_type is None and len(tokens) > 1:
            # Попробуем поискать тип в конце, как "тверская улица"
            last = tokens[-1].strip('.').lower()
            if last in self.ABBREVIATIONS:
                canon = self.ABBREVIATIONS[last]
                if canon in {'ул', 'пр-кт', 'пер', 'пр-д', 'б-р', 'пл', 'наб', 'ш', 'ал', 'туп', 'лин', 'кв-л'}:
                    street_type = canon
                    tokens[-1] = ''

        street_name = ' '.join(t for t in tokens if t).strip()

        result["type"] = street_type
        result["name"] = street_name
        return result
    
    # --- Работа с номером дома (дом / корпус / строение) ---

    def parse_house_number(self, text: str) -> Dict[str, Optional[str]]:
        """
        Парсинг номера дома на составные части.
        
        Поддерживает несколько «ступеней» (корпус, строение, литера и т.п.), например:
            "2 к2 с1" -> main="2", extras=[("к","2"), ("с","1")]
        
        Возвращает словарь:
            {
                "main": "10",          # основной номер
                "extra_type": "к",     # тип первой доп. части (для обратной совместимости)
                "extra_number": "1",   # номер первой доп. части
                "extras": [("к","1"), ("с","4"), ...]  # полный список доп. частей
            }
        """
        result: Dict[str, Optional[str]] = {
            "main": None,
            "extra_type": None,
            "extra_number": None,
        }
        # Полный список доп. частей (тип, номер)
        extras = []

        if not text:
            result["extras"] = extras
            return result

        t = str(text).lower().strip()
        # Убираем лишние пробелы
        t = re.sub(r'\s+', ' ', t)

        # Приводим обозначения корпуса/строения/литеры к короткой форме
        # корпус / корп. / к. -> к
        t = re.sub(r'\b(корпус|корп\.?|к)\b', 'к', t)
        # строение / стр. / с. -> с
        t = re.sub(r'\b(строение|стр\.?|с)\b', 'с', t)
        # литера / лит. / л. -> л
        t = re.sub(r'\b(литера|лит\.?|л)\b', 'л', t)

        # Основной номер дома: первое число (возможно с буквой)
        m_main = re.search(r'\d+[а-яa-z]?', t)
        if not m_main:
            # Если не распознали шаблон, считаем всё основным номером
            result["main"] = t or None
            result["extras"] = extras
            return result

        main = m_main.group(0)
        result["main"] = main

        # Остаток строки после основного номера
        rest = t[m_main.end():]

        # Ищем все дополнительные части вида "к2", "с7", "л1" и т.п.
        for m in re.finditer(r'(к|с|л)\s*(\d+[а-яa-z]?)', rest):
            extra_type = m.group(1)
            extra_num = m.group(2)
            extras.append((extra_type, extra_num))

        if extras:
            # Для обратной совместимости заполняем первую доп. часть
            result["extra_type"] = extras[0][0]
            result["extra_number"] = extras[0][1]

        # Полный список доп. частей
        result["extras"] = extras
        return result

    def normalize_house_number(self, text: str) -> str:
        """
        Нормализация номера дома в каноническую строку.

        Примеры:
            "10" -> "10"
            "10 к1", "10 корп 1" -> "10к1"
            "10 стр 2", "10 строение 2" -> "10с2"
        """
        if not text:
            return ""

        info = self.parse_house_number(text)
        main = info.get("main")
        if not main:
            return ""

        # Собираем полный канонический номер:
        #   "2", [("к","2"), ("с","1")] -> "2к2с1"
        parts = [main]
        extras = info.get("extras") or []
        for extra_type, extra_num in extras:
            if extra_type and extra_num:
                parts.append(f"{extra_type}{extra_num}")

        return "".join(parts)
    
    def parse_address(self, address: str) -> Dict[str, Optional[str]]:
        """
        Парсинг адреса на компоненты.
        
        Args:
            address: Адресная строка
            
        Returns:
            Словарь с компонентами адреса
        """
        normalized = self.normalize(address)
        
        result = {
            'locality': None,
            'street': None,
            'number': None,
            'original': address
        }
        
        # Разбиваем по запятым
        parts = [p.strip() for p in normalized.split(',') if p.strip()]
        
        # Ищем город (обычно первая часть)
        if len(parts) > 0:
            first_part = parts[0]
            # Проверяем, является ли это городом
            if any(marker in first_part for marker in ['москва', 'г ', 'город']):
                result['locality'] = 'Москва'
                parts = parts[1:] if len(parts) > 1 else []
        
        # Если после города ничего не осталось — возвращаем то, что есть
        if not parts:
            return result

        # Пытаемся интерпретировать ПОСЛЕДНЮЮ часть как номер дома.
        # Это важно для случаев вида "МКАД, 78-й километр, 2 к2 с1":
        #   улица = "МКАД, 78-й километр"
        #   номер = "2к2с1"
        street_parts = parts[:]
        last_part = parts[-1]
        candidate_number = self.normalize_house_number(last_part)
        if candidate_number:
            result['number'] = candidate_number
            street_parts = parts[:-1]  # всё до номера считаем улицей
        
        # Ищем улицу (всё, что осталось до номера)
        if street_parts:
            street_raw = ', '.join(street_parts)
            street_info = self.parse_street(street_raw)
            # Каноническое имя улицы: "тверская пл", "тверская ул", "мкад 78-й километр"
            if street_info["name"]:
                street_name = street_info["name"]
                if street_info["type"]:
                    street_name = f"{street_name} {street_info['type']}"
                result['street'] = street_name
            else:
                result['street'] = street_raw
        
        # Если номер не нашли во второй части, попробуем поискать по всей строке
        if result['number'] is None:
            normalized_number = self.normalize_house_number(normalized)
            if normalized_number:
                result['number'] = normalized_number
        
        return result
    
    def expand_abbreviations(self, text: str) -> str:
        """
        Расшифровка сокращений в тексте.
        
        Args:
            text: Текст с сокращениями
            
        Returns:
            Текст с расшифрованными сокращениями
        """
        words = text.split()
        expanded = []
        
        for word in words:
            # Убираем точку в конце
            word_clean = word.rstrip('.')
            if word_clean.lower() in self.EXPANSIONS:
                expanded.append(self.EXPANSIONS[word_clean.lower()])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def format_house_number_display(self, text: str) -> str:
        """
        Форматирование номера дома для вывода в человеко‑читаемом виде.
        
        Требуемый формат:
            "{номер дома} {кN} {сM} ..."
        Пример:
            raw:  "50к1с15" или "50 к1 с15" -> "50 к1 с15"
        """
        if not text:
            return ""
        
        info = self.parse_house_number(text)
        main = info.get("main")
        if not main:
            return ""
        
        parts = [main]
        extras = info.get("extras") or []
        for extra_type, extra_num in extras:
            if extra_type and extra_num:
                parts.append(f"{extra_type}{extra_num}")
        
        return " ".join(parts)
    
    def normalize_components_for_output(self, locality: str, street: str, number: str) -> Dict[str, str]:
        """
        Нормализация компонентов адреса к формату, описанному в задании:
            \"{город}, {улица}, {номер дома} {номер корпус} {строение}\"
        
        - Названия объектов без сокращений (ул. -> улица, пр-т -> проспект и т.д.).
        - Порядок слов в названии улицы сохраняем как в исходных данных,
          но убираем сокращения.
        """
        loc = (locality or "").strip() or "Москва"
        street_raw = (street or "").strip()
        number_raw = (number or "").strip()

        street_out = self.expand_abbreviations(street_raw) if street_raw else ""
        number_out = self.format_house_number_display(number_raw) if number_raw else ""

        return {
            "locality": loc,
            "street": street_out,
            "number": number_out,
        }
    
    def create_full_address(self, locality: str, street: str, number: str) -> str:
        """
        Создание полного адреса из компонентов.
        
        Args:
            locality: Населенный пункт
            street: Улица
            number: Номер дома
            
        Returns:
            Полный адрес
        """
        parts = []
        if locality:
            parts.append(locality)
        if street:
            parts.append(street)
        if number:
            parts.append(f"д. {number}")
        
        return ", ".join(parts)
    
    def canonicalize_street(self, text: str) -> str:
        """
        Приведение улицы к каноническому виду для индексов и сравнения.
        
        Примеры:
            "Тверская площадь"   -> "тверская пл"
            "пл. Тверская"       -> "тверская пл"
            "ул. Твардовского"   -> "твардовского ул"
        """
        if not text:
            return ""
        
        info = self.parse_street(text)
        if not info["name"]:
            # Если не смогли распарсить, просто нормализуем строку
            return self.normalize(text)
        
        name = info["name"]
        if info["type"]:
            return f"{name} {info['type']}"
        
        return name
    
    def calculate_similarity(self, addr1: str, addr2: str) -> float:
        """
        Вычисление схожести двух адресов (0-1).
        
        Args:
            addr1: Первый адрес
            addr2: Второй адрес
            
        Returns:
            Коэффициент схожести от 0 до 1
        """
        from rapidfuzz import fuzz
        
        # Нормализуем оба адреса
        norm1 = self.normalize(addr1)
        norm2 = self.normalize(addr2)
        
        # Используем ratio из rapidfuzz (аналог нормализованного Левенштейна)
        similarity = fuzz.ratio(norm1, norm2) / 100.0
        
        return similarity

