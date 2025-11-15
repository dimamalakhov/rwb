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

        Возвращает словарь:
            {
                "main": "10",         # основной номер
                "extra_type": "к"     # тип доп. части: к (корпус), с (строение), л (литера) и т.п.
                "extra_number": "1"   # номер корпуса/строения
            }
        """
        result = {
            "main": None,
            "extra_type": None,
            "extra_number": None,
        }
        if not text:
            return result

        t = str(text).lower().strip()
        # Убираем лишние пробелы
        t = re.sub(r'\s+', ' ', t)

        # Общий шаблон: <число/число+буква> [тип] [номер]
        pattern = re.compile(
            r'(?P<main>\d+[а-яa-z]?)'
            r'(?:\s*(?P<extra_type>к|корп|корпус|с|стр|строение|лит|литера)\.?\s*'
            r'(?P<extra_num>\d+[а-яa-z]?)?)?',
            re.IGNORECASE,
        )

        m = pattern.search(t)
        if not m:
            # Если не распознали шаблон, считаем всё основным номером
            result["main"] = t or None
            return result

        main = m.group("main")
        extra_type = m.group("extra_type")
        extra_num = m.group("extra_num")

        if not main:
            return result

        result["main"] = main

        if extra_type:
            extra_type = extra_type.lower()
            if extra_type.startswith("кор"):
                result["extra_type"] = "к"
            elif extra_type.startswith("стр") or extra_type.startswith("стро"):
                result["extra_type"] = "с"
            elif extra_type.startswith("лит"):
                result["extra_type"] = "л"
            else:
                result["extra_type"] = extra_type

        if extra_num:
            result["extra_number"] = extra_num

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
        main = info["main"]
        if not main:
            return ""

        extra_type = info["extra_type"]
        extra_num = info["extra_number"]

        if extra_type and extra_num:
            return f"{main}{extra_type}{extra_num}"

        return main
    
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
        parts = [p.strip() for p in normalized.split(',')]
        
        # Ищем город (обычно первая часть)
        if len(parts) > 0:
            first_part = parts[0]
            # Проверяем, является ли это городом
            if any(marker in first_part for marker in ['москва', 'г ', 'город']):
                result['locality'] = 'Москва'
                parts = parts[1:] if len(parts) > 1 else []
        
        # Ищем улицу
        if len(parts) > 0:
            street_part = parts[0]
            street_info = self.parse_street(street_part)
            # Сохраняем только имя улицы (без типа) как основной компонент
            result['street'] = street_info["name"] or street_part
        
        # Ищем номер дома
        if len(parts) > 1:
            number_part = parts[1]
            # Извлекаем номер дома и нормализуем
            normalized_number = self.normalize_house_number(number_part)
            if normalized_number:
                result['number'] = normalized_number
        
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

