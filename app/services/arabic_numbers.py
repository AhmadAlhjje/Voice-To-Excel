"""
Arabic number conversion service.
Converts spoken Arabic numbers to actual digits.
Handles compound numbers, sequential digits, and preserves leading zeros.
"""

import re
from typing import Optional, Tuple, List


class ArabicNumberConverter:
    """
    Converts Arabic spoken numbers to digits.

    Examples:
        - "خمسة وأربعون" → "45"
        - "صفر تسعة تسعة ثمانية" → "0998"
        - "مئة وخمسة وعشرون" → "125"
    """

    # Basic Arabic numbers (0-10)
    BASIC_NUMBERS = {
        "صفر": 0,
        "واحد": 1, "واحدة": 1,
        "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2,
        "ثلاثة": 3, "ثلاث": 3,
        "أربعة": 4, "أربع": 4, "اربعة": 4, "اربع": 4,
        "خمسة": 5, "خمس": 5,
        "ستة": 6, "ست": 6,
        "سبعة": 7, "سبع": 7,
        "ثمانية": 8, "ثماني": 8, "ثمان": 8,
        "تسعة": 9, "تسع": 9,
        "عشرة": 10, "عشر": 10,
    }

    # Teens (11-19)
    TEENS = {
        "أحد عشر": 11, "احد عشر": 11,
        "اثنا عشر": 12, "اثني عشر": 12, "اثنى عشر": 12,
        "ثلاثة عشر": 13, "ثلاث عشر": 13,
        "أربعة عشر": 14, "اربعة عشر": 14, "أربع عشر": 14, "اربع عشر": 14,
        "خمسة عشر": 15, "خمس عشر": 15,
        "ستة عشر": 16, "ست عشر": 16,
        "سبعة عشر": 17, "سبع عشر": 17,
        "ثمانية عشر": 18, "ثماني عشر": 18,
        "تسعة عشر": 19, "تسع عشر": 19,
    }

    # Tens (20-90)
    TENS = {
        "عشرون": 20, "عشرين": 20,
        "ثلاثون": 30, "ثلاثين": 30,
        "أربعون": 40, "أربعين": 40, "اربعون": 40, "اربعين": 40,
        "خمسون": 50, "خمسين": 50,
        "ستون": 60, "ستين": 60,
        "سبعون": 70, "سبعين": 70,
        "ثمانون": 80, "ثمانين": 80,
        "تسعون": 90, "تسعين": 90,
    }

    # Hundreds
    HUNDREDS = {
        "مئة": 100, "مائة": 100,
        "مئتان": 200, "مئتين": 200, "مائتان": 200, "مائتين": 200,
        "ثلاثمئة": 300, "ثلاثمائة": 300, "ثلاث مئة": 300, "ثلاث مائة": 300,
        "أربعمئة": 400, "أربعمائة": 400, "أربع مئة": 400, "أربع مائة": 400, "اربعمئة": 400, "اربعمائة": 400,
        "خمسمئة": 500, "خمسمائة": 500, "خمس مئة": 500, "خمس مائة": 500,
        "ستمئة": 600, "ستمائة": 600, "ست مئة": 600, "ست مائة": 600,
        "سبعمئة": 700, "سبعمائة": 700, "سبع مئة": 700, "سبع مائة": 700,
        "ثمانمئة": 800, "ثمانمائة": 800, "ثمان مئة": 800, "ثمان مائة": 800,
        "تسعمئة": 900, "تسعمائة": 900, "تسع مئة": 900, "تسع مائة": 900,
    }

    # Thousands
    THOUSANDS = {
        "ألف": 1000, "الف": 1000,
        "ألفان": 2000, "ألفين": 2000, "الفان": 2000, "الفين": 2000,
    }

    # All number words combined for detection
    ALL_NUMBER_WORDS = set()

    def __init__(self):
        """Initialize the converter with all number mappings."""
        # Build the complete set of number words
        self.ALL_NUMBER_WORDS = set()
        for d in [self.BASIC_NUMBERS, self.TEENS, self.TENS, self.HUNDREDS, self.THOUSANDS]:
            self.ALL_NUMBER_WORDS.update(d.keys())
        # Add connecting words
        self.ALL_NUMBER_WORDS.add("و")

    def is_phone_or_id_column(self, column_name: str) -> bool:
        """
        Check if a column typically contains phone numbers or IDs.
        These require preserving leading zeros and sequential digit handling.
        """
        phone_keywords = [
            "هاتف", "تلفون", "جوال", "موبايل", "رقم الهاتف", "رقم الجوال",
            "phone", "mobile", "tel", "telephone"
        ]
        id_keywords = [
            "رقم وطني", "الرقم الوطني", "هوية", "رقم الهوية", "بطاقة",
            "id", "national", "identity"
        ]

        column_lower = column_name.lower()
        for keyword in phone_keywords + id_keywords:
            if keyword in column_lower:
                return True
        return False

    def convert_sequential_digits(self, text: str) -> str:
        """
        Convert sequential spoken digits to a number string.
        Used for phone numbers and IDs where each digit is spoken separately.

        Example: "صفر تسعة تسعة ثمانية واحد صفر سبعة سبعة اثنين اثنين"
                 → "0998107722"
        """
        result = []
        words = text.split()

        for word in words:
            # Skip connecting words
            if word == "و":
                continue

            # Check if it's a basic digit (0-9)
            if word in self.BASIC_NUMBERS:
                value = self.BASIC_NUMBERS[word]
                if value <= 9:
                    result.append(str(value))
                else:
                    # For 10, treat as "1" and "0" in sequential mode
                    result.append("10")

        return "".join(result)

    def convert_compound_number(self, text: str) -> Optional[int]:
        """
        Convert a compound Arabic number to an integer.

        Example: "خمسة وأربعون" → 45
                 "مئة وخمسة وعشرون" → 125
        """
        # Normalize text
        text = text.strip()
        if not text:
            return None

        # Check for teens first (they're compound words)
        for teen, value in self.TEENS.items():
            if teen in text:
                return value

        # Parse the number components
        total = 0
        current = 0

        # Remove "و" and split
        words = re.split(r'\s+', text.replace(" و ", " ").replace("و", " "))

        for word in words:
            word = word.strip()
            if not word:
                continue

            # Check thousands
            if word in self.THOUSANDS:
                if current == 0:
                    current = 1
                total += current * self.THOUSANDS[word]
                current = 0
            # Check hundreds
            elif word in self.HUNDREDS:
                current += self.HUNDREDS[word]
            # Check tens
            elif word in self.TENS:
                current += self.TENS[word]
            # Check basic numbers
            elif word in self.BASIC_NUMBERS:
                current += self.BASIC_NUMBERS[word]

        total += current
        return total if total > 0 else None

    def convert(self, text: str, column_name: str = "") -> str:
        """
        Main conversion method. Determines the best conversion strategy
        based on the column type.

        Args:
            text: The Arabic text containing numbers
            column_name: The column name to determine conversion strategy

        Returns:
            Converted number as string
        """
        if not text or not text.strip():
            return text

        text = text.strip()

        # For phone numbers and IDs, use sequential digit conversion
        if self.is_phone_or_id_column(column_name):
            return self.convert_sequential_digits(text)

        # Try compound number conversion first
        result = self.convert_compound_number(text)
        if result is not None:
            return str(result)

        # If compound conversion fails, try sequential
        sequential_result = self.convert_sequential_digits(text)
        if sequential_result:
            return sequential_result

        # Return original text if no conversion possible
        return text

    def process_extracted_data(self, data: dict, headers: List[str]) -> dict:
        """
        Process all extracted data and convert Arabic numbers.

        Args:
            data: Dictionary of column_name -> value
            headers: List of column headers

        Returns:
            Processed data with converted numbers
        """
        processed = {}

        for key, value in data.items():
            if value is None:
                processed[key] = None
                continue

            if isinstance(value, str):
                # Check if the value contains Arabic number words
                has_arabic_numbers = any(
                    word in value for word in self.BASIC_NUMBERS.keys()
                )

                if has_arabic_numbers:
                    processed[key] = self.convert(value, key)
                else:
                    processed[key] = value
            else:
                processed[key] = value

        return processed


# Create a global instance
arabic_number_converter = ArabicNumberConverter()


def convert_arabic_number(text: str, column_name: str = "") -> str:
    """Convenience function to convert Arabic numbers."""
    return arabic_number_converter.convert(text, column_name)


def process_extracted_data(data: dict, headers: List[str]) -> dict:
    """Convenience function to process extracted data."""
    return arabic_number_converter.process_extracted_data(data, headers)
