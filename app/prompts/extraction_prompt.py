"""
LLM Prompts for data extraction from Arabic speech transcriptions.
"""

from typing import List

# System prompt for the LLM
SYSTEM_PROMPT = """أنت مساعد متخصص في استخراج البيانات من النصوص العربية المنطوقة.
مهمتك هي تحليل النص الصوتي المحول وربط كل قيمة بالعمود المناسب.

القواعد الصارمة التي يجب اتباعها:
1. استخرج فقط البيانات الموجودة في النص - لا تخترع بيانات
2. اربط كل قيمة بالعمود الصحيح من القائمة المتاحة فقط
3. حوّل الأرقام المنطوقة إلى أرقام عربية:
   - "خمسة وأربعون" → "45"
   - "مئة وعشرون" → "120"
4. للهواتف والأرقام الوطنية - أرقام متتالية:
   - "صفر تسعة تسعة ثمانية" → "0998"
   - حافظ دائماً على الأصفار في البداية
5. إذا لم تجد قيمة لعمود، اجعله null
6. لا تضف أعمدة جديدة غير موجودة في القائمة المتاحة
7. إذا ذُكر عمود في النص بشكل مختلف قليلاً، طابقه مع أقرب عمود متاح

مهم جداً:
- أرجع JSON فقط بدون أي نص إضافي أو شرح
- لا تضع الـ JSON داخل code blocks
- تأكد من أن JSON صالح ويمكن تحليله"""


def get_extraction_prompt(headers: List[str], transcription: str) -> str:
    """
    Generate the user prompt for data extraction.

    Args:
        headers: List of column headers from the Excel file
        transcription: The transcribed text from speech

    Returns:
        Formatted prompt string
    """
    headers_str = "، ".join(headers)

    prompt = f"""الأعمدة المتاحة في ملف Excel:
[{headers_str}]

النص المنطوق من المستخدم:
"{transcription}"

استخرج البيانات من النص وأرجعها بصيغة JSON.
كل مفتاح يجب أن يكون اسم عمود من الأعمدة المتاحة بالضبط.
القيم يجب أن تكون نصية (string) أو null إذا لم تُذكر.

مثال على الصيغة المطلوبة:
{{
  "الاسم": "محمد",
  "الرقم": "45",
  "الهاتف": "0998107722",
  "العنوان": null
}}

أرجع JSON فقط:"""

    return prompt


def get_correction_prompt(headers: List[str], original_data: dict, correction: str) -> str:
    """
    Generate a prompt for correcting previously extracted data.

    Args:
        headers: List of column headers
        original_data: The original extracted data
        correction: User's correction instruction

    Returns:
        Formatted prompt string
    """
    import json
    headers_str = "، ".join(headers)
    data_str = json.dumps(original_data, ensure_ascii=False, indent=2)

    prompt = f"""الأعمدة المتاحة:
[{headers_str}]

البيانات الحالية:
{data_str}

طلب التصحيح من المستخدم:
"{correction}"

قم بتصحيح البيانات وفقاً لطلب المستخدم وأرجع JSON المصحح فقط:"""

    return prompt
