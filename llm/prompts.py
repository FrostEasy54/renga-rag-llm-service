from models.building import BuildingData

SYSTEM_PROMPT = """Ты — эксперт по календарному планированию строительных проектов.
Ты помогаешь составлять календарные планы на основе данных о здании.
Всегда отвечай только на русском языке.
Когда тебя просят составить календарный план, возвращай ответ строго в формате JSON.
"""


def build_schedule_prompt(building: BuildingData) -> str:
    """
    Convert BuildingData into a detailed Russian prompt for the LLM.

    Prompt engineering notes:
    - Floor sequence is spelled out explicitly — LLMs follow explicit steps
      better than implicit domain knowledge
    - Windows are inserted after walls of each floor
    - External decoration after roof, internal after external
    - Engineering systems last (requires finished walls + roof)
    """
    if building.elements:
        lines = [
            f"  - {el.count}x {el.type} ({el.material}, объём: {el.volume} м³)"
            for el in building.elements
        ]
        elements_text = "Элементы здания:\n" + "\n".join(lines)
    else:
        elements_text = "Элементы здания: данные не предоставлены."

    # Spell out the per-floor sequence explicitly so the LLM doesn't guess
    floor_steps = []
    step = 2
    for floor in range(1, building.floors + 1):
        floor_steps.append(f"  {step}. Стены {floor}-го этажа")
        step += 1
        floor_steps.append(f"  {step}. Окна {floor}-го этажа (после стен {floor}-го этажа)")
        step += 1
        floor_steps.append(f"  {step}. Перекрытие {floor}-го этажа (опирается на стены {floor}-го этажа)")
        step += 1

    floor_sequence = "\n".join(floor_steps)

    finishing_start = step
    return f"""Составь календарный план строительства следующего объекта.

Название проекта: {building.name}
Тип здания: {building.building_type}
Количество этажей: {building.floors}
Общая площадь: {building.total_area} м²
{elements_text}
Дополнительные заметки: {building.notes or 'нет'}

ВАЖНО: Строительство ведётся поэтажно в следующем порядке:
  1. Фундамент
{floor_sequence}
  {finishing_start}. Кровля (после последнего перекрытия)
  {finishing_start + 1}. Внешняя отделка (после кровли)
  {finishing_start + 2}. Внутренняя отделка (после внешней отделки)
  {finishing_start + 3}. Инженерные сети (после внутренней отделки)
  {finishing_start + 4}. Благоустройство территории (параллельно с инженерными сетями)

Правила:
- Стены этажа возводятся ДО перекрытия этого же этажа
- Перекрытие опирается на стены — нельзя делать перекрытие до стен
- Каждый следующий этаж начинается только после укладки перекрытия предыдущего
- Окна устанавливаются сразу после стен этажа
- task_name должен быть коротким и чистым, без скобок и технических деталей

Верни ответ ТОЛЬКО в формате JSON без дополнительного текста, строго по следующей схеме:
{{
  "project_name": "...",
  "total_duration_days": <число>,
  "tasks": [
    {{
      "task_name": "...",
      "duration_days": <число>,
      "depends_on": ["название предыдущей задачи"],
      "workers_required": <число>,
      "description": "..."
    }}
  ],
  "notes": "..."
}}
"""