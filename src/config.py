import os
from pathlib import Path

# Корень проекта (папка, где находится src)
ROOT_DIR = Path(__file__).parent.parent

# Пути для сохранения моделей и результатов
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

# Создаём папки, если их нет
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Параметры бенчмарка
BENCHMARK_RUNS = 50          # количество замеров
BENCHMARK_WARMUP = 5         # прогрев перед замерами
MAX_NEW_TOKENS = 50          # длина генерации для LLM

# Тестовые промпты (для русского языка)
TEST_PROMPTS = [
    "Привет, как дела?",
    "Напиши краткую историю о космосе",
    "Объясни, что такое квантование нейронных сетей"
]

# Настройки устройства
FORCE_CPU = os.getenv("FORCE_CPU", "0") == "1"

# Настройки логирования
LOG_LEVEL = "INFO"