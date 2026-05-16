# src/optimizers/tensorrt_optimizer.py
import sys
from pathlib import Path
import numpy as np

# Добавляем путь к скомпилированному C++ модулю
CPP_BUILD_DIR = Path(__file__).parent.parent.parent / "cpp" / "build" / "Release"
if str(CPP_BUILD_DIR) not in sys.path:
    sys.path.append(str(CPP_BUILD_DIR))

class TensorRTOptimizer:
    """Оптимизатор, использующий нашу C++ обертку."""
    def __init__(self, engine_path):
        """
        Args:
            engine_path (str): Путь к .engine файлу.
        """
        try:
            # Импортируем наш C++ модуль
            import cpp_inference
            self.engine = cpp_inference.TensorRTEngine(engine_path)
            print(f"[OK] TensorRT Engine loaded from {engine_path}")
        except ImportError:
            print("[ERROR] Модуль 'cpp_inference' не найден. Запустите сборку C++ кода.")
            raise
        except Exception as e:
            print(f"[ERROR] Ошибка загрузки TensorRT Engine: {e}")
            raise

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Выполняет инференс.
        Args:
            input_data (np.ndarray): Входные данные (float32).
        Returns:
            np.ndarray: Результаты инференса.
        """
        # Конвертируем numpy массив в список C++
        input_list = input_data.flatten().tolist()
        # Вызываем C++ метод
        output_list = self.engine.infer(input_list)
        # Возвращаем результат как numpy массив
        return np.array(output_list, dtype=np.float32)