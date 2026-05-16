#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.device_manager import DeviceManager
from src.model_loader import SberLightningLoader
from src.benchmark import InferenceBenchmark
from src.optimizers.tensorrt_optimizer import TensorRTOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("=== Запуск расширенного бенчмаркинга ===")
    
    device_manager = DeviceManager(force_cpu=True)
    logger.info(f"Используется устройство: {device_manager.device_name}")
    
    loader = SberLightningLoader()
    model, tokenizer = loader.load_fp16()
    bench = InferenceBenchmark(model, tokenizer)
    
    # Разные промпты для теста
    test_prompts = [
        "Привет, как дела?",
        "Напиши краткую историю о космосе",
        "Объясни, что такое квантование нейронных сетей",
        "Посчитай 15 + 27",
    ]
    
    results = []
    for prompt in test_prompts:
        logger.info(f"Промпт: {prompt[:50]}...")
        latency = bench.measure_latency(prompt, runs=10, warmup=3)
        results.append({
            "prompt": prompt,
            "latency_ms": latency['mean_ms'],
            "std_ms": latency['std_ms']
        })
        logger.info(f"  → {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
    
    # Замер памяти
    memory = bench.measure_memory()
    if memory:
        logger.info(f"Потребление памяти: {memory.get('allocated_mb', 0):.2f} MB")
    
    # Вывод сводной таблицы
    print("\n" + "="*60)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*60)
    for r in results:
        print(f"{r['prompt'][:30]:30} | {r['latency_ms']:7.2f} ± {r['std_ms']:.2f} ms")
    print("="*60)

    ENGINE_PATH = "models/sber_lightning_fp16.engine"

    trt_optimizer = TensorRTOptimizer(ENGINE_PATH)

    input_numpy = np.random.randn(1, 512).astype(np.float32)

    output = trt_optimizer.infer(input_numpy)
    print("Output shape:", output.shape)

if __name__ == "__main__":
    main()