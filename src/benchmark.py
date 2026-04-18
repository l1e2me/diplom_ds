import time
import torch
import numpy as np
from typing import List, Dict
from .device_manager import device_manager

class InferenceBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device_manager.device

    def measure_latency(self, prompt: str, runs: int = 100, warmup: int = 10) -> Dict[str, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=50)

        latencies = []
        for _ in range(runs):
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=50)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_ms": np.mean(latencies),
            "std_ms": np.std(latencies),
            "p95_ms": np.percentile(latencies, 95)
        }

    def measure_throughput(self, prompts: List[str], batch_size: int = 4) -> Dict[str, float]:
        # ... (логика замера пропускной способности)
        pass

    def measure_memory(self) -> Dict[str, float]:
        if self.device.type == 'cuda':
            return {
                "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            }
        return {}