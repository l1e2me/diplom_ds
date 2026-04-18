import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from .device_manager import device_manager

logger = logging.getLogger(__name__)

class SberLightningLoader:
    """Загрузчик моделей — на CPU всегда использует маленькую тестовую модель."""

    def load_fp16(self):
        """Загружает модель. На CPU всегда загружает distilgpt2."""
        if not device_manager.is_cuda:
            logger.info("Загружаем тестовую модель 'distilgpt2'.")
            return self._load_mock_model()

        logger.info("Режим GPU: 'ai-sb/GigaChat-3-Lightning'")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                "ai-sb/GigaChat-3-Lightning",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "ai-sb/GigaChat-3-Lightning",
                trust_remote_code=True
            )
            return model, tokenizer
        except Exception as e:
            logger.error(f"Не удалось загрузить Sber Lightning: {e}")
            logger.info("Падаем в тестовую модель 'distilgpt2'.")
            return self._load_mock_model()

    def _load_mock_model(self):
        """Загружает маленькую модель distilgpt2 для тестирования."""
        logger.info("Загрузка тестовой модели 'distilgpt2'...")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        # Устанавливаем pad_token, если его нет
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer