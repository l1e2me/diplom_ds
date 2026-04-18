import torch
import logging

logger = logging.getLogger(__name__)

class DeviceManager:
    """Абстракция устройства CPU/GPU с автоматическим определением."""
    
    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self._device = self._detect_device()
    
    def _detect_device(self):
        if self.force_cpu:
            logger.info("CPU")
            return torch.device('cpu')
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")
            return torch.device('cuda:0')
        
        return torch.device('cpu')
    
    @property
    def device(self):
        return self._device
    
    @property
    def is_cuda(self):
        return self._device.type == 'cuda'
    
    @property
    def device_name(self):
        return torch.cuda.get_device_name(0) if self.is_cuda else "CPU"
    
    def to_device(self, obj):
        """Переносит объект (модель, тензор) на целевое устройство."""
        if hasattr(obj, 'to'):
            return obj.to(self._device)
        return obj

device_manager = DeviceManager()