import os
import json
import numpy as np
import torch
from torch.utils.hooks import RemovableHandle
from typing import List, Dict
import torch.distributed as dist

COLLECT_TYPE = ['q', 'k', 'v', 'kbase', 'vbase', 'latents']

class Collector:
    def __init__(self, save_dir: str, target_steps: List[int] = None, target_layers: List[int] = None, enabled: bool = False, rank: int = 0):
        self.save_dir = save_dir
        self.target_steps = target_steps # None means all steps
        self.target_layers = target_layers # None means all layers
        self.enabled = enabled
        self.rank = rank
        os.makedirs(save_dir, exist_ok=True)
    
    def collect(self, tensor: torch.Tensor, type: str, step: int, layer: int):
        if not self.enabled:
            return
        
        if type not in COLLECT_TYPE:
            raise ValueError(f"Invalid collect type: {type}")
        if self.target_steps is not None and step not in self.target_steps:
            return
        if self.target_layers is not None and layer not in self.target_layers:
            return
        
        if type == 'latents':
            assert layer is None # latents are not layer-specific
            save_dir = os.path.join(self.save_dir, f"rank_{self.rank}", f"step_{step}")
            filename = f"{type}.pt"
            self._save_activation(tensor, save_dir, filename)
        else:
            save_dir = os.path.join(self.save_dir, f"rank_{self.rank}", f"step_{step}", f"layer_{layer}")
            filename = f"{type}.pt"
            self._save_activation(tensor, save_dir, filename)
    
    def _save_activation(self, tensor: torch.Tensor, save_dir: str, filename: str):
        """
        保存激活值到文件
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        tensor_cpu = tensor.detach().cpu()
        torch.save(tensor_cpu, os.path.join(save_dir, filename))

instance = None

def init(collector: Collector):
    global instance
    instance = collector

def collect(tensor: torch.Tensor, type: str, step: int, layer: int):
    global instance
    if instance is None:
        raise ValueError("Collector not initialized")
    instance.collect(tensor, type, step, layer)