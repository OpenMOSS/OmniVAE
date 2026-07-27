from typing import Dict

import torch
import torch.nn as nn


class StreamingTrainState:
    """训练状态，支持流式数据集的精确恢复"""

    def __init__(self):
        self.step: int = 0
        self.dataset_state_dict: Dict = {
            "consumed_samples": None,
            "rng_state": None,
            "used_epochs": None,
            "dataset_state_dict": {},
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.dataset_state_dict = state_dict["dataset_state_dict"]

    def state_dict(self):
        return {
            "step": self.step,
            "dataset_state_dict": self.dataset_state_dict,
        }

    def update(self, state_dict):
        self.step += 1
        self.dataset_state_dict["consumed_samples"] = state_dict.get("consumed_samples")
        self.dataset_state_dict["used_epochs"] = state_dict.get("used_epochs")
        self.dataset_state_dict["rng_state"] = state_dict.get("rng_state")
        self.dataset_state_dict["dataset_state_dict"].update(
            state_dict.get("dataset_state_dict", {})
        )


class EMA:
    """指数移动平均"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """更新 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self):
        """应用 EMA 参数（用于评估）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """恢复原始参数（评估后）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state_dict):
        self.shadow = state_dict["shadow"]
        self.decay = state_dict.get("decay", self.decay)
