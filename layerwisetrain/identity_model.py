from typing import Tuple, List
import torch
import torch.nn as nn


class IdentityLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        return hidden_states,
    
    @staticmethod
    def patch_model(model: nn.Module) -> nn.Module:
        if hasattr(model, "layers"):
            patch_layers: List[nn.Module] = model.layers
            for i in range(len(patch_layers)):
                patch_layers[i] = IdentityLayer()
        elif hasattr(model, "model"):
            IdentityLayer.patch_model(model.model)
        else:
            raise ValueError(f"Model {model} has no layers or model.layers")
