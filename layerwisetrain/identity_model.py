import copy
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


class IdentityLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor]:
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
        return model


class IdentityWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super(IdentityWrapper, self).__init__()
        self.model = IdentityLayer.patch_model(copy.deepcopy(model))
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        cache_position = None,
        logits_to_keep = 0,
        **kwargs,
    ):
        assert labels is not None
        response = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs
        )
        batch_size = response.logits.shape[0]
        logits_flat = response.logits.view((-1, response.logits.shape[-1]))
        labels_flat = labels.view((-1))
        loss = F.cross_entropy(logits_flat, labels_flat, reduction='sum') / batch_size
        return CausalLMOutputWithPast(
            loss=loss,
            logits=response.logits,
            past_key_values=response.past_key_values,
            hidden_states=response.hidden_states,
            attentions=response.attentions,
        )