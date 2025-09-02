import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
        return config["char_freq_values"]


char_freq_values = read_config()


class BaseHangmanModel(nn.Module):
    def __init__(self):
        super(BaseHangmanModel, self).__init__()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def reset_last_embedding(self):
        """the last embedding should be reset to make it a sort of null embedding"""
        mask = torch.eye(self.embed_layer.weight.shape[0])
        last_row = char_freq_values + [0, 0]
        last_row = torch.tensor(last_row, dtype=torch.float32)
        last_row /= last_row.sum()
        mask[-1] = last_row.to(self.device)
        self.embed_layer.weight.data = torch.matmul(mask, self.embed_layer.weight)
