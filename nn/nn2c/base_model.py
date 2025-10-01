import json
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
        return config["char_freq"], config["most_coocuring_values"]


char_freq, most_cooccuring_values = read_config()
