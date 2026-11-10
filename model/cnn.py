"""
Model architecture definition and loading utility for MNIST Handwritten Digit Recognition.
Reconstructs the fully connected neural network (MLP) matching trained weights in model.pth.
"""

import os
from typing import Optional
import torch
import torch.nn as nn


def make_model(activation_name: str = "ReLU") -> nn.Sequential:
    """
    Constructs the reconstructed neural network architecture.
    
    Layers:
      0. Flatten: [B, 1, 28, 28] -> [B, 784]
      1. Linear: 784 -> 128 (matches parameter key '1.weight', '1.bias')
      2. Activation: ReLU / Sigmoid / Tanh
      3. Linear: 128 -> 10 (matches parameter key '3.weight', '3.bias')
    """
    activation_choices = {
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh
    }

    if activation_name not in activation_choices:
        raise ValueError(
            f"Unsupported activation '{activation_name}'. "
            f"Supported activations: {list(activation_choices.keys())}"
        )

    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        activation_choices[activation_name](),
        nn.Linear(128, 10)
    )


def resolve_model_weights_path(custom_path: Optional[str] = None) -> str:
    """
    Locates the model.pth file across standard project locations.
    """
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        custom_path,
        os.path.join(base_dir, "model.pth"),
        os.path.join(base_dir, "..", "model.pth"),
        os.path.join(os.getcwd(), "model.pth"),
        os.path.join(base_dir, "best_model.pth"),
        os.path.join(base_dir, "..", "best_model.pth"),
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"Could not locate 'model.pth' in any standard location: {candidates}"
    )


def load_trained_model(
    weights_path: Optional[str] = None,
    activation_name: str = "ReLU",
    device: str = "cpu"
) -> nn.Sequential:
    """
    Instantiates the reconstructed model and loads weights with strict verification.
    """
    resolved_path = resolve_model_weights_path(weights_path)

    # 1. Instantiate reconstructed model
    model = make_model(activation_name=activation_name)

    # 2. Load checkpoint using weights_only=True
    state_dict = torch.load(
        resolved_path,
        map_location=device,
        weights_only=True
    )

    # 3. Strictly load weights
    model.load_state_dict(state_dict, strict=True)

    # 4. Set to evaluation mode
    model.eval()
    model.to(device)

    return model

