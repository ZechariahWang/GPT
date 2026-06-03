import torch
import torch.nn as nn
from typing import List, Dict

class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)
                    if x.dim() >= 2:
                        is_negative = (x <= 0)
                        dead_units = is_negative.all(dim = 0)
                        dead_frac = round(dead_units.float().mean().item(), 4)
                    else:
                        is_negative = (x <= 0)
                        dead_frac = round(is_negative.float().mean().item(), 4)
                    stats.append({"mean" : mean, "std" : std, "dead_fraction" : dead_frac})
        return stats 

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        # forward pass
        output = model(x)

        # back prop
        loss = nn.MSELoss()(output, y)
        loss.backward()

        stats = []
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean = round(grad.mean().item(), 4)
                std = round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                stats.append({"mean" : mean, "std" : std, "norm" : norm})
        return stats 

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for stats in activation_stats:
            if stats["dead_fraction"] > 0.5:
                return "dead_neurons"
            if stats["std"] < 0.1:
                return "vanishing_gradients"  
            elif stats["std"] > 10:
                return "exploding_gradients"

        for stats in gradient_stats:
            if stats["norm"] > 1000:
                return "exploding_gradients"
            if stats["norm"] <1e-5:
                return "vanishing_gradients"

        return "healthy"
            


