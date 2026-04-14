import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from typing import Dict, Callable

# -----------------------------
# Metrics for (r_norm, cos_theta, sin_theta)
# -----------------------------
@torch.no_grad()
def metrics_r_cos_sin(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable = None,
    angle_deg_thresh: float = 10.0,
    r_abs_thresh: float = 0.05,
) -> Dict[str, float]:

    y_hat = y_hat.float()
    y = y.float()

    # ---------- loss ----------
    if loss_fn is None:
        loss_val = F.mse_loss(y_hat, y).item()
    else:
        loss_val = loss_fn(y_hat, y).item()

    # ---------- r error ----------
    r_hat, r = y_hat[:, 0], y[:, 0]
    r_mae = (r_hat - r).abs().mean().item()

    # ---------- direction ----------
    if y_hat.shape[1] >= 3:
        c_hat, s_hat = y_hat[:, 1], y_hat[:, 2]
        c, s = y[:, 1], y[:, 2]

        norm = torch.sqrt(c_hat**2 + s_hat**2).clamp_min(1e-6)
        c_hat_n = c_hat / norm
        s_hat_n = s_hat / norm

        theta_hat = torch.atan2(s_hat_n, c_hat_n)
        theta = torch.atan2(s, c)

        dtheta = torch.atan2(
            torch.sin(theta_hat - theta),
            torch.cos(theta_hat - theta)
        )

        angle_mae_deg = dtheta.abs().mean().item() * (180.0 / math.pi)

        angle_deg = dtheta.abs() * (180.0 / math.pi)
        angle_acc = (angle_deg <= angle_deg_thresh).float().mean().item()

        joint_acc = (
            (angle_deg <= angle_deg_thresh)
            & ((r_hat - r).abs() <= r_abs_thresh)
        ).float().mean().item()
    else:
        angle_mae_deg = float('nan')
        angle_acc = float('nan')
        joint_acc = float('nan')

    return {
        "loss": loss_val,
        "r_mae": r_mae,
        "angle_mae_deg": angle_mae_deg,
        "angle_acc": angle_acc,
        "joint_acc": joint_acc,
    }