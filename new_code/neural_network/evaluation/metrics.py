import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from typing import Dict, Callable

def _metrics_1d(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
    }

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

import torch
import numpy as np

def get_predictions(model, loader, device="cpu"):
    model.eval()

    y_true_all = []
    y_pred_all = []
    sub_all = []
    index_all = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            y = batch["y"]

            # move x to device
            if isinstance(x, dict):
                x = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in x.items()
                }
            else:
                x = x.to(device)

            # y may be dict: {"gaze": [B,2], "pupil": [B,1]}
            if isinstance(y, dict):
                y_gaze = y["gaze"]
                y_pupil = y["pupil"]

                if not torch.is_tensor(y_gaze) or not torch.is_tensor(y_pupil):
                    raise TypeError("y['gaze'] and y['pupil'] must be tensors.")

                y_gaze = y_gaze.to(device)
                y_pupil = y_pupil.to(device)

                y_tensor = torch.cat([y_gaze, y_pupil], dim=-1)   # [B, 3]
            else:
                y_tensor = y.to(device)

            _, y_hat = model(x)

            y_true_all.append(y_tensor.detach().cpu().numpy())
            y_pred_all.append(y_hat.detach().cpu().numpy())

            if "sub" in batch:
                sub_all.extend(batch["sub"])

            if "index" in batch:
                idx = batch["index"]
                if torch.is_tensor(idx):
                    index_all.extend(idx.cpu().numpy().tolist())
                else:
                    index_all.extend(list(idx))

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    return {
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "sub": sub_all,
        "index": index_all,
    }

def split_outputs(y):
    gaze = y[:, :2]      # [N, 2]
    pupil = y[:, 2:]     # [N, 1]
    return gaze, pupil

def compute_metrics(y_true, y_pred, name=""):
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print(f"\n{name} Metrics:")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {"mae": mae, "mse": mse, "rmse": rmse}

import matplotlib.pyplot as plt

def plot_gaze(y_true_gaze, y_pred_gaze, n_samples=500):
    idx = np.random.choice(len(y_true_gaze), size=min(n_samples, len(y_true_gaze)), replace=False)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_true_gaze[idx, 0],
        y_true_gaze[idx, 1],
        alpha=0.5,
        label="True"
    )
    plt.scatter(
        y_pred_gaze[idx, 0],
        y_pred_gaze[idx, 1],
        alpha=0.5,
        label="Pred"
    )

    plt.xlabel("Gaze X")
    plt.ylabel("Gaze Y")
    plt.title("Gaze: True vs Pred")
    plt.legend()
    plt.grid()
    plt.show()

def plot_pupil(y_true_pupil, y_pred_pupil, n_samples=500):
    idx = np.random.choice(len(y_true_pupil), size=min(n_samples, len(y_true_pupil)), replace=False)

    plt.figure(figsize=(6, 4))
    plt.plot(y_true_pupil[idx], label="True", alpha=0.7)
    plt.plot(y_pred_pupil[idx], label="Pred", alpha=0.7)

    plt.title("Pupil: True vs Pred")
    plt.legend()
    plt.grid()
    plt.show()

def evaluate_eye_model(model, test_loader, device="cpu", max_plot_points=500):
    out = get_predictions(model, test_loader, device=device)

    y_true = out["y_true"]   # [N, 3]
    y_pred = out["y_pred"]   # [N, 3]

    # split
    gaze_true = y_true[:, :2]
    gaze_pred = y_pred[:, :2]

    pupil_true = y_true[:, 2]
    pupil_pred = y_pred[:, 2]

    metrics = {
        "gaze_x": _metrics_1d(gaze_true[:, 0], gaze_pred[:, 0]),
        "gaze_y": _metrics_1d(gaze_true[:, 1], gaze_pred[:, 1]),
        "pupil": _metrics_1d(pupil_true, pupil_pred),
        "gaze_all": {
            "mae": float(np.mean(np.abs(gaze_true - gaze_pred))),
            "mse": float(np.mean((gaze_true - gaze_pred) ** 2)),
            "rmse": float(np.sqrt(np.mean((gaze_true - gaze_pred) ** 2))),
        },
    }

    print("=== Test metrics ===")
    for name, vals in metrics.items():
        print(f"\n{name}")
        for k, v in vals.items():
            print(f"  {k}: {v:.6f}")

    # sample for plotting
    n = len(y_true)
    m = min(max_plot_points, n)
    idx = np.arange(m)

    # gaze_x
    plt.figure(figsize=(8, 4))
    plt.plot(idx, gaze_true[:m, 0], label="true gaze_x", alpha=0.8)
    plt.plot(idx, gaze_pred[:m, 0], label="pred gaze_x", alpha=0.8)
    plt.title("Gaze X: true vs pred")
    plt.legend()
    plt.grid(True)
    plt.show()

    # gaze_y
    plt.figure(figsize=(8, 4))
    plt.plot(idx, gaze_true[:m, 1], label="true gaze_y", alpha=0.8)
    plt.plot(idx, gaze_pred[:m, 1], label="pred gaze_y", alpha=0.8)
    plt.title("Gaze Y: true vs pred")
    plt.legend()
    plt.grid(True)
    plt.show()

    # pupil
    plt.figure(figsize=(8, 4))
    plt.plot(idx, pupil_true[:m], label="true pupil", alpha=0.8)
    plt.plot(idx, pupil_pred[:m], label="pred pupil", alpha=0.8)
    plt.title("Pupil: true vs pred")
    plt.legend()
    plt.grid(True)
    plt.show()

    # scatter: pred vs true
    plt.figure(figsize=(5, 5))
    plt.scatter(gaze_true[:, 0], gaze_pred[:, 0], alpha=0.3)
    mn = min(gaze_true[:, 0].min(), gaze_pred[:, 0].min())
    mx = max(gaze_true[:, 0].max(), gaze_pred[:, 0].max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.title("Gaze X: pred vs true")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(gaze_true[:, 1], gaze_pred[:, 1], alpha=0.3)
    mn = min(gaze_true[:, 1].min(), gaze_pred[:, 1].min())
    mx = max(gaze_true[:, 1].max(), gaze_pred[:, 1].max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.title("Gaze Y: pred vs true")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(pupil_true, pupil_pred, alpha=0.3)
    mn = min(pupil_true.min(), pupil_pred.min())
    mx = max(pupil_true.max(), pupil_pred.max())
    plt.plot([mn, mx], [mn, mx], "--")
    plt.title("Pupil: pred vs true")
    plt.xlabel("true")
    plt.ylabel("pred")
    plt.grid(True)
    plt.show()

    return metrics, out