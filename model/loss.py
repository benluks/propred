import torch
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x:    [B, T]
    mask: [B, T] float/bool (1=valid, 0=pad)
    """
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / (mask.sum().clamp_min(eps))


def masked_mse(
    pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mean((pred - tgt) ** 2, mask)


def masked_l1(
    pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mean((pred - tgt).abs(), mask)


def masked_bce_with_logits(
    logits: torch.Tensor, tgt01: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    logits: [B, T]
    tgt01:  [B, T] float in [0,1]
    """
    # per-element BCE (no reduction), then masked mean
    per = F.binary_cross_entropy_with_logits(logits, tgt01, reduction="none")
    return masked_mean(per, mask)


def prosody_loss(
    pred_log_f0: torch.Tensor,
    pred_vuv: torch.Tensor,  # logits if using BCEWithLogits
    pred_rms: torch.Tensor,
    tgt_log_f0: torch.Tensor,
    tgt_vuv: torch.Tensor,  # float in [0,1]
    tgt_rms: torch.Tensor,
    x_mask: torch.Tensor,  # [B,1,T]
    *,
    w_f0: float = 1.0,
    w_vuv: float = 1.0,
    w_rms: float = 1.0,
    f0_loss: str = "mse",  # "mse" or "l1"
    rms_loss: str = "mse",  # "mse" or "l1"
) -> tuple[torch.Tensor, dict]:
    """
    Returns: (total_loss, logs_dict)
    """
    mask = x_mask.squeeze(1)  # [B,T]
    mask = mask.to(dtype=pred_log_f0.dtype)

    # Optional: if your targets might contain NaNs in padded areas, this helps.
    # But you should still have mask correct.
    # mask = mask * torch.isfinite(tgt_log_f0).to(mask.dtype) * torch.isfinite(tgt_rms).to(mask.dtype)

    if f0_loss == "mse":
        lf0 = masked_mse(pred_log_f0, tgt_log_f0, mask)
    elif f0_loss == "l1":
        lf0 = masked_l1(pred_log_f0, tgt_log_f0, mask)
    else:
        raise ValueError("f0_loss must be 'mse' or 'l1'")

    lvuv = masked_bce_with_logits(pred_vuv, tgt_vuv, mask)

    if rms_loss == "mse":
        lrms = masked_mse(pred_rms, tgt_rms, mask)
    elif rms_loss == "l1":
        lrms = masked_l1(pred_rms, tgt_rms, mask)
    else:
        raise ValueError("rms_loss must be 'mse' or 'l1'")

    total = w_f0 * lf0 + w_vuv * lvuv + w_rms * lrms

    logs = {
        "loss/total": total.detach(),
        "loss/log_f0": lf0.detach(),
        "loss/vuv": lvuv.detach(),
        "loss/rms": lrms.detach(),
        "mask/valid_frames": mask.sum().detach(),
    }
    return total, logs
