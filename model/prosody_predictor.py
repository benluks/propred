import torch
import torch.nn as nn
import logging

from .conv_decoder import ConvDecoder


class ProsodyPredictor(nn.Module):
    """
    Wrapper that:
      1) embeds speaker ids (from an index) + optionally projects reps
      2) concatenates reps and speaker embedding along channel dim -> [B, D_rep + D_spk, T]
      3) feeds trunk (duration-predictor-style conv stack)
      4) predicts log-f0, vuv, rms energy as [B, T]
    """

    def __init__(
        self,
        *,
        rep_dim: int,
        n_speakers: int,
        spk_dim: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        vuv_output: str = "logits",  # "logits" or "prob"
        rep_proj_dim: int | None = None,  # set to rep_dim to keep same; or None to skip
    ):
        super().__init__()

        self.spk_emb = nn.Embedding(n_speakers, spk_dim)

        # Optional rep projection (handy if your reps are huge and you want a smaller trunk)
        representation_dimension = rep_dim
        if rep_proj_dim is not None:
            self.rep_proj = nn.Conv1d(rep_dim, rep_proj_dim, kernel_size=1)
            representation_dimension = rep_proj_dim
        else:
            self.rep_proj = None

        self.duration_predictor = ConvDecoder(
            representation_dimension + spk_dim, filter_channels, kernel_size, p_dropout
        )

        # Heads: Conv1d -> 1 channel
        self.f0_head = nn.Conv1d(filter_channels, 1, kernel_size=1)  # log-f0 regression
        self.vuv_head = nn.Conv1d(
            filter_channels, 1, kernel_size=1
        )  # vuv classification
        self.rms_head = nn.Conv1d(filter_channels, 1, kernel_size=1)  # rms regression

        if vuv_output not in ("logits", "prob"):
            raise ValueError("vuv_output must be 'logits' or 'prob'")
        self.vuv_output = vuv_output

    def forward(self, wavs, spk_id, x_mask):
        """
        reps:   [B, D_rep, T]
        spk_id: [B] (int64 speaker indices)
        x_mask: [B, 1, T] (1 for valid frames, 0 for padding)

        returns:
          log_f0: [B, T]
          vuv:    [B, T] (logits or prob depending on vuv_output)
          rms:    [B, T]
        """

        logging.info(
            f"BN extractor device: {next(self.bn_extractor.parameters()).device}"
        )
        with torch.no_grad():
            reps = self.bn_extractor.get_bn(wavs)

        *_, T = reps.shape

        if self.rep_proj is not None:
            reps = self.rep_proj(reps)

        spk = self.spk_emb(spk_id)  # [B, D_spk]
        spk = spk.unsqueeze(-1).expand(-1, -1, T)  # [B, D_spk, T]

        x = torch.cat([reps, spk], dim=1)  # [B, D_rep' + D_spk, T]

        # reps dimension and f0/vuv/energy dimenstion don't always match; trim to smallest length
        min_T = min(x.shape[-1], x_mask.shape[-1])
        x = x[..., :min_T]
        x_mask = x_mask[..., :min_T]

        x = self.duration_predictor(x, x_mask)

        log_f0 = self.f0_head(x) * x_mask  # [B, 1, T]
        vuv = self.vuv_head(x) * x_mask  # [B, 1, T]
        rms = self.rms_head(x) * x_mask  # [B, 1, T]

        # squeeze channel -> [B, T]
        log_f0 = log_f0.squeeze(1)
        vuv = vuv.squeeze(1)
        rms = rms.squeeze(1)

        if self.vuv_output == "prob":
            vuv = torch.sigmoid(vuv)

        return log_f0, vuv, rms


if __name__ == "__main__":
    dp = ConvDecoder(256, 256, 3, 0.1)
