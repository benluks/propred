from typing import List, Optional, Tuple

from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Batch:
    values: torch.Tensor
    durations: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor
    spk_ids: Optional[torch.Tensor] = None


class CollateValuesDurations:
    def __init__(self, pad_value: int = 0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Batch:

        if len(batch[0]) == 2:
            values_list, durs_list = zip(*batch)
            spk_ids_list = None
        else:
            values_list, durs_list, spk_ids_list = zip(*batch)

        values = pad_sequence(
            values_list, batch_first=True, padding_value=self.pad_value
        )
        durations = pad_sequence(
            durs_list, batch_first=True, padding_value=self.pad_value
        )
        spk_ids = (
            torch.tensor(spk_ids_list).to(torch.long)
            if spk_ids_list is not None
            else None
        )

        mask = (values != self.pad_value).float()
        lengths = mask.sum(-1).to(torch.long)

        return Batch(
            values=values.to(torch.long),
            durations=durations.to(torch.long),
            spk_ids=spk_ids,
            lengths=lengths,
            mask=mask,
        )
