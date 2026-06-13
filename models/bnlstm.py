import torch
import torch.nn as nn

from models.separated_batchnorm import SeparatedBatchNorm1d


class BNLSTMCell(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        max_length
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input → Gates
        self.weight_ih = nn.Parameter(
            torch.Tensor(
                input_size,
                4 * hidden_size
            )
        )

        # Hidden → Gates
        self.weight_hh = nn.Parameter(
            torch.Tensor(
                hidden_size,
                4 * hidden_size
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(
                4 * hidden_size
            )
        )

        self.bn_ih = SeparatedBatchNorm1d(
            4 * hidden_size,
            max_length
        )

        self.bn_hh = SeparatedBatchNorm1d(
            4 * hidden_size,
            max_length
        )

        self.bn_c = SeparatedBatchNorm1d(
            hidden_size,
            max_length
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.orthogonal_(
            self.weight_hh
        )

        nn.init.xavier_uniform_(
            self.weight_ih
        )

    def forward(
        self,
        x,
        hx,
        cx,
        time_step
    ):

        raise NotImplementedError