import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparatedBatchNorm1d(nn.Module):
    """
    Batch Normalization for Recurrent Neural Networks.

    BN-LSTM uses different running statistics for each timestep.

    Reference:
    Recurrent Batch Normalization
    (Cooijmans et al., 2017)
    """

    def __init__(
        self,
        num_features: int,
        max_length: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()

        self.num_features = num_features
        self.max_length = max_length
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        # Separate running statistics per timestep
        for t in range(max_length):
            self.register_buffer(
                f"running_mean_{t}",
                torch.zeros(num_features)
            )

            self.register_buffer(
                f"running_var_{t}",
                torch.ones(num_features)
            )

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

        for t in range(self.max_length):
            getattr(self, f"running_mean_{t}").zero_()
            getattr(self, f"running_var_{t}").fill_(1)

    def forward(self, x, time_step: int):
        """
        Parameters
        ----------
        x : Tensor
            Shape: [batch_size, num_features]

        time_step : int
            Current timestep in sequence.
        """

        if time_step >= self.max_length:
            time_step = self.max_length - 1

        running_mean = getattr(
            self,
            f"running_mean_{time_step}"
        )

        running_var = getattr(
            self,
            f"running_var_{time_step}"
        )

        return F.batch_norm(
            x,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

    def extra_repr(self):
        return (
            f"num_features={self.num_features}, "
            f"max_length={self.max_length}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}"
        )