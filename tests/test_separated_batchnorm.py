import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.separated_batchnorm import SeparatedBatchNorm1d

bn = SeparatedBatchNorm1d(
    num_features=128,
    max_length=200
)

x = torch.randn(64, 128)

y = bn(x, time_step=0)

print("Input shape :", x.shape)
print("Output shape:", y.shape)