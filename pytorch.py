"""conda create --name segfault matplotlib pandas pytorch cpuonly
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

START_DATE = pd.Timestamp("2020-01-01")
N_PERIODS = 4
N_EXAMPLES_PER_BATCH = 32
N_FEATURES = 16


class MyDataset(Dataset):
    def __init__(self, n_batches_per_epoch: int):
        self.n_batches_per_epoch = n_batches_per_epoch

    def __len__(self):
        return self.n_batches_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(N_EXAMPLES_PER_BATCH, N_FEATURES, 1)
        y = torch.rand(N_EXAMPLES_PER_BATCH, 1)
        return x, y


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.nn = nn.Linear(in_features=N_FEATURES, out_features=1)

    def forward(self, x):
        x = self.flatten(x)
        return self.nn(x)


dataloader = DataLoader(
    MyDataset(n_batches_per_epoch=1024),
    batch_size=None,
    num_workers=2,
)


# Training loop
model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for batch_idx, (x, y) in enumerate(dataloader):
    pred = model(x)
    loss = loss_fn(pred, y)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Plot random data
    fig, axes = plt.subplots(ncols=32)
    for ax in axes:
        start_date = START_DATE + pd.Timedelta(
            np.random.randint(low=0, high=2000), unit="hour")
        range = pd.date_range(start_date, periods=N_PERIODS, freq="30 min")
        ax.plot(range, np.random.randint(low=0, high=10, size=N_PERIODS))
    plt.close(fig)
