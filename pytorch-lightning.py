import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


START_DATE = pd.Timestamp("2020-01-01")
N_PERIODS = 4
N_EXAMPLES_PER_BATCH = 32
N_FEATURES = 1
SEGFAULT = True


class MyDataset(Dataset):
    def __init__(self, n_batches_per_epoch: int):
        self.n_batches_per_epoch = n_batches_per_epoch

    def __len__(self):
        return self.n_batches_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(N_EXAMPLES_PER_BATCH, N_FEATURES, 1)
        y = torch.rand(N_EXAMPLES_PER_BATCH, 1)
        return x, y


class LitNeuralNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.nn = nn.Linear(in_features=N_FEATURES, out_features=1)

    def forward(self, x):
        x = self.flatten(x)
        return self.nn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        plot_random_data()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def plot_random_data():
    # Plot random data
    fig, axes = plt.subplots(ncols=32)
    for ax in axes:
        range = pd.date_range(START_DATE, periods=N_PERIODS, freq="30 min")
        if not SEGFAULT:
            range = mdates.date2num(range)
        ax.plot(range, np.random.randint(low=0, high=10, size=N_PERIODS))
    plt.close(fig)


dataloader = DataLoader(
    MyDataset(n_batches_per_epoch=1024),
    batch_size=None,
    num_workers=2,
)

model = LitNeuralNetwork()
trainer = pl.Trainer()
trainer.fit(model=model, train_dataloader=dataloader)
