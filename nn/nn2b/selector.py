import os
import numpy as np

from torch import nn
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

os.makedirs("data", exist_ok=True)
base_data_filepath = os.path.join("data", "selector.csv")


class SimpleDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def append_to_file(x: list[float], y: list[int]):
    with open(base_data_filepath, "a") as f:
        f.write(
            ",".join([str(v) for v in x]) + "," + ",".join([str(v) for v in y]) + "\n"
        )


def load_data(batch_size: int) -> DataLoader:
    data = np.loadtxt(base_data_filepath, delimiter=",")
    print(f"data shape: {data.shape}")
    x, y = data[:, :-6], data[:, -6:]

    return DataLoader(
        SimpleDataset(x, y),
        batch_size,
    )


class SelectorNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_cardinality: int):
        super(SelectorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_cardinality)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_selector_nn():
    n_epochs = 200

    model = SelectorNN(input_dim=12, hidden_dim=48, output_cardinality=6)
    dataloader = load_data(batch_size=32)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    for epoch in trange(n_epochs):
        model.train()

        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "selector_nn.pth")
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train_selector_nn()
