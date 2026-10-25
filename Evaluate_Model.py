#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

MODEL_PATH = "models/checkpoints/best_model.pt"
DATASET_PATH = "data/labeled/dataset.pt"
BATCH_SIZE = 256


class ChessEvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x).squeeze(1)


def main():
    print("Loading dataset...")
    data = torch.load(DATASET_PATH)

    train_dataset = TensorDataset(data["train_boards"], data["train_labels"])
    val_dataset = TensorDataset(data["val_boards"], data["val_labels"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ChessEvalNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    loss_fn = nn.MSELoss()

    def compute_loss(loader, name):
        total_loss = 0.0
        total_abs_error = 0.0
        n = 0
        with torch.no_grad():
            for boards, labels in loader:
                predictions = model(boards)
                loss = loss_fn(predictions, labels)
                total_loss += loss.item() * boards.size(0)
                total_abs_error += (predictions - labels).abs().sum().item()
                n += boards.size(0)
        mse = total_loss / n
        mae = total_abs_error / n
        print(f"{name}: MSE={mse:.5f} | MAE={mae:.5f} | n={n}")
        return mse, mae

    # A naive baseline: always predicting 0.5 (i.e. "always equal").
    # If the model's MSE isn't meaningfully better than this, it hasn't learned much.
    baseline_mse = ((data["val_labels"] - 0.5) ** 2).mean().item()
    print(f"Naive baseline (always predict 0.5): MSE={baseline_mse:.5f}")
    print()

    compute_loss(train_loader, "Train")
    compute_loss(val_loader, "Val")


if __name__ == "__main__":
    main()

