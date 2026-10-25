#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import os

# ---- Configuration ----
DATASET_PATH = "data/labeled/dataset.pt"
CHECKPOINT_DIR = "models/checkpoints"
BEST_MODEL_PATH = "models/checkpoints/best_model.pt"

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ChessEvalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch, 12, 8, 8)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)

        # After 3 conv layers with padding=1, spatial size stays 8x8
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
        return torch.sigmoid(x).squeeze(1)  # output in 0-1 range, matches normalized eval labels


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    for boards, labels in loader:
        boards, labels = boards.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        predictions = model(boards)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * boards.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for boards, labels in loader:
            boards, labels = boards.to(DEVICE), labels.to(DEVICE)
            predictions = model(boards)
            loss = loss_fn(predictions, labels)
            total_loss += loss.item() * boards.size(0)
    return total_loss / len(loader.dataset)


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading dataset...")
    data = torch.load(DATASET_PATH)

    train_dataset = TensorDataset(data["train_boards"], data["train_labels"])
    val_dataset = TensorDataset(data["val_boards"], data["val_labels"])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train positions: {len(train_dataset)} | Val positions: {len(val_dataset)}")

    model = ChessEvalNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)
        elapsed = time.time() - start

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.5f} | "
              f"val_loss={val_loss:.5f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model saved (val_loss={val_loss:.5f})")

    print(f"Training complete. Best val_loss: {best_val_loss:.5f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()

