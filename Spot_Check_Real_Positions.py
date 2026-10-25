#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "models/checkpoints/best_model.pt"
DATASET_PATH = "data/labeled/dataset.pt"
NUM_SAMPLES = 15


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
    data = torch.load(DATASET_PATH)
    val_boards = data["val_boards"]
    val_labels = data["val_labels"]

    model = ChessEvalNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Sort validation examples by label so we see a spread across the eval spectrum:
    # some clearly lost, some equal, some clearly won.
    sorted_indices = torch.argsort(val_labels)
    step = len(sorted_indices) // NUM_SAMPLES
    sample_indices = sorted_indices[::step][:NUM_SAMPLES]

    print(f"{'True label':>12s} | {'Prediction':>12s} | {'Abs error':>10s}")
    print("-" * 42)

    with torch.no_grad():
        for idx in sample_indices:
            board = val_boards[idx].unsqueeze(0)
            true_label = val_labels[idx].item()
            prediction = model(board).item()
            error = abs(prediction - true_label)
            print(f"{true_label:12.4f} | {prediction:12.4f} | {error:10.4f}")


if __name__ == "__main__":
    main()

