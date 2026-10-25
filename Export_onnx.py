#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MODEL_PATH = "models/checkpoints/best_model.pt"
ONNX_PATH = "models/checkpoints/chess_eval.onnx"


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
    # ---- Load the trained model ----
    model = ChessEvalNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # ---- Export to ONNX ----
    # A dummy input matching the real input shape: (batch, 12 planes, 8x8 board)
    dummy_input = torch.randn(1, 12, 8, 8)

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["board_planes"],
        output_names=["eval_score"],
        dynamic_axes={
            "board_planes": {0: "batch_size"},  # allow variable batch sizes
            "eval_score": {0: "batch_size"},
        },
        opset_version=17,
    )
    print(f"Exported ONNX model to {ONNX_PATH}")

    # ---- Verify: compare PyTorch output vs ONNX Runtime output on the same input ----
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nonnxruntime not installed, skipping verification.")
        print("Install it with: python -m pip install onnxruntime")
        return

    test_input = torch.randn(4, 12, 8, 8)  # a small batch to also confirm dynamic batching works

    with torch.no_grad():
        torch_output = model(test_input).numpy()

    ort_session = ort.InferenceSession(ONNX_PATH)
    onnx_output = ort_session.run(
        None, {"board_planes": test_input.numpy()}
    )[0]

    max_diff = np.abs(torch_output - onnx_output).max()
    print(f"\nMax difference between PyTorch and ONNX outputs: {max_diff:.8f}")

    if max_diff < 1e-4:
        print("Verification PASSED: ONNX model matches PyTorch model.")
    else:
        print("Verification WARNING: outputs differ more than expected. Investigate before relying on the ONNX model.")


if __name__ == "__main__":
    main()

