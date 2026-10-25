#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import chess
import json
import math
import numpy as np
import torch

# ---- Configuration ----
INPUT_PATH = "data/labeled/positions.jsonl"
OUTPUT_PATH = "data/labeled/dataset.pt"
VAL_FRACTION = 0.1
SIGMOID_SCALE = 400.0  # controls how sharply eval_cp squashes into 0-1; 400 is a common choice

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def fen_to_tensor(fen):
    """Encodes a board position as a (12, 8, 8) float32 array of piece-presence planes."""
    board = chess.Board(fen)
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)  # rank 8 -> row 0, rank 1 -> row 7
        col = chess.square_file(square)
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        planes[plane, row, col] = 1.0

    return planes


def normalize_eval(eval_cp):
    """Squashes a centipawn score (White's perspective) into a smooth 0-1 range."""
    return 1.0 / (1.0 + math.exp(-eval_cp / SIGMOID_SCALE))


def main():
    boards = []
    labels = []
    skipped = 0

    with open(INPUT_PATH, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                tensor = fen_to_tensor(record["fen"])
                label = normalize_eval(record["eval_cp"])
                boards.append(tensor)
                labels.append(label)
            except Exception:
                skipped += 1
                continue

            if line_num % 50000 == 0:
                print(f"{line_num} lines processed, {len(boards)} kept, {skipped} skipped")

    print(f"Total usable positions: {len(boards)} (skipped {skipped})")

    boards = np.stack(boards)          # shape: (N, 12, 8, 8)
    labels = np.array(labels, dtype=np.float32)  # shape: (N,)

    # Shuffle before splitting so train/val aren't biased by file order (e.g. game order)
    rng = np.random.default_rng(seed=42)
    perm = rng.permutation(len(boards))
    boards = boards[perm]
    labels = labels[perm]

    split_idx = int(len(boards) * (1 - VAL_FRACTION))
    train_boards, val_boards = boards[:split_idx], boards[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    dataset = {
        "train_boards": torch.from_numpy(train_boards),
        "train_labels": torch.from_numpy(train_labels),
        "val_boards": torch.from_numpy(val_boards),
        "val_labels": torch.from_numpy(val_labels),
    }

    torch.save(dataset, OUTPUT_PATH)
    print(f"Saved dataset to {OUTPUT_PATH}")
    print(f"Train: {len(train_boards)} positions | Val: {len(val_boards)} positions")


if __name__ == "__main__":
    main()

