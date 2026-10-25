#!/usr/bin/env python
# coding: utf-8

# In[3]:


import chess
import chess.pgn
import chess.engine
import random
import json
import os

# ---- Configuration: update these two paths for your machine ----
PGN_PATH = r"C:\lichess_elite_2021-10.pgn"
STOCKFISH_PATH = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"

OUTPUT_PATH = "data/labeled/positions.jsonl"

MAX_GAMES = 100000          # how many games to sample from the file
POSITIONS_PER_GAME = 8     # sampled plies per game
MIN_PLY = 10                # skip early opening moves
DEPTH = 14                  # Stockfish search depth (speed/quality tradeoff)


def sample_plies(game_length, count, min_ply):
    if game_length <= min_ply:
        return []
    candidates = range(min_ply, game_length)
    return sorted(random.sample(list(candidates), min(count, len(candidates))))


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    games_processed = 0
    positions_written = 0

    with open(PGN_PATH, encoding="utf-8", errors="ignore") as pgn_file, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:

        while games_processed < MAX_GAMES:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # end of file reached

            moves = list(game.mainline_moves())
            if len(moves) < MIN_PLY + 5:
                games_processed += 1
                continue

            target_plies = sample_plies(len(moves), POSITIONS_PER_GAME, MIN_PLY)
            target_set = set(target_plies)

            board = game.board()
            ply_index = 0

            for move in moves:
                board.push(move)
                ply_index += 1

                if ply_index in target_set:
                    info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                    score = info["score"].white().score(mate_score=100000)
                    if score is not None:
                        record = {
                            "fen": board.fen(),
                            "eval_cp": score
                        }
                        out_file.write(json.dumps(record) + "\n")
                        positions_written += 1

            games_processed += 1
            if games_processed % 500 == 0:
                print(f"{games_processed} games processed, {positions_written} positions labeled")

    engine.quit()
    print(f"Done. {positions_written} positions written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


# In[ ]:




