#pragma once
#include "Board.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <string>
#include <stdexcept>

// Loads the exported chess_eval.onnx model and evaluates Board positions,
// producing the same 0-1 "White winning probability" score used in training.
class ChessEvaluator {
public:
    explicit ChessEvaluator(const std::wstring& modelPath)
        : env_(ORT_LOGGING_LEVEL_WARNING, "ChessEval"),
        session_(env_, modelPath.c_str(), Ort::SessionOptions{ nullptr }),
        memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    }

    // Returns a score in [0, 1]: 0 = Black winning, 1 = White winning, 0.5 = equal.
    float evaluate(const Board& board) {
        std::array<float, 12 * 8 * 8> input{};
        encodeBoard(board, input);

        std::array<int64_t, 4> inputShape = { 1, 12, 8, 8 };

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo_, input.data(), input.size(),
            inputShape.data(), inputShape.size());

        const char* inputNames[] = { "board_planes" };
        const char* outputNames[] = { "eval_score" };

        auto outputTensors = session_.Run(
            Ort::RunOptions{ nullptr },
            inputNames, &inputTensor, 1,
            outputNames, 1);

        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        return outputData[0];
    }

private:
    Ort::Env env_;
    Ort::Session session_;
    const Ort::MemoryInfo memoryInfo_;

    // Maps (PieceType, Color) to the plane index used in training.
    // Must exactly match PIECE_TO_PLANE in build_dataset.py.
    static int planeIndex(PieceType type, Color color) {
        int base;
        switch (type) {
        case PieceType::Pawn:   base = 0; break;
        case PieceType::Knight: base = 1; break;
        case PieceType::Bishop: base = 2; break;
        case PieceType::Rook:   base = 3; break;
        case PieceType::Queen:  base = 4; break;
        case PieceType::King:   base = 5; break;
        default: throw std::runtime_error("planeIndex called with PieceType::None");
        }
        return (color == Color::White) ? base : base + 6;
    }

    // Fills a flat (12, 8, 8) row-major float buffer matching fen_to_tensor() in Python:
    // index = plane * 64 + row * 8 + col
    static void encodeBoard(const Board& board, std::array<float, 12 * 8 * 8>& out) {
        out.fill(0.0f);
        for (int row = 0; row < 8; ++row) {
            for (int col = 0; col < 8; ++col) {
                Piece p = board.at(row, col);
                if (p.type == PieceType::None) continue;
                int plane = planeIndex(p.type, p.color);
                out[plane * 64 + row * 8 + col] = 1.0f;
            }
        }
    }
};