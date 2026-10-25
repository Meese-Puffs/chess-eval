#pragma once
#include <array>
#include <vector>
#include <string>

enum class PieceType { None, Pawn, Knight, Bishop, Rook, Queen, King };
enum class Color { None, White, Black };

struct Piece {
    PieceType type = PieceType::None;
    Color color = Color::None;
};

struct Move {
    int fromRow = -1, fromCol = -1, toRow = -1, toCol = -1;
    PieceType promotion = PieceType::None;
    bool isCastleKingSide = false;
    bool isCastleQueenSide = false;
    bool isEnPassant = false;
    bool isDoublePawnPush = false;

    bool sameEndpoints(const Move& other) const {
        return fromRow == other.fromRow && fromCol == other.fromCol &&
               toRow == other.toRow && toCol == other.toCol &&
               promotion == other.promotion;
    }
};

// 8x8 grid. grid[0] = rank 8 (Black's back rank), grid[7] = rank 1 (White's back rank).
// grid[row][0] = file a, grid[row][7] = file h.
class Board {
public:
    Board();

    void setupStartingPosition();
    void print() const;

    Color sideToMove() const { return sideToMove_; }
    Piece at(int row, int col) const { return grid_[row][col]; }

    // All fully legal moves for `color` (own king can never end up in check).
    std::vector<Move> generateLegalMoves(Color color) const;

    // Applies `move` if it is legal for the side to move; switches turns.
    // Returns false (no state change) if the move is not legal.
    bool makeMove(const Move& move);

    bool isInCheck(Color color) const;
    bool isCheckmate(Color color) const;
    bool isStalemate(Color color) const;
    bool hasAnyLegalMove(Color color) const;

    // Parses input like "e2e4" or "e7e8q" (promotion suffix q/r/b/n),
    // matches it against the legal move list, and applies it if found.
    bool tryUserMove(const std::string& input, std::string& errorMsg);

    static bool parseSquare(const std::string& s, int& row, int& col);
    static std::string squareToString(int row, int col);

private:
    std::array<std::array<Piece, 8>, 8> grid_{};
    Color sideToMove_ = Color::White;

    bool whiteKingSideCastle_ = true;
    bool whiteQueenSideCastle_ = true;
    bool blackKingSideCastle_ = true;
    bool blackQueenSideCastle_ = true;

    // Square a pawn can capture into via en passant this move, or (-1,-1) if none.
    int enPassantRow_ = -1;
    int enPassantCol_ = -1;

    std::vector<Move> generatePseudoLegalMoves(Color color) const;
    void generatePawnMoves(int row, int col, std::vector<Move>& moves) const;
    void generateKnightMoves(int row, int col, std::vector<Move>& moves) const;
    void generateSlidingMoves(int row, int col,
                               const std::vector<std::pair<int, int>>& directions,
                               std::vector<Move>& moves) const;
    void generateKingMoves(int row, int col, std::vector<Move>& moves) const;

    bool isSquareAttacked(int row, int col, Color byColor) const;
    void findKing(Color color, int& row, int& col) const;

    // Applies a move to the grid/state with no legality checking.
    void applyMoveRaw(const Move& move);

    static bool inBounds(int row, int col);
    static Color opposite(Color c) { return c == Color::White ? Color::Black : Color::White; }
    char pieceChar(const Piece& p) const;
};
