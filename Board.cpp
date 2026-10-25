#include "Board.h"
#include <iostream>
#include <cctype>

Board::Board() { 
    setupStartingPosition(); 
}

bool Board::inBounds(int row, int col) {
    return row >= 0 && row < 8 && col >= 0 && col < 8;
}

void Board::setupStartingPosition() {
    for (auto& row : grid_)
        for (auto& sq : row)
            sq = Piece{};

    static const std::array<PieceType, 8> backRank = {
        PieceType::Rook, PieceType::Knight, PieceType::Bishop, PieceType::Queen,
        PieceType::King, PieceType::Bishop, PieceType::Knight, PieceType::Rook
    };

    for (int col = 0; col < 8; ++col) {
        grid_[0][col] = Piece{backRank[col], Color::Black};
        grid_[1][col] = Piece{PieceType::Pawn, Color::Black};
        grid_[6][col] = Piece{PieceType::Pawn, Color::White};
        grid_[7][col] = Piece{backRank[col], Color::White};
    }

    sideToMove_ = Color::White;
    whiteKingSideCastle_ = whiteQueenSideCastle_ = true;
    blackKingSideCastle_ = blackQueenSideCastle_ = true;
    enPassantRow_ = enPassantCol_ = -1;
}

char Board::pieceChar(const Piece& p) const {
    char c = '.';
    switch (p.type) {
        case PieceType::Pawn:   c = 'p'; break;
        case PieceType::Knight: c = 'n'; break;
        case PieceType::Bishop: c = 'b'; break;
        case PieceType::Rook:   c = 'r'; break;
        case PieceType::Queen:  c = 'q'; break;
        case PieceType::King:   c = 'k'; break;
        default: return '.';
    }
    return p.color == Color::White ? static_cast<char>(std::toupper(c)) : c;
}

void Board::print() const {
    std::cout << "\n  a b c d e f g h\n";
    for (int row = 0; row < 8; ++row) {
        std::cout << (8 - row) << " ";
        for (int col = 0; col < 8; ++col)
            std::cout << pieceChar(grid_[row][col]) << ' ';
        std::cout << (8 - row) << "\n";
    }
    std::cout << "  a b c d e f g h\n\n";
}

bool Board::parseSquare(const std::string& s, int& row, int& col) {
    if (s.size() != 2) return false;
    char file = s[0], rank = s[1];
    if (file < 'a' || file > 'h') return false;
    if (rank < '1' || rank > '8') return false;
    col = file - 'a';
    row = 8 - (rank - '0');
    return true;
}

std::string Board::squareToString(int row, int col) {
    std::string s;
    s += static_cast<char>('a' + col);
    s += static_cast<char>('0' + (8 - row));
    return s;
}

void Board::findKing(Color color, int& row, int& col) const {
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            if (grid_[r][c].type == PieceType::King && grid_[r][c].color == color) {
                row = r; col = c;
                return;
            }
    row = col = -1; // should never happen in a valid game
}

bool Board::isSquareAttacked(int row, int col, Color byColor) const {
    // Pawn attacks: a byColor pawn attacks diagonally "forward" for its color.
    int pawnDir = (byColor == Color::White) ? -1 : 1; // white pawns move toward row 0
    for (int dc : {-1, 1}) {
        int r = row - pawnDir; // square the attacking pawn would stand on
        int c = col + dc;
        if (inBounds(r, c) && grid_[r][c].type == PieceType::Pawn && grid_[r][c].color == byColor)
            return true;
    }

    static const std::vector<std::pair<int, int>> knightOffsets = {
        {1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}
    };
    for (auto [dr, dc] : knightOffsets) {
        int r = row + dr, c = col + dc;
        if (inBounds(r, c) && grid_[r][c].type == PieceType::Knight && grid_[r][c].color == byColor)
            return true;
    }

    static const std::vector<std::pair<int, int>> kingOffsets = {
        {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
    };
    for (auto [dr, dc] : kingOffsets) {
        int r = row + dr, c = col + dc;
        if (inBounds(r, c) && grid_[r][c].type == PieceType::King && grid_[r][c].color == byColor)
            return true;
    }

    static const std::vector<std::pair<int, int>> diagonals = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const std::vector<std::pair<int, int>> straights = {{1,0},{-1,0},{0,1},{0,-1}};

    for (auto [dr, dc] : diagonals) {
        int r = row + dr, c = col + dc;
        while (inBounds(r, c)) {
            const Piece& p = grid_[r][c];
            if (p.type != PieceType::None) {
                if (p.color == byColor && (p.type == PieceType::Bishop || p.type == PieceType::Queen))
                    return true;
                break;
            }
            r += dr; c += dc;
        }
    }
    for (auto [dr, dc] : straights) {
        int r = row + dr, c = col + dc;
        while (inBounds(r, c)) {
            const Piece& p = grid_[r][c];
            if (p.type != PieceType::None) {
                if (p.color == byColor && (p.type == PieceType::Rook || p.type == PieceType::Queen))
                    return true;
                break;
            }
            r += dr; c += dc;
        }
    }
    return false;
}

void Board::generatePawnMoves(int row, int col, std::vector<Move>& moves) const {
    const Piece& p = grid_[row][col];
    int dir = (p.color == Color::White) ? -1 : 1;
    int startRow = (p.color == Color::White) ? 6 : 1;
    int promoRow = (p.color == Color::White) ? 0 : 7;

    auto addWithPromotion = [&](int toRow, int toCol, bool isEP) {
        Move m; m.fromRow = row; m.fromCol = col; m.toRow = toRow; m.toCol = toCol;
        m.isEnPassant = isEP;
        if (toRow == promoRow) {
            for (PieceType promo : {PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight}) {
                Move pm = m; pm.promotion = promo;
                moves.push_back(pm);
            }
        } else {
            moves.push_back(m);
        }
    };

    // Single push
    int r1 = row + dir;
    if (inBounds(r1, col) && grid_[r1][col].type == PieceType::None) {
        addWithPromotion(r1, col, false);
        // Double push
        int r2 = row + 2 * dir;
        if (row == startRow && grid_[r2][col].type == PieceType::None) {
            Move m; m.fromRow = row; m.fromCol = col; m.toRow = r2; m.toCol = col;
            m.isDoublePawnPush = true;
            moves.push_back(m);
        }
    }

    // Captures
    for (int dc : {-1, 1}) {
        int r = row + dir, c = col + dc;
        if (!inBounds(r, c)) continue;
        const Piece& target = grid_[r][c];
        if (target.type != PieceType::None && target.color != p.color) {
            addWithPromotion(r, c, false);
        } else if (r == enPassantRow_ && c == enPassantCol_) {
            addWithPromotion(r, c, true);
        }
    }
}

void Board::generateKnightMoves(int row, int col, std::vector<Move>& moves) const {
    const Piece& p = grid_[row][col];
    static const std::vector<std::pair<int, int>> offsets = {
        {1,2},{2,1},{-1,2},{-2,1},{1,-2},{2,-1},{-1,-2},{-2,-1}
    };
    for (auto [dr, dc] : offsets) {
        int r = row + dr, c = col + dc;
        if (!inBounds(r, c)) continue;
        if (grid_[r][c].type == PieceType::None || grid_[r][c].color != p.color) {
            Move m; m.fromRow = row; m.fromCol = col; m.toRow = r; m.toCol = c;
            moves.push_back(m);
        }
    }
}

void Board::generateSlidingMoves(int row, int col,
                                  const std::vector<std::pair<int, int>>& directions,
                                  std::vector<Move>& moves) const {
    const Piece& p = grid_[row][col];
    for (auto [dr, dc] : directions) {
        int r = row + dr, c = col + dc;
        while (inBounds(r, c)) {
            if (grid_[r][c].type == PieceType::None) {
                Move m; m.fromRow = row; m.fromCol = col; m.toRow = r; m.toCol = c;
                moves.push_back(m);
            } else {
                if (grid_[r][c].color != p.color) {
                    Move m; m.fromRow = row; m.fromCol = col; m.toRow = r; m.toCol = c;
                    moves.push_back(m);
                }
                break;
            }
            r += dr; c += dc;
        }
    }
}

void Board::generateKingMoves(int row, int col, std::vector<Move>& moves) const {
    const Piece& p = grid_[row][col];
    static const std::vector<std::pair<int, int>> offsets = {
        {1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}
    };
    for (auto [dr, dc] : offsets) {
        int r = row + dr, c = col + dc;
        if (!inBounds(r, c)) continue;
        if (grid_[r][c].type == PieceType::None || grid_[r][c].color != p.color) {
            Move m; m.fromRow = row; m.fromCol = col; m.toRow = r; m.toCol = c;
            moves.push_back(m);
        }
    }

    Color enemy = opposite(p.color);
    int homeRow = (p.color == Color::White) ? 7 : 0;
    if (row != homeRow || col != 4) return; // king must be on its starting square

    bool kingSideRight = (p.color == Color::White) ? whiteKingSideCastle_ : blackKingSideCastle_;
    bool queenSideRight = (p.color == Color::White) ? whiteQueenSideCastle_ : blackQueenSideCastle_;

    if (isSquareAttacked(row, col, enemy)) return; // can't castle out of check

    if (kingSideRight &&
        grid_[homeRow][5].type == PieceType::None && grid_[homeRow][6].type == PieceType::None &&
        grid_[homeRow][7].type == PieceType::Rook && grid_[homeRow][7].color == p.color &&
        !isSquareAttacked(homeRow, 5, enemy) && !isSquareAttacked(homeRow, 6, enemy)) {
        Move m; m.fromRow = row; m.fromCol = col; m.toRow = homeRow; m.toCol = 6;
        m.isCastleKingSide = true;
        moves.push_back(m);
    }
    if (queenSideRight &&
        grid_[homeRow][1].type == PieceType::None && grid_[homeRow][2].type == PieceType::None &&
        grid_[homeRow][3].type == PieceType::None &&
        grid_[homeRow][0].type == PieceType::Rook && grid_[homeRow][0].color == p.color &&
        !isSquareAttacked(homeRow, 3, enemy) && !isSquareAttacked(homeRow, 2, enemy)) {
        Move m; m.fromRow = row; m.fromCol = col; m.toRow = homeRow; m.toCol = 2;
        m.isCastleQueenSide = true;
        moves.push_back(m);
    }
}

std::vector<Move> Board::generatePseudoLegalMoves(Color color) const {
    std::vector<Move> moves;
    static const std::vector<std::pair<int, int>> diagonals = {{1,1},{1,-1},{-1,1},{-1,-1}};
    static const std::vector<std::pair<int, int>> straights = {{1,0},{-1,0},{0,1},{0,-1}};

    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            const Piece& p = grid_[r][c];
            if (p.color != color) continue;
            switch (p.type) {
                case PieceType::Pawn:   generatePawnMoves(r, c, moves); break;
                case PieceType::Knight: generateKnightMoves(r, c, moves); break;
                case PieceType::Bishop: generateSlidingMoves(r, c, diagonals, moves); break;
                case PieceType::Rook:   generateSlidingMoves(r, c, straights, moves); break;
                case PieceType::Queen: {
                    std::vector<std::pair<int, int>> all = diagonals;
                    all.insert(all.end(), straights.begin(), straights.end());
                    generateSlidingMoves(r, c, all, moves);
                    break;
                }
                case PieceType::King: generateKingMoves(r, c, moves); break;
                default: break;
            }
        }
    }
    return moves;
}

void Board::applyMoveRaw(const Move& move) {
    Piece moving = grid_[move.fromRow][move.fromCol];

    // Reset en passant target; re-set below only for a fresh double push.
    enPassantRow_ = enPassantCol_ = -1;

    if (move.isEnPassant) {
        grid_[move.toRow][move.toCol] = moving;
        grid_[move.fromRow][move.fromCol] = Piece{};
        // Captured pawn sits beside the destination, on the origin's row.
        grid_[move.fromRow][move.toCol] = Piece{};
    } else if (move.isCastleKingSide || move.isCastleQueenSide) {
        grid_[move.toRow][move.toCol] = moving;
        grid_[move.fromRow][move.fromCol] = Piece{};
        int homeRow = move.fromRow;
        if (move.isCastleKingSide) {
            grid_[homeRow][5] = grid_[homeRow][7];
            grid_[homeRow][7] = Piece{};
        } else {
            grid_[homeRow][3] = grid_[homeRow][0];
            grid_[homeRow][0] = Piece{};
        }
    } else {
        grid_[move.toRow][move.toCol] = moving;
        grid_[move.fromRow][move.fromCol] = Piece{};
        if (move.promotion != PieceType::None)
            grid_[move.toRow][move.toCol].type = move.promotion;
        if (move.isDoublePawnPush) {
            enPassantRow_ = (move.fromRow + move.toRow) / 2;
            enPassantCol_ = move.fromCol;
        }
    }

    // Update castling rights if king or rook moved, or a rook was captured on its home square.
    if (moving.type == PieceType::King) {
        if (moving.color == Color::White) whiteKingSideCastle_ = whiteQueenSideCastle_ = false;
        else blackKingSideCastle_ = blackQueenSideCastle_ = false;
    }
    auto clearRookRight = [&](int row, int col) {
        if (row == 7 && col == 0) whiteQueenSideCastle_ = false;
        if (row == 7 && col == 7) whiteKingSideCastle_ = false;
        if (row == 0 && col == 0) blackQueenSideCastle_ = false;
        if (row == 0 && col == 7) blackKingSideCastle_ = false;
    };
    clearRookRight(move.fromRow, move.fromCol);
    clearRookRight(move.toRow, move.toCol);

    sideToMove_ = opposite(sideToMove_);
}

std::vector<Move> Board::generateLegalMoves(Color color) const {
    std::vector<Move> pseudo = generatePseudoLegalMoves(color);
    std::vector<Move> legal;
    for (const Move& m : pseudo) {
        Board copy = *this; // deep copy: plain arrays/POD members only
        copy.applyMoveRaw(m);
        int kr, kc;
        copy.findKing(color, kr, kc);
        if (kr == -1 || !copy.isSquareAttacked(kr, kc, opposite(color)))
            legal.push_back(m);
    }
    return legal;
}

bool Board::isInCheck(Color color) const {
    int kr, kc;
    findKing(color, kr, kc);
    if (kr == -1) return false;
    return isSquareAttacked(kr, kc, opposite(color));
}

bool Board::hasAnyLegalMove(Color color) const {
    return !generateLegalMoves(color).empty();
}

bool Board::isCheckmate(Color color) const {
    return isInCheck(color) && !hasAnyLegalMove(color);
}

bool Board::isStalemate(Color color) const {
    return !isInCheck(color) && !hasAnyLegalMove(color);
}

bool Board::makeMove(const Move& move) {
    std::vector<Move> legal = generateLegalMoves(sideToMove_);
    for (const Move& m : legal) {
        if (m.sameEndpoints(move)) {
            applyMoveRaw(m);
            return true;
        }
    }
    return false;
}

bool Board::tryUserMove(const std::string& input, std::string& errorMsg) {
    if (input.size() < 4 || input.size() > 5) {
        errorMsg = "Enter a move like e2e4 (or e7e8q for promotion).";
        return false;
    }
    int fr, fc, tr, tc;
    if (!parseSquare(input.substr(0, 2), fr, fc) || !parseSquare(input.substr(2, 2), tr, tc)) {
        errorMsg = "Could not parse squares.";
        return false;
    }
    PieceType promo = PieceType::None;
    if (input.size() == 5) {
        switch (std::tolower(input[4])) {
            case 'q': promo = PieceType::Queen; break;
            case 'r': promo = PieceType::Rook; break;
            case 'b': promo = PieceType::Bishop; break;
            case 'n': promo = PieceType::Knight; break;
            default:
                errorMsg = "Unknown promotion piece (use q, r, b, or n).";
                return false;
        }
    }

    std::vector<Move> legal = generateLegalMoves(sideToMove_);
    for (const Move& m : legal) {
        if (m.fromRow == fr && m.fromCol == fc && m.toRow == tr && m.toCol == tc) {
            if (m.promotion != PieceType::None && m.promotion != promo) continue;
            applyMoveRaw(m);
            return true;
        }
    }
    errorMsg = "That's not a legal move.";
    return false;
}
