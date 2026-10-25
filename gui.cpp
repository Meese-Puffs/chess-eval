#include "Board.h"
#include "ChessEvaluator.h"
#include <SFML/Graphics.hpp>
#include <optional>
#include <string>
#if defined(_WIN32)
#include <Windows.h>
#endif

namespace {

    constexpr int SQUARE = 80;
    constexpr int BOARD_PIXELS = SQUARE * 8;
    constexpr int STATUS_HEIGHT = 50;
    constexpr int EVAL_BAR_WIDTH = 30;
    constexpr int BOARD_OFFSET_X = EVAL_BAR_WIDTH; // board is shifted right to make room for the bar

    const sf::Color LIGHT_SQUARE(240, 217, 181);
    const sf::Color DARK_SQUARE(181, 136, 99);
    const sf::Color SELECTED_OVERLAY(246, 246, 105, 180);
    const sf::Color LEGAL_DOT(20, 90, 20, 140);
    const sf::Color LAST_MOVE_OVERLAY(170, 200, 100, 140);
    const sf::Color WHITE_PIECE_FILL(250, 250, 250);
    const sf::Color BLACK_PIECE_FILL(40, 40, 40);
    const sf::Color BOARD_BG(30, 30, 30);
    const sf::Color CHECK_OVERLAY(220, 30, 30, 200);
    const sf::Color CHECKMATE_OVERLAY(10, 10, 10, 230);
    const sf::Color EVAL_BAR_BG(25, 25, 25);
    const sf::Color EVAL_BAR_WHITE(235, 235, 235);

    char pieceLetter(PieceType t) {
        switch (t) {
        case PieceType::King:   return 'K';
        case PieceType::Queen:  return 'Q';
        case PieceType::Rook:   return 'R';
        case PieceType::Bishop: return 'B';
        case PieceType::Knight: return 'N';
        case PieceType::Pawn:   return 'P';
        default: return '?';
        }
    }

    std::string findFont() {
#if defined(_WIN32)
        const char* candidates[] = {
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\segoeuib.ttf",
            "C:\\Windows\\Fonts\\segoeui.ttf",
        };
#else
        const char* candidates[] = {
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        };
#endif
        for (const char* path : candidates) {
            sf::Font test;
            if (test.loadFromFile(path)) return path;
        }
        return "";
    }

    // Draws one piece centered at (cx, cy) as a filled circle with its letter on top.
    void drawPiece(sf::RenderWindow& window, const sf::Font& font, const Piece& piece,
        float cx, float cy) {
        float radius = SQUARE * 0.38f;

        sf::CircleShape body(radius);
        body.setOrigin(radius, radius);
        body.setPosition(cx, cy);
        body.setFillColor(piece.color == Color::White ? WHITE_PIECE_FILL : BLACK_PIECE_FILL);
        body.setOutlineThickness(2.f);
        body.setOutlineColor(piece.color == Color::White ? sf::Color(60, 60, 60) : sf::Color(220, 220, 220));
        window.draw(body);

        sf::Text label;
        label.setFont(font);
        label.setString(std::string(1, pieceLetter(piece.type)));
        label.setCharacterSize(static_cast<unsigned int>(SQUARE * 0.4f));
        label.setFillColor(piece.color == Color::White ? sf::Color::Black : sf::Color::White);
        sf::FloatRect bounds = label.getLocalBounds();
        label.setOrigin(bounds.left + bounds.width / 2.f, bounds.top + bounds.height / 2.f);
        label.setPosition(cx, cy);
        window.draw(label);
    }

    struct Selection {
        bool active = false;
        int row = -1, col = -1;
        std::vector<Move> legalMoves; // legal moves originating from the selected square
    };

    // x/y here are raw window pixel coordinates; the board itself starts at BOARD_OFFSET_X.
    bool pixelToSquare(int x, int y, int& row, int& col) {
        int boardX = x - BOARD_OFFSET_X;
        if (boardX < 0 || y < 0 || boardX >= BOARD_PIXELS || y >= BOARD_PIXELS) return false;
        col = boardX / SQUARE;
        row = y / SQUARE;
        return true;
    }

    bool hasLegalMoveTo(const Selection& sel, int row, int col) {
        for (const Move& m : sel.legalMoves)
            if (m.toRow == row && m.toCol == col) return true;
        return false;
    }

    bool findKingSquare(const Board& board, Color color, int& row, int& col) {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                Piece p = board.at(r, c);
                if (p.type == PieceType::King && p.color == color) {
                    row = r; col = c;
                    return true;
                }
            }
        }
        return false;
    }

} // namespace

int main() {
    Board board;
    Selection selection;
    Move lastMove; // for highlighting; lastMove.fromRow == -1 means "none yet"
    lastMove.fromRow = -1;

    // Eval score: 0 = Black winning, 1 = White winning, 0.5 = equal.
    // Update this path to wherever chess_eval.onnx actually lives on your machine.
    float evalScore = 0.5f;
    ChessEvaluator* evaluatorPtr = nullptr;
    try {
        evaluatorPtr = new ChessEvaluator(L"C:\\Users\\whati\\source\\repos\\Project_C\\Project_C\\models\\checkpoints\\chess_eval.onnx");
        evalScore = evaluatorPtr->evaluate(board); // initial read for the starting position
    }
    catch (const std::exception& e) {
        MessageBoxA(nullptr, e.what(), "Failed to load chess_eval.onnx", MB_OK | MB_ICONERROR);
        return 1;
    }
    ChessEvaluator& evaluator = *evaluatorPtr;

    std::string fontPath = findFont();
    sf::Font font;
    if (!fontPath.empty()) font.loadFromFile(fontPath);

    sf::RenderWindow window(sf::VideoMode(EVAL_BAR_WIDTH + BOARD_PIXELS, BOARD_PIXELS + STATUS_HEIGHT),
        "Chess", sf::Style::Titlebar | sf::Style::Close);
    window.setFramerateLimit(60);

    bool dragging = false;
    sf::Vector2i mousePos;
    std::string statusMessage;

    auto refreshSelectionMoves = [&]() {
        selection.legalMoves.clear();
        if (!selection.active) return;
        std::vector<Move> all = board.generateLegalMoves(board.sideToMove());
        for (const Move& m : all)
            if (m.fromRow == selection.row && m.fromCol == selection.col)
                selection.legalMoves.push_back(m);
        };

    auto attemptMove = [&](int fr, int fc, int tr, int tc) {
        std::string moveStr = Board::squareToString(fr, fc) + Board::squareToString(tr, tc);
        // Auto-promote to queen when a pawn reaches the back rank; a promotion
        // picker (choosing Q/R/B/N) is a natural follow-up enhancement.
        Piece moving = board.at(fr, fc);
        if (moving.type == PieceType::Pawn && (tr == 0 || tr == 7))
            moveStr += 'q';

        std::string err;
        if (board.tryUserMove(moveStr, err)) {
            lastMove.fromRow = fr; lastMove.fromCol = fc;
            lastMove.toRow = tr; lastMove.toCol = tc;
            statusMessage.clear();
            evalScore = evaluator.evaluate(board); // refresh eval bar after every successful move
        }
        else {
            statusMessage = err;
        }
        };

    auto onPress = [&](int x, int y) {
        int row, col;
        if (!pixelToSquare(x, y, row, col)) return;

        if (selection.active) {
            if (row == selection.row && col == selection.col) {
                selection.active = false; // clicked same square: deselect
                return;
            }
            Piece target = board.at(row, col);
            if (target.color == board.sideToMove()) {
                selection.row = row; selection.col = col; selection.active = true;
                refreshSelectionMoves();
                dragging = true;
                return;
            }
            attemptMove(selection.row, selection.col, row, col);
            selection.active = false;
            return;
        }

        Piece p = board.at(row, col);
        if (p.color == board.sideToMove()) {
            selection.row = row; selection.col = col; selection.active = true;
            refreshSelectionMoves();
            dragging = true;
        }
        };

    auto onRelease = [&](int x, int y) {
        if (!dragging) return;
        dragging = false;
        int row, col;
        if (!pixelToSquare(x, y, row, col)) return; // released off-board: keep selection
        if (row == selection.row && col == selection.col) return; // click-style: keep selection

        Piece target = board.at(row, col);
        if (target.color == board.sideToMove()) {
            selection.row = row; selection.col = col; selection.active = true;
            refreshSelectionMoves();
            return;
        }
        attemptMove(selection.row, selection.col, row, col);
        selection.active = false;
        };

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            else if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                onPress(event.mouseButton.x, event.mouseButton.y);
            }
            else if (event.type == sf::Event::MouseMoved) {
                mousePos = { event.mouseMove.x, event.mouseMove.y };
            }
            else if (event.type == sf::Event::MouseButtonReleased &&
                event.mouseButton.button == sf::Mouse::Left) {
                onRelease(event.mouseButton.x, event.mouseButton.y);
            }
        }

        window.clear(BOARD_BG);

        Color turnColor = board.sideToMove();
        bool inCheckNow = board.isInCheck(turnColor);
        bool inCheckmateNow = board.isCheckmate(turnColor);
        int kingRow = -1, kingCol = -1;
        if (inCheckNow) findKingSquare(board, turnColor, kingRow, kingCol);

        // ---- Eval bar (drawn in the strip to the left of the board) ----
        sf::RectangleShape evalBarBg(sf::Vector2f(EVAL_BAR_WIDTH, BOARD_PIXELS));
        evalBarBg.setPosition(0, 0);
        evalBarBg.setFillColor(EVAL_BAR_BG);
        window.draw(evalBarBg);

        // White's portion fills from the bottom upward, same convention as chess.com's bar.
        float whiteHeight = evalScore * BOARD_PIXELS;
        sf::RectangleShape evalBarWhite(sf::Vector2f(EVAL_BAR_WIDTH, whiteHeight));
        evalBarWhite.setPosition(0, BOARD_PIXELS - whiteHeight);
        evalBarWhite.setFillColor(EVAL_BAR_WHITE);
        window.draw(evalBarWhite);

        // Board squares
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                sf::RectangleShape sq(sf::Vector2f(SQUARE, SQUARE));
                sq.setPosition(BOARD_OFFSET_X + c * SQUARE, r * SQUARE);
                sq.setFillColor((r + c) % 2 == 0 ? LIGHT_SQUARE : DARK_SQUARE);
                window.draw(sq);

                if (lastMove.fromRow != -1 &&
                    ((r == lastMove.fromRow && c == lastMove.fromCol) ||
                        (r == lastMove.toRow && c == lastMove.toCol))) {
                    sf::RectangleShape overlay(sf::Vector2f(SQUARE, SQUARE));
                    overlay.setPosition(BOARD_OFFSET_X + c * SQUARE, r * SQUARE);
                    overlay.setFillColor(LAST_MOVE_OVERLAY);
                    window.draw(overlay);
                }

                if (selection.active && r == selection.row && c == selection.col) {
                    sf::RectangleShape overlay(sf::Vector2f(SQUARE, SQUARE));
                    overlay.setPosition(BOARD_OFFSET_X + c * SQUARE, r * SQUARE);
                    overlay.setFillColor(SELECTED_OVERLAY);
                    window.draw(overlay);
                }

                if (kingRow == r && kingCol == c) {
                    sf::RectangleShape overlay(sf::Vector2f(SQUARE, SQUARE));
                    overlay.setPosition(BOARD_OFFSET_X + c * SQUARE, r * SQUARE);
                    overlay.setFillColor(inCheckmateNow ? CHECKMATE_OVERLAY : CHECK_OVERLAY);
                    window.draw(overlay);
                }

                if (selection.active && hasLegalMoveTo(selection, r, c)) {
                    Piece occupant = board.at(r, c);
                    if (occupant.type != PieceType::None) {
                        sf::RectangleShape overlay(sf::Vector2f(SQUARE, SQUARE));
                        overlay.setPosition(BOARD_OFFSET_X + c * SQUARE, r * SQUARE);
                        overlay.setFillColor(sf::Color(220, 60, 60, 110));
                        window.draw(overlay);
                    }
                    else {
                        sf::CircleShape dot(SQUARE * 0.15f);
                        dot.setFillColor(LEGAL_DOT);
                        dot.setOrigin(SQUARE * 0.15f, SQUARE * 0.15f);
                        dot.setPosition(BOARD_OFFSET_X + c * SQUARE + SQUARE / 2.f, r * SQUARE + SQUARE / 2.f);
                        window.draw(dot);
                    }
                }
            }
        }

        // Pieces (the one being dragged follows the cursor instead of sitting in its square)
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                Piece p = board.at(r, c);
                if (p.type == PieceType::None) continue;
                if (dragging && selection.active && r == selection.row && c == selection.col)
                    continue;
                drawPiece(window, font, p, BOARD_OFFSET_X + c * SQUARE + SQUARE / 2.f, r * SQUARE + SQUARE / 2.f);
            }
        }
        if (dragging && selection.active) {
            Piece p = board.at(selection.row, selection.col);
            drawPiece(window, font, p, static_cast<float>(mousePos.x), static_cast<float>(mousePos.y));
        }

        // Status bar
        sf::RectangleShape statusBar(sf::Vector2f(EVAL_BAR_WIDTH + BOARD_PIXELS, STATUS_HEIGHT));
        statusBar.setPosition(0, BOARD_PIXELS);
        statusBar.setFillColor(sf::Color(20, 20, 20));
        window.draw(statusBar);

        Color turn = board.sideToMove();
        std::string turnName = (turn == Color::White) ? "White" : "Black";
        std::string text = turnName + " to move";
        if (board.isCheckmate(turn)) {
            text = turnName + " is checkmated - " + (turn == Color::White ? "Black" : "White") + " wins!";
        }
        else if (board.isStalemate(turn)) {
            text = "Stalemate - draw.";
        }
        else if (board.isInCheck(turn)) {
            text += "  (check)";
        }
        if (!statusMessage.empty()) text += "   [" + statusMessage + "]";

        sf::Text statusText;
        statusText.setFont(font);
        statusText.setString(text);
        statusText.setCharacterSize(20);
        statusText.setFillColor(sf::Color::White);
        statusText.setPosition(BOARD_OFFSET_X + 10, BOARD_PIXELS + 12);
        window.draw(statusText);

        window.display();
    }

    return 0;
}