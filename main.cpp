#include "Board.h"
#include <iostream>
#include <string>

int main() {
    Board board;

    std::cout << "Simple console chess. Enter moves like 'e2e4', or 'e7e8q' for promotion.\n"
                 "Type 'quit' to exit.\n";

    while (true) {
        board.print();

        Color turn = board.sideToMove();
        std::string turnName = (turn == Color::White) ? "White" : "Black";

        if (board.isCheckmate(turn)) {
            std::cout << turnName << " is checkmated. "
                      << (turn == Color::White ? "Black" : "White") << " wins!\n";
            break;
        }
        if (board.isStalemate(turn)) {
            std::cout << "Stalemate. It's a draw.\n";
            break;
        }
        if (board.isInCheck(turn)) {
            std::cout << turnName << " is in check.\n";
        }

        std::cout << turnName << " to move: ";
        std::string input;
        if (!(std::cin >> input)) break;
        if (input == "quit") break;

        std::string error;
        if (!board.tryUserMove(input, error)) {
            std::cout << error << "\n";
        }
    }

    return 0;
}
