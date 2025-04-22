import numpy as np
import random
import math

ROWS = 6
COLUMNS = 7
PLAYER = 1
AI = 2
EMPTY = 0
WINDOW_LENGTH = 4
DEPTH = 5

def create_board():
    return np.zeros((ROWS, COLUMNS), dtype=int)

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[0][col] == 0

def get_valid_locations(board):
    return [c for c in range(COLUMNS) if is_valid_location(board, c)]

def get_next_open_row(board, col):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == 0:
            return r

def print_board(board):
    print("\nCurrent board:")
    for r in range(ROWS):
        row_str = "| " + " | ".join(str(int(cell)) if cell != 0 else " " for cell in board[r]) + " |"
        print(row_str)
    print("  " + "   ".join(str(i) for i in range(COLUMNS)))

def winning_move(board, piece):
    for c in range(COLUMNS - 3):
        for r in range(ROWS):
            if all(board[r, c+i] == piece for i in range(4)):
                return True
    for c in range(COLUMNS):
        for r in range(ROWS - 3):
            if all(board[r+i, c] == piece for i in range(4)):
                return True
    for c in range(COLUMNS - 3):
        for r in range(ROWS - 3):
            if all(board[r+i, c+i] == piece for i in range(4)):
                return True
    for c in range(COLUMNS - 3):
        for r in range(3, ROWS):
            if all(board[r-i, c+i] == piece for i in range(4)):
                return True
    return False

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER if piece == AI else AI
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 10
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 5
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 80
    return score

def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMNS//2])]
    score += center_array.count(piece) * 3

    for r in range(ROWS):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMNS - 3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLUMNS):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER) or winning_move(board, AI) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI):
                return (None, 1000000000000)
            elif winning_move(board, PLAYER):
                return (None, -1000000000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI))

    if maximizingPlayer:
        value = -math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, AI)
            new_score = minimax(temp_board, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value

    else:
        value = math.inf
        best_col = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp_board = board.copy()
            drop_piece(temp_board, row, col, PLAYER)
            new_score = minimax(temp_board, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

def main():
    board = create_board()
    game_over = False
    print_board(board)
    turn = random.randint(PLAYER, AI)

    while not game_over:
        if turn == PLAYER:
            col = int(input("Player 1, choose a column (0-6): "))
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER)

                if winning_move(board, PLAYER):
                    print_board(board)
                    print("Player 1 wins!")
                    game_over = True
                turn = AI
            else:
                print("Column full. Try again.")

        else:
            print("AI is thinking...")
            col, _ = minimax(board, DEPTH, -math.inf, math.inf, True)
            print(f"AI chooses column {col}")
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI)

                if winning_move(board, AI):
                    print_board(board)
                    print("AI wins!")
                    game_over = True
                turn = PLAYER

        print_board(board)

        if not game_over and len(get_valid_locations(board)) == 0:
            print("It's a tie!")
            game_over = True

if __name__ == "__main__":
    main()
