import time
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
from connect_four_ai import (
    create_board, drop_piece, is_valid_location, get_next_open_row,
    winning_move, get_valid_locations, minimax, ROWS, COLUMNS, PLAYER, AI
)

def random_move(board):
    return random.choice(get_valid_locations(board))

def run_vs_random(n_games, depth):
    ai_wins = 0
    random_wins = 0
    draws = 0
    move_times = []

    for _ in range(n_games):
        board = create_board()
        game_over = False
        turn = PLAYER if random.random() < 0.5 else AI

        while not game_over:
            if turn == PLAYER:
                col = random_move(board)
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER)
                if winning_move(board, PLAYER):
                    random_wins += 1
                    game_over = True
            else:
                start = time.time()
                col, _ = minimax(board, depth, -float("inf"), float("inf"), True)
                move_times.append(time.time() - start)
                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, AI)
                    if winning_move(board, AI):
                        ai_wins += 1
                        game_over = True

            if not game_over and len(get_valid_locations(board)) == 0:
                draws += 1
                game_over = True

            turn = PLAYER if turn == AI else AI

    return {
        "depth": depth,
        "games": n_games,
        "ai_wins": ai_wins,
        "random_wins": random_wins,
        "draws": draws,
        "avg_move_time_sec": round(sum(move_times) / len(move_times), 4)
    }

def run_vs_random_all_depths(n_games=100, depths=(1, 2, 3, 4, 5)):
    results = []
    for d in depths:
        print(f"Running AI vs Random at depth {d}...")
        result = run_vs_random(n_games, d)
        results.append(result)
    return pd.DataFrame(results)

def run_ai_vs_ai_matrix(n_games=30, depths=(2, 3, 4, 5)):
    size = len(depths)
    win_matrix = np.zeros((size, size))
    draw_matrix = np.zeros((size, size))  # To store draw percentages

    full_stats = {}

    for i, d1 in enumerate(depths):
        for j, d2 in enumerate(depths):
            if j <= i:
                continue
            print(f"Running AI_1 (depth {d1}) vs AI_2 (depth {d2})...")

            ai1_wins = 0
            ai2_wins = 0
            draws = 0

            for _ in range(n_games):
                board = create_board()
                game_over = False
                turn = PLAYER if random.random() < 0.5 else AI

                while not game_over:
                    if turn == PLAYER:
                        col, _ = minimax(board, d1, -float("inf"), float("inf"), True)
                        if is_valid_location(board, col):
                            row = get_next_open_row(board, col)
                            drop_piece(board, row, col, PLAYER)
                            if winning_move(board, PLAYER):
                                ai1_wins += 1
                                game_over = True
                    else:
                        col, _ = minimax(board, d2, -float("inf"), float("inf"), True)
                        if is_valid_location(board, col):
                            row = get_next_open_row(board, col)
                            drop_piece(board, row, col, AI)
                            if winning_move(board, AI):
                                ai2_wins += 1
                                game_over = True

                    if not game_over and len(get_valid_locations(board)) == 0:
                        draws += 1
                        game_over = True

                    turn = PLAYER if turn == AI else AI

            ai1_pct = 100 * ai1_wins / n_games
            ai2_pct = 100 * ai2_wins / n_games
            draw_pct = 100 * draws / n_games

            win_matrix[i, j] = ai1_pct
            draw_matrix[i, j] = draw_pct

            full_stats[(i, j)] = {
                "ai1": ai1_pct,
                "ai2": ai2_pct,
                "draw": draw_pct
            }

    return depths, win_matrix, draw_matrix, full_stats


if __name__ == "__main__":
    # ---- AI vs RANDOM across depths 1–5 ----
    df_random = run_vs_random_all_depths(n_games=100, depths=(1, 2, 3, 4, 5))
    df_random.to_csv("ai_vs_random_by_depth.csv", index=False)
    print("✅ Saved ai_vs_random_by_depth.csv")

    # ---- Plot win rate vs depth ----
    plt.figure(figsize=(6, 4))
    plt.plot(df_random["depth"], 100 * df_random["ai_wins"] / df_random["games"], marker="o")
    plt.title("AI Win Rate vs Depth (vs Random)")
    plt.xlabel("Search Depth")
    plt.ylabel("Win Rate (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ai_vs_random_winrate.png")
    print("✅ Saved ai_vs_random_winrate.png")
    plt.show()

    # ---- Plot move time vs depth ----
    plt.figure(figsize=(6, 4))
    plt.plot(df_random["depth"], df_random["avg_move_time_sec"], marker="o", color="orange")
    plt.title("Average AI Move Time vs Depth")
    plt.xlabel("Search Depth")
    plt.ylabel("Avg Time per Move (sec)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ai_vs_random_movetime.png")
    print("✅ Saved ai_vs_random_movetime.png")
    plt.show()

    # ---- AI vs AI Matrix (4x4: depths 2–5) ----
    depths, win_matrix, draw_matrix, full_stats = run_ai_vs_ai_matrix(n_games=100, depths=(2, 3, 4, 5))
    df_matrix = pd.DataFrame(win_matrix, columns=[f'Depth {d}' for d in depths], index=[f'Depth {d}' for d in depths])
    df_matrix.to_csv("ai_vs_ai_matrix.csv")
    print("✅ Saved ai_vs_ai_matrix.csv")

    plt.figure(figsize=(8, 6))

    masked_matrix = np.ma.array(win_matrix, mask=np.eye(len(depths), dtype=bool))
    for i in range(len(depths)):
        for j in range(len(depths)):
            if j > i and (j, i) not in full_stats:
                full_stats[(j, i)] = {
                    "ai1": full_stats[(i, j)]["ai2"],
                    "ai2": full_stats[(i, j)]["ai1"],
                    "draw": full_stats[(i, j)]["draw"]
                }
                win_matrix[j, i] = full_stats[(j, i)]["ai1"]


    # Plot the masked matrix and set black for masked cells
    im = plt.imshow(masked_matrix, cmap='coolwarm', interpolation='nearest')
    im.cmap.set_bad(color='black')

    plt.colorbar(label='Win % for AI_1 (Row) vs AI_2 (Col)')
    plt.xticks(np.arange(len(depths)), [f'Depth {d}' for d in depths])
    plt.yticks(np.arange(len(depths)), [f'Depth {d}' for d in depths])
    plt.title("AI vs AI Heatmap (Annotated)")

    for i in range(len(depths)):
        for j in range(len(depths)):
            if i == j:
                continue  # skip diagonal
            stats = full_stats[(i, j)]
            text = f"A1: {stats['ai1']:.0f}%\nDraw: {stats['draw']:.0f}%\nA2: {stats['ai2']:.0f}%"
            plt.text(j, i, text, ha='center', va='center', color='black', fontsize=8)

    plt.xlabel("AI_2 Depth")
    plt.ylabel("AI_1 Depth")
    plt.tight_layout()
    plt.savefig("ai_vs_ai_heatmap_annotated.png")
    print("✅ Saved ai_vs_ai_heatmap_annotated.png")
    plt.show()

