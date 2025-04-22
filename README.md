# Connect Four AI

This project implements a command-line Connect Four game powered by an AI agent using the Minimax algorithm with Alpha-Beta pruning. It includes simulations for AI vs Random and AI vs AI players across different depths, and generates statistics and annotated heatmaps for analysis.

---

## 📁 Project Files

- `connect_four_ai.py` – Game logic and AI implementation (Minimax + Alpha-Beta Pruning)
- `experiments.py` – Simulation runner and visualization generator
- `ai_vs_random_by_depth.csv` – Win/loss/draw stats for AI vs Random across depths
- `ai_vs_random_winrate.png` – AI win rate vs depth plot
- `ai_vs_random_movetime.png` – Average AI move time vs depth
- `ai_vs_ai_matrix.csv` – Matrix of win percentages in AI vs AI matchups
- `ai_vs_ai_heatmap_annotated.png` – Annotated heatmap with win/draw rates

---

## 🛠 Requirements

Install required packages (Python 3.9+ recommended):

```bash
pip install numpy matplotlib pandas
```

---

## 🚀 How to Run

Run all experiments and generate plots:

```bash
python experiments.py
```

This will:
- Simulate 100 games of AI vs Random at depths 1–5
- Simulate 100 games of AI vs AI for all unique depth combinations (e.g., 2 vs 3, 2 vs 4…)
- Generate and save CSVs and plots with the results
- Display win rate plots, average move times, and a heatmap annotated with win/draw rates
- Black out heatmap diagonal cells (same-depth matchups), since those are skipped

