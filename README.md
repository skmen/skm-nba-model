## SKM NBA Model

SKM NBA Model is a comprehensive Python-based framework for predicting NBA player statistics, simulating game outcomes, and evaluating betting opportunities. It leverages the `nba_api` to fetch real-time data, a custom-built machine learning pipeline using `xgboost` and `scikit-learn` to generate predictions, and a suite of scripts to manage the end-to-end workflow from data acquisition to performance analysis.

### Table of Contents
* [Setup Guide](#setup-guide)
* [Workflow Routine](#workflow-routine)
* [Scripts](#scripts)
  * [automated\_predictions.py](#automated_predictionspy)
  * [run\_simulation.py](#run_simulationpy)
  * [generate\_full\_report.py](#generate_full_reportpy)
  * [bet\_analyzer.py](#bet_analyzerpy)
  * [evaluate\_accuracy.py](#evaluate_accuracypy)
  * [train\_global\_models.py](#train_global_modelspy)
  * [generate\_accuracy\_report.py](#generate_accuracy_reportpy)
  * [generate\_dvp\_stats.py](#generate_dvp_statspy)
* [Core Modules](#core-modules)
* [Prediction Pipeline](#prediction-pipeline)

### Setup Guide

Follow these steps to set up the project environment.

1.  **Prerequisites:**
    *   Python 3.x

2.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd skm-nba-model
    ```

3.  **Create and Activate a Virtual Environment:**
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS/Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The required packages are:
    *   `pandas`
    *   `nba_api`
    *   `xgboost`
    *   `scikit-learn`
    *   `matplotlib`
    *   `pyarrow`
    *   `requests`

### Workflow Routine

This is the typical step-by-step workflow for using the model.

1.  **Run Training (Weekly):**
    Train the models on the latest data. This should be done periodically (e.g., weekly) to keep the models current.
    ```bash
    python scripts/train_global_models.py
    ```

2.  **Generate Raw Predictions:**
    On game days, generate the raw player stat predictions for all scheduled games.
    ```bash
    python scripts/automated_predictions.py --run-once
    ```
    This creates a file like `data/predictions/predictions_YYYY-MM-DD.csv`.

3.  **Run Game Simulation:**
    Use the raw predictions to simulate team scores and game outcomes.
    ```bash
    python scripts/run_simulation.py
    ```
    This generates `game_predictions_YYYY-MM-DD.json` and other related simulation files.

4.  **Generate Betting Sheets:**
    Create a detailed betting sheet from the raw predictions, including context like model trust, MAE, and opponent stats.
    ```bash
    python scripts/generate_full_report.py data/predictions/predictions_YYYY-MM-DD.csv
    ```
    This produces a `betting_sheet_...csv` file.

5.  **Analysis and Betting:**
    Use the interactive `bet_analyzer` to compare model predictions against live sportsbook lines.
    ```bash
    python scripts/bet_analyzer.py --date YYYY-MM-DD
    ```

6.  **Get "Answer Key" (Day After):**
    The day after the games, fetch the actual player stats.
    ```bash
    python scripts/automated_predictions.py --get-actuals YYYY-MM-DD
    ```
    This saves the results to `data/predictions/actuals_YYYY-MM-DD.csv`.

7.  **Evaluate Accuracy:**
    Compare the predictions with the actuals to evaluate model performance.
    ```bash
    python scripts/evaluate_accuracy.py YYYY-MM-DD
    ```

### Scripts

#### `automated_predictions.py`
Handles the generation of daily predictions and the retrieval of actual game results.

*   **Purpose:**
    *   `--run-once`: Fetches all players playing on a given day and runs the prediction pipeline for each of them.
    *   `--get-actuals`: Fetches the actual game statistics for all players on a given date.
*   **Arguments:**
    *   `--run-once`: Runs predictions for the current day (or a specified `--date`).
    *   `--get-actuals YYYY-MM-DD`: Retrieves actual stats for the specified date.
    *   `--date YYYY-MM-DD` (optional): Specifies the date for predictions. Defaults to the current day.
    *   `--team <team_name>` (optional): Filters predictions for a specific team.
    *   `--season <season>` (optional): Specifies the NBA season (e.g., `2024-25`).
*   **Examples:**
    *   Run predictions for today:
        ```bash
        python scripts/automated_predictions.py --run-once
        ```
    *   Get actuals for November 28, 2025:
        ```bash
        python scripts/automated_predictions.py --get-actuals 2025-11-28
        ```

#### `run_simulation.py`
Simulates game outcomes based on the generated player predictions.

*   **Purpose:** Aggregates individual player predictions to forecast team scores, game totals, and spreads.
*   **Arguments:**
    *   `[date]` (optional): The date to run the simulation for, in `YYYY-MM-DD` format. Defaults to the current date.
*   **Example:**
    *   Run simulation for November 28, 2025:
        ```bash
        python scripts/run_simulation.py 2025-11-28
        ```

#### `generate_full_report.py`
Creates a "Master Betting Sheet" from a raw predictions CSV file.

*   **Purpose:** Enriches raw predictions with model trust scores, Mean Absolute Error (MAE), and contextual data like player usage, minutes, and opponent DvP (Defense vs. Position).
*   **Arguments:**
    *   `file_path`: The path to the raw predictions CSV file.
*   **Example:**
    ```bash
    python scripts/generate_full_report.py data/predictions/predictions_2025-11-28.csv
    ```

#### `bet_analyzer.py`
An interactive tool to compare model predictions against live sportsbook lines.

*   **Purpose:** Allows for quick analysis of betting opportunities by calculating the "edge" between the model's prediction and a given betting line.
*   **Arguments:**
    *   `--date YYYY-MM-DD`: Loads the daily betting sheet for the specified date.
    *   `--file <file_path>` (optional): Direct path to a prediction JSON or CSV file.
*   **Example:**
    *   Analyze the betting sheet for a specific date:
        ```bash
        python scripts/bet_analyzer.py --date 2025-11-28
        ```

#### `evaluate_accuracy.py`
Compares generated predictions with actual results to produce a console-based accuracy report.

*   **Purpose:** Calculates MAE (Mean Absolute Error) and bias (over/under prediction tendency) for each stat category.
*   **Arguments:**
    *   `[date]` (optional): The date to evaluate, in `YYYY-MM-DD` format. Defaults to yesterday.
*   **Example:**
    ```bash
    python scripts/evaluate_accuracy.py 2025-11-28
    ```

#### `train_global_models.py`
Trains the machine learning models for different player position buckets.

*   **Purpose:** Loads all available historical data, separates players into positional buckets (Guards, Wings, Bigs), and trains a unique model for each stat target (PTS, REB, AST, etc.) within each bucket. The trained models are saved to the `models/` directory.
*   **Arguments:** None.
*   **Example:**
    ```bash
    python scripts/train_global_models.py
    ```

#### `generate_accuracy_report.py`
Generates an HTML report to visualize model performance.

*   **Purpose:** Creates a user-friendly HTML dashboard (`docs/accuracy_report.html`) that shows a trust scorecard, bias analysis, and a "bad beats" section for the biggest prediction misses.
*   **Arguments:**
    *   `--date YYYY-MM-DD` (optional): The date to analyze. Defaults to yesterday.
*   **Example:**
    ```bash
    python scripts/generate_accuracy_report.py --date 2025-11-28
    ```

#### `generate_dvp_stats.py`
Generates Defense vs. Position (DvP) statistics.

*   **Purpose:** Fetches league-wide data to calculate the average stats allowed by each team to each position (Guard, Forward, Center). This is a crucial feature for the prediction models.
*   **Arguments:** None.
*   **Example:**
    ```bash
    python scripts/generate_dvp_stats.py
    ```

### Core Modules

The `src/` directory contains the core logic for the prediction system.

*   **`config.py`**: Central configuration for features, targets, file paths, and constants.
*   **`data_fetcher.py`**: Handles all data acquisition from `nba_api`.
*   **`feature_engineer.py`**: Creates the features used by the models.
*   **`model.py`**: Defines the `NBAPlayerStack` model, which uses an `xgboost` regressor. Includes training, evaluation, and prediction functions.
*   **`batch_predictor.py`**: Manages the process of making predictions for a large batch of players.
*   **`game_predictor.py` / `game_simulator.py`**: Contains logic for simulating game outcomes.
*   **`prediction_pipeline.py`**: Orchestrates the end-to-end process for predicting a single player's performance.

### Prediction Pipeline

The `src/prediction_pipeline.py` script is the heart of the single-player prediction process. While not typically run directly in the daily workflow, it encapsulates the key steps:

1.  **Acquire Data**: Fetches game logs, opponent defensive ratings, and player usage stats.
2.  **Engineer Features**: Calculates rolling averages, opponent-adjusted stats, and other advanced features.
3.  **Prepare Simulation Context**: Estimates the player's minutes, the game's pace, and the player's likely possession count.
4.  **Train & Predict Loop**: For each target statistic (e.g., points, rebounds):
    *   It trains a fresh, temporary `NBAPlayerStack` model on that player's historical data.
    *   It predicts the stat as a *rate* (e.g., points per 100 possessions).
    *   It converts the rate into a *total* prediction based on the simulation context (e.g., 25.5 total points).
5.  **Output**: Saves the final predictions, model errors (MAE), and simulation context to a JSON file.