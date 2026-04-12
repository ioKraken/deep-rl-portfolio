import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import config
from data_pipeline import prepare_data
from train import train
from backtest import run_backtest, generate_plots, save_metrics, print_results


def main():
    print("=" * 72)
    print("   DEEP RL PORTFOLIO OPTIMIZATION")
    print("  LSTM-PPO | 50 Equities | Sharpe Ratio Objective")
    print("=" * 72)
    print()

    overall_start = time.time()

    #  Step 1: Data 
    print("━" * 72)
    print("  STEP 1: DATA PIPELINE")
    print("━" * 72)
    (
        train_features,
        test_features,
        train_prices,
        test_prices,
        benchmark_train,
        benchmark_test,
        prices_df,
    ) = prepare_data()
    print()

    # Step 2: Training
    print("━" * 72)
    print("  STEP 2: PPO TRAINING")
    print("━" * 72)
    agent = train(train_features, train_prices)
    print()

    # Step 3: Backtesting
    print("━" * 72)
    print("  STEP 3: BACKTESTING ON HELD-OUT DATA")
    print("━" * 72)
    backtest_result = run_backtest(agent, test_features, test_prices, benchmark_test)

    # Print results
    print_results(backtest_result["metrics"])

    # Step 4: Generate Report
    print("━" * 72)
    print("  STEP 4: GENERATING REPORT & VISUALIZATIONS")
    print("━" * 72)
    generate_plots(backtest_result, config.RESULTS_DIR)
    save_metrics(backtest_result["metrics"], config.RESULTS_DIR)

    total_time = time.time() - overall_start
    print(f"\n Pipeline complete in {total_time:.1f}s")
    print(f" Results saved to: {config.RESULTS_DIR}/")
    print(f"   • cumulative_returns.png")
    print(f"   • drawdown.png")
    print(f"   • rolling_sharpe.png")
    print(f"   • weight_allocation.png")
    print(f"   • monthly_returns_heatmap.png")
    print(f"   • performance_summary.png")
    print(f"   • training_progress.png")
    print(f"   • metrics.json")
    print()


if __name__ == "__main__":
    main()
