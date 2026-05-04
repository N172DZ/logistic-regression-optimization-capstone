from train import train_model
from evaluation import compare_runs
from plots import plot_loss_curves, plot_accuracy_curves

import numpy as np
import random
import pandas as pd


def main():
    seeds = [0, 1, 2, 3, 4]  # repeat trials with these random seeds

    settings = [
        {"optimizer": "bgd", "learning_rate": 0.05, "epochs": 200, "batch_size": 16},
        {"optimizer": "sgd", "learning_rate": 0.01, "epochs": 200, "batch_size": 1},
        {"optimizer": "mbgd", "learning_rate": 0.03, "epochs": 200, "batch_size": 16},
    ]

    results = []
    last_histories = {}

    for seed in seeds:
        # ensure reproducible split / RNG for each trial
        np.random.seed(seed)
        random.seed(seed)

        X_train, X_test, y_train, y_test = load_data(test_size=0.2, random_state=seed)

        for cfg in settings:
            _, history = train_model(
                X_train,
                y_train,
                X_test,
                y_test,
                optimizer=cfg["optimizer"],
                learning_rate=cfg["learning_rate"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
            )

            results.append(
                {
                    "seed": seed,
                    "optimizer": cfg["optimizer"],
                    "final_train_loss": history["train_loss"][-1],
                    "final_test_loss": history["test_loss"][-1],
                    "final_train_acc": history["train_acc"][-1],
                    "final_test_acc": history["test_acc"][-1],
                    "avg_epoch_time": float(np.mean(history.get("epoch_time", [0]))),
                }
            )

            # keep last history per optimizer for quick plotting
            last_histories[cfg["optimizer"]] = history

    df = pd.DataFrame(results)
    summary_df = (
        df.groupby("optimizer")
        .agg(
            mean_final_test_acc=("final_test_acc", "mean"),
            std_final_test_acc=("final_test_acc", "std"),
            mean_final_test_loss=("final_test_loss", "mean"),
            std_final_test_loss=("final_test_loss", "std"),
            mean_epoch_time=("avg_epoch_time", "mean"),
        )
        .reset_index()
    )

    print("\nRepeated-trials summary (mean ± std over seeds):\n")
    print(summary_df)

    # plot curves for the last seed run (one per optimizer)
    histories = [last_histories[cfg["optimizer"]] for cfg in settings]
    plot_loss_curves(histories)
    plot_accuracy_curves(histories)


if __name__ == "__main__":
    main()