import numpy as np
import pandas as pd


def summarize_history(history):
    df = pd.DataFrame(history)
    summary = {
        "optimizer": history["optimizer"],
        "learning_rate": history["learning_rate"],
        "final_train_loss": df["train_loss"].iloc[-1],
        "final_test_loss": df["test_loss"].iloc[-1],
        "final_train_acc": df["train_acc"].iloc[-1],
        "final_test_acc": df["test_acc"].iloc[-1],
        "avg_epoch_time": df["epoch_time"].mean(),
        "loss_std": df["train_loss"].std(),
    }
    return summary


def compare_runs(histories):
    results = [summarize_history(h) for h in histories]
    return pd.DataFrame(results)