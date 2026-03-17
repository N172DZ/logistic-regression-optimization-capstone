from data_loader import load_data
from train import train_model
from evaluation import compare_runs
from plots import plot_loss_curves, plot_accuracy_curves


def main():
    X_train, X_test, y_train, y_test = load_data()

    settings = [
        {"optimizer": "bgd", "learning_rate": 0.05, "epochs": 200, "batch_size": 16},
        {"optimizer": "sgd", "learning_rate": 0.01, "epochs": 200, "batch_size": 1},
        {"optimizer": "mbgd", "learning_rate": 0.03, "epochs": 200, "batch_size": 16},
    ]

    histories = []

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
        histories.append(history)

    summary_df = compare_runs(histories)
    print("\nSummary of Optimization Comparison:\n")
    print(summary_df)

    plot_loss_curves(histories)
    plot_accuracy_curves(histories)


if __name__ == "__main__":
    main()