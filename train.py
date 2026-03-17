import numpy as np
import time
from model import LogisticRegressionScratch


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def create_batches(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    batches = []
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        batches.append((X_shuffled[start:end], y_shuffled[start:end]))
    return batches


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    optimizer="bgd",
    learning_rate=0.05,
    epochs=200,
    batch_size=16,
    random_state=42,
):
    np.random.seed(random_state)

    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = LogisticRegressionScratch(n_features, n_classes)

    history = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "epoch_time": [],
        "optimizer": optimizer,
        "learning_rate": learning_rate,
    }

    for epoch in range(epochs):
        start_time = time.perf_counter()

        if optimizer == "bgd":
            dW, db = model.compute_gradients(X_train, y_train)
            model.W -= learning_rate * dW
            model.b -= learning_rate * db

        elif optimizer == "sgd":
            batches = create_batches(X_train, y_train, 1)
            for Xi, yi in batches:
                dW, db = model.compute_gradients(Xi, yi)
                model.W -= learning_rate * dW
                model.b -= learning_rate * db

        elif optimizer == "mbgd":
            batches = create_batches(X_train, y_train, batch_size)
            for Xi, yi in batches:
                dW, db = model.compute_gradients(Xi, yi)
                model.W -= learning_rate * dW
                model.b -= learning_rate * db

        else:
            raise ValueError("optimizer must be 'bgd', 'sgd', or 'mbgd'")

        train_loss = model.compute_loss(X_train, y_train)
        test_loss = model.compute_loss(X_test, y_test)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        elapsed = time.perf_counter() - start_time

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(elapsed)

    return model, history