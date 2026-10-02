"""PyTorch multilayer perceptron with a scikit-learn style interface."""

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def build_mlp(n_features, hidden_dims=(128, 64), dropout=0.1):
    layers = []
    width = n_features
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)]
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class TorchMLPRegressor:
    """MLP regressor trained on standardized inputs and targets.

    ``fit`` holds out ``val_fraction`` of the training data for early stopping
    and learning-rate scheduling, then restores the weights from the epoch with
    the lowest validation loss. Predictions are returned in the original units.
    """

    def __init__(
        self,
        hidden_dims=(128, 64),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=32,
        max_epochs=1000,
        patience=60,
        lr_factor=0.5,
        lr_patience=20,
        val_fraction=0.1,
        random_state=42,
        verbose=False,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_fraction, random_state=self.random_state
        )

        self.x_scaler_ = StandardScaler().fit(X_train)
        self.y_scaler_ = StandardScaler().fit(y_train)

        train_set = TensorDataset(
            self._to_tensor(self.x_scaler_.transform(X_train)),
            self._to_tensor(self.y_scaler_.transform(y_train)),
        )
        loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        X_val_t = self._to_tensor(self.x_scaler_.transform(X_val))
        y_val_t = self._to_tensor(self.y_scaler_.transform(y_val))

        self.net_ = build_mlp(X.shape[1], self.hidden_dims, self.dropout)
        optimizer = torch.optim.AdamW(
            self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.lr_factor, patience=self.lr_patience
        )
        loss_fn = nn.MSELoss()

        self.history_ = {"train_loss": [], "val_loss": [], "lr": []}
        best_loss, best_state, stale = np.inf, None, 0

        for epoch in range(self.max_epochs):
            self.net_.train()
            total = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.net_(xb), yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(xb)
            train_loss = total / len(train_set)

            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(X_val_t), y_val_t).item()
            scheduler.step(val_loss)

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["lr"].append(optimizer.param_groups[0]["lr"])

            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = copy.deepcopy(self.net_.state_dict())
                self.best_epoch_ = epoch
            else:
                stale += 1

            if self.verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}, train {train_loss:.4f}, val {val_loss:.4f}")

            if stale >= self.patience:
                break

        self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def predict(self, X):
        X = self.x_scaler_.transform(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            scaled = self.net_(self._to_tensor(X)).numpy()
        return self.y_scaler_.inverse_transform(scaled).reshape(-1)

    @staticmethod
    def _to_tensor(array):
        return torch.as_tensor(np.asarray(array, dtype=np.float32))
