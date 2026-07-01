"""
HireSense AI — PyTorch MLP estimator.

Kept in its own module (not deep_model.py) so the pickled class is always
referenced as `torch_mlp.TorchMLP` regardless of how training was launched.
This lets a saved deep-model bundle be unpickled from any entry point (API,
Streamlit, predict CLI) as long as `src` is on the path and torch is present.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE

import torch
import torch.nn as nn

HIDDEN_LAYERS = (128, 64, 32)


class _Net(nn.Module):
    """Feed-forward network: Linear -> ReLU -> Dropout blocks -> 2-class logits."""

    def __init__(self, n_features: int, hidden=HIDDEN_LAYERS, dropout=0.3):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLP(BaseEstimator, ClassifierMixin):
    """A scikit-learn-compatible wrapper around a torch MLP with early stopping."""

    def __init__(self, hidden=HIDDEN_LAYERS, dropout=0.3, lr=1e-3,
                 epochs=300, batch_size=64, patience=20, seed=RANDOM_STATE):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=self.seed, stratify=y
        )
        Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr)
        Xv, yv = torch.tensor(X_val), torch.tensor(y_val)

        self.net_ = _Net(X.shape[1], self.hidden, self.dropout)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        best_val, best_state, no_improve = float("inf"), None, 0
        n = len(Xt)
        epoch = 0
        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net_(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()

            self.net_.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net_(Xv), yv).item()
            if val_loss < best_val - 1e-4:
                best_val, best_state, no_improve = val_loss, self.net_.state_dict(), 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break
        self.n_epochs_ = epoch + 1
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            logits = self.net_(torch.tensor(X))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
