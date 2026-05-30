"""
Baseline models module
Corresponds to Section 2 of the paper: LR, DT, SVM, NN (MLP), ICNN, RNN (LSTM)
All models implement a unified fit / predict_proba / predict interface.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ── Traditional machine learning models ──────────────────────────────────────

def build_lr():
    """Logistic Regression (LR) — L2 regularization, lbfgs solver."""
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42
    )


def build_dt():
    """Decision Tree (DT) — max_depth=8, Gini criterion."""
    return DecisionTreeClassifier(
        max_depth=8,
        criterion="gini",
        random_state=42
    )


def build_svm():
    """Support Vector Machine (SVM) — RBF kernel."""
    return SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )


def build_nn():
    """Neural Network / MLP (NN) — 3 hidden layers (128-64-32)."""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )


# ── Deep learning models (Keras wrapper) ─────────────────────────────────────

class KerasClassifierWrapper:
    """Wraps a Keras model with a sklearn-compatible interface."""

    def __init__(self, build_fn, epochs=50, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None
        self.n_features_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.model_ = self.build_fn(self.n_features_)
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.1,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        return self

    def predict_proba(self, X):
        prob = self.model_.predict(X, verbose=0).flatten()
        return np.column_stack([1 - prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_icnn(n_features):
    """
    Improved CNN (ICNN) — adaptive multi-scale convolution strategy.
    Tabular features are reshaped into an approximate square 2D matrix
    to simulate image convolution.
    """
    # Pad features to the nearest perfect square
    side = int(np.ceil(np.sqrt(n_features)))
    padded = side * side

    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(padded, activation="relu")(inp)
    x = layers.Reshape((side, side, 1))(x)

    # Multi-scale convolution: parallel 1x1 and 3x3 kernels
    x1 = layers.Conv2D(32, (1, 1), padding="same", activation="relu")(x)
    x2 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x1, x2])
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _build_rnn(n_features):
    """
    RNN / LSTM — features serialized as time steps, two LSTM layers.
    """
    time_steps = n_features
    inp = keras.Input(shape=(time_steps, 1))

    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


class ICNNClassifier(KerasClassifierWrapper):
    """Improved CNN classifier."""

    def __init__(self, **kwargs):
        super().__init__(build_fn=_build_icnn, **kwargs)


class RNNClassifier(KerasClassifierWrapper):
    """LSTM classifier — features reshaped as a time series."""

    def __init__(self, **kwargs):
        super().__init__(build_fn=_build_rnn, **kwargs)

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.model_ = self.build_fn(self.n_features_)
        # reshape: (samples, features, 1) as a time series
        X_seq = X.reshape(X.shape[0], X.shape[1], 1)
        self.model_.fit(
            X_seq, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=0.1,
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        return self

    def predict_proba(self, X):
        X_seq = X.reshape(X.shape[0], X.shape[1], 1)
        prob = self.model_.predict(X_seq, verbose=0).flatten()
        return np.column_stack([1 - prob, prob])


# ── Model factory ─────────────────────────────────────────────────────────────

def get_all_models():
    """Return a dict of all baseline models used in the paper."""
    return {
        "LR":   build_lr(),
        "DT":   build_dt(),
        "SVM":  build_svm(),
        "NN":   build_nn(),
        "ICNN": ICNNClassifier(epochs=80, batch_size=32),
        "RNN":  RNNClassifier(epochs=80, batch_size=32),
    }
