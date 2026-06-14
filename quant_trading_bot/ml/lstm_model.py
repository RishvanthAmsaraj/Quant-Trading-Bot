"""LSTM-based price-prediction model.

The model is intentionally lightweight so that the whole project can
be trained on a laptop CPU in a few minutes.  It predicts the *next
day's close* given a sliding window of past closes (and optional
features).  Predictions are then mapped to a long/flat signal in the
strategy layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class LSTMTrainingResult:
    train_loss: list
    val_loss: list
    history: "object"  # tf.keras History


class LSTMPricePredictor:
    """Train, predict and persist an LSTM price model.

    Parameters
    ----------
    lookback:
        Number of past bars used as input.
    hidden_units:
        Number of LSTM units per layer.
    dropout:
        Dropout rate after the LSTM layer.
    learning_rate:
        Optimiser learning rate.
    """

    def __init__(
        self,
        lookback: int = 60,
        hidden_units: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
    ) -> None:
        self.lookback = lookback
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_mean: Optional[float] = None
        self.scaler_std: Optional[float] = None

    # ------------------------------------------------------------------ #
    def _build_model(self, n_features: int) -> "object":  # type: ignore[name-defined]
        """Lazily build the keras model to keep TF imports optional."""
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        inputs = Input(shape=(self.lookback, n_features), name="input")
        x = LSTM(self.hidden_units, return_sequences=False)(inputs)
        x = Dropout(self.dropout)(x)
        x = Dense(32, activation="relu")(x)
        out = Dense(1, name="yhat")(x)
        model = Model(inputs=inputs, outputs=out)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mse")
        return model

    # ------------------------------------------------------------------ #
    def _make_windows(
        self, series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(series)):
            X.append(series[i - self.lookback : i])
            y.append(series[i, 0])  # first column = close (normalised)
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32)
        return X_arr, y_arr

    # ------------------------------------------------------------------ #
    def _prepare_features(self, close: pd.Series) -> np.ndarray:
        """Build a (T, F) feature matrix: close + returns + rolling vol."""
        df = pd.DataFrame({"close": close.astype(float)})
        df["ret_1"] = df["close"].pct_change().fillna(0.0)
        df["ret_5"] = df["close"].pct_change(5).fillna(0.0)
        df["vol_10"] = df["ret_1"].rolling(10).std().fillna(0.0)
        df = df.dropna()
        return df.values

    # ------------------------------------------------------------------ #
    def fit(
        self,
        close: pd.Series,
        epochs: int = 30,
        batch_size: int = 32,
        train_split: float = 0.8,
        early_stopping_patience: int = 5,
        verbose: int = 0,
    ) -> LSTMTrainingResult:
        """Train the LSTM on the provided close-price series."""
        features = self._prepare_features(close)
        self.scaler_mean = features.mean(axis=0, keepdims=True)
        self.scaler_std = features.std(axis=0, keepdims=True) + 1e-9
        normed = (features - self.scaler_mean) / self.scaler_std

        X, y = self._make_windows(normed)
        split = int(len(X) * train_split)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        LOGGER.info(
            "Training LSTM: X_train=%s, X_val=%s, epochs=%d, batch=%d",
            X_train.shape, X_val.shape, epochs, batch_size,
        )

        self.model = self._build_model(n_features=features.shape[1])
        from tensorflow.keras.callbacks import EarlyStopping

        cb = [EarlyStopping(patience=early_stopping_patience, restore_best_weights=True)] if len(X_val) > 0 else []
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb,
            verbose=verbose,
        )
        return LSTMTrainingResult(
            train_loss=history.history["loss"],
            val_loss=history.history.get("val_loss", []),
            history=history,
        )

    # ------------------------------------------------------------------ #
    def predict(self, close: pd.Series) -> pd.Series:
        """Return the model's predicted next-day *normalised* close."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet - call .fit() first")
        features = self._prepare_features(close)
        normed = (features - self.scaler_mean) / self.scaler_std
        X, _ = self._make_windows(normed)
        preds_norm = self.model.predict(X, verbose=0).flatten()
        # Denormalise the first column (close)
        preds = preds_norm * self.scaler_std[0, 0] + self.scaler_mean[0, 0]
        idx = close.index[self.lookback:]
        return pd.Series(preds, index=idx, name="lstm_pred")

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save: model is not trained")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        # Save scaler params next to the model
        import json

        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lookback": self.lookback,
                    "hidden_units": self.hidden_units,
                    "dropout": self.dropout,
                    "learning_rate": self.learning_rate,
                    "scaler_mean": self.scaler_mean.tolist(),
                    "scaler_std": self.scaler_std.tolist(),
                },
                f,
            )
        LOGGER.info("Saved LSTM model to %s", path)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> "LSTMPricePredictor":
        from tensorflow.keras.models import load_model as keras_load
        import json

        path = Path(path)
        cfg = json.loads(path.with_suffix(".json").read_text())
        predictor = cls(
            lookback=cfg["lookback"],
            hidden_units=cfg["hidden_units"],
            dropout=cfg["dropout"],
            learning_rate=cfg["learning_rate"],
        )
        predictor.model = keras_load(path)
        predictor.scaler_mean = np.array(cfg["scaler_mean"])
        predictor.scaler_std = np.array(cfg["scaler_std"])
        return predictor
