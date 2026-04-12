"""
LSTM Multi-Horizon Model
==========================
Bidirectional LSTM trained end-to-end for all horizons.
Output shape: (batch, n_horizons) for multi-task learning.
"""

import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class BiLSTMModel:
    """
    Bidirectional LSTM for multi-horizon forecasting.
    Input: (batch, context_len, 1)
    Output: (batch, n_horizons)
    """
    
    def __init__(
        self,
        context_len: int,
        n_horizons: int,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        dense_units: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """
        Args:
            context_len: input sequence length (e.g., 168 for 7 days)
            n_horizons: number of horizons to predict
            lstm_units: LSTM hidden units per direction
            dropout_rate: dropout fraction
            dense_units: dense layer size
            learning_rate: Adam learning rate
            random_state: seed
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for LSTM")
        
        self.context_len = context_len
        self.n_horizons = n_horizons
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.is_fitted = False
        self._build_model()
    
    def _build_model(self):
        """Build model architecture."""
        inputs = layers.Input(shape=(self.context_len, 1))
        
        # Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True)
        )(inputs)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Another Bidirectional LSTM
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=False)
        )(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Dense layers
        x = layers.Dense(self.dense_units, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output: one per horizon
        outputs = layers.Dense(self.n_horizons, activation="linear")(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 0
    ) -> "BiLSTMModel":
        """
        Train model.
        
        Args:
            X_train: (n_train, context_len, 1)
            y_train: (n_train, n_horizons)
            X_val: validation features
            y_val: validation targets
            epochs: training epochs
            batch_size: batch size
            verbose: verbosity (0, 1, or 2)
        
        Returns:
            self
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=self.dropout_rate,  # Reuse dropout_rate as patience proxy
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict all horizons.
        
        Args:
            X: (n_samples, context_len, 1)
        
        Returns:
            (n_samples, n_horizons)
        """
        assert self.is_fitted, "Model not fitted"
        return self.model.predict(X, verbose=0)
    
    def predict_single_horizon(self, X: np.ndarray, h_idx: int) -> np.ndarray:
        """
        Predict single horizon.
        
        Args:
            X: (n_samples, context_len, 1)
            h_idx: horizon index (0-indexed)
        
        Returns:
            (n_samples,)
        """
        assert self.is_fitted, "Model not fitted"
        pred = self.model.predict(X, verbose=0)
        return pred[:, h_idx]
    
    def save(self, dirpath: Path) -> None:
        """Save model."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        self.model.save(str(dirpath / "lstm_model.h5"))
    
    def load(self, dirpath: Path) -> "BiLSTMModel":
        """Load model."""
        dirpath = Path(dirpath)
        self.model = models.load_model(str(dirpath / "lstm_model.h5"))
        self.is_fitted = True
        return self
