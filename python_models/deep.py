"""Deep candidates for the Python model comparison (SRS Step 7, FR xxiii).

A 1-D CNN over the mel axis, a CNN + bidirectional GRU, and a frozen MobileNetV3Small with
a new head. They use the 128 mel-band columns of the same 254-column feature vector as the
classical models, and the same split.

Each has the sklearn interface (fit, predict, predict_proba, classes_) so the tuning code
treats it like any other model. The Keras model stores its weights as bytes because Keras 3
models can't be pickled.

None of these were selected; the served model is in train_transfer.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


# Which of the 254 columns are the mel bands


def melband_columns(columns: Sequence[str]) -> list[int]:
    """Indices of the melband_* columns, found by name."""
    idx = [i for i, name in enumerate(columns) if str(name).startswith("melband_")]
    if len(idx) < 8:
        raise ValueError(
            f"expected at least 8 'melband_*' columns to build the spatial view, found "
            f"{len(idx)} in {len(list(columns))} columns"
        )
    return idx


def default_mel_indices() -> list[int]:
    from feature_extraction.features import feature_columns

    return melband_columns(feature_columns())


# Candidate spec (same shape as classical.CandidateSpec)


@dataclass
class CandidateSpec:
    """One deep candidate: how to build it and what to search."""

    name: str
    builder: Callable[[Mapping[str, Any], int, np.random.Generator | None], Any]
    param_grid: dict[str, Sequence[Any]] = field(default_factory=dict)
    preprocess: str = "none"
    notes: str = ""
    n_jobs: int = 1
    backend: str = "torch"

    def build(self, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
        return self.builder(dict(params or {}), seed, None)


# Torch modules are defined at module level so joblib can pickle them.


class _MelCNNNet(nn.Module):
    """1-D CNN over the mel frequency axis (see MelCNN)."""

    def __init__(self, n_classes: int, channels: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(channels * 2, n_classes)

    def forward(self, x):
        return self.head(self.body(x))


class _MelCRNNNet(nn.Module):
    """CNN + bidirectional GRU over frequency (see MelCRNN)."""

    def __init__(self, n_classes: int, channels: int, hidden: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        # (batch, freq, channels): frequency is the sequence axis.
        self.rnn = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        h = self.conv(x)                     # (B, C, F/2)
        h = h.transpose(1, 2)                # (B, F/2, C)
        out, _ = self.rnn(h)                 # (B, F/2, 2H)
        # first + last states of both directions (keeps order info, unlike average pooling)
        pooled = out[:, -1, :].add(out[:, 0, :])
        return self.head(self.drop(pooled))


# sklearn-style base class


class DeepModel:
    """Base class for the deep candidates.

    Store __init__ arguments under the same names; sklearn's get_params relies on it.
    """

    classes_: np.ndarray
    n_features_in_: int

    def _label_encode(self, y: Sequence[str]) -> np.ndarray:
        """Sorted like sklearn's LabelEncoder."""
        classes = np.array(sorted(set(str(v) for v in y)), dtype=object)
        self.classes_ = classes
        lookup = {c: i for i, c in enumerate(classes)}
        return np.asarray([lookup[str(v)] for v in y], dtype=np.int64)


    def _build(self) -> Any:
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _standardise(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Standardise with the training stats (zero-variance columns become 0)."""
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (X - mean) / scale


# 1. MelCNN: 1-D CNN over the mel axis


class MelCNN(DeepModel):
    """1-D CNN over the mel spectrum: two strided conv + batch-norm blocks, pooling, linear head."""

    def __init__(
        self,
        *,
        channels: int = 32,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        label_smoothing: float = 0.0,
        seed: int = 0,
        mel_indices: Sequence[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.label_smoothing = float(label_smoothing)
        self.seed = int(seed)
        self.mel_indices = list(mel_indices) if mel_indices is not None else None


    def _spatial(self, X: np.ndarray) -> np.ndarray:
        """(n, n_mels) -> (n, 1, freq) for Conv1d."""
        idx = self._mel_index_array(X.shape[1])
        block = np.ascontiguousarray(X[:, idx], dtype=np.float32)
        return block[:, None, :]  # (n, channels=1, freq)

    def _mel_index_array(self, n_features: int) -> np.ndarray:
        if self.mel_indices is None or len(self.mel_indices) > n_features:
            self.mel_indices = default_mel_indices()
        if max(self.mel_indices) >= n_features:
            raise ValueError(
                f"the mel block reaches column {max(self.mel_indices)} but the input has only "
                f"{n_features}; the feature vector and the model don't match"
            )
        return np.asarray(self.mel_indices, dtype=np.int64)

    def _build(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return _MelCNNNet(len(self.classes_), self.channels, self.dropout)

    def fit(self, X, y, sample_weight=None):
        return _train_torch(self, X, y, sample_weight)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature drift: MelCNN was fitted on {self.n_features_in_} columns but "
                f"received {X.shape[1]}"
            )
        Xs = _standardise(X, self.mean_, self.scale_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(self._spatial(Xs)))
        return torch.softmax(logits, dim=1).numpy()

    def predict_proba(self, X) -> np.ndarray:
        proba = self._forward(X)
        return _aligned(proba, self.classes_)


# 2. MelCRNN: CNN + bidirectional GRU over frequency


class MelCRNN(DeepModel):
    """CNN, then a bidirectional GRU that reads the frequency axis as a sequence."""

    def __init__(
        self,
        *,
        channels: int = 32,
        hidden: int = 96,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        label_smoothing: float = 0.0,
        seed: int = 0,
        mel_indices: Sequence[int] | None = None,
    ) -> None:
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.label_smoothing = float(label_smoothing)
        self.seed = int(seed)
        self.mel_indices = list(mel_indices) if mel_indices is not None else None

    def _spatial(self, X: np.ndarray) -> np.ndarray:
        idx = self._mel_index_array(X.shape[1])
        block = np.ascontiguousarray(X[:, idx], dtype=np.float32)
        return block[:, None, :]

    def _mel_index_array(self, n_features: int) -> np.ndarray:
        if self.mel_indices is None or len(self.mel_indices) > n_features:
            self.mel_indices = default_mel_indices()
        if max(self.mel_indices) >= n_features:
            raise ValueError(
                f"the mel block reaches column {max(self.mel_indices)} but the input has only "
                f"{n_features} columns"
            )
        return np.asarray(self.mel_indices, dtype=np.int64)

    def _build(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        return _MelCRNNNet(len(self.classes_), self.channels, self.hidden, self.dropout)

    def fit(self, X, y, sample_weight=None):
        return _train_torch(self, X, y, sample_weight)

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature drift: MelCRNN was fitted on {self.n_features_in_} columns but "
                f"received {X.shape[1]}"
            )
        Xs = _standardise(X, self.mean_, self.scale_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(self._spatial(Xs)))
        return torch.softmax(logits, dim=1).numpy()

    def predict_proba(self, X) -> np.ndarray:
        return _aligned(self._forward(X), self.classes_)


# Training loop for MelCNN and MelCRNN


def _train_torch(model: Any, X, y, sample_weight=None):
    """Training loop shared by the two torch models."""
    import torch.nn as nn

    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    model.n_features_in_ = int(X.shape[1])
    targets = model._label_encode(y)

    model.mean_ = X.mean(axis=0)
    model.scale_ = X.std(axis=0)
    model.model_ = model._build()

    Xs = _standardise(X, model.mean_, model.scale_)
    feats = torch.from_numpy(model._spatial(Xs))
    labels = torch.from_numpy(targets)
    weights = None
    if sample_weight is not None:
        weights = torch.as_tensor(
            np.asarray(sample_weight, dtype=np.float32) / float(np.mean(sample_weight)),
            dtype=torch.float32,
        )

    opt = torch.optim.AdamW(list(model.model_.parameters()), lr=model.lr, weight_decay=model.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, model.epochs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=model.label_smoothing, reduction="none")
    n = feats.shape[0]
    gen = torch.Generator().manual_seed(model.seed)

    model.model_.train()
    for _ in range(model.epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, model.batch_size):
            pick = perm[start : start + model.batch_size]
            opt.zero_grad(set_to_none=True)
            per_row = loss_fn(model.model_(feats[pick]), labels[pick])
            loss = (per_row * weights[pick]).mean() if weights is not None else per_row.mean()
            loss.backward()
            opt.step()
        sched.step()
    return model


def _aligned(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Clean (n, n_classes) probabilities that sum to 1."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    if proba.shape[1] != len(classes):
        raise ValueError(
            f"model produced {proba.shape[1]} confidence columns but there are "
            f"{len(classes)} classes"
        )
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    rowsum = proba.sum(axis=1, keepdims=True)
    dead = np.squeeze(rowsum <= 0, axis=1)
    if dead.any():
        # Rows with no output become uniform instead of NaN.
        proba[dead] = 1.0 / proba.shape[1]
        rowsum = proba.sum(axis=1, keepdims=True)
    return proba / rowsum


# 3. TransferMobileNet: pretrained ImageNet backbone with a new head


class TransferMobileNet(DeepModel):
    """Frozen ImageNet MobileNetV3Small with a new classification head.

    The 128-bin spectrum is tiled to 16 x 8, upsampled to 96 x 96 and copied to three
    channels. It did not beat the models trained from scratch (python_models/metrics/).
    """

    def __init__(
        self,
        *,
        head_units: int = 128,
        epochs: int = 40,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        label_smoothing: float = 0.0,
        seed: int = 0,
        input_size: int = 96,
        mel_indices: Sequence[int] | None = None,
        trainable_backbone: bool = False,
    ) -> None:
        self.head_units = int(head_units)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)
        self.label_smoothing = float(label_smoothing)
        self.seed = int(seed)
        self.input_size = int(input_size)
        self.mel_indices = list(mel_indices) if mel_indices is not None else None
        self.trainable_backbone = bool(trainable_backbone)


    def _mel_index_array(self, n_features: int) -> np.ndarray:
        if self.mel_indices is None or len(self.mel_indices) > n_features:
            self.mel_indices = default_mel_indices()
        if max(self.mel_indices) >= n_features:
            raise ValueError(
                f"the mel block reaches column {max(self.mel_indices)} but the input has only "
                f"{n_features} columns"
            )
        return np.asarray(self.mel_indices, dtype=np.int64)

    def _spatial(self, X: np.ndarray) -> np.ndarray:
        """``(n, 128)`` mel block -> ``(n, 16, 8, 1)`` image, ready for the resizing layer."""
        idx = self._mel_index_array(X.shape[1])
        block = np.asarray(X[:, idx], dtype=np.float32)
        n_mels = block.shape[1]
        h = 16
        w = max(1, n_mels // h)
        if h * w != n_mels:
            w = int(np.ceil(n_mels / h))
            padded = np.zeros((block.shape[0], h * w), dtype=np.float32)
            padded[:, :n_mels] = block
            block = padded
        return block.reshape(block.shape[0], h, w, 1)


    def _build(self):
        import os

        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        from keras import layers

        keras.utils.set_random_seed(self.seed)

        n_classes = len(self.classes_)
        inputs = keras.Input(shape=(16, None, 1), name="mel_tile")
        x = layers.Resizing(self.input_size, self.input_size, interpolation="bilinear")(inputs)
        x = layers.Concatenate(axis=-1)([x, x, x])  # 1 -> 3 channels

        from keras.applications import MobileNetV3Small

        # 96x96 instead of 224x224 to keep CPU training fast.
        backbone = MobileNetV3Small(
            input_shape=(self.input_size, self.input_size, 3),
            include_top=False,
            weights="imagenet",
            include_preprocessing=False,
        )
        backbone.trainable = bool(self.trainable_backbone)
        x = backbone(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(self.head_units, activation="relu")(x)
        outputs = layers.Dense(n_classes, activation="softmax", name="head")(x)

        model = keras.Model(inputs, outputs, name="ss_transfer_mobilenetv3")
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.weight_decay),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        self.backbone_name_ = backbone.name
        return model

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.n_features_in_ = int(X.shape[1])
        targets = self._label_encode(y)

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        Xs = _standardise(X, self.mean_, self.scale_)
        feats = self._spatial(Xs)

        self.model_ = self._build()
        # Shuffling uses Keras' seeded RNG.
        self.model_.fit(
            feats,
            targets,
            batch_size=min(self.batch_size, feats.shape[0]),
            epochs=self.epochs,
            shuffle=True,
            verbose=0,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32),
        )
        self._fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"feature drift: TransferMobileNet was fitted on {self.n_features_in_} columns "
                f"but received {X.shape[1]}"
            )
        Xs = _standardise(X, self.mean_, self.scale_)
        proba = np.asarray(self.model_.predict(self._spatial(Xs), verbose=0), dtype=np.float64)
        return _aligned(proba, self.classes_)


    def __getstate__(self) -> dict[str, Any]:
        """Save the weights as bytes, since Keras 3 models can't be pickled."""
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            # Keras 3 can only save to a .keras path, so go through a temp file.
            import os
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".keras")
            try:
                os.close(fd)
                model.save(tmp_path)
                with open(tmp_path, "rb") as fh:
                    state["_weights_bytes_"] = fh.read()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        weights = state.pop("_weights_bytes_", None)
        self.__dict__.update(state)
        if weights is not None:
            import os

            os.environ.setdefault("KERAS_BACKEND", "tensorflow")
            import keras

            # Same on load.
            import tempfile

            fd, tmp_path = tempfile.mkstemp(suffix=".keras")
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(weights)
                self.model_ = keras.models.load_model(tmp_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            self._fitted_ = True


# Registry


def _cnn1d(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    return MelCNN(
        channels=int(params.get("channels", 32)),
        epochs=int(params.get("epochs", 60)),
        lr=float(params.get("lr", 1e-3)),
        dropout=float(params.get("dropout", 0.3)),
        seed=seed,
    )


def _crnn(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    return MelCRNN(
        channels=int(params.get("channels", 32)),
        hidden=int(params.get("hidden", 96)),
        epochs=int(params.get("epochs", 60)),
        lr=float(params.get("lr", 1e-3)),
        dropout=float(params.get("dropout", 0.3)),
        seed=seed,
    )


def _transfer(params: Mapping[str, Any], seed: int, _rng: Any) -> Any:
    return TransferMobileNet(
        head_units=int(params.get("head_units", 128)),
        epochs=int(params.get("epochs", 40)),
        lr=float(params.get("lr", 1e-3)),
        dropout=float(params.get("dropout", 0.3)),
        input_size=int(params.get("input_size", 96)),
        seed=seed,
    )


# Small grids so a run finishes on a laptop CPU.
DEEP_CANDIDATES: dict[str, CandidateSpec] = {
    "cnn1d": CandidateSpec(
        name="cnn1d",
        builder=_cnn1d,
        backend="torch",
        param_grid={"channels": [32, 64], "epochs": [60]},
        notes=(
            "1-D CNN over the 128-bin mel spectrum from the locked feature vector: two strided "
            "convolutions with batch norm, adaptive average pooling, linear head. Frequency-"
            "translation equivariance is the inductive bias this model is here to test."
        ),
    ),
    "crnn": CandidateSpec(
        name="crnn",
        builder=_crnn,
        backend="torch",
        param_grid={"hidden": [96], "channels": [32], "epochs": [60]},
        notes=(
            "Convolutional front-end feeding a bidirectional GRU over the frequency axis with "
            "recurrent pooling (first + last state).  Tests whether the ORDER of frequency "
            "bins carries signal a purely local receptive field misses."
        ),
    ),
    "transfer": CandidateSpec(
        name="transfer",
        builder=_transfer,
        backend="keras",
        param_grid={"head_units": [128], "epochs": [40]},
        notes=(
            "Transfer learning: frozen ImageNet-pretrained MobileNetV3Small backbone, new head "
            "trained on our mel spectra.  The 128-bin spectrum is upsampled to 96x96, which "
            "loses information; the comparison table shows whether reused filters beat the "
            "from-scratch CNNs anyway. Needs the cached pretrained weights (no network "
            "access at train or inference time)."
        ),
    ),
}


def get_candidate(name: str) -> CandidateSpec:
    """Look up a deep candidate by name."""
    try:
        return DEEP_CANDIDATES[name]
    except KeyError:
        raise KeyError(
            f"unknown deep candidate {name!r}; available: {sorted(DEEP_CANDIDATES)}"
        ) from None


def build_estimator(spec: CandidateSpec, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
    """Build an unfitted deep estimator."""
    resolved = dict(params or {})
    resolved.pop("_seed", None)
    return spec.build({k: v for k, v in resolved.items() if not k.startswith("_")}, seed)


def fit_estimator(
    estimator: Any,
    X: np.ndarray,
    y: Sequence[str],
    *,
    class_weights: Mapping[str, float] | None = None,
) -> Any:
    """Fit, passing class weights as sample_weight."""
    if class_weights:
        sample_weight = sample_weights_for(y, class_weights)
        return estimator.fit(X, list(y), sample_weight=sample_weight)
    return estimator.fit(X, list(y))


def sample_weights_for(y: Sequence[str], class_weights: Mapping[str, float]) -> np.ndarray:
    """Per-row weights from a class -> weight map."""
    return np.asarray([float(class_weights.get(str(label), 1.0)) for label in y], dtype=np.float32)


def supports_sample_weight(estimator: Any) -> bool:
    """Always True for the deep models."""
    import inspect

    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def weighted_variant(spec: CandidateSpec, weights: Mapping[str, float] | None) -> CandidateSpec:
    """The same candidate with critical-class weights."""
    note = (
        spec.notes
        + " Critical-class weighted variant: critical classes are boosted "
        + f"{weights if weights else ''} via sample_weight."
    )
    return CandidateSpec(
        name=f"{spec.name}+cw",
        builder=spec.builder,
        param_grid=spec.param_grid,
        preprocess=spec.preprocess,
        notes=note,
        n_jobs=spec.n_jobs,
        backend=spec.backend,
    )


def describe_zoo() -> list[dict[str, Any]]:
    """Deep candidates, for the comparison report."""
    return [
        {
            "name": spec.name,
            "backend": spec.backend,
            "param_grid": {k: list(v) for k, v in spec.param_grid.items()},
            "notes": spec.notes,
        }
        for spec in DEEP_CANDIDATES.values()
    ]


def inference_latency_ms(estimator: Any, X: np.ndarray, repeats: int = 5) -> dict[str, float]:
    """Single-row prediction latency after one warm-up call."""
    row = np.asarray(X[:1], dtype=np.float32)
    estimator.predict_proba(row)  # warm-up
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        estimator.predict_proba(row)
        samples.append((time.perf_counter() - started) * 1000.0)
    arr = np.asarray(samples)
    return {
        "single_row_mean_ms": float(arr.mean()),
        "single_row_max_ms": float(arr.max()),
        "repeats": repeats,
    }
