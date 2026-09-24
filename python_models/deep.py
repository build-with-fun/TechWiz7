"""Deep model zoo for SonicSentinel AI -- SRS Step 7, FR xxiii-xxvi.

Owner: nadia (deep audio models).

WHAT THIS IS
------------
The SRS requires "at least three models trained and compared".  ``python_models/classical.py``
defines the classical half; this module defines the DEEP half: a convolutional network, a
convolutional-recurrent network and an ImageNet transfer-learning model, all consuming the
SAME locked 254-column feature vector on the SAME frozen split.

WHY THE SAME VECTOR AS THE CLASSICAL MODELS
-------------------------------------------
The web app's inference path is:

    PythonModelPredictor.load(model_dir, FeatureExtractor())      # src/services/pipeline.py
    predictor.predict(source, preprocessor)                        # -> PredictionResult

``FeatureExtractor.extract`` returns the locked ``(254,)`` vector, and
``PythonModelPredictor._predict_features`` hands the estimator a ``(1, 254)`` matrix after
checking its width against the saved bundle.  So a model that wanted the raw ``(128, 94)``
log-mel tensor at predict time would need a *different* extractor wired into the app, and
it would no longer be comparable to the classical models -- the comparison table would mix
two input representations.

Instead every candidate here accepts ``(n, 254)`` and slices its own spatial view out of
that vector internally.  The spatial models use the 128 melband columns (``melband_000_mean``
.. ``melband_127_mean``), which ARE a mel-spaced spectrum, as their frequency axis.  The
column indices are looked up from ``feature_extraction.feature_columns()`` at fit time, never
hard-coded, so a config edit that moves the block is caught rather than silently reading the
wrong columns.

THE THREE HYPOTHESES
-------------------
* ``cnn1d``    -- a 1-D CNN over the mel frequency axis.  Convolutional layers give
  translation equivariance across frequency, which is the inductive bias that matters for a
  spectrum: a formant shifted up a few bins is the same event.
* ``crnn``    -- CNN front-end followed by a bidirectional GRU over the frequency axis with
  recurrent pooling.  Tests whether the *ordering* of frequency bins carries information a
  purely local receptive field misses.
* ``transfer`` -- a MobileNetV3Small backbone pretrained on ImageNet, frozen, with a new
  classification head trained on our mel spectra.  Genuine transfer learning: the
  low-level filters (edges, textures) are reused and only the head is learned from our data.

All three are trained from scratch or from published pretrained weights on this machine.
Nothing here calls an external generative-AI API -- the SRS forbids that for the final
classification, and the final decision in this project always comes from one of these.

THE WRAPPER CONTRACT
--------------------
Each candidate is a sklearn-compatible estimator exposing ``fit``, ``predict``,
``predict_proba`` and ``classes_``, because three pieces of infrastructure depend on exactly
that shape and must not be special-cased for deep models:

* ``tuning.build_feature_matrix`` produces ``(X, y)`` and hands it to ``fit``;
* ``tuning.TuningProtocol._predictions`` calls ``predict_proba`` and requires ``(n, n_classes)``
  in ``class_names`` order;
* ``src.inference.predictor.save_bundle`` / ``PythonModelPredictor.load`` persist and reload
  the estimator via joblib and verify ``classes_`` against the saved label order.

Standardisation is composed INSIDE the wrapper and fitted on the training rows only -- the
same guarantee ``classical.build_estimator`` gives via its Pipeline, so a scaler fitted on
the whole dataset cannot leak test statistics into training.

PICKLING
--------
``joblib.dump`` is what ``save_bundle`` uses, so both backends must survive a joblib
round-trip.  Torch modules round-trip natively; Keras 3 models do not, so the Keras wrapper
serialises its weights to bytes in ``__getstate__`` and rebuilds the architecture in
``__setstate__``.  Either way the saved bundle is loadable by the one writer, one format the
app already speaks.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

REPO_ROOT_SENTINEL = None  # replaced at first use; keeps import side effects out

# --------------------------------------------------------------------------------------
# The spatial slice: which of the 254 columns form the mel spectrum
# --------------------------------------------------------------------------------------


def melband_columns(columns: Sequence[str]) -> list[int]:
    """Indices of the ``melband_*`` columns inside the locked feature vector.

    Derived from the column list rather than fixed numbers, so the spatial models read the
    spectrum the extractor actually produced.  Raises if the block is absent -- a deep model
    with no frequency axis is not trainable and failing loudly beats training on the wrong
    columns.
    """
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


# --------------------------------------------------------------------------------------
# Candidate specification (mirrors classical.CandidateSpec so the grid builders are shared)
# --------------------------------------------------------------------------------------


@dataclass
class CandidateSpec:
    """One comparable deep model: how to build it, what to search, and what it costs."""

    name: str
    builder: Callable[[Mapping[str, Any], int, np.random.Generator | None], Any]
    param_grid: dict[str, Sequence[Any]] = field(default_factory=dict)
    preprocess: str = "none"
    notes: str = ""
    n_jobs: int = 1
    backend: str = "torch"

    def build(self, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
        return self.builder(dict(params or {}), seed, None)


# --------------------------------------------------------------------------------------
# Torch architectures, defined at MODULE level so joblib can pickle a fitted estimator.
#
# A class declared inside a method has no importable qualified name, so pickle cannot express
# "rebuild this object" -- and ``save_bundle`` joblib-dumps the estimator.  Keeping them here
# also makes the two architectures diffable side by side, which is what the comparison
# between cnn1d and crnn actually turns on.
# --------------------------------------------------------------------------------------


class _MelCNNNet(nn.Module):
    """1-D CNN over the mel frequency axis.  See :class:`MelCNN` for the rationale."""

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
    """CNN front-end + bidirectional GRU over frequency.  See :class:`MelCRNN`."""

    def __init__(self, n_classes: int, channels: int, hidden: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        # (batch, freq, channels) -- the frequency axis becomes the sequence.
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
        h = h.transpose(1, 2)                # (B, F/2, C) -- sequence over frequency
        out, _ = self.rnn(h)                 # (B, F/2, 2H)
        # first + last state of both directions: recurrent pooling, which keeps the ordering
        # information a global average pool would discard.
        pooled = out[:, -1, :].add(out[:, 0, :])
        return self.head(self.drop(pooled))


# --------------------------------------------------------------------------------------
# Shared sklearn-compat machinery
# --------------------------------------------------------------------------------------


class DeepModel:
    """Base class giving every deep candidate the sklearn-shaped surface the harness needs.

    Subclasses implement :meth:`_build`, :meth:`_fit_backend` and :meth:`_proba_backend`.
    ``__init__`` parameters must be stored as same-named attributes -- sklearn's
    ``BaseEstimator.get_params`` introspects the signature, and ``save_bundle``'s callers
    expect ``get_params`` to round-trip.
    """

    # Declared so mypy/inspect see the public sklearn surface even before fitting.
    classes_: np.ndarray
    n_features_in_: int

    def _label_encode(self, y: Sequence[str]) -> np.ndarray:
        """``np.unique`` on strings is sorted, so ``classes_`` has a canonical order that is
        identical to sklearn's ``LabelEncoder`` and therefore to the classical models' order.
        That is what makes ``set(estimator.classes_) == set(saved_classes)`` hold at load."""
        classes = np.array(sorted(set(str(v) for v in y)), dtype=object)
        self.classes_ = classes
        lookup = {c: i for i, c in enumerate(classes)}
        return np.asarray([lookup[str(v)] for v in y], dtype=np.int64)

    # -- to be provided by each backend ---------------------------------------------

    def _build(self) -> Any:
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError

    def predict_proba(self, X) -> np.ndarray:
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _standardise(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply the training statistics.  Zero-variance columns are left at 0, not divided by 0."""
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    return (X - mean) / scale


# --------------------------------------------------------------------------------------
# 1. MelCNN -- a 1-D CNN over the mel frequency axis
# --------------------------------------------------------------------------------------


class MelCNN(DeepModel):
    """Convolutional network over the mel spectrum.

    The 128 melband columns become a single-channel signal along the frequency axis; two
    strided convolutions with batch norm and ReLU build a hierarchy of frequency-local
    features, an adaptive average pool collapses the axis, and a linear head classifies.
    Batch norm here is what makes a small CNN trainable on ~2,000 rows without careful
    initialisation.
    """

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

    # -- backend ---------------------------------------------------------------------

    def _spatial(self, X: np.ndarray) -> np.ndarray:
        """``(n, n_mels)`` mel block from the locked vector, as a torch-shaped input."""
        idx = self._mel_index_array(X.shape[1])
        block = np.ascontiguousarray(X[:, idx], dtype=np.float32)
        return block[:, None, :]  # (n, channels=1, freq)

    def _mel_index_array(self, n_features: int) -> np.ndarray:
        if self.mel_indices is None or len(self.mel_indices) > n_features:
            self.mel_indices = default_mel_indices()
        if max(self.mel_indices) >= n_features:
            raise ValueError(
                f"the mel block reaches column {max(self.mel_indices)} but the input has only "
                f"{n_features} columns -- the feature vector and the model are out of sync"
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


# --------------------------------------------------------------------------------------
# 2. MelCRNN -- CNN front-end + bidirectional GRU over frequency, recurrent pooling
# --------------------------------------------------------------------------------------


class MelCRNN(DeepModel):
    """Convolutional-recurrent network.

    The same mel view, but after one strided convolution the frequency axis is treated as a
    SEQUENCE and a bidirectional GRU reads it bin by bin.  The final hidden states of both
    directions are concatenated -- recurrent pooling, which keeps order information that a
    global average pool destroys.  This is the model that tests whether frequency *ordering*
    is informative beyond local texture.
    """

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


# --------------------------------------------------------------------------------------
# Torch training loop, shared by MelCNN and MelCRNN
# --------------------------------------------------------------------------------------


def _train_torch(model: Any, X, y, sample_weight=None):
    """The shared torch optimisation loop.

    Factored out because MelCNN and MelCRNN differ only in architecture; sharing the loop
    means the optimiser, the class-weight handling and the seeding cannot drift between them.
    """
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
    """Guarantee ``(n, len(classes))`` finite probabilities summing to 1.

    ``normalise_confidences`` on the app side clamps NaN/inf, but the harness checks the
    column count strictly, so a degenerate row here would raise deep inside selection.
    """
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
        # A row the model is genuinely silent on becomes uniform rather than a NaN-laden
        # spike; the report then sees "Uncertain", which is the honest reading.
        proba[dead] = 1.0 / proba.shape[1]
        rowsum = proba.sum(axis=1, keepdims=True)
    return proba / rowsum


# --------------------------------------------------------------------------------------
# 3. TransferMobileNet -- ImageNet-pretrained backbone, new head on our spectra
# --------------------------------------------------------------------------------------


class TransferMobileNet(DeepModel):
    """Transfer learning: a frozen MobileNetV3Small backbone with a new classification head.

    The 128-bin mel spectrum is reshaped to ``(16, 8, 1)`` and bilinearly upsampled inside
    the graph to the backbone's ``96x96`` input, then tiled to three channels so the
    pretrained first-layer convolutions see the shape they were trained on.  The backbone is
    FROZEN -- its weights are the ImageNet ones downloaded once and cached under
    ``~/.keras/models`` -- and only the head is optimised on our labels.

    That is what makes this transfer learning rather than training from scratch: the
    low-level filters are reused unchanged and the number of parameters we must learn from
    ~2,000 rows is small.  Upsampling a spectrum is a loss of information, and the docstring
    here says so rather than hiding it -- the comparison table will show whether the reused
    filters beat the from-scratch CNNs on this data.
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

    # -- spatial view ----------------------------------------------------------------

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
        h = int(np.sqrt(n_mels / 2))  # 128 -> 8 ... but we want (16, 8)
        # Deterministic reshape of the spectrum into a small 2-D tile.
        h = 16
        w = max(1, n_mels // h)
        if h * w != n_mels:
            w = int(np.ceil(n_mels / h))
            padded = np.zeros((block.shape[0], h * w), dtype=np.float32)
            padded[:, :n_mels] = block
            block = padded
        return block.reshape(block.shape[0], h, w, 1)

    # -- backend ---------------------------------------------------------------------

    def _build(self):
        import os

        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        from keras import layers

        keras.utils.set_random_seed(self.seed)

        n_classes = len(self.classes_)
        inputs = keras.Input(shape=(16, None, 1), name="mel_tile")
        x = layers.Resizing(self.input_size, self.input_size, interpolation="bilinear")(inputs)
        x = layers.Concatenate(axis=-1)([x, x, x])  # 1 -> 3 channels for the pretrained convs

        from keras.applications import MobileNetV3Small

        # The published ImageNet weights were trained at 224x224.  Feeding a smaller spatial
        # size still loads them -- Keras handles the shape mismatch -- but warns loudly, which
        # is correct behaviour, not noise: 96x96 is a deliberate accuracy/speed trade for a
        # CPU box, and the comparison table shows what it costs.
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
        # The backbone is frozen, so a fit-time shuffle uses Keras' own seeded RNG.
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

    # -- joblib round-trip -----------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Keras 3 models are not joblib-picklable, so serialise weights to bytes.

        ``save_bundle`` joblib-dumps the estimator, and the saved bundle must be loadable by
        ``PythonModelPredictor.load`` without a network connection -- so the architecture is
        rebuilt from the stored params and only the weights travel as bytes.
        """
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            # Keras 3 refuses to save to a file-like object (it dispatches on the path's
            # extension), so the model is written to a temporary ``.keras`` file and only
            # the bytes travel inside the joblib bundle.
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

            # Same constraint on load: a ``.keras`` path, never a BytesIO.
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


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------


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


#: The comparable deep candidates.  Grids are deliberately small -- on a CPU box with ~3.5 h
#: left in the competition, one well-chosen point per axis beats a sweep that does not finish.
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
            "from-scratch CNNs anyway.  Requires the cached pretrained weights -- no network "
            "access at train or inference time."
        ),
    ),
}


def get_candidate(name: str) -> CandidateSpec:
    """Fetch a registered deep candidate.  Raises rather than returning a half-built model."""
    try:
        return DEEP_CANDIDATES[name]
    except KeyError:
        raise KeyError(
            f"unknown deep candidate {name!r}; available: {sorted(DEEP_CANDIDATES)}"
        ) from None


def build_estimator(spec: CandidateSpec, params: Mapping[str, Any] | None = None, seed: int = 0) -> Any:
    """A ready-to-fit deep estimator.  Standardisation is inside ``fit``, so no outer pipeline."""
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
    """Fit, routing class weights through ``sample_weight`` -- the mechanism every deep model
    shares, so critical-class boosting applies uniformly and cannot be silently dropped.

    ``sample_weight`` is normalised to mean 1 inside the training loop, so passing weights
    does not also change the effective learning rate.
    """
    if class_weights:
        sample_weight = sample_weights_for(y, class_weights)
        return estimator.fit(X, list(y), sample_weight=sample_weight)
    return estimator.fit(X, list(y))


def sample_weights_for(y: Sequence[str], class_weights: Mapping[str, float]) -> np.ndarray:
    """Per-row weights from a class -> weight map.  Critical classes get the boost."""
    return np.asarray([float(class_weights.get(str(label), 1.0)) for label in y], dtype=np.float32)


def supports_sample_weight(estimator: Any) -> bool:
    """Deep models all accept ``sample_weight``; reported for symmetry with the classical zoo."""
    import inspect

    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False


def weighted_variant(spec: CandidateSpec, weights: Mapping[str, float] | None) -> CandidateSpec:
    """The same candidate with critical-class weights applied.  Mirrors ``classical.weighted_variant``
    so the deep and classical grids are built by identical code and the comparison table can
    ask 'did weighting help?' of both families."""
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
    """What the comparison report lists as the deep models we trained and compared."""
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
    """Single-row latency, after a warm-up call.

    The SRS gives a live window a 3-second budget and an upload 8 seconds.  A deep model that
    wins on macro-F1 but blows the latency budget is not deployable, so this is measured and
    reported next to the score rather than assumed.
    """
    row = np.asarray(X[:1], dtype=np.float32)
    estimator.predict_proba(row)  # warm up lazy imports and thread pools
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
