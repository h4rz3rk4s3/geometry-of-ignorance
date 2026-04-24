"""
probes.py — Probe definitions for (non-)linearity experiments.

Probes available:
  Linear:
    - LRProbe          Logistic Regression (AdamW, BCE loss)
    - RidgeProbe       Ridge Regression    (AdamW, MSE loss)
    - LinearSVMProbe   Linear SVM          (sklearn LinearSVC)

  Non-linear:
    - NonLinearProbe   Configurable-depth MLP (ReLU activations)

Factory:
    - make_nonlinear_probe(hidden_dim, depth)
        Returns a fully-configured class whose `from_data`, `pred`, and
        `__str__` behave identically to the linear probes, allowing it to
        be dropped into any ProbeClasses list without special-casing.

All probes expose the same interface:
    ProbeClass.from_data(acts, labels, **kwargs) -> probe instance
    probe.pred(acts)                             -> torch.Tensor (float, 0/1)
    ProbeClass.__str__()                         -> str name (class-level)
"""

import torch as t
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(x, device='cpu'):
    if isinstance(x, np.ndarray):
        return t.from_numpy(x).float().to(device)
    return x.float().to(device)


# ---------------------------------------------------------------------------
# Linear probes
# ---------------------------------------------------------------------------

class LRProbe(t.nn.Module):
    """
    Logistic Regression probe trained with AdamW + BCE loss.
    Equivalent to L2-regularised logistic regression.
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid(),
        )

    def forward(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self.net(x).squeeze(-1)

    def pred(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self(x).round()

    @staticmethod
    def from_data(
        acts: t.Tensor,
        labels: t.Tensor,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        epochs: int = 1000,
        device: str = 'cpu',
    ) -> 'LRProbe':
        acts, labels = _to_tensor(acts, device), _to_tensor(labels, device)
        probe = LRProbe(acts.shape[-1]).to(device)
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            t.nn.BCELoss()(probe(acts), labels).backward()
            opt.step()
        return probe

    @staticmethod
    def __str__() -> str:
        return "LRProbe"

    @property
    def direction(self) -> t.Tensor:
        return self.net[0].weight.data[0]


class RidgeProbe(t.nn.Module):
    """
    Ridge (L2-regularised linear) probe trained with AdamW + MSE loss.
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.net = t.nn.Linear(d_in, 1, bias=True)

    def forward(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self.net(x).squeeze(-1)

    def pred(self, x: t.Tensor, iid=None) -> t.Tensor:
        return (self(x) > 0).float()

    @staticmethod
    def from_data(
        acts: t.Tensor,
        labels: t.Tensor,
        lr: float = 1e-3,
        weight_decay: float = 1.0,
        epochs: int = 1000,
        device: str = 'cpu',
    ) -> 'RidgeProbe':
        acts, labels = _to_tensor(acts, device), _to_tensor(labels, device)
        probe = RidgeProbe(acts.shape[-1]).to(device)
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            t.nn.MSELoss()(probe(acts), labels).backward()
            opt.step()
        return probe

    @staticmethod
    def __str__() -> str:
        return "RidgeProbe"

    @property
    def direction(self) -> t.Tensor:
        return self.net.weight.data[0]


class LinearSVMProbe:
    """
    Linear SVM probe using sklearn's LinearSVC.

    Uses CalibratedClassifierCV to expose probability estimates (not strictly
    needed for binary accuracy but keeps the interface consistent).

    Note: this probe is NOT a torch.nn.Module — it wraps a sklearn estimator.
    It exposes the same from_data / pred / __str__ interface as the torch probes.
    """

    def __init__(self, clf):
        self._clf = clf  # fitted CalibratedClassifierCV

    def pred(self, x: t.Tensor, iid=None) -> t.Tensor:
        x_np = x.detach().cpu().numpy() if isinstance(x, t.Tensor) else x
        preds = self._clf.predict(x_np)
        return t.from_numpy(preds.astype(np.float32))

    @staticmethod
    def from_data(
        acts: t.Tensor,
        labels: t.Tensor,
        C: float = 1.0,
        max_iter: int = 2000,
        device: str = 'cpu',   # kept for API compatibility, unused
    ) -> 'LinearSVMProbe':
        acts_np = acts.detach().cpu().numpy() if isinstance(acts, t.Tensor) else acts
        labels_np = labels.detach().cpu().numpy().astype(int) if isinstance(labels, t.Tensor) else labels.astype(int)

        base = LinearSVC(C=C, max_iter=max_iter, dual='auto')
        clf = CalibratedClassifierCV(base, cv=3)
        clf.fit(acts_np, labels_np)
        return LinearSVMProbe(clf)

    @staticmethod
    def __str__() -> str:
        return "LinearSVMProbe"


# ---------------------------------------------------------------------------
# Non-linear probe
# ---------------------------------------------------------------------------

class NonLinearProbe(t.nn.Module):
    """
    Configurable-depth MLP probe for binary classification.

    Architecture:  Linear → (ReLU → Dropout → Linear) × (depth-1) → Linear(1)

    Args:
        input_dim  : Dimension of the input activation space.
        hidden_dim : Width of every hidden layer.
        depth      : Number of hidden layers (default 2 matches the paper).
        dropout    : Dropout rate applied after each hidden activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: list = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(t.nn.Linear(in_dim, hidden_dim))
            layers.append(t.nn.ReLU())
            layers.append(t.nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(t.nn.Linear(hidden_dim, 1))

        self.net = t.nn.Sequential(*layers)
        self._hidden_dim = hidden_dim
        self._depth = depth

    def forward(self, x: t.Tensor, iid=None) -> t.Tensor:
        return self.net(x).squeeze(-1)

    def pred(self, x: t.Tensor, iid=None) -> t.Tensor:
        with t.no_grad():
            probs = t.sigmoid(self(x))
        return (probs > 0.5).float()

    @staticmethod
    def from_data(
        acts: t.Tensor,
        labels: t.Tensor,
        hidden_dim: int = 512,
        depth: int = 2,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.1,
        device: str = 'cpu',
    ) -> 'NonLinearProbe':
        acts = _to_tensor(acts, device)
        labels = _to_tensor(labels, device)

        probe = NonLinearProbe(
            acts.shape[-1],
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
        ).to(device)

        loader = DataLoader(
            TensorDataset(acts, labels),
            batch_size=batch_size,
            shuffle=True,
        )
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = t.nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            probe.train()
            for batch_acts, batch_labels in loader:
                opt.zero_grad()
                criterion(probe(batch_acts), batch_labels).backward()
                opt.step()

        probe.eval()
        return probe

    @staticmethod
    def __str__() -> str:
        return "NonLinearProbe"

    @property
    def direction(self) -> t.Tensor:
        # First weight matrix — used for e.g. activation patching compatibility
        return self.net[0].weight.data[0]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_nonlinear_probe(hidden_dim: int = 512, depth: int = 2):
    """
    Returns a fully-configured NonLinearProbe *class* (not instance) baking in
    the given hidden_dim and depth.

    The returned class:
      - can be used anywhere a ProbeClass is expected
      - has a descriptive __str__() → e.g. "NonLinearProbe_128x2"
      - passes hidden_dim / depth automatically to from_data and __init__

    Example
    -------
    >>> NLP_128 = make_nonlinear_probe(128, depth=2)
    >>> probe = NLP_128.from_data(train_acts, train_labels)
    >>> NLP_128.__str__()
    'NonLinearProbe_128x2'
    """
    _hd = hidden_dim
    _d  = depth
    _name = f"NonLinearProbe_{_hd}x{_d}"

    class _ConfiguredNLP(NonLinearProbe):

        def __init__(self, input_dim: int):
            super().__init__(input_dim, hidden_dim=_hd, depth=_d)

        @staticmethod
        def from_data(
            acts: t.Tensor,
            labels: t.Tensor,
            **kwargs,
        ) -> '_ConfiguredNLP':
            # Allow caller to override, but default to baked-in values
            kwargs.setdefault('hidden_dim', _hd)
            kwargs.setdefault('depth', _d)
            device = kwargs.pop('device', 'cpu')
            return NonLinearProbe.from_data(acts, labels, device=device, **kwargs)

        @staticmethod
        def __str__() -> str:
            return _name

    _ConfiguredNLP.__name__     = _name
    _ConfiguredNLP.__qualname__ = _name
    return _ConfiguredNLP


# ---------------------------------------------------------------------------
# Convenience: default capacity-scaling suite used in the paper
# ---------------------------------------------------------------------------

#: Five MLP widths used for the non-linearity capacity sweep (paper §X).
NL_HIDDEN_DIMS = [32, 64, 128, 256, 512]

#: Two depths used for the depth-scaling ablation (paper §X).
NL_DEPTHS = [1, 2, 3]

#: Ready-made probe classes for the capacity sweep at depth=2.
CAPACITY_SWEEP_PROBES = [make_nonlinear_probe(hd, depth=2) for hd in NL_HIDDEN_DIMS]

#: All linear baselines in one list.
LINEAR_PROBES = [LRProbe, RidgeProbe, LinearSVMProbe]
