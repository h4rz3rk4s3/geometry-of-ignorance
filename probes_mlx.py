import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Optional, Tuple
import numpy as np


class LRProbe(nn.Module):
    """Logistic Regression Probe using MLX for Apple Silicon optimization."""

    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1, bias=False)

    def __call__(self, x: mx.array, iid: Optional[bool] = None) -> mx.array:
        """Forward pass through the probe."""
        logits = self.linear(x)
        return mx.sigmoid(logits.squeeze(-1))

    def pred(self, x: mx.array, iid: Optional[bool] = None) -> mx.array:
        """Get binary predictions."""
        probs = self(x, iid)
        return mx.round(probs)

    @staticmethod
    def from_data(acts: mx.array, labels: mx.array, lr: float = 0.001,
                  weight_decay: float = 0.1, epochs: int = 1000) -> 'LRProbe':
        """Create and train a logistic regression probe from data."""

        # Ensure inputs are MLX arrays
        if not isinstance(acts, mx.array):
            acts = mx.array(acts)
        if not isinstance(labels, mx.array):
            labels = mx.array(labels)

        probe = LRProbe(acts.shape[-1])
        optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

        def loss_fn(model, x, y):
            predictions = model(x)
            # Binary cross-entropy loss
            eps = 1e-15  # Small epsilon to prevent log(0)
            predictions = mx.clip(predictions, eps, 1 - eps)
            return -mx.mean(y * mx.log(predictions) + (1 - y) * mx.log(1 - predictions))

        # Training loop
        for epoch in range(epochs):
            loss, grads = mx.value_and_grad(loss_fn)(probe, acts, labels)
            optimizer.update(probe, grads)
            mx.eval(probe.parameters(), optimizer.state)

        return probe

    def __str__(self) -> str:
        return "LRProbe"

    @property
    def direction(self) -> mx.array:
        """Get the learned direction vector."""
        return self.linear.weight[0]


class MMProbe(nn.Module):
    """Mass Mean Probe using MLX for Apple Silicon optimization."""

    def __init__(self, direction: mx.array, covariance: Optional[mx.array] = None,
                 inv: Optional[mx.array] = None, atol: float = 1e-3):
        super().__init__()

        # Convert to MLX arrays if needed
        if not isinstance(direction, mx.array):
            direction = mx.array(direction)

        self.direction = direction

        if inv is None and covariance is not None:
            if not isinstance(covariance, mx.array):
                covariance = mx.array(covariance)
            # MLX doesn't have pinv, so we'll use a custom implementation
            self.inv = self._pseudo_inverse(covariance, atol=atol)
        elif inv is not None:
            if not isinstance(inv, mx.array):
                inv = mx.array(inv)
            self.inv = inv
        else:
            self.inv = None

    def _pseudo_inverse(self, matrix: mx.array, atol: float = 1e-3) -> mx.array:
        """Compute pseudo-inverse using SVD decomposition."""
        try:
            # Use SVD for pseudo-inverse calculation
            U, s, Vt = mx.linalg.svd(matrix)

            # Filter out small singular values
            s_inv = mx.where(s > atol, 1.0 / s, 0.0)

            # Reconstruct pseudo-inverse
            return Vt.T @ mx.diag(s_inv) @ U.T
        except:
            # Fallback: use regular inverse if matrix is square and well-conditioned
            try:
                return mx.linalg.inv(matrix)
            except:
                # Last resort: return identity scaled by a small factor
                return mx.eye(matrix.shape[0]) * 0.01

    def __call__(self, x: mx.array, iid: bool = False) -> mx.array:
        """Forward pass through the probe."""
        if iid and self.inv is not None:
            result = x @ self.inv @ self.direction
        else:
            result = x @ self.direction
        return mx.sigmoid(result)

    def pred(self, x: mx.array, iid: bool = False) -> mx.array:
        """Get binary predictions."""
        probs = self(x, iid=iid)
        return mx.round(probs)

    @staticmethod
    def from_data(acts: mx.array, labels: mx.array, atol: float = 1e-3) -> 'MMProbe':
        """Create a Mass Mean probe from data."""

        # Ensure inputs are MLX arrays
        if not isinstance(acts, mx.array):
            acts = mx.array(acts)
        if not isinstance(labels, mx.array):
            labels = mx.array(labels)

        # Separate positive and negative examples
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_acts = acts[pos_mask]
        neg_acts = acts[neg_mask]

        # Calculate means
        pos_mean = mx.mean(pos_acts, axis=0)
        neg_mean = mx.mean(neg_acts, axis=0)
        direction = pos_mean - neg_mean

        # Calculate covariance matrix
        pos_centered = pos_acts - pos_mean
        neg_centered = neg_acts - neg_mean
        centered_data = mx.concatenate([pos_centered, neg_centered], axis=0)
        covariance = (centered_data.T @ centered_data) / acts.shape[0]

        probe = MMProbe(direction, covariance=covariance, atol=atol)
        return probe

    def __str__(self) -> str:
        return "MMProbe"


def ccs_loss(probe: 'CCSProbe', acts: mx.array, neg_acts: mx.array) -> mx.array:
    """Compute the Contrast Consistent Search (CCS) loss."""
    p_pos = probe(acts)
    p_neg = probe(neg_acts)

    # Consistency loss: p(x) + p(not x) should equal 1
    consistency_losses = (p_pos - (1 - p_neg)) ** 2

    # Confidence loss: min(p(x), p(not x)) should be small
    stacked = mx.stack([p_pos, p_neg], axis=-1)
    confidence_losses = mx.min(stacked, axis=-1) ** 2

    return mx.mean(consistency_losses + confidence_losses)


class CCSProbe(nn.Module):
    """Contrast Consistent Search Probe using MLX for Apple Silicon optimization."""

    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1, bias=False)

    def __call__(self, x: mx.array, iid: Optional[bool] = None) -> mx.array:
        """Forward pass through the probe."""
        logits = self.linear(x)
        return mx.sigmoid(logits.squeeze(-1))

    def pred(self, acts: mx.array, iid: Optional[bool] = None) -> mx.array:
        """Get binary predictions."""
        probs = self(acts, iid)
        return mx.round(probs)

    @staticmethod
    def from_data(acts: mx.array, neg_acts: mx.array, labels: Optional[mx.array] = None,
                  lr: float = 0.001, weight_decay: float = 0.1, epochs: int = 1000) -> 'CCSProbe':
        """Create and train a CCS probe from data."""

        # Ensure inputs are MLX arrays
        if not isinstance(acts, mx.array):
            acts = mx.array(acts)
        if not isinstance(neg_acts, mx.array):
            neg_acts = mx.array(neg_acts)
        if labels is not None and not isinstance(labels, mx.array):
            labels = mx.array(labels)

        probe = CCSProbe(acts.shape[-1])
        optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

        def loss_fn(model):
            return ccs_loss(model, acts, neg_acts)

        # Training loop
        for epoch in range(epochs):
            loss, grads = mx.value_and_grad(loss_fn)(probe)
            optimizer.update(probe, grads)
            mx.eval(probe.parameters(), optimizer.state)

        # Flip direction if needed based on labels
        if labels is not None:
            predictions = probe.pred(acts)
            accuracy = mx.mean((predictions == labels).astype(mx.float32))

            if accuracy < 0.5:
                # Flip the weight direction
                probe.linear.weight = -probe.linear.weight

        return probe

    def __str__(self) -> str:
        return "CCSProbe"

    @property
    def direction(self) -> mx.array:
        """Get the learned direction vector."""
        return self.linear.weight[0]


# Utility functions for working with the probes

def evaluate_probe(probe, test_acts: mx.array, test_labels: mx.array,
                   neg_acts: Optional[mx.array] = None, iid: bool = False) -> dict:
    """Evaluate a probe on test data and return metrics."""

    if not isinstance(test_acts, mx.array):
        test_acts = mx.array(test_acts)
    if not isinstance(test_labels, mx.array):
        test_labels = mx.array(test_labels)

    # Get predictions
    if isinstance(probe, MMProbe):
        predictions = probe.pred(test_acts, iid=iid)
        probabilities = probe(test_acts, iid=iid)
    else:
        predictions = probe.pred(test_acts)
        probabilities = probe(test_acts)

    # Calculate metrics
    accuracy = mx.mean((predictions == test_labels).astype(mx.float32))

    # Calculate precision, recall, F1 for binary classification
    true_positives = mx.sum((predictions == 1) & (test_labels == 1))
    false_positives = mx.sum((predictions == 1) & (test_labels == 0))
    false_negatives = mx.sum((predictions == 0) & (test_labels == 1))

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'predictions': predictions,
        'probabilities': probabilities
    }


def save_probe(probe, filepath: str):
    """Save a trained probe to disk."""
    import pickle

    # Convert MLX arrays to numpy for saving
    probe_data = {
        'type': type(probe).__name__,
        'state_dict': {}
    }

    if hasattr(probe, 'linear'):
        probe_data['state_dict']['linear_weight'] = np.array(probe.linear.weight)
        if probe.linear.bias is not None:
            probe_data['state_dict']['linear_bias'] = np.array(probe.linear.bias)

    if isinstance(probe, MMProbe):
        probe_data['state_dict']['direction'] = np.array(probe.direction)
        if probe.inv is not None:
            probe_data['state_dict']['inv'] = np.array(probe.inv)

    with open(filepath, 'wb') as f:
        pickle.dump(probe_data, f)


def load_probe(filepath: str):
    """Load a trained probe from disk."""
    import pickle

    with open(filepath, 'rb') as f:
        probe_data = pickle.load(f)

    probe_type = probe_data['type']
    state_dict = probe_data['state_dict']

    if probe_type == 'LRProbe':
        d_in = state_dict['linear_weight'].shape[1]
        probe = LRProbe(d_in)
        probe.linear.weight = mx.array(state_dict['linear_weight'])
        if 'linear_bias' in state_dict:
            probe.linear.bias = mx.array(state_dict['linear_bias'])

    elif probe_type == 'CCSProbe':
        d_in = state_dict['linear_weight'].shape[1]
        probe = CCSProbe(d_in)
        probe.linear.weight = mx.array(state_dict['linear_weight'])
        if 'linear_bias' in state_dict:
            probe.linear.bias = mx.array(state_dict['linear_bias'])

    elif probe_type == 'MMProbe':
        direction = mx.array(state_dict['direction'])
        inv = mx.array(state_dict['inv']) if 'inv' in state_dict else None
        probe = MMProbe(direction, inv=inv)

    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    return probe