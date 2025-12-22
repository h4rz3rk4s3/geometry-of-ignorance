import torch as t
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

class LRProbe(t.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 1, bias=False),
            t.nn.Sigmoid()
        )

    def forward(self, x, iid=None):
        return self.net(x).squeeze(-1)

    def pred(self, x, iid=None):
        return self(x).round()
    
    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device='cpu'):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            opt.zero_grad()
            loss = t.nn.BCELoss()(probe(acts), labels) #BCEWithLogitsLoss, BCELoss
            loss.backward()
            opt.step()
        
        return probe

    def __str__():
        return "LRProbe"

    @property
    def direction(self):
        return self.net[0].weight.data[0]


class MMProbe(t.nn.Module):
    def __init__(self, direction, covariance=None, inv=None, atol=1e-3):
        super().__init__()
        self.direction = t.nn.Parameter(direction, requires_grad=False).to("cpu")
        # if covariance:
        #     covariance = covariance.to("cpu")
        #TODO: Move eigenvalue calc to CPU, as the MPS backend is not available in PyTorch.
        if inv is None:
            self.inv = t.nn.Parameter(t.linalg.pinv(covariance.cpu(), hermitian=True, atol=atol), requires_grad=False).to("cpu")
        else:
            self.inv = t.nn.Parameter(inv, requires_grad=False).to("cpu")

    def forward(self, x, iid=False):
        if iid:
            return t.nn.Sigmoid()(x @ self.inv @ self.direction).to("cpu")
        else:
            return t.nn.Sigmoid()(x @ self.direction).to("cpu")

    def pred(self, x, iid=False):
        return self(x.to("cpu"), iid=iid).round()

    def from_data(acts, labels, atol=1e-3, device='cpu'):
        acts, labels
        pos_acts, neg_acts = acts[labels==1], acts[labels==0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
        direction = pos_mean - neg_mean

        centered_data = t.cat([pos_acts - pos_mean, neg_acts - neg_mean], 0)
        covariance = centered_data.t() @ centered_data / acts.shape[0]
        
        probe = MMProbe(direction, covariance=covariance).to(device)

        return probe
    
    def __str__():
        return "MMProbe"

class NonLinearProbe(t.nn.Module):
    """
    Two-layer MLP probe for toxicity classification.
    
    Args:
        input_dim: Dimension of the input activation space
        hidden_dim: Dimension of the hidden layer (default: 512)
        activation: Activation function ('relu', 'gelu', 'silu') (default: 'relu')
        dropout: Dropout rate (default: 0.1)
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 512,
        activation: str = 'silu',
        dropout: float = 0.1
    ):
        super(NonLinearProbe, self).__init__()
        
        # Choose activation function
        activations = {
            'relu': t.nn.ReLU(),
            'gelu': t.nn.GELU(),
            'silu': t.nn.SiLU()
        }
        self.activation_fn = activations.get(activation.lower(), t.nn.ReLU())
        
        # Two-layer MLP
        self.fc1 = t.nn.Linear(input_dim, hidden_dim)
        self.dropout = t.nn.Dropout(dropout)
        self.fc2 = t.nn.Linear(hidden_dim, 1)  # Binary classification
        
    def forward(self, x: t.Tensor, iid=None) -> t.Tensor:
        """
        Forward pass through the probe.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            logits: Output tensor of shape (batch_size, 1)
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def pred(
        self,
        x: t.Tensor,
        device: str = 'mps' if t.mps.is_available() else 'cpu',
        return_probs: bool = False,
        iid=None
    ) -> np.ndarray:
        """
        Run inference with the probe on new activations.
        
        Args:
            probe: The trained NonLinearProbe model
            activations: Input activations (n_samples, hidden_dim)
            device: Device to run inference on
            return_probs: If True, return probabilities; if False, return binary predictions
            
        Returns:
            predictions: Binary predictions (0/1) or probabilities, shape (n_samples,)
        """
        x = x.to(device)
        
        with t.no_grad():
            logits = self(x).squeeze()
            probs = t.sigmoid(logits)
            
            if return_probs:
                return probs.cpu().numpy()
            else:
                preds = (probs > 0.5).float()
                return preds.cpu().numpy()
            
    def from_data(
            acts: t.Tensor, 
            labels: t.Tensor,
            epochs: int = 10,
            batch_size: int = 32,
            lr: float = 1e-4,
            weight_decay: float = 1e-4, 
            device: str = 'mps' if t.mps.is_available() else 'cpu'):
       
        acts, labels = acts.to(device), labels.to(device).float()
        probe = NonLinearProbe(acts.shape[-1]).to(device)

        history = {
            'train_loss': [],
            'train_acc': [],
        }

        train_dataset = TensorDataset(acts, labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        opt = t.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = t.nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            probe.train()
            train_loss = 0.0
            train_preds = []
            train_true = []

            for batch_acts, batch_labels in train_loader:
                opt.zero_grad()
                logits = probe(batch_acts).squeeze()
                loss = criterion(logits, batch_labels)
                loss.backward()
                opt.step()
                
                train_loss += loss.item()
                preds = (t.sigmoid(logits) > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(batch_labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_true, train_preds)
            #print(f"Train Loss: {train_loss}.  Train Acc: {train_acc}. Epoch: {_}/{epochs}.")
        
        return probe
    
    def __str__():
        return "NonLinearProbe"