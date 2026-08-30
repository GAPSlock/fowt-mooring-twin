import torch
import torch.nn as nn
import numpy as np

class FourierFeatureEmbedding(nn.Module):
    """
    Random Fourier Feature Embedding layer.
    Solves the spectral bias problem in MLPs, allowing them to learn 
    high-frequency mooring line tension variations.
    """
    def __init__(self, in_features, mapping_size=64, scale=5.0):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        
        # Frozen random normal matrix (B from Tancik et al. 2020)
        # Shape: (in_features, mapping_size)
        B = torch.randn((in_features, mapping_size)) * scale
        self.register_buffer("B", B)
        
    def forward(self, x):
        # x is (batch_size, in_features)
        # x_proj is (batch_size, mapping_size)
        x_proj = 2.0 * np.pi * x @ self.B
        
        # Output is (batch_size, 2 * mapping_size) [cos(2pi Bx), sin(2pi Bx)]
        out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return out

class MooringPINN(nn.Module):
    """
    Feedforward MLP for FOWT Mooring Tension.
    Inputs: 18 features (Pos, Vel, Acc)
    Outputs: 3 Fairlead Tensions [Line1, Line2, Line3]
    """
    def __init__(self, in_features=18, hidden_dim=256, out_features=3, fourier_features=None):
        super().__init__()
        
        # Standard Feedforward MLP (Smooth interpolation)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, x):
        return self.mlp(x)

class SoftAdaptLoss(nn.Module):
    """
    SoftAdapt dynamic loss weighting for multi-objective PINN training.
    Balances the data loss and physics losses dynamically based on their rates of change.
    """
    def __init__(self, num_losses=2, init_weights=None):
        super().__init__()
        self.num_losses = num_losses
        if init_weights is None:
            self.weights = torch.ones(num_losses)
        else:
            self.weights = torch.tensor(init_weights, dtype=torch.float32)
            
        self.loss_history = []
        
    def update_weights(self, current_losses, beta=0.1):
        """
        Update the adaptive weights.
        current_losses: list or tensor of the current epoch's unweighted losses
        """
        self.loss_history.append(current_losses)
        
        if len(self.loss_history) > 2:
            # Need at least two steps to compute rate of change
            loss_t = torch.tensor(self.loss_history[-1])
            loss_t_1 = torch.tensor(self.loss_history[-2])
            
            # Rate of change: how much the loss has decreased (or increased)
            rate_of_change = loss_t / (loss_t_1 + 1e-8)
            
            # Softmax to get new weights
            new_weights = torch.softmax(rate_of_change / beta, dim=0)
            
            # Scale by num_losses so the sum of weights = num_losses
            self.weights = new_weights * self.num_losses
            
            # Keep history manageable
            self.loss_history = self.loss_history[-5:]
            
    def get_weighted_loss(self, current_losses):
        total_loss = 0
        for i in range(self.num_losses):
            total_loss += self.weights[i] * current_losses[i]
        return total_loss
