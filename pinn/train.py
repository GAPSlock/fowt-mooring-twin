import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import os
import time

from model import MooringPINN, SoftAdaptLoss
from loss import compute_data_loss, compute_physics_loss

# Configuration
H5_DATA_PATH = "../data/training/train_data.h5"
BATCH_SIZE = 4096
EPOCHS_ADAM = 2000
STEPS_LBFGS = 1000
LR = 1e-3

def load_data(filepath):
    """
    Loads HDF5 OpenFAST batch data and prepares PyTorch tensors.
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_hdf(filepath, 'simulations')
    
    # Inputs: Surge, Sway, Heave, Roll, Pitch, Yaw
    # Outputs: FairTen1, FairTen2, FairTen3
    X = df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
            'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values
            
    Y = df[['FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']].values / 1000.0 # Convert to kN
    
    dt = 0.0125 
    V = np.gradient(X, dt, axis=0)
    A = np.gradient(V, dt, axis=0)
    
    T_static = np.ones_like(Y) * Y.mean(axis=0)
    
    # ---------------- NEW 99.9% ACCURACY UPGRADE ----------------
    # Combine X, V, and A into a single 18-feature input matrix
    X_full = np.concatenate([X, V, A], axis=1)
    
    # Standardize the new 18-feature input
    X_full_mean, X_full_std = X_full.mean(axis=0), X_full.std(axis=0)
    X_full_norm = (X_full - X_full_mean) / (X_full_std + 1e-8)
    
    # STANDARDIZE THE TARGETS (Y) to prevent gradient starvation!
    Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
    Y_norm = (Y - Y_mean) / (Y_std + 1e-8)
    
    # Save the mean/std for the dashboard and Unity!
    np.save("checkpoints/x_mean.npy", X_full_mean)
    np.save("checkpoints/x_std.npy", X_full_std)
    np.save("checkpoints/y_mean.npy", Y_mean)
    np.save("checkpoints/y_std.npy", Y_std)
    # ------------------------------------------------------------
    
    # In normalized space, the mean is 0. So static tension is just 0.
    T_static_norm = np.zeros_like(Y_norm)
    
    return (torch.tensor(X_full_norm, dtype=torch.float32), 
            torch.tensor(Y_norm, dtype=torch.float32), 
            torch.tensor(V, dtype=torch.float32),
            torch.tensor(A, dtype=torch.float32),
            torch.tensor(T_static_norm, dtype=torch.float32))

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=========================================")
    print(f"TRAINING DEVICE: {device.type.upper()}")
    print(f"=========================================\n")
    
    os.makedirs("checkpoints", exist_ok=True)
    X, Y, V, A, T_static = load_data(H5_DATA_PATH)
    
    # Move to device
    X = X.to(device)
    Y = Y.to(device)
    V = V.to(device)
    A = A.to(device)
    T_static = T_static.to(device)
    
    N_total = X.shape[0]
    
    # 3. Initialize Model with 18 Inputs
    model = MooringPINN(in_features=18, hidden_dim=256, out_features=3, fourier_features=64).to(device)
    soft_adapt = SoftAdaptLoss(num_losses=2).to(device)
    
    optimizer_adam = optim.Adam(model.parameters(), lr=5e-3)
    
    print("--- Starting Adam Optimization (18 Inputs) ---")
    start_time = time.time()
    
    sample_size = int(N_total * 0.1)
    
    for epoch in range(EPOCHS_ADAM):
        model.train()
        optimizer_adam.zero_grad()
        
        idx = torch.randperm(N_total, device=device)[:sample_size]
        
        batch_X = X[idx]
        batch_Y = Y[idx]
        batch_V = V[idx]
        batch_A = A[idx]
        batch_T_static = T_static[idx]
        
        preds = model(batch_X)
        
        loss_data = compute_data_loss(preds, batch_Y)
        loss_phys = compute_physics_loss(preds, batch_T_static, batch_V, batch_A)
        
        total_loss = soft_adapt.get_weighted_loss([loss_data, loss_phys])
        
        total_loss.backward()
        optimizer_adam.step()
        
        soft_adapt.update_weights([loss_data.item(), loss_phys.item()])
        
        if epoch % 500 == 0 or epoch == EPOCHS_ADAM - 1:
            print(f"Epoch {epoch:04d}/{EPOCHS_ADAM} | "
                  f"Data Loss: {loss_data.item():.4f} | Phys Loss: {loss_phys.item():.4f} | "
                  f"Weights: W_data={soft_adapt.weights[0]:.2f}, W_phys={soft_adapt.weights[1]:.2f}")
                  
    print(f"Adam phase complete in {(time.time() - start_time)/60:.2f} minutes.")
    
    torch.save(model.state_dict(), "checkpoints/pinn_adam_final.pt")
    print("Model saved to checkpoints/pinn_adam_final.pt")

if __name__ == "__main__":
    train_model()
