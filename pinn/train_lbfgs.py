import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time

from model import MooringPINN, SoftAdaptLoss
from loss import compute_data_loss, compute_physics_loss

# Configuration
H5_DATA_PATH = "../data/training/train_data.h5"
STEPS_LBFGS = 500

def load_data(filepath):
    print(f"Loading dataset from {filepath}...")
    df = pd.read_hdf(filepath, 'simulations')
    
    X = df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
            'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values
    Y = df[['FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']].values / 1000.0
    
    dt = 0.0125 
    V = np.gradient(X, dt, axis=0)
    A = np.gradient(V, dt, axis=0)
    T_static = np.ones_like(Y) * Y.mean(axis=0)
    
    X_full = np.concatenate([X, V, A], axis=1)
    
    X_full_mean, X_full_std = X_full.mean(axis=0), X_full.std(axis=0)
    X_full_norm = (X_full - X_full_mean) / (X_full_std + 1e-8)
    
    return (torch.tensor(X_full_norm, dtype=torch.float32), 
            torch.tensor(Y, dtype=torch.float32),
            torch.tensor(V, dtype=torch.float32),
            torch.tensor(A, dtype=torch.float32),
            torch.tensor(T_static, dtype=torch.float32))

def train_lbfgs():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=========================================")
    print(f"L-BFGS FINE-TUNING DEVICE: {device.type.upper()}")
    print(f"=========================================\n")
    
    X, Y, V, A, T_static = load_data(H5_DATA_PATH)
    
    # Subsample to avoid memory explosion with LBFGS history (e.g. 50k points)
    N_total = X.shape[0]
    sample_size = min(N_total, 100000)
    idx = torch.randperm(N_total)[:sample_size]
    
    X = X[idx].to(device)
    Y = Y[idx].to(device)
    V = V[idx].to(device)
    A = A[idx].to(device)
    T_static = T_static[idx].to(device)
    
    model = MooringPINN(in_features=18, hidden_dim=256, out_features=3, fourier_features=64).to(device)
    
    checkpoint_path = "checkpoints/pinn_adam_final.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded Adam weights for fine-tuning.")
    else:
        print("ERROR: Adam weights not found.")
        return

    # Standard PINN L-BFGS settings
    optimizer = optim.LBFGS(
        model.parameters(), 
        lr=1.0, 
        max_iter=STEPS_LBFGS, 
        max_eval=STEPS_LBFGS * 1.25, 
        history_size=50, 
        tolerance_grad=1e-7, 
        tolerance_change=1e-9, 
        line_search_fn="strong_wolfe"
    )
    
    soft_adapt = SoftAdaptLoss(num_losses=2).to(device)
    
    print(f"--- Starting L-BFGS Optimization ({STEPS_LBFGS} steps) ---")
    start_time = time.time()
    
    step_count = [0]
    
    def closure():
        optimizer.zero_grad()
        preds = model(X)
        loss_data = compute_data_loss(preds, Y)
        loss_phys = compute_physics_loss(preds, T_static, V, A)
        
        # Keep weights fixed during closure for stability
        total_loss = loss_data + loss_phys
        total_loss.backward()
        
        if step_count[0] % 10 == 0:
            print(f"L-BFGS Step {step_count[0]:04d} | Data Loss: {loss_data.item():.4f} | Phys Loss: {loss_phys.item():.4f}")
        step_count[0] += 1
        return total_loss
        
    model.train()
    optimizer.step(closure)
                  
    print(f"L-BFGS phase complete in {(time.time() - start_time)/60:.2f} minutes.")
    
    torch.save(model.state_dict(), "checkpoints/pinn_lbfgs_final.pt")
    print("Model saved to checkpoints/pinn_lbfgs_final.pt")

if __name__ == "__main__":
    train_lbfgs()
