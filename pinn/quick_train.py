import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('../pinn')
from model import MooringPINN

print('Loading data...')
df = pd.read_hdf('../data/training/batch_01.h5', 'simulations')
X = df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values
Y = df[['FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']].values / 1000.0

dt = 0.0125
V = np.gradient(X, dt, axis=0)
A = np.gradient(V, dt, axis=0)
X_full = np.concatenate([X, V, A], axis=1)

X_mean, X_std = X_full.mean(axis=0), X_full.std(axis=0)
X_norm = (X_full - X_mean) / (X_std + 1e-8)

Y_mean, Y_std = Y.mean(axis=0), Y.std(axis=0)
Y_norm = (Y - Y_mean) / (Y_std + 1e-8)

np.save('checkpoints/y_mean.npy', Y_mean)
np.save('checkpoints/y_std.npy', Y_std)

device = torch.device('cpu')
X_t = torch.tensor(X_norm, dtype=torch.float32).to(device)
Y_t = torch.tensor(Y_norm, dtype=torch.float32).to(device)

print('Training...')
model = MooringPINN(in_features=18, hidden_dim=256, out_features=3, fourier_features=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-3)

for epoch in range(1500):
    optimizer.zero_grad()
    idx = torch.randperm(X_t.shape[0])[:2048]
    preds = model(X_t[idx])
    loss = F.mse_loss(preds, Y_t[idx])
    loss.backward()
    optimizer.step()
    if epoch % 300 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'checkpoints/pinn_adam_final.pt')
print('Exporting ONNX...')
dummy = torch.randn(1, 18, device=device)
torch.onnx.export(model, dummy, 'checkpoints/fowt_mooring_twin.onnx', input_names=['kinematics'], output_names=['tensions'], dynamic_axes={'kinematics': {0: 'batch'}, 'tensions': {0: 'batch'}})
print('Done!')
