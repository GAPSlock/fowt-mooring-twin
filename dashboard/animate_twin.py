import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pinn')))
from model import MooringPINN

# 1. Load Data
H5_PATH = "../data/training/batch_01.h5"
df = pd.read_hdf(H5_PATH, 'simulations')
# Pick the first sea state
case_df = df[df['Meta_CaseID'] == df['Meta_CaseID'].unique()[0]].copy()

# Downsample for faster GIF generation (from 80Hz to 10Hz)
case_df = case_df.iloc[::8].reset_index(drop=True)
# Take 20 seconds of data (200 frames)
case_df = case_df.iloc[:200]

# 2. Extract inputs
X_case = case_df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
                  'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values
dt = 0.1 # because we downsampled by 8 (8 * 0.0125 = 0.1)
V_case = np.gradient(X_case, dt, axis=0)
A_case = np.gradient(V_case, dt, axis=0)
X_full_case = np.concatenate([X_case, V_case, A_case], axis=1)

# Load normalization stats
X_mean = np.load("../pinn/checkpoints/x_mean.npy")
X_std = np.load("../pinn/checkpoints/x_std.npy")
X_norm = (X_full_case - X_mean) / (X_std + 1e-8)

# 3. Predict Tensions using PINN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MooringPINN(in_features=18, hidden_dim=256, out_features=3, fourier_features=64).to(device)
model.load_state_dict(torch.load("../pinn/checkpoints/pinn_adam_final.pt", map_location=device))
model.eval()

with torch.no_grad():
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    Y_pred = model(X_tensor).cpu().numpy() # Shape: (200, 3) in kN

# Normalize tension for color mapping (assuming max 2000 kN)
T_max = 2000.0

# 4. Setup 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0f172a') # Dark background
fig.patch.set_facecolor('#0f172a')

# Geometry (Approximate OC4 DeepCwind Semi-submersible)
# Fairleads relative to platform center (radius 40.87, depth -14)
angles = np.array([0, 120, 240]) * np.pi / 180.0
r_f = 40.87
fairleads_local = np.array([
    [r_f * np.cos(angles[0]), r_f * np.sin(angles[0]), -14],
    [r_f * np.cos(angles[1]), r_f * np.sin(angles[1]), -14],
    [r_f * np.cos(angles[2]), r_f * np.sin(angles[2]), -14]
])

# Anchors (radius 837.6, depth -200)
r_a = 837.6
anchors = np.array([
    [r_a * np.cos(angles[0]), r_a * np.sin(angles[0]), -200],
    [r_a * np.cos(angles[1]), r_a * np.sin(angles[1]), -200],
    [r_a * np.cos(angles[2]), r_a * np.sin(angles[2]), -200]
])

def euler_to_matrix(roll_deg, pitch_deg, yaw_deg):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

lines = []
for i in range(3):
    line, = ax.plot([], [], [], lw=2)
    lines.append(line)
    
# Draw anchors
ax.scatter(anchors[:,0], anchors[:,1], anchors[:,2], color='white', s=50, marker='^')

# Plot limits
ax.set_xlim([-900, 900])
ax.set_ylim([-900, 900])
ax.set_zlim([-250, 50])
ax.set_axis_off() # Hide grid for cinematic look

# Normalize tension dynamically so the wave pulses are visually obvious!
# Baseline catenary tension is huge, so we need to map the color to the *dynamic variation*
T_min = np.min(Y_pred, axis=0)
T_max = np.max(Y_pred, axis=0)

def get_color(tension_kn, line_idx):
    # Map tension to a scale from Green to Red based strictly on its dynamic range
    ratio = (tension_kn - T_min[line_idx]) / (T_max[line_idx] - T_min[line_idx] + 1e-5)
    ratio = np.clip(ratio, 0, 1)
    return (ratio, 1.0 - ratio, 0.0)

def animate(frame):
    # Get 6-DOF
    surge, sway, heave, roll, pitch, yaw = X_case[frame]
    
    # Exaggerate motion by 10x for visual effect in the GIF so the rotation is obvious
    surge *= 10
    sway *= 10
    heave *= 10
    roll *= 10
    pitch *= 10
    yaw *= 10
    
    R = euler_to_matrix(roll, pitch, yaw)
    T = np.array([surge, sway, heave])
    
    for i in range(3):
        # Transform fairlead
        fl_global = R @ fairleads_local[i] + T
        anc = anchors[i]
        
        # Simple straight line for mooring
        xs = [fl_global[0], anc[0]]
        ys = [fl_global[1], anc[1]]
        zs = [fl_global[2], anc[2]]
        
        lines[i].set_data(np.array(xs), np.array(ys))
        lines[i].set_3d_properties(np.array(zs))
        
        # Set color based on PINN prediction (using dynamic scaling)
        lines[i].set_color(get_color(Y_pred[frame, i], i))
        
    ax.set_title(f"PINN Digital Twin - Time: {case_df['Time_[s]'].iloc[frame]:.1f}s", color='white')
    return lines

print("Generating 3D Animation GIF...")
ani = animation.FuncAnimation(fig, animate, frames=len(case_df), interval=100, blit=False)
ani.save('twin_simulation.gif', writer='pillow', fps=10)
print("Saved twin_simulation.gif!")
