import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort

print("Loading dataset...")
# Load the actual HDF5 dataset that contains the ground truth
df = pd.read_hdf('../data/training/batch_01.h5')

# The ONNX model expects 18 inputs.
# Let's rebuild the 18 inputs from the raw positions.
dt = 0.0125 # True OpenFAST dt

positions = df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
                'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values

# Calculate velocities (1st derivative)
velocities = np.gradient(positions, dt, axis=0)

# Calculate accelerations (2nd derivative)
accelerations = np.gradient(velocities, dt, axis=0)

# Combine into 18 features
inputs_18 = np.hstack([positions, velocities, accelerations]).astype(np.float32)

# Normalization (Using the exact same hardcoded values from the Unity script)
x_mean = np.array([6.874, -0.121, -0.027, 0.009, 2.467, -0.387, 0.0001, -0.00001, 0.00002, 0.0000009, 0.00005, -0.00004, -0.00001, 0.00000005, -0.000006, -0.00000007, 0.000001, 0.000001], dtype=np.float32)
x_std = np.array([0.552, 0.045, 0.206, 0.003, 0.182, 0.088, 0.562, 0.038, 0.096, 0.002, 0.155, 0.105, 31.146, 2.199, 3.016, 0.132, 7.576, 5.943], dtype=np.float32)

inputs_18_norm = (inputs_18 - x_mean) / (x_std + 1e-8)

print("Running ONNX inference...")
# Run Inference
ort_session = ort.InferenceSession('../pinn/checkpoints/fowt_mooring_twin.onnx')
input_name = ort_session.get_inputs()[0].name
predictions_norm = ort_session.run(None, {input_name: inputs_18_norm})[0]

y_mean = np.load('../pinn/checkpoints/y_mean.npy')
y_std = np.load('../pinn/checkpoints/y_std.npy')

predictions = (predictions_norm * y_std) + y_mean

# The ground truth MoorDyn tensions
true_tensions = df[['FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']].values / 1000.0 # Convert to kN

# We focus on Line 2 for the plots (highest tension line usually)
true_line2 = true_tensions[:, 1]
pred_line2 = predictions[:, 1]

# Exclude the first few seconds where derivatives are noisy due to initial zero-padding
start_idx = 100
true_line2 = true_line2[start_idx:]
pred_line2 = pred_line2[start_idx:]
time_steps = df['Time_[s]'].values[start_idx:]

print("Generating plots...")
# 1. Scatter Parity Plot
plt.figure(figsize=(8, 8))
plt.scatter(true_line2, pred_line2, alpha=0.3, color='blue', s=2)
# Draw y=x reference line
min_val = min(true_line2.min(), pred_line2.min())
max_val = max(true_line2.max(), pred_line2.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('MoorDyn Reference Tension (kN)', fontsize=14)
plt.ylabel('Neural Surrogate Prediction (kN)', fontsize=14)
plt.title('Tension Prediction Parity (Line 2)', fontsize=16)
plt.grid(True)
plt.savefig('../assets/scatter_parity_plot.png', dpi=300, bbox_inches='tight')
print("Saved scatter_parity_plot.png")

# 2. Time-Series Overlay (Plotting 60 seconds of data)
subset_len = int(60 / dt) # 60 seconds
plt.figure(figsize=(12, 4))
plt.plot(time_steps[:subset_len], true_line2[:subset_len], label='MoorDyn Ground Truth', color='black', lw=2)
plt.plot(time_steps[:subset_len], pred_line2[:subset_len], label='Neural Surrogate', color='red', linestyle='dashed', lw=2)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Tension (kN)', fontsize=14)
plt.title('Real-Time Mooring Tension Tracking (60s Window)', fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig('../assets/time_series_overlay.png', dpi=300, bbox_inches='tight')
print("Saved time_series_overlay.png")

# Optional: Print real RMSE
rmse = np.sqrt(np.mean((true_line2 - pred_line2)**2))
print(f"ACTUAL RMSE on this batch: {rmse:.2f} kN")
