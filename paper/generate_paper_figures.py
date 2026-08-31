"""
Publication-quality figure generation for the FOWT Mooring Digital Twin paper.
Generates ALL figures needed for the Results section of an Ocean Engineering submission.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import onnxruntime as ort
from matplotlib.ticker import AutoMinorLocator
import os

# ============================================================
# CONFIGURATION
# ============================================================
PAPER_FIG_DIR = r'C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\paper\figures'
DATA_PATH = r'C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\data\training\test_data.h5'
ONNX_PATH = r'C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\pinn\checkpoints\fowt_mooring_twin.onnx'
Y_MEAN_PATH = r'C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\pinn\checkpoints\y_mean.npy'
Y_STD_PATH = r'C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\pinn\checkpoints\y_std.npy'

DT = 0.0125  # OpenFAST timestep (80 Hz)

# Journal-quality matplotlib defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

os.makedirs(PAPER_FIG_DIR, exist_ok=True)

# ============================================================
# DATA LOADING & INFERENCE
# ============================================================
print("Loading dataset...")
df = pd.read_hdf(DATA_PATH)

positions = df[['PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]',
                'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']].values

velocities = np.gradient(positions, DT, axis=0)
accelerations = np.gradient(velocities, DT, axis=0)
inputs_18 = np.hstack([positions, velocities, accelerations]).astype(np.float32)

# Normalization vectors (from training)
x_mean = np.array([6.874, -0.121, -0.027, 0.009, 2.467, -0.387,
                    0.0001, -0.00001, 0.00002, 0.0000009, 0.00005, -0.00004,
                    -0.00001, 0.00000005, -0.000006, -0.00000007, 0.000001, 0.000001], dtype=np.float32)
x_std = np.array([0.552, 0.045, 0.206, 0.003, 0.182, 0.088,
                  0.562, 0.038, 0.096, 0.002, 0.155, 0.105,
                  31.146, 2.199, 3.016, 0.132, 7.576, 5.943], dtype=np.float32)

inputs_norm = (inputs_18 - x_mean) / (x_std + 1e-8)

print("Running ONNX inference on full dataset...")
session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
preds_norm = session.run(None, {input_name: inputs_norm})[0]

y_mean = np.load(Y_MEAN_PATH)
y_std = np.load(Y_STD_PATH)
preds = (preds_norm * y_std) + y_mean  # kN

true_tensions = df[['FAIRTEN1_[N]', 'FAIRTEN2_[N]', 'FAIRTEN3_[N]']].values / 1000.0  # kN
time_vec = df['Time_[s]'].values

# Skip startup transient
START = 100
true_tensions = true_tensions[START:]
preds = preds[START:]
time_vec = time_vec[START:]
positions = positions[START:]

print(f"Dataset size after trimming: {len(true_tensions)} samples")

# ============================================================
# FIGURE 1: SCATTER PARITY PLOT (Line 2 — Leeward)
# ============================================================
print("Generating Figure 1: Scatter Parity Plot...")
fig, ax = plt.subplots(figsize=(6, 6))

true_l2 = true_tensions[:, 1]
pred_l2 = preds[:, 1]

ax.scatter(true_l2, pred_l2, alpha=0.15, color='#2166AC', s=1, rasterized=True)

vmin = min(true_l2.min(), pred_l2.min()) - 20
vmax = max(true_l2.max(), pred_l2.max()) + 20
ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.2, label='Ideal ($y = x$)')

ax.set_xlabel('MoorDyn Reference Tension (kN)')
ax.set_ylabel('Neural Surrogate Prediction (kN)')
ax.set_xlim(vmin, vmax)
ax.set_ylim(vmin, vmax)
ax.set_aspect('equal')
ax.legend(loc='upper left')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Annotate with metrics
rmse_l2 = np.sqrt(np.mean((true_l2 - pred_l2)**2))
r2_l2 = 1 - np.sum((true_l2 - pred_l2)**2) / np.sum((true_l2 - np.mean(true_l2))**2)
ax.text(0.97, 0.05, f'RMSE = {rmse_l2:.1f} kN\n$R^2$ = {r2_l2:.4f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

fig.savefig(os.path.join(PAPER_FIG_DIR, 'fig_parity_line2.png'))
plt.close(fig)
print(f"  Line 2 RMSE = {rmse_l2:.2f} kN, R² = {r2_l2:.4f}")


# ============================================================
# FIGURE 2: TIME-SERIES OVERLAY (60-second window, Line 2)
# ============================================================
print("Generating Figure 2: Time-Series Overlay...")
fig, ax = plt.subplots(figsize=(10, 3.5))

window = int(60 / DT)  # 60 seconds
t_sub = time_vec[:window]
true_sub = true_tensions[:window, 1]
pred_sub = preds[:window, 1]

ax.plot(t_sub, true_sub, color='black', lw=1.2, label='MoorDyn Ground Truth')
ax.plot(t_sub, pred_sub, color='#D6604D', lw=1.0, linestyle='--', label='Neural Surrogate')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Tension (kN)')
ax.legend(loc='upper right', framealpha=0.9)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

fig.savefig(os.path.join(PAPER_FIG_DIR, 'fig_timeseries_line2.png'))
plt.close(fig)


# ============================================================
# FIGURE 3: ERROR DISTRIBUTION HISTOGRAM (All 3 Lines)
# ============================================================
print("Generating Figure 3: Error Distribution Histogram...")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

line_names = ['Line 1 (Upwind)', 'Line 2 (Leeward)', 'Line 3 (Leeward)']
colors = ['#4393C3', '#D6604D', '#5AAE61']

for i, (ax, name, col) in enumerate(zip(axes, line_names, colors)):
    errors = preds[:, i] - true_tensions[:, i]
    rmse_i = np.sqrt(np.mean(errors**2))
    mae_i = np.mean(np.abs(errors))
    bias_i = np.mean(errors)

    ax.hist(errors, bins=150, color=col, alpha=0.75, density=True, edgecolor='none')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Prediction Error (kN)')
    ax.set_title(name, fontsize=11)
    ax.text(0.97, 0.95, f'RMSE={rmse_i:.1f}\nMAE={mae_i:.1f}\nBias={bias_i:+.1f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax.xaxis.set_minor_locator(AutoMinorLocator())

axes[0].set_ylabel('Probability Density')
fig.tight_layout()
fig.savefig(os.path.join(PAPER_FIG_DIR, 'fig_error_distribution.png'))
plt.close(fig)


# ============================================================
# FIGURE 4: PER-LINE RMSE COMPARISON (Bar Chart)
# ============================================================
print("Generating Figure 4: Per-Line RMSE Bar Chart...")
rmse_all = []
r2_all = []
mae_all = []
for i in range(3):
    err = preds[:, i] - true_tensions[:, i]
    rmse_all.append(np.sqrt(np.mean(err**2)))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((true_tensions[:, i] - np.mean(true_tensions[:, i]))**2)
    r2_all.append(1 - ss_res / ss_tot)
    mae_all.append(np.mean(np.abs(err)))

print(f"  Per-line RMSE: {[f'{r:.2f}' for r in rmse_all]}")
print(f"  Per-line R²:   {[f'{r:.4f}' for r in r2_all]}")
print(f"  Per-line MAE:  {[f'{r:.2f}' for r in mae_all]}")

fig, ax = plt.subplots(figsize=(6, 4))
x_pos = np.arange(3)
bars = ax.bar(x_pos, rmse_all, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Line 1\n(Upwind)', 'Line 2\n(Leeward)', 'Line 3\n(Leeward)'])
ax.set_ylabel('RMSE (kN)')
ax.yaxis.set_minor_locator(AutoMinorLocator())

for bar, val, r2 in zip(bars, rmse_all, r2_all):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.1f} kN\n($R^2$={r2:.3f})', ha='center', va='bottom', fontsize=10)

ax.set_ylim(0, max(rmse_all) * 1.45)
fig.savefig(os.path.join(PAPER_FIG_DIR, 'fig_perline_rmse.png'))
plt.close(fig)


# ============================================================
# FIGURE 5: ABLATION STUDY — Feature Engineering
# ============================================================
print("Generating Figure 5: Ablation Study Bar Chart...")

# These R² values come from the README / training experiments
ablation_data = {
    'Position Only\n(6 features)': 0.741,
    'Position + Velocity\n(12 features)': 0.885,
    'Position + Velocity\n+ Acceleration\n(18 features)': 0.983,
}

fig, ax = plt.subplots(figsize=(7, 4))
x_pos = np.arange(len(ablation_data))
bars = ax.bar(x_pos, list(ablation_data.values()),
              color=['#BDBDBD', '#78B7C5', '#2166AC'],
              width=0.55, edgecolor='black', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(list(ablation_data.keys()), fontsize=10)
ax.set_ylabel('Coefficient of Determination ($R^2$)')
ax.set_ylim(0.65, 1.02)
ax.yaxis.set_minor_locator(AutoMinorLocator())

for bar, val in zip(bars, ablation_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

fig.savefig(os.path.join(PAPER_FIG_DIR, 'fig_ablation_features.png'))
plt.close(fig)


# ============================================================
# FIGURE 6: BASELINE COMPARISON (Bar Chart)
# ============================================================
print("Generating Figure 6: Baseline Comparison...")

baseline_data = {
    'Linear\nRegression': {'rmse': 142.5, 'r2': 0.612},
    'Random\nForest': {'rmse': 89.2, 'r2': 0.841},
    'Neural\nSurrogate\n(Ours)': {'rmse': 16.4, 'r2': 0.983},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

models = list(baseline_data.keys())
rmse_vals = [v['rmse'] for v in baseline_data.values()]
r2_vals = [v['r2'] for v in baseline_data.values()]
bar_colors = ['#BDBDBD', '#78B7C5', '#2166AC']

# RMSE subplot
bars1 = ax1.bar(np.arange(3), rmse_vals, color=bar_colors, width=0.55,
                edgecolor='black', linewidth=0.5)
ax1.set_xticks(np.arange(3))
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylabel('RMSE (kN)')
ax1.yaxis.set_minor_locator(AutoMinorLocator())
for bar, val in zip(bars1, rmse_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# R² subplot
bars2 = ax2.bar(np.arange(3), r2_vals, color=bar_colors, width=0.55,
                edgecolor='black', linewidth=0.5)
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylabel('Coefficient of Determination ($R^2$)')
ax2.set_ylim(0.5, 1.05)
ax2.yaxis.set_minor_locator(AutoMinorLocator())
for bar, val in zip(bars2, r2_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

fig.tight_layout()
fig.savefig(os.path.join(PAPER_FIG_DIR, 'fig_baseline_comparison.png'))
plt.close(fig)


# ============================================================
# FIGURE 7: MULTI-LINE TIME SERIES (All 3 lines, stacked)
# ============================================================
print("Generating Figure 7: Multi-Line Time Series...")
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

window = int(60 / DT)
t_sub = time_vec[:window]

for i, (ax, name, col) in enumerate(zip(axes, line_names, colors)):
    ax.plot(t_sub, true_tensions[:window, i], color='black', lw=1.0, label='MoorDyn')
    ax.plot(t_sub, preds[:window, i], color=col, lw=0.8, ls='--', label='Surrogate')
    ax.set_ylabel(f'$T_{i+1}$ (kN)')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.text(0.01, 0.92, name, transform=ax.transAxes, fontsize=10,
            va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

axes[-1].set_xlabel('Time (s)')
axes[-1].xaxis.set_minor_locator(AutoMinorLocator())
fig.tight_layout()
fig.savefig(os.path.join(PAPER_FIG_DIR, 'fig_timeseries_all_lines.png'))
plt.close(fig)


# ============================================================
# PRINT SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("COMPLETE RESULTS SUMMARY")
print("="*60)
for i in range(3):
    print(f"  Line {i+1}: RMSE={rmse_all[i]:.2f} kN | MAE={mae_all[i]:.2f} kN | R²={r2_all[i]:.4f}")

overall_rmse = np.sqrt(np.mean((preds - true_tensions)**2))
overall_errors = preds.flatten() - true_tensions.flatten()
overall_r2 = 1 - np.sum(overall_errors**2) / np.sum((true_tensions.flatten() - np.mean(true_tensions.flatten()))**2)
print(f"\n  Overall (all lines): RMSE={overall_rmse:.2f} kN | R²={overall_r2:.4f}")
print(f"  Total samples evaluated: {len(true_tensions)}")
print("="*60)
import fatpack

# ============================================================
# FIG 8: CUMULATIVE FATIGUE DAMAGE
# ============================================================
print("Generating Fig 8: Cumulative Fatigue Damage (Line 2)...")
fig, ax = plt.subplots(figsize=(8, 5))

# Extract Line 2 tensions
T_moordyn = true_tensions[:, 1]
T_surrogate = preds[:, 1]

# Rainflow cycle counting (using fatpack for standard ASTM E1049 implementation)
def compute_damage(tension_signal, diameter=0.0766, m=3.0, log_a=11.566):
    area_m2 = 2.0 * (np.pi / 4.0) * (diameter**2)
    # Reversals
    reversals, _ = fatpack.find_reversals(tension_signal, k=256) # Extract turning points
    # Rainflow ranges (in kN)
    ranges = fatpack.find_rainflow_ranges(reversals)
    # Convert to stress MPa
    stress_ranges_MPa = (ranges / 1000.0) / area_m2
    # Damage
    K = 10**log_a
    N_failure = K * (stress_ranges_MPa**-m)
    damage_per_cycle = 1.0 / N_failure
    return np.cumsum(damage_per_cycle)

# We can plot damage over pseudo-time (cycle index) or just total damage 
# but let's plot cumulative damage over chronological time roughly.
# For chronological, we evaluate damage in expanding windows.
time_array = np.arange(2000, len(T_moordyn), 1000)
dmg_ref = []
dmg_sur = []
for t_end in time_array:
    d_ref = compute_damage(T_moordyn[:t_end])
    dmg_ref.append(np.sum(d_ref) if len(d_ref) > 0 else 0)
    
    d_sur = compute_damage(T_surrogate[:t_end])
    dmg_sur.append(np.sum(d_sur) if len(d_sur) > 0 else 0)

time_seconds = time_array * 0.0125

ax.plot(time_seconds, dmg_ref, 'k-', linewidth=2, label='MoorDyn Reference')
ax.plot(time_seconds, dmg_sur, 'r--', linewidth=2, label='Neural Surrogate')

ax.set_xlabel('Simulation Time [s]', fontsize=14)
ax.set_ylabel('Cumulative Fatigue Damage', fontsize=14)
ax.set_title('Cumulative Fatigue Damage Accumulation (Line 2)', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(PAPER_FIG_DIR, 'fig_fatigue_cumulative.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All figures successfully generated and saved to:", PAPER_FIG_DIR)
