import os
import matplotlib.pyplot as plt
from openfast_toolbox.io import FASTOutputFile

sim_dir = os.path.expanduser("~/simulations/fowt-mooring-twin/openfast/r-test/glue-codes/openfast/5MW_OC4Semi_WSt_WavesWN")
out_file_path = os.path.join(sim_dir, "5MW_OC4Semi_WSt_WavesWN.outb")
plot_save_path = "/mnt/c/Users/Guramrit Pal Singh/OneDrive/Desktop/Research/fowt-mooring-twin/openfast/results/first_run_plot.png"

print("Reading output data...")
out_data = FASTOutputFile(out_file_path).toDataFrame()

# Extract variables
time = out_data['Time_[s]']
surge = out_data['PtfmSurge_[m]']
heave = out_data['PtfmHeave_[m]']
pitch = out_data['PtfmPitch_[deg]']

# Mooring line tensions (Fairlead) - note uppercase for v5.0.0
ten1 = out_data['FAIRTEN1_[N]'] / 1000.0  
ten2 = out_data['FAIRTEN2_[N]'] / 1000.0
ten3 = out_data['FAIRTEN3_[N]'] / 1000.0

print("Generating plots...")
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(time, surge, label='Surge (m)', color='blue')
axs[0].plot(time, heave, label='Heave (m)', color='green')
axs[0].set_ylabel('Displacement [m]')
axs[0].set_title('Platform Motion & Mooring Tension (OC4-DeepCwind)')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.5)

axs[1].plot(time, pitch, label='Pitch (deg)', color='orange')
axs[1].set_ylabel('Rotation [deg]')
axs[1].legend(loc='upper right')
axs[1].grid(True, alpha=0.5)

axs[2].plot(time, ten1, label='Line 1 (Upwind)', color='red')
axs[2].plot(time, ten2, label='Line 2', color='purple')
axs[2].plot(time, ten3, label='Line 3', color='brown')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Fairlead Tension [kN]')
axs[2].legend(loc='upper right')
axs[2].grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig(plot_save_path, dpi=300)
print(f"Success! Plot saved to: {plot_save_path}")
