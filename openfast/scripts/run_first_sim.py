import os
import subprocess
import matplotlib.pyplot as plt
from openfast_toolbox.io import FASTInputFile, FASTOutputFile

# Define paths
# Note: This script runs natively in WSL, so we use Linux paths
sim_dir = os.path.expanduser("~/simulations/fowt-mooring-twin/openfast/r-test/glue-codes/openfast/5MW_OC4Semi_WSt_WavesWN")
fst_file_path = os.path.join(sim_dir, "5MW_OC4Semi_WSt_WavesWN.fst")
out_file_path = os.path.join(sim_dir, "5MW_OC4Semi_WSt_WavesWN.outb")
plot_save_path = "/mnt/c/Users/Guramrit Pal Singh/OneDrive/Desktop/Research/fowt-mooring-twin/openfast/results/first_run_plot.png"

print("1. Modifying FAST input file for a 600-second run...")
fst = FASTInputFile(fst_file_path)
fst['TMax'] = 600.0  # Set simulation time to 600 seconds
fst['OutFileFmt'] = 2  # Binary output

# Also we must disable the external Windows DLL controller since we are on Linux
servo_file_path = os.path.join(sim_dir, fst['ServoFile'].strip('"\''))
servo = FASTInputFile(servo_file_path)
servo['PCMode'] = 0     # Disable pitch control DLL
servo['VSContrl'] = 0   # Disable generator torque DLL
servo.write(servo_file_path)

fst.write(fst_file_path)

print("\n2. Running OpenFAST simulation (this will take 5-10 minutes)...")
# Run openfast in the simulation directory
result = subprocess.run(["openfast", "5MW_OC4Semi_WSt_WavesWN.fst"], cwd=sim_dir, capture_output=True, text=True)

if result.returncode != 0:
    print("Error running OpenFAST!")
    print(result.stdout)
    exit(1)

print("\n3. Simulation complete! Reading output data...")
# Read the binary output file
out_data = FASTOutputFile(out_file_path).toDataFrame()

# Extract variables
time = out_data['Time_[s]']
surge = out_data['PtfmSurge_[m]']
heave = out_data['PtfmHeave_[m]']
pitch = out_data['PtfmPitch_[deg]']

# Mooring line tensions (Fairlead)
ten1 = out_data['FAIRTEN1_[N]'] / 1000.0  # Convert to kN
ten2 = out_data['FAIRTEN2_[N]'] / 1000.0
ten3 = out_data['FAIRTEN3_[N]'] / 1000.0

print("\n4. Generating plots...")
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: Platform Surge & Heave
axs[0].plot(time, surge, label='Surge (m)', color='blue')
axs[0].plot(time, heave, label='Heave (m)', color='green')
axs[0].set_ylabel('Displacement [m]')
axs[0].set_title('Platform Motion')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.5)

# Plot 2: Platform Pitch
axs[1].plot(time, pitch, label='Pitch (deg)', color='orange')
axs[1].set_ylabel('Rotation [deg]')
axs[1].legend(loc='upper right')
axs[1].grid(True, alpha=0.5)

# Plot 3: Mooring Tensions
axs[2].plot(time, ten1, label='Line 1 (Upwind)', color='red')
axs[2].plot(time, ten2, label='Line 2', color='purple')
axs[2].plot(time, ten3, label='Line 3', color='brown')
axs[2].set_xlabel('Time [s]')
axs[2].set_ylabel('Fairlead Tension [kN]')
axs[2].legend(loc='upper right')
axs[2].grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig(plot_save_path, dpi=300)
print(f"\nSuccess! Plot saved to Windows at: {plot_save_path}")
