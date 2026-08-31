import os
import subprocess
import sys

print("Running pipeline...")
os.chdir(r"C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\pinn")

def run(cmd):
    print(f"\n>>> Running: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

run("python train.py")
run("python train_lbfgs.py")
run("python train_baselines.py")

os.chdir("../paper")
run("python generate_paper_figures.py")
print("\nPipeline complete!")
