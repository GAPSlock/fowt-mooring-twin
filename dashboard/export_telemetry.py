import pandas as pd
import numpy as np
import os

# Paths
H5_PATH = "../data/training/batch_01.h5"
CSV_PATH = "telemetry_feed.csv"

print(f"Loading data from {H5_PATH}...")
df = pd.read_hdf(H5_PATH, 'simulations')

# Get the first sea state
case_id = df['Meta_CaseID'].unique()[0]
case_df = df[df['Meta_CaseID'] == case_id].copy()

# Downsample from 80Hz to something manageable for Unity (e.g., 20Hz or 10Hz)
# Let's use 10Hz to make the file lightweight for the game engine
case_df = case_df.iloc[::8].reset_index(drop=True)

# Select only the telemetry data (Time + 6-DOF)
# We do NOT include the tension, because the PINN in Unity has to predict that!
telemetry_df = case_df[['Time_[s]', 'PtfmSurge_[m]', 'PtfmSway_[m]', 'PtfmHeave_[m]', 
                        'PtfmRoll_[deg]', 'PtfmPitch_[deg]', 'PtfmYaw_[deg]']]

print(f"Exporting {len(telemetry_df)} telemetry frames to {CSV_PATH}...")
telemetry_df.to_csv(CSV_PATH, index=False)
print("Done! Unity can now read this file to simulate live IoT sensors.")
