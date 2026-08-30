import os
import shutil
import subprocess
import multiprocessing
import numpy as np
import pandas as pd
import h5py
from openfast_toolbox.io import FASTInputFile, FASTOutputFile

# Define base paths (Linux paths for WSL)
BASE_TEST_DIR = os.path.expanduser("~/simulations/fowt-mooring-twin/openfast/r-test/glue-codes/openfast")
BASE_MODEL_DIR = os.path.join(BASE_TEST_DIR, "5MW_OC4Semi_WSt_WavesWN")
FST_BASENAME = "5MW_OC4Semi_WSt_WavesWN.fst"
WINDOWS_H5_PATH = "/mnt/c/Users/Guramrit Pal Singh/OneDrive/Desktop/Research/fowt-mooring-twin/data/training/batch_01.h5"

# Define the sea states we want to simulate
# We'll do a 4x4 matrix = 16 simulations. 
# On an 8-core machine, this is 2 batches, taking ~10-12 minutes total.
WAVE_HEIGHTS = [2.0, 4.0, 6.0, 8.0]   # Hs in meters
WAVE_PERIODS = [6.0, 8.0, 10.0, 12.0] # Tp in seconds

def setup_and_run_case(args):
    idx, hs, tp = args
    
    # Create a new directory for this case alongside the base model
    # so relative paths (../../) to the 5MW_Baseline stay valid!
    case_name = f"batch_case_{idx:03d}"
    case_dir = os.path.join(BASE_TEST_DIR, case_name)
    
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(BASE_MODEL_DIR, case_dir)
    
    fst_path = os.path.join(case_dir, FST_BASENAME)
    fst = FASTInputFile(fst_path)
    
    try:
        # 1. Modify FST
        fst['TMax'] = 600.0
        fst['OutFileFmt'] = 2
        fst.write(fst_path)
        
        # 2. Modify ServoDyn (Disable Windows DLL)
        servo_path = os.path.join(case_dir, fst['ServoFile'].strip('"\''))
        servo = FASTInputFile(servo_path)
        servo['PCMode'] = 0
        servo['VSContrl'] = 0
        servo.write(servo_path)
        
        # 3. Modify SeaState (Set wave conditions for v5.0.0)
        seast_path = os.path.join(case_dir, fst['SeaStFile'].strip('"\''))
        seast = FASTInputFile(seast_path)
        seast['WaveMod'] = 2      # 2 = JONSWAP spectrum
        seast['WaveHs'] = hs      # Significant wave height
        seast['WaveTp'] = tp      # Peak spectral period
        seast['WaveSeed(1)'] = int(np.random.randint(1, 999999)) # Randomize phase (cast to int!)
        seast.write(seast_path)
        
        # 4. Run OpenFAST
        print(f"[{case_name}] Starting OpenFAST (Hs={hs}m, Tp={tp}s)...", flush=True)
        result = subprocess.run(["openfast", FST_BASENAME], cwd=case_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[{case_name}] ERROR running OpenFAST:\n{result.stdout}", flush=True)
            return None
            
        print(f"[{case_name}] Finished OpenFAST successfully.", flush=True)
        
        # 5. Read output and return dataframe
        outb_path = os.path.join(case_dir, FST_BASENAME.replace('.fst', '.outb'))
        df = FASTOutputFile(outb_path).toDataFrame()
        
        # Add metadata columns
        df['Meta_Hs'] = hs
        df['Meta_Tp'] = tp
        df['Meta_CaseID'] = idx
        
        return df
    except Exception as e:
        import traceback
        print(f"[{case_name}] EXCEPTION: {e}\n{traceback.format_exc()}", flush=True)
        return None

if __name__ == '__main__':
    # Generate case combinations
    cases = []
    idx = 1
    for hs in WAVE_HEIGHTS:
        for tp in WAVE_PERIODS:
            cases.append((idx, hs, tp))
            idx += 1
            
    print(f"Starting batch generation of {len(cases)} simulations...", flush=True)
    print("Using 3 CPU cores to prevent memory overload. This will take ~20-30 minutes.", flush=True)
    
    # Run in parallel using 3 workers to prevent WSL crash
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(setup_and_run_case, cases)
        
    # Filter out any failed runs
    valid_dfs = [df for df in results if df is not None]
    print(f"\nCompleted {len(valid_dfs)}/{len(cases)} simulations successfully.")
    
    if len(valid_dfs) > 0:
        print(f"Saving compiled dataset to HDF5: {WINDOWS_H5_PATH}")
        # Combine all dataframes
        master_df = pd.concat(valid_dfs, ignore_index=True)
        
        # Save to HDF5 (highly compressed, fast to read in PyTorch)
        master_df.to_hdf(WINDOWS_H5_PATH, key='simulations', mode='w', format='table', data_columns=True)
        print("Dataset saved! Phase 2 Batch Generation Complete.")
