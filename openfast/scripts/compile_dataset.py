import os
import pandas as pd
from openfast_toolbox.io import FASTOutputFile
import glob

BASE_TEST_DIR = os.path.expanduser("~/simulations/fowt-mooring-twin/openfast/r-test/glue-codes/openfast")
WINDOWS_H5_PATH = "/mnt/c/Users/Guramrit Pal Singh/OneDrive/Desktop/Research/fowt-mooring-twin/data/training/batch_01.h5"

print("Compiling dataset from generated OpenFAST output files...")

# Find all generated outb files
outb_files = glob.glob(os.path.join(BASE_TEST_DIR, "batch_case_*", "*.outb"))

if not outb_files:
    print("No .outb files found. Did the simulations run?")
    exit(1)

print(f"Found {len(outb_files)} output files.")

dataframes = []
for file_path in outb_files:
    try:
        # Extract case info from path
        case_name = os.path.basename(os.path.dirname(file_path))
        print(f"Processing {case_name}...")
        
        # Read the binary output
        df = FASTOutputFile(file_path).toDataFrame()
        
        # We need to extract the metadata (Hs, Tp) which we can infer, 
        # but since we didn't save it separately, let's just use it 
        # for raw training (the PINN takes platform motions as input, 
        # so Hs/Tp aren't strictly required for the PINN inputs).
        # We will just append the case name for tracking.
        df['Meta_CaseID'] = case_name
        
        dataframes.append(df)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

if dataframes:
    print("\nConcatenating dataframes...")
    master_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Saving to {WINDOWS_H5_PATH}...")
    # Using format='fixed' to avoid issues with '/' in column names
    master_df.to_hdf(WINDOWS_H5_PATH, key='simulations', mode='w', format='fixed')
    print("Success! Dataset compiled and saved.")
else:
    print("No data extracted.")
