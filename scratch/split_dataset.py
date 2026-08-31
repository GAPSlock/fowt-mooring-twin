import pandas as pd
import numpy as np

H5_PATH = r"C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\data\training\batch_01.h5"

print(f"Loading {H5_PATH}...")
df = pd.read_hdf(H5_PATH)

print("Unique cases:", df['Meta_CaseID'].unique())

# Split based on Meta_CaseID. Let's reserve 15 and 16 for testing.
train_mask = ~df['Meta_CaseID'].isin(['batch_case_005', 'batch_case_006'])
test_mask = df['Meta_CaseID'].isin(['batch_case_005', 'batch_case_006'])

df_train = df[train_mask].copy()
df_test = df[test_mask].copy()

print(f"Train samples: {len(df_train)}")
print(f"Test samples: {len(df_test)}")

df_train.to_hdf(r"C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\data\training\train_data.h5", key="simulations", mode="w")
df_test.to_hdf(r"C:\Users\Guramrit Pal Singh\OneDrive\Desktop\Research\fowt-mooring-twin\data\training\test_data.h5", key="simulations", mode="w")

print("Split completed successfully!")
