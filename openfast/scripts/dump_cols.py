import os
from openfast_toolbox.io import FASTOutputFile

out_file_path = os.path.expanduser("~/simulations/fowt-mooring-twin/openfast/r-test/glue-codes/openfast/5MW_OC4Semi_WSt_WavesWN/5MW_OC4Semi_WSt_WavesWN.outb")
df = FASTOutputFile(out_file_path).toDataFrame()

with open("/mnt/c/Users/Guramrit Pal Singh/OneDrive/Desktop/Research/fowt-mooring-twin/openfast/results/cols.txt", "w") as f:
    for col in df.columns:
        f.write(col + "\n")
