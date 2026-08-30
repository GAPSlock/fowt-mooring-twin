import os, sys
from openfast_toolbox.io import FASTInputFile

try:
    BASE_MODEL_DIR = os.path.expanduser('~/simulations/fowt-mooring-twin/openfast/r-test/glue-codes/openfast/5MW_OC4Semi_WSt_WavesWN')
    fst_path = os.path.join(BASE_MODEL_DIR, '5MW_OC4Semi_WSt_WavesWN.fst')
    fst = FASTInputFile(fst_path)
    
    seast_path = os.path.join(BASE_MODEL_DIR, fst['SeaStFile'].strip('"\''))
    print('SeaSt path:', seast_path)
    
    seast = FASTInputFile(seast_path)
    print('WaveMod exists?', 'WaveMod' in seast.keys())
    print('WaveHs exists?', 'WaveHs' in seast.keys())
except Exception as e:
    import traceback
    traceback.print_exc()
