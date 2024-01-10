import pandas as pd
import numpy as np
from pathlib import Path
from utils import *
from convert_demog import jfs_demog

# Path to where the data files are located
pa = Path("/home/kshedden/data/Tawanna_Dillahunt")

fn = ["JFS_Raw_Data.csv.gz", "JFS_Cleaned_Sheet.csv.gz"]

# Use cleaned data
da = pd.read_csv(pa / Path(fn[1]))
da_header = da.iloc[0, :].tolist()
da = da.iloc[1:, :]

with open("jfs_convert_log.txt", "w") as out:
    da = do_recode(out, da)

# The capacity items
z = [x for x in da.columns if x.startswith("a_") or x.startswith("k_") or x.startswith("i_") or x == "Response_Id"]
dm = da[z].copy()
for x in dm.columns[1:]:
    dm[x] = pd.to_numeric(dm[x])
