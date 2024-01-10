import pandas as pd
from pathlib import Path
import numpy as np
from utils import *
from convert_demog import fop_demog as demog

# Path to where the data files are located
pa = Path("/home/kshedden/data/Tawanna_Dillahunt")

# Names of the files created by convert.py.
fn = ["Michiganders.csv.gz", "Detroiters.csv.gz", "Non-Michiganders.csv.gz",
      "Parkside-Residents.csv.gz", "All-Valid-Responses.csv.gz"]

out = open("fop_convert_log.txt", "w")

dx = []
for f in fn:
    df = pd.read_csv(pa / Path(f))
    dx.append(df)

out.close()

dm = {"mich": dx[0].copy(), "detroit": dx[1].copy(), "nonmich": dx[2].copy(),
      "parkside": dx[3].copy(), "allvalid": dx[4].copy()}

demog["agegrp"] = 10 * np.floor(demog["age"] / 10)
demogv = ["sex", "age", "agegrp", "race", "hhs", "money", "education", "work"]
demog = demog[["Response_Id"] + demogv].copy()

for c in demog.columns:
    demog.loc[:, c] = [x.strip() if type(x) is str else x for x in demog[c]]

for k in dm.keys():
    dm[k] = simplify_columns(dm[k])

# Align demographic data with capacity data.
dmp = {}
for k in dm.keys():
    dx = pd.merge(dm[k], demog, on = "Response_Id", how="left")
    dmp[k] = dx[["Response_Id"] + demogv]

# Save these datafiles
for k in ["allvalid", "parkside"]:
    dm[k].to_csv(Path("data") / Path("%s.csv.gz" % k), index=None)
    dmp[k].to_csv(Path("data") / Path("%s_demog.csv.gz" % k), index=None)
