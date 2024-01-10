import pandas as pd
import os
import numpy as np

# Path to where the data files are located
pa = "/home/kshedden/data/Tawanna_Dillahunt"

# Names of the files created by convert.py.
fn = ["Michiganders.csv.gz", "Detroiters.csv.gz", "Non-Michiganders.csv.gz",
      "Parkside-Residents.csv.gz", "All-Valid-Responses.csv.gz"]

dx = []
for f in fn:
    df = pd.read_csv(os.path.join(pa, f))
    dx.append(df)

dm = {"mich": dx[0].copy(), "detroit": dx[1].copy(), "nonmich": dx[2].copy(),
      "parkside": dx[3].copy(), "allvalid": dx[4].copy()}

demog = pd.read_csv(os.path.join(pa, "demog.csv.gz"))

demog["agegrp"] = 10 * np.floor(demog["age"] / 10)
demogv = ["sex", "age", "agegrp", "race", "hhs", "money", "education", "work"]
demog = demog[["Response_Id"] + demogv]

for c in demog.columns:
    demog[c] = [x.strip() if type(x) is str else x for x in demog[c]]

# Align demographic data with capacity data.
dmp = {}
for k in dm.keys():
    dx = pd.merge(dm[k], demog, on = "Response_Id", how="left")
    dmp[k] = dx[["Response_Id"] + demogv]

# Save these datafiles
for k in ["allvalid", "parkside"]:
    dm[k].to_csv(os.path.join("data", "%s.csv.gz" % k), index=None)
    dmp[k].to_csv(os.path.join("data", "%s_demog.csv.gz" % k), index=None)
