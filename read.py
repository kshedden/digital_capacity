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
demogv = ["sex", "agegrp", "race"]
demog = demog[["Response_Id"] + demogv]
demog = demog.rename({"agegrp": "age"}, axis=1)
demogv = [x if x != "agegrp" else "age" for x in demogv]

# Align demographic data with capacity data.
dmp = {}
for k in dm.keys():
    dx = pd.merge(dm[k], demog, on = "Response_Id", how="left")
    dmp[k] = dx[["Response_Id"] + demogv]

