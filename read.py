import pandas as pd
import os

# Path to where the data files are located
pa = "/home/kshedden/data/Tawanna_Dillahunt"

# Names of the files created by convert.py.
fn = ["Michiganders.csv.gz", "Detroiters.csv.gz", "Non-Michiganders.csv.gz",
      "Parkside-Residents.csv.gz"]

dx = []
for f in fn:
    df = pd.read_csv(os.path.join(pa, f))
    dx.append(df)

dm = {"mich": dx[0].copy(), "detroit": dx[1].copy(), "nonmich": dx[2].copy(),
      "parkside": dx[3].copy()}

# Create a pooled dataset with all cohorts and a region indicator.
for j in range(4):
    dx[j]["mi"] = 0
    dx[j]["nm"] = 0
    dx[j]["pa"] = 0
    dx[j]["dt"] = 0
    if j == 0:
        dx[j]["mi"] = 1
    elif j == 1:
        dx[j]["nm"] = 1
    elif j == 2:
        dx[j]["pa"] = 1
    elif j == 3:
        dx[j]["dt"] = 1

da = pd.concat(dx, axis=0)
dm["all"] = da
