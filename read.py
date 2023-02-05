import pandas as pd
import os

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
