import pandas as pd
import os
import numpy as np
from utils import do_recode

# Path where data are located
pa = "/home/kshedden/data/Tawanna_Dillahunt"

# Read all sheets into a dictionary
dfm = pd.read_excel(os.path.join(pa, "nsf.xlsx"), sheet_name=None)

out = open("convert_fop_log.txt", "w")

for (k, df) in dfm.items():

    if k.startswith("Unfiltered"):
        continue

    if k.startswith("All Valid"):
        df = df.iloc[1:, :]

    # Standardize the filename
    kx = k.replace(" ", "-")

    # Some of the files have spaces in variable names, others have underscores,
    # change everything to underscores.
    df = df.rename({x: x.replace(" ", "_") for x in df.columns}, axis=1)

    out.write("%s\n" % k)
    df = do_recode(out, df)

    # Retain only the items to be analyzed.
    vx = [x for x in df.columns if x.startswith("a_") or x.startswith("k_") or x.startswith("i_")]
    vx = ["Response_Id"] + vx
    df = df[vx]

    df.to_csv(os.path.join(pa, "%s.csv.gz" % kx), index=None)

out.close()
