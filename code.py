import pandas as pd
import os
import numpy as np

pa = "/home/kshedden/data/Tawanna_Dillahunt"

df = pd.read_csv(os.path.join(pa, "All-Valid-Responses.csv.gz"))

items = [x for x in df.columns if x.startswith("Q")]

out = open("codes.txt", "w")
for item in items:
    u = df[item].unique()
    u = [x for x in u if not (type(x) is float and np.isnan(x))]
    m = {x: 0 for x in u}
    out.write("%s\n" % item)
    for v in m:
        out.write("  %s\n" % v)
out.close()
