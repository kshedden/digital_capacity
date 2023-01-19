"""
Calculate the mean of each item within each cohort and statistically
compare response means between cohorts.
"""

import numpy as np
import pandas as pd
from read import dm
from scipy.stats import t as tdist

def compare(d1, d2, labs, out):
    # Complement of d1 in d2
    d3 = d2.loc[~d2.Response_Id.isin(d1.Response_Id), :]

    ii = [x for x in d3.columns if x not in ["Response_Id", "mi", "nm", "pa", "dt"]]
    d1x = d1[ii]
    d3x = d3[ii]

    n1 = d1x.notnull().sum(0)
    n3 = d3x.notnull().sum(0)

    # Get the pooled standard deviation
    v1 = d1x.var(0)
    v3 = d3x.var(0)
    sp = np.sqrt(((n1-1)*v1 + (n3-1)*v3)/(n1+n3-2))

    # Compare two proportions
    mdiff = d1x.mean(0) - d3x.mean(0)
    se = sp * np.sqrt(1/n1 + 1/n3)
    z = mdiff / se
    pval = 2*tdist.cdf(-np.abs(z), n1 + n3 - 2)

    tst = pd.DataFrame(mdiff / se)
    tst.columns = ["Z-score"]
    tst[labs[0]] = d1x.mean(0)
    tst[labs[1]] = d3x.mean(0)
    tst["mean_diff"] = mdiff
    tst["pvalue"] = pval
    tst["pvalue_adj"] = np.clip(pval * tst.shape[0], 0, 1)

    out.write("==== %s (n=%d) versus %s (n=%d) ====\n" %
              (labs[0], d1x.shape[0], labs[1], d3x.shape[0]))
    out.write(tst.to_string())
    out.write("\n\n")


out = open("summary.txt", "w")

# Generate means for all items in all cohorts.
for (ky, df) in dm.items():
    df = df.drop("Response_Id", axis=1)
    n0 = df.shape[0]
    mn = df.mean(0)

    out.write("==== %s (n=%d) ====\n\n" % (ky, n0))
    out.write(mn.to_string())
    out.write("\n\n")

# Compare item means between selected pairs of cohorts
compare(dm["detroit"], dm["all"], ["Detroit", "Non-Detroit"], out)
compare(dm["mich"], dm["all"], ["Michigan", "Non-Michigan"], out)
compare(dm["parkside"], dm["mich"], ["Parkside", "Michigan"], out)

out.close()
