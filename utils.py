import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

cm = plt.get_cmap("tab10")
colors = {"i": cm(0), "k": cm(1/10), "a": cm(2/10)}

# Map from question numbers to item labels.
vnmap = {"Q5-1_1": "a_text",
         "Q5-1_2": "a_keyboard",
         "Q5-1_3": "a_download",
         "Q5-1_4": "a_watch",
         "Q5-1_5": "a_record",
         "Q5-1_6": "a_openread",
         "Q5-1_7": "a_email",
         "Q5-2_1": "a_search",
         "Q5-2_2": "a_calendar",
         "Q5-2_3": "a_videocall",
         "Q5-2_4": "a_payment",
         "Q5-2_5": "a_purchase",
         "Q5-3_1": "a_socialmedia",
         "Q5-3_2": "a_prvsetting",
         "Q5-3_3": "a_spreadsheet",
         "Q6-1_1": "k_once",
         "Q6-1_2": "k_daily",
         "Q6-2_1": "k_fewprob",
         "Q6-2_2": "k_allprob",
         "Q6-3_1": "k_teach",
         "Q6-3_2": "k_anytask",
         "Q6-3_3": "k_provide",
         "Q6-4_1": "k_loan",
         "Q6-4_2": "k_comeover",
         "Q6-4_3": "k_place",
         "Q8_1": "i_address",
         "Q8_2": "i_reliable",
         "Q8_3": "i_place",
         "Q8_4": "i_outside"}

# Flip for visualization
def flip(load, scr):
    for j in range(load.shape[1]):
        if (load[:, j] < 0).mean() > (load[:, j] > 0).mean():
            load[:, j] *= -1
            scr[:, j] *= -1
    return load, scr

def fit_fa(df, d, cfa=None):

    df = df.copy()
    for x in df.columns:
        df[x] = df[x] - df[x].mean()
        df[x] = df[x] / df[x].std()

    if cfa is None:
        mod = sm.Factor(df, d, method="ml")
        rslt = mod.fit()
    else:
        mod = sm.Factor(df, 2, method="ml", cfa=cfa)
        rslt = mod.fit()

    return mod, rslt

def do_recode(out, df):

    df = df.copy()

    # Standardize names
    df = df.rename({"IP Address": "IP_Address", "IPAddress": "IP_Address"}, axis=1)
    df = df.rename({"ResponseId": "Response_Id"}, axis=1)

    # Initial sample size
    n0 = df.shape[0]

    n1 = len(df["IP_Address"].unique())

    # Get the total number of items with a response per submission
    v = [x for x in df.columns if any([x.startswith(y) for y in ["Q5", "Q6", "Q8"]])]
    df["nresp"] = pd.notnull(df[v]).sum(1)

    # Final sample size
    n2 = df.shape[0]

    out.write("%d initial sample size\n" % n0)
    out.write("%d unique IP addresses\n" % n1)
    out.write("%d sample size after dropping people with insufficient data\n\n" % n2)

    # Numerically recode all questions 1, 2, ... so that higher numbers correspond to
    # more positively-valenced responses.
    for (k, v) in enumerate(df.columns):
        if v.startswith("Q5"):
            df[v] = df[v].replace({"I don't know how to do this / I have never tried": 1,
                                   "I can do this, but it can be difficult": 2,
                                   "I can do this easily": 3})
        elif v.startswith("Q6") or v.startswith("Q8"):
            if df[v].dtype is np.dtype('float64'):
                df[v] += 1
            else:
                df[v] = df[v].replace({False: 1, True: 2, "False": 1, "True": 2})

    df = df.rename(vnmap, axis=1)

    for x in df.columns:
        if x.startswith("a_") or x.startswith("k_") or x.startswith("i_"):
            df[x] = pd.to_numeric(df[x])

    # Combine these two items
    df["k_fewprob"] = df["k_fewprob"] + df["k_allprob"] - 1
    df = df.drop("k_allprob", axis=1)
    df = df.rename({"k_fewprob": "k_prob"}, axis=1)

    return df
