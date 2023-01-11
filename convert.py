import pandas as pd
import os
import numpy as np

# Path where data are located
pa = "/home/kshedden/data/Tawanna_Dillahunt"

# Read all sheets into a dictionary
dfm = pd.read_excel(os.path.join(pa, "nsf.xlsx"), sheet_name=None)

out = open("convert_log.txt", "w")

# Map from question numbers to item labels.
vn = {"Q5-1_1": "a_text",
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
      "Q6-1_1": "know_once",
      "Q6-1_2": "know_daily",
      "Q6-2_1": "know_fewprob",
      "Q6-2_2": "know_allprob",
      "Q6-3_1": "know_teach",
      "Q6-3_2": "know_anytask",
      "Q6-3_3": "know_provide",
      "Q6-4_1": "know_loan",
      "Q6-4_2": "know_comeover",
      "Q6-4_3": "know_place",
      "Q8_1": "internet_address",
      "Q8_2": "internet_reliable",
      "Q8_3": "internet_place",
      "Q8_4": "internet_outside"}

def do_recode(k, df):

    # Standardize names
    df = df.rename({"IP Address": "IP_Address", "IPAddress": "IP_Address"}, axis=1)
    df = df.rename({"ResponseId": "Response_Id"}, axis=1)

    # Initial sample size
    n0 = df.shape[0]

    n1 = len(df["IP_Address"].unique())

    # Get the total number of items with a response per submission
    v = [x for x in df.columns if any([x.startswith(y) for y in ["Q5", "Q6", "Q8"]])]
    df["nresp"] = pd.notnull(df[v]).sum(1)

    # Require at least 20 items
    df = df.loc[df.nresp >= 20, :]

    # Final sample size
    n2 = df.shape[0]

    out.write("%s\n" % k)
    out.write("%d initial sample size\n" % n0)
    out.write("%d unique IP addresses\n" % n1)
    out.write("%d sample size after dropping people with insufficient data\n\n" % n2)

    # Numerically recode all questions 1, 2, ... so that higher numbers correpsond to
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
                df[v] = df[v].replace({False: 1, True: 2})
    return df

for (k, df) in dfm.items():
    kx = k.replace(" ", "-")
    # Some of the files have spaces in variable names, others have underscores,
    # change everything to underscores.
    df = df.rename({x: x.replace(" ", "_") for x in df.columns}, axis=1)
    df = do_recode(k, df)
    df = df.rename(vn, axis=1)

    # Retain only the items to be analyzed.
    vx = [x for x in df.columns if x.startswith("a_") or x.startswith("know_") or x.startswith("internet_")]
    vx = ["Response_Id"] + vx
    df = df[vx]
    df.to_csv(os.path.join(pa, "%s.csv.gz" % kx), index=None)

out.close()
