import pandas as pd
import numpy as np
import os
from pathlib import Path

# Path where data are located
pa = "/home/kshedden/data/Tawanna_Dillahunt"

# Variable locations
v_id = "ResponseId"
v_yob = "Q16"
v_sex = "Q17"
v_money = "Q22"
v_employ = "Q23"
v_educ = "Q104"
v_hhs = "Q21"
v_hispanic = "Q20"
v_race = "Q19"

def clean_demog(df):

    df = df.copy()

    def f(x):
        if type(x) is str:
            return x.split("/")[-1]
        else:
            return x
    df["Q16"] = [f(x) for x in df["Q16"]]

    for v in [v_yob, v_sex, v_hhs, v_money, v_educ]:
        df[v] = pd.to_numeric(df[v], errors="coerce")

    vna = [v_id, v_yob, v_sex, v_money, v_educ, v_employ, v_hhs, v_race]
    if "UserLanguage" in df.columns:
        vna.append("UserLanguage")
    demog = df[vna].copy()

    # Clean YOB
    demog.loc[:, "age"] = 2022 - demog["Q16"]

    demog = demog.rename({"ResponseId": "Response_Id", "Q16": "yob", "Q17": "sex", "Q22": "money",
                          "Q104": "education", "Q21": "hhs", "Q20": "hispanic", "Q19": "race",
                          "Q23": "work"}, axis=1)

    demog["race"] = demog["race"].replace({"(blank)": "", "American Indian or Alaska Native": "AmerInd",
                     "Middle Eastern or North African": "MidEast", "Native Hawaiian or Other Pacific Islander": "Pacific"})

    demog["sex"] = demog["sex"].replace({1: "Male", 2: "Female", 3: "Nonbinary", 4: "Decline", 5: "Self-describe"})

    demog["money"] = demog["money"].replace({9: np.nan})

    work = []
    for i,v in enumerate(demog["work"]):
        if type(v) is str:
            w = [int(x) for x in v.split(",")]
            work.append(1*(len(set([1, 2, 3, 10]) & set(w)) > 0))
        else:
            work.append(np.nan)

    demog["work"] = work

    return demog

# Get FOP demographics
f = "Demographics_community_digital_capacity_f_fixed.xlsx"
df = pd.read_excel(os.path.join(pa, f), sheet_name=None)
df = df["All Valid Responses"]
df.columns = df.iloc[0,:]
df = df.iloc[1:,:]
fop_demog = clean_demog(df)

# Get JFS demographics
pa = Path("/home/kshedden/data/Tawanna_Dillahunt")
fn = ["JFS_Raw_Data.csv.gz", "JFS_Cleaned_Sheet.csv.gz"]
da = pd.read_csv(pa / Path(fn[1]))
da_header = da.iloc[0, :].tolist()
da = da.iloc[1:, :]
jfs_demog = clean_demog(da)
