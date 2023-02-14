import pandas as pd
import numpy as np
import os

# Path where data are located
pa = "/home/kshedden/data/Tawanna_Dillahunt"

# Data file with demographics
f = "Demographics_community_digital_capacity_f_fixed.xlsx"

df = pd.read_excel(os.path.join(pa, f), sheet_name=None)
df = df["All Valid Responses"]

df.columns = df.iloc[0,:]
df = df.iloc[1:,:]

# Variable locations
cols = df.columns.tolist()
v_id = "ResponseId"
v_yob = "Q16"
v_sex = "Q17"
v_money = "Q22"
v_employ = "Q23"
v_educ = "Q104"
v_hhs = "Q21"
v_hispanic = "Q20"
v_race = "Q19"

for v in [v_yob, v_sex, v_hhs, v_money, v_educ]:
    df[v] = pd.to_numeric(df[v], errors="coerce")

demog = df[[v_id, v_yob, v_sex, v_money, v_educ, v_money, v_hhs, v_race]]
demog["age"] = 2022 - demog["Q16"]

demog = demog.rename({"ResponseId": "Response_Id", "Q16": "yob", "Q17": "sex", "Q22": "money",
                      "Q104": "education", "Q21": "hhs", "Q20": "hispanic", "Q19": "race"}, axis=1)

demog["race"] = demog["race"].replace({"(blank)": "", "American Indian or Alaska Native": "AmerInd",
                 "Middle Eastern or North African": "MidEast", "Native Hawaiian or Other Pacific Islander": "Pacific"})

demog["sex"] = demog["sex"].replace({1: "Male", 2: "Female", 3: "Nonbinary", 4: "Decline", 5: "Self-describe"})

demog["money"] = demog["money"].replace({9: np.nan})

demog.to_csv(os.path.join(pa, "demog.csv.gz"), index=None)

