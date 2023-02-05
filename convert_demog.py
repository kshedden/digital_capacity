import pandas as pd
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
ix_id = cols.index("ResponseId")
ix_yob = cols.index("Q16")
ix_sex = cols.index("Q17")
ix_money = cols.index("Q22")
#ix_employ = cols.index("Q23")
ix_educ = cols.index("Q104")
ix_hhs = cols.index("Q21")
ix_hispanic = cols.index("Q20")
ix_race = cols.index("Q19")

for v in [ix_yob, ix_sex, ix_hhs, ix_money, ix_educ]:
    df.iloc[:, v] = pd.to_numeric(df.iloc[:, v], errors="coerce")

demog = df.iloc[:, [ix_id, ix_yob, ix_sex, ix_money, ix_educ, ix_hhs, ix_race]]
demog["age"] = 2022 - demog["Q16"]

demog = demog.rename({"ResponseId": "Response_Id", "Q16": "yob", "Q17": "sex", "Q22": "money",
                      "Q104": "education", "Q21": "hhs", "Q20": "hispanic", "Q19": "race"}, axis=1)

demog["race"] = demog["race"].replace({"(blank)": "", "American Indian or Alaska Native": "AmerInd",
                 "Middle Eastern or North African": "MidEast", "Native Hawaiian or Other Pacific Islander": "Pacific"})

demog.to_csv(os.path.join(pa, "demog.csv.gz"), index=None)

