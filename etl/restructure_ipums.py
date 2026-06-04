import pandas as pd
import os

# Paths
INPUT_FILE  = os.path.expanduser("~/datasets/usa_00001.csv.gz")
OUTPUT_DIR  = os.path.expanduser("~/repoad688-employability-sum26-groupB/data/processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load raw data
print("Loading IPUMS data")
df = pd.read_csv(INPUT_FILE, compression="gzip")
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

# Clean data
df["INCWAGE"] = df["INCWAGE"].replace(999999, None)
df = df[df["GQ"].isin([1, 2])]

# Table 1: Household
print("Creating Household table")
household_df = df[["SERIAL", "STATEFIP", "GQ", "HHWT"]].drop_duplicates(subset=["SERIAL"])
state_map = {
    1:"Alabama", 2:"Alaska", 4:"Arizona", 5:"Arkansas", 6:"California",
    8:"Colorado", 9:"Connecticut", 10:"Delaware", 11:"District of Columbia",
    12:"Florida", 13:"Georgia", 15:"Hawaii", 16:"Idaho", 17:"Illinois",
    18:"Indiana", 19:"Iowa", 20:"Kansas", 21:"Kentucky", 22:"Louisiana",
    23:"Maine", 24:"Maryland", 25:"Massachusetts", 26:"Michigan",
    27:"Minnesota", 28:"Mississippi", 29:"Missouri", 30:"Montana",
    31:"Nebraska", 32:"Nevada", 33:"New Hampshire", 34:"New Jersey",
    35:"New Mexico", 36:"New York", 37:"North Carolina", 38:"North Dakota",
    39:"Ohio", 40:"Oklahoma", 41:"Oregon", 42:"Pennsylvania",
    44:"Rhode Island", 45:"South Carolina", 46:"South Dakota",
    47:"Tennessee", 48:"Texas", 49:"Utah", 50:"Vermont", 51:"Virginia",
    53:"Washington", 54:"West Virginia", 55:"Wisconsin", 56:"Wyoming"
}
household_df = household_df.copy()
household_df["STATE_NAME"] = household_df["STATEFIP"].map(state_map)
household_df.to_csv(f"{OUTPUT_DIR}/household.csv", index=False)
print(f"  Saved household.csv ({len(household_df):,} rows)")

# Table 2: Person
print("Creating Person table")
sex_map = {1: "Male", 2: "Female"}
race_map = {
    1: "White", 2: "Black", 3: "American Indian",
    4: "Chinese", 5: "Japanese", 6: "Other Asian",
    7: "Other", 8: "Two races", 9: "Three or more races"
}
person_df = df[["SERIAL", "PERNUM", "SEX", "AGE", "RACE", "RACED", "PERWT"]].copy()
person_df["SEX_LABEL"]  = person_df["SEX"].map(sex_map)
person_df["RACE_LABEL"] = person_df["RACE"].map(race_map)
person_df.to_csv(f"{OUTPUT_DIR}/person.csv", index=False)
print(f"  Saved person.csv ({len(person_df):,} rows)")

# Table 3: Employment
print("Creating Employment table")
empstat_map = {1: "Employed", 2: "Unemployed", 3: "Not in labor force"}
employment_df = df[["SERIAL", "PERNUM", "EMPSTAT", "EMPSTATD", "OCC", "IND", "INCWAGE"]].copy()
employment_df["EMPSTAT_LABEL"] = employment_df["EMPSTAT"].map(empstat_map)
employment_df.to_csv(f"{OUTPUT_DIR}/employment.csv", index=False)
print(f"  Saved employment.csv ({len(employment_df):,} rows)")

# Table 4: Employed Only (for gender analysis)
print("Creating Employed_Only table")
employed_df = df[df["EMPSTAT"] == 1][
    ["SERIAL", "PERNUM", "SEX", "AGE", "RACE", "STATEFIP", "OCC", "IND", "INCWAGE", "PERWT"]
].copy()
employed_df["SEX_LABEL"]  = employed_df["SEX"].map(sex_map)
employed_df["RACE_LABEL"] = employed_df["RACE"].map(race_map)
employed_df["STATE_NAME"] = employed_df["STATEFIP"].map(state_map)
employed_df = employed_df[employed_df["INCWAGE"].notna()]
employed_df.to_csv(f"{OUTPUT_DIR}/employed_only.csv", index=False)
print(f"  Saved employed_only.csv ({len(employed_df):,} rows)")

print("\nAll tables saved to:", OUTPUT_DIR)
print("Done!")

# Table 5: Occupation Lookup
print("Creating Occupation lookup table")
occ_df = employed_df[["OCC"]].drop_duplicates().copy()
occ_df.to_csv(f"{OUTPUT_DIR}/occupation_codes.csv", index=False)
print(f"  Saved occupation_codes.csv ({len(occ_df):,} rows)")

# Table 6: Industry Lookup
print("Creating Industry lookup table")
ind_df = employed_df[["IND"]].drop_duplicates().copy()
ind_df.to_csv(f"{OUTPUT_DIR}/industry_codes.csv", index=False)
print(f"  Saved industry_codes.csv ({len(ind_df):,} rows)")

