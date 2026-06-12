import pandas as pd

# Load data
industry = pd.read_csv("data/processed/lightcast_industry.csv")
household = pd.read_csv("data/processed/household.csv")

print("Original industry shape:", industry.shape)
print("Original household shape:", household.shape)

# -------------------------
# Clean lightcast_industry
# -------------------------
industry_clean = industry.copy()

columns_to_drop = [
    "NAICS2", "NAICS2_NAME",
    "NAICS4", "NAICS4_NAME",
    "NAICS6", "NAICS6_NAME",
    "NAICS_2022_2", "NAICS_2022_2_NAME",
    "NAICS_2022_4", "NAICS_2022_4_NAME",
    "LIGHTCAST_SECTORS_NAME"
]

industry_clean = industry_clean.drop(columns=columns_to_drop, errors="ignore")

industry_clean["NAICS_2022_6"] = industry_clean["NAICS_2022_6"].fillna("Unknown")
industry_clean["NAICS_2022_6_NAME"] = industry_clean["NAICS_2022_6_NAME"].fillna("Unknown")

industry_clean = industry_clean.drop_duplicates()

industry_clean.to_csv("data/processed/lightcast_industry_clean.csv", index=False)

print("Clean industry shape:", industry_clean.shape)


# -------------------------
# Clean household
# -------------------------
household_clean = household.copy()

household_clean = household_clean.drop_duplicates()

household_clean["STATE_NAME"] = household_clean["STATE_NAME"].fillna("Unknown")

household_clean = household_clean[
    ["SERIAL", "STATEFIP", "STATE_NAME", "HHWT"]
]

household_clean.to_csv("data/processed/household_clean.csv", index=False)

print("Clean household shape:", household_clean.shape)

print("Cleaning complete.")
