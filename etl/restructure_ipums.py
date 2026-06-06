import pan
das as pd
import os

# ── Constants ──────────────────────────────────────────────────────────────
INPUT_FILE = os.path.expanduser("~/datasets/usa_00001.csv.gz")
OUTPUT_DIR = os.path.expanduser("~/repoad688-employability-sum26-groupB/data/processed")

# ── Lookup Maps ────────────────────────────────────────────────────────────
STATE_MAP = {
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

SEX_MAP = {1: "Male", 2: "Female"}

RACE_MAP = {
    1: "White", 2: "Black", 3: "American Indian",
    4: "Chinese", 5: "Japanese", 6: "Other Asian",
    7: "Other", 8: "Two races", 9: "Three or more races"
}

EMPSTAT_MAP = {1: "Employed", 2: "Unemployed", 3: "Not in labor force"}


# ── Helper Functions ───────────────────────────────────────────────────────
def load_data(input_file, compression="gzip"):
    """Load raw data from a file."""
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, compression=compression)
    print(f"  Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def clean_data(df):
    """Apply general cleaning steps to the raw dataframe."""
    print("Cleaning data...")
    df["INCWAGE"] = df["INCWAGE"].replace(999999, None)
    df = df[df["GQ"].isin([1, 2])]
    print(f"  Cleaned data: {len(df):,} rows remaining.")
    return df


def save_table(df, output_dir, filename):
    """Save a dataframe to CSV and print confirmation."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = f"{output_dir}/{filename}"
    df.to_csv(filepath, index=False)
    print(f"  Saved {filename} ({len(df):,} rows)")
    return filepath


def apply_labels(df, col, label_map, label_col=None):
    """Map a column's codes to human-readable labels."""
    if label_col is None:
        label_col = f"{col}_LABEL"
    df[label_col] = df[col].map(label_map)
    return df


# ── Table Functions ────────────────────────────────────────────────────────
def create_household_table(df, output_dir):
    """Create and save the Household table."""
    print("Creating Household table...")
    household_df = df[["SERIAL", "STATEFIP", "GQ", "HHWT"]].drop_duplicates(subset=["SERIAL"]).copy()
    household_df = apply_labels(household_df, "STATEFIP", STATE_MAP, "STATE_NAME")
    save_table(household_df, output_dir, "household.csv")
    return household_df


def create_person_table(df, output_dir):
    """Create and save the Person table."""
    print("Creating Person table...")
    person_df = df[["SERIAL", "PERNUM", "SEX", "AGE", "RACE", "RACED", "PERWT"]].copy()
    person_df = apply_labels(person_df, "SEX", SEX_MAP)
    person_df = apply_labels(person_df, "RACE", RACE_MAP)
    save_table(person_df, output_dir, "person.csv")
    return person_df


def create_employment_table(df, output_dir):
    """Create and save the Employment table."""
    print("Creating Employment table...")
    employment_df = df[["SERIAL", "PERNUM", "EMPSTAT", "EMPSTATD", "OCC", "IND", "INCWAGE"]].copy()
    employment_df = apply_labels(employment_df, "EMPSTAT", EMPSTAT_MAP)
    save_table(employment_df, output_dir, "employment.csv")
    return employment_df


def create_employed_only_table(df, output_dir):
    """Create and save the Employed Only table (for gender analysis)."""
    print("Creating Employed Only table...")
    employed_df = df[df["EMPSTAT"] == 1][
        ["SERIAL", "PERNUM", "SEX", "AGE", "RACE", "STATEFIP", "OCC", "IND", "INCWAGE", "PERWT"]
    ].copy()
    employed_df = apply_labels(employed_df, "SEX", SEX_MAP)
    employed_df = apply_labels(employed_df, "RACE", RACE_MAP)
    employed_df = apply_labels(employed_df, "STATEFIP", STATE_MAP, "STATE_NAME")
    employed_df = employed_df[employed_df["INCWAGE"].notna()]
    save_table(employed_df, output_dir, "employed_only.csv")
    return employed_df


def create_lookup_table(df, col, output_dir, filename):
    """Create and save a lookup table for a given column."""
    print(f"Creating {col} lookup table...")
    lookup_df = df[[col]].drop_duplicates().copy()
    save_table(lookup_df, output_dir, filename)
    return lookup_df


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Load and clean
    df = load_data(INPUT_FILE)
    df = clean_data(df)

    # Create all tables
    create_household_table(df, OUTPUT_DIR)
    create_person_table(df, OUTPUT_DIR)
    create_employment_table(df, OUTPUT_DIR)
    employed_df = create_employed_only_table(df, OUTPUT_DIR)

    # Lookup tables
    create_lookup_table(employed_df, "OCC", OUTPUT_DIR, "occupation_codes.csv")
    create_lookup_table(employed_df, "IND", OUTPUT_DIR, "industry_codes.csv")

    print(f"\nAll tables saved to: {OUTPUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
