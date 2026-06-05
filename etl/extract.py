import pandas as pd
import os

def extract(filepath: str) -> pd.DataFrame:
    """
    Load a single CSV file, print shape and head for inspection.

    Args:
        filepath: path to raw CSV

    Returns:
        df: raw DataFrame, unmodified
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[extract] File not found: {filepath}\n"
            f"Check that 01_extract.ipynb has been run and the file was saved to data/raw/"
        )

    df = pd.read_csv(filepath)
    print(f"[extract] Loaded: {filepath}")
    print(f"[extract] Shape : {df.shape}")
    print(df.head(2))
    return df


def extract_acs(path_a: str, path_b: str) -> pd.DataFrame:
    """
    Load and combine the two ACS PUMS 2024 files (split A-M / N-Z by state).

    Args:
        path_a: path to psam_pusa.csv
        path_b: path to psam_pusb.csv

    Returns:
        df: combined DataFrame, unmodified
    """
    for p in [path_a, path_b]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[extract_acs] File not found: {p}")

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)
    df   = pd.concat([df_a, df_b], ignore_index=True)

    print(f"[extract_acs] Loaded A: {df_a.shape}, B: {df_b.shape}")
    print(f"[extract_acs] Combined : {df.shape}")
    return df
