import pandas as pd
import os

def load(df: pd.DataFrame, output_path: str) -> None:
    """
    Save a cleaned DataFrame to CSV in data/processed/.
    Creates the output directory if it doesn't exist.
    Prints confirmation with row/col count and file size.

    Args:
        df          : cleaned DataFrame from transform step
        output_path : destination path, e.g. "data/processed/lightcast_clean.csv"
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"[load] Saved  : {output_path}")
    print(f"[load] Shape  : {df.shape}")
    print(f"[load] Size   : {size_kb:.1f} KB")
