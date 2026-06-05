import pandas as pd
import numpy as np
import os


def merge_red_blue_state(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Left-join a red/blue state classification table onto df using STATE_NAME.
    Adds a 'POLITICAL_LEAN' column ('Red' / 'Blue' / 'Purple').
    Skipped if 'red_blue_path' is not in config.

    The red/blue CSV must have columns: state_name, political_lean

    Args:
        df    : input DataFrame (must contain STATE_NAME)
        config: dataset config dict

    Returns:
        df with POLITICAL_LEAN column added (or unchanged if skipped)
    """
    path = config.get("red_blue_path")

    if not path:
        print("[merge_red_blue_state] Skipped — 'red_blue_path' not set in config")
        return df

    if not os.path.exists(path):
        print(f"[merge_red_blue_state] WARNING: file not found at '{path}' — skipped")
        return df

    if "STATE_NAME" not in df.columns:
        print("[merge_red_blue_state] WARNING: STATE_NAME not in DataFrame — skipped")
        return df

    rb = pd.read_csv(path)
    rb.columns = rb.columns.str.strip().str.lower()

    if "state_name" not in rb.columns or "political_lean" not in rb.columns:
        raise ValueError(
            f"Red/blue CSV must have columns 'state_name' and 'political_lean'. "
            f"Found: {rb.columns.tolist()}"
        )

    rb = rb[["state_name", "political_lean"]].rename(
        columns={"state_name": "STATE_NAME", "political_lean": "POLITICAL_LEAN"}
    )

    before = len(df)
    df = df.merge(rb, on="STATE_NAME", how="left")

    matched  = df["POLITICAL_LEAN"].notna().sum()
    unmatched = df["POLITICAL_LEAN"].isna().sum()
    print(f"[merge_red_blue_state] Joined POLITICAL_LEAN: {matched:,} matched, {unmatched:,} unmatched")
    if before != len(df):
        print(f"[merge_red_blue_state] WARNING: row count changed {before} → {len(df)} — check for duplicate state names in red/blue file")

    return df


def drop_columns(df: pd.DataFrame, cols_to_drop: list) -> pd.DataFrame:
    """
    Safely drop a list of columns from df.
    Skips any column that doesn't exist — no KeyError.
    Prints a summary of dropped vs skipped.

    Args:
        df          : input DataFrame
        cols_to_drop: list of column name strings to remove

    Returns:
        df with specified columns removed
    """
    existing  = [c for c in cols_to_drop if c in df.columns]
    not_found = [c for c in cols_to_drop if c not in df.columns]

    df = df.drop(columns=existing)

    print(f"[drop_columns] Dropped  : {len(existing)} columns")
    if not_found:
        print(f"[drop_columns] Not found (skipped): {not_found}")
    print(f"[drop_columns] Remaining: {df.shape[1]} columns")
    return df


def audit_missing(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Report missing value counts and percentages per column.
    Only shows columns with at least 1 missing value, sorted highest → lowest.

    Args:
        df   : input DataFrame
        label: optional label printed in the header (e.g. 'before' / 'after')

    Returns:
        summary DataFrame with columns [column, missing_count, missing_pct, dtype]
        Returns empty DataFrame (not None) if no missing values found.
    """
    total          = len(df)
    missing_counts = df.isnull().sum()
    missing_pct    = (missing_counts / total * 100).round(2)

    summary = pd.DataFrame({
        "column"       : missing_counts.index,
        "missing_count": missing_counts.values,
        "missing_pct"  : missing_pct.values,
        "dtype"        : df.dtypes.values
    })
    summary = (
        summary[summary["missing_count"] > 0]
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    tag = f" ({label})" if label else ""
    print(f"\n[audit_missing{tag}] {len(summary)} columns with missing values (of {df.shape[1]})")
    print(f"[audit_missing{tag}] Total rows: {total:,}\n")
    if len(summary) > 0:
        print(summary.to_string(index=False))
    else:
        print("  No missing values found.")
    return summary


def handle_missing(df: pd.DataFrame, strategy_map: dict) -> pd.DataFrame:
    """
    Apply per-column missing value strategies.

    Supported strategies:
        'drop_row'     — drop any row where this column is null
        'fill_mode'    — fill with the most frequent value
        'fill_median'  — fill with the median (numeric columns only)
        'fill_mean'    — fill with the mean (numeric columns only)
        'fill_unknown' — fill with the string 'Unknown'
        'fill_zero'    — fill with 0
        'leave'        — do nothing (document intentionally)

    Args:
        df           : input DataFrame
        strategy_map : dict mapping column name -> strategy string

    Returns:
        df with missing values handled per strategy
    """
    df = df.copy()
    rows_before = len(df)

    for col, strategy in strategy_map.items():
        if col not in df.columns:
            print(f"[handle_missing] WARNING: '{col}' not in DataFrame — skipped")
            continue

        missing_n = df[col].isnull().sum()
        if missing_n == 0:
            continue

        if strategy == "drop_row":
            df = df.dropna(subset=[col])
            print(f"[handle_missing] '{col}': dropped {missing_n} rows")

        elif strategy == "fill_mode":
            mode_series = df[col].mode()
            if mode_series.empty:
                print(f"[handle_missing] WARNING: '{col}' is entirely null — fill_mode skipped")
            else:
                mode_val = mode_series[0]
                df[col]  = df[col].fillna(mode_val)
                print(f"[handle_missing] '{col}': filled {missing_n} nulls with mode='{mode_val}'")

        elif strategy == "fill_median":
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            print(f"[handle_missing] '{col}': filled {missing_n} nulls with median={median_val}")

        elif strategy == "fill_mean":
            mean_val = df[col].mean()
            df[col]  = df[col].fillna(mean_val)
            print(f"[handle_missing] '{col}': filled {missing_n} nulls with mean={mean_val:.2f}")

        elif strategy == "fill_unknown":
            df[col] = df[col].fillna("Unknown")
            print(f"[handle_missing] '{col}': filled {missing_n} nulls with 'Unknown'")

        elif strategy == "fill_zero":
            df[col] = df[col].fillna(0)
            print(f"[handle_missing] '{col}': filled {missing_n} nulls with 0")

        elif strategy == "leave":
            print(f"[handle_missing] '{col}': {missing_n} nulls — intentionally left as-is")

        else:
            raise ValueError(f"Unknown strategy '{strategy}' for column '{col}'")

    rows_after = len(df)
    if rows_before != rows_after:
        print(f"\n[handle_missing] Rows: {rows_before:,} → {rows_after:,} (removed {rows_before - rows_after:,})")
    return df


def transform(df: pd.DataFrame, config: dict):
    """
    Master transform function. Runs the full pipeline in order:
      0. merge_red_blue_state  (joins POLITICAL_LEAN before STATE_NAME is dropped)
      1. drop_columns
      2. audit_missing         (before)
      3. handle_missing
      4. audit_missing         (after)

    Config keys:
        dataset_name     (str)  : label for logging
        cols_to_drop     (list) : columns to drop — must not be empty for non-placeholder configs
        missing_strategy (dict) : {col: strategy} passed to handle_missing
        red_blue_path    (str)  : optional path to red/blue CSV; skipped if absent
        is_placeholder   (bool) : if True, skip the empty-config guard (for stub configs)

    Args:
        df    : raw DataFrame from extract step
        config: dataset-specific config dict

    Returns:
        tuple: (df_clean, audit_before, audit_after)
            df_clean     : transformed DataFrame
            audit_before : audit_missing summary before handle_missing
            audit_after  : audit_missing summary after handle_missing
    """
    name = config.get("dataset_name", "UNNAMED")

    # Guard: catch accidental runs of placeholder configs against real data
    if not config.get("is_placeholder", False):
        if not config.get("cols_to_drop") and not config.get("missing_strategy"):
            raise ValueError(
                f"Config '{name}' has empty cols_to_drop and missing_strategy. "
                f"If this is intentional (stub config), set is_placeholder=True."
            )

    print(f"\n{'='*60}")
    print(f" TRANSFORM START: {name}")
    print(f" Input shape    : {df.shape}")
    print(f"{'='*60}")

    # Step 0: join red/blue state BEFORE dropping STATE_NAME
    print("\n--- Step 0: Red/Blue State Join ---")
    df = merge_red_blue_state(df, config)

    # Step 1: drop columns
    print("\n--- Step 1: Drop Columns ---")
    df = drop_columns(df, config.get("cols_to_drop", []))

    # Step 2: audit before
    print("\n--- Step 2: Missing Value Audit (before) ---")
    audit_before = audit_missing(df, label="before")

    # Step 3: handle missing
    print("\n--- Step 3: Handle Missing Values ---")
    df = handle_missing(df, config.get("missing_strategy", {}))

    # Step 4: audit after
    print("\n--- Step 4: Missing Value Audit (after) ---")
    audit_after = audit_missing(df, label="after")

    print(f"\n{'='*60}")
    print(f" TRANSFORM DONE : {name}")
    print(f" Output shape   : {df.shape}")
    print(f"{'='*60}\n")

    return df, audit_before, audit_after


# ─────────────────────────────────────────────────────────────────────────────
# LIGHTCAST CONFIG
# Source: Lightcast 2024 Job Postings (1000 rows, 131 cols)
# red_blue_path: set this once the red/blue state CSV is ready
# ─────────────────────────────────────────────────────────────────────────────
LIGHTCAST_CONFIG = {
    "dataset_name" : "LIGHTCAST_2024",
    "red_blue_path": None,  # TODO: set to "data/external/state_political_lean.csv" when ready

    "cols_to_drop": [
        # Metadata / tracking
        "LAST_UPDATED_TIMESTAMP",
        "DUPLICATES", "EXPIRED", "DURATION",
        "SOURCE_TYPES", "SOURCES", "URL",
        "ACTIVE_URLS", "ACTIVE_SOURCES_INFO",
        "MODELED_EXPIRED", "MODELED_DURATION",
        # Redundant company / raw fields
        "COMPANY_RAW",
        # Education — keep NAME versions
        "MIN_EDULEVELS", "MAX_EDULEVELS",
        # Employment / remote type — keep NAME, drop code
        "EMPLOYMENT_TYPE", "REMOTE_TYPE",
        # Salary — keep SALARY_FROM / SALARY_TO
        "SALARY",
        # Pay period
        "ORIGINAL_PAY_PERIOD",
        # Geography codes — keep NAME versions
        "LOCATION", "CITY",
        "COUNTY", "COUNTY_NAME",
        "MSA", "MSA_NAME",
        "STATE",
        # STATE_NAME is dropped AFTER the red/blue join in Step 0
        "STATE_NAME",
        # Commute flow geography
        "COUNTY_OUTGOING", "COUNTY_NAME_OUTGOING",
        "COUNTY_INCOMING", "COUNTY_NAME_INCOMING",
        "MSA_OUTGOING", "MSA_NAME_OUTGOING", "MSA_INCOMING",
        # NAICS — keep 2/4/6-digit; drop 3/5
        "NAICS3", "NAICS3_NAME", "NAICS5", "NAICS5_NAME",
        # Title — keep TITLE_CLEAN
        "TITLE", "TITLE_NAME",
        # Skills — drop sparse; keep SKILLS_NAME, SOFTWARE
        "SPECIALIZED_SKILLS", "SPECIALIZED_SKILLS_NAME",
        "CERTIFICATIONS", "CERTIFICATIONS_NAME",
        # ONET legacy
        "ONET_2019", "ONET_2019_NAME",
        # CIP codes
        "CIP6", "CIP6_NAME", "CIP4", "CIP4_NAME", "CIP2", "CIP2_NAME",
        # SOC 2021 — all levels dropped (use SOC_2 / SOC_5 legacy below)
        "SOC_2021_2", "SOC_2021_2_NAME",
        "SOC_2021_3", "SOC_2021_3_NAME",
        "SOC_2021_4", "SOC_2021_4_NAME",
        "SOC_2021_5", "SOC_2021_5_NAME",
        # LOT occupation taxonomy — drop detailed levels
        "LOT_CAREER_AREA",
        "LOT_OCCUPATION", "LOT_OCCUPATION_NAME",
        "LOT_SPECIALIZED_OCCUPATION", "LOT_SPECIALIZED_OCCUPATION_NAME",
        "LOT_OCCUPATION_GROUP", "LOT_OCCUPATION_GROUP_NAME",
        "LOT_V6_SPECIALIZED_OCCUPATION", "LOT_V6_SPECIALIZED_OCCUPATION_NAME",
        "LOT_V6_OCCUPATION", "LOT_V6_OCCUPATION_NAME",
        # SOC legacy — drop mid-levels; keep SOC_2 and SOC_5
        "SOC_3", "SOC_3_NAME", "SOC_4", "SOC_4_NAME",
        # Lightcast sector code — keep NAME
        "LIGHTCAST_SECTORS",
        # NAICS 2022 — keep 2/4/6; drop 3/5
        "NAICS_2022_3", "NAICS_2022_3_NAME",
        "NAICS_2022_5", "NAICS_2022_5_NAME",
    ],

    "missing_strategy": {
        # Salary — fill with median (right-skewed distribution)
        "SALARY_FROM"          : "fill_median",
        "SALARY_TO"            : "fill_median",
        # Employment attributes
        "EMPLOYMENT_TYPE_NAME" : "fill_unknown",
        "REMOTE_TYPE_NAME"     : "fill_unknown",
        # Education
        "MIN_EDULEVELS_NAME"   : "fill_unknown",
        "MAX_EDULEVELS_NAME"   : "fill_unknown",
        # Experience — sparse by design; null = not specified
        "MIN_YEARS_EXPERIENCE" : "leave",
        "MAX_YEARS_EXPERIENCE" : "leave",
        # Skills — sparse by design
        "SKILLS_NAME"          : "leave",
        "COMMON_SKILLS_NAME"   : "leave",
        "SOFTWARE_SKILLS_NAME" : "leave",
        # Company
        "COMPANY_NAME"         : "fill_unknown",
        # Occupation
        "ONET_NAME"            : "fill_unknown",
        "NAICS2_NAME"          : "fill_unknown",
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# CPS CONFIG — stub, is_placeholder=True prevents accidental run
# Source: fedesoriano/gender-pay-gap-dataset → CurrentPopulationSurvey.csv
# TODO: run audit_missing on CPS data, then populate cols_to_drop
# ─────────────────────────────────────────────────────────────────────────────
CPS_CONFIG = {
    "dataset_name"  : "CPS_GENDER_PAY_GAP",
    "is_placeholder": True,
    "red_blue_path" : None,

    "cols_to_drop": [],  # TODO: populate after audit_missing

    "missing_strategy": {
        "sex"      : "drop_row",    # core variable — must be present
        "statefip" : "drop_row",    # needed for red/blue join
        "incwage"  : "fill_median",
        "ind"      : "fill_unknown",
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# ACS PUMS CONFIG — stub, is_placeholder=True prevents accidental run
# Source: psam_pusa.csv + psam_pusb.csv (combined in 01_extract)
# TODO: ACS has 200+ cols — run audit_missing after extract, then populate
# ─────────────────────────────────────────────────────────────────────────────
ACS_CONFIG = {
    "dataset_name"  : "ACS_PUMS_2024",
    "is_placeholder": True,
    "red_blue_path" : None,

    "cols_to_drop": [],  # TODO: populate after audit_missing

    "missing_strategy": {}  # TODO: populate after audit_missing
}