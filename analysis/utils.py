# ── 1. Imports ────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── 2. PATHS ──────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
FIGURES   = BASE / "assets" / "figures"

# ── 3. PLOTLY_THEME ───────────────────────────
# px-compatible kwargs only (template, color_discrete_sequence)
PLOTLY_THEME = dict(
    template="plotly_white",
    color_discrete_sequence=["#2c7bb6", "#d7191c", "#fdae61", "#1a9641"]
)

# Layout-level settings — apply via apply_theme(fig) after creation
_LAYOUT_THEME = dict(
    font=dict(family="Arial", size=13),
    title_font=dict(size=16, color="#2c3e50"),
    colorway=["#2c7bb6", "#d7191c", "#fdae61", "#1a9641"]
)

def apply_theme(fig):
    """Apply shared layout theme to any plotly figure."""
    fig.update_layout(**_LAYOUT_THEME)
    return fig

# ── 4. _base_load() ───────────────────────────
def _base_load(filepath: Path) -> pd.DataFrame:
    df = pd.read_parquet(filepath)
    if "ID" in df.columns:
        df = df.dropna(subset=["ID"])
        df = df.drop_duplicates(subset=["ID"])
    return df


# ── 5a. map_ind_to_industry() — Julie ─────────────────
def map_ind_to_industry(ind_series: pd.Series) -> pd.Series:
    bins = [0, 290, 490, 770, 3990, 4090, 4590, 4690,
            5790, 6390, 6470, 6780, 7190, 7580, 7790, 8470,
            8590, 9290, 9590, 9870, 9999]
    labels = [
        "Agriculture", "Mining", "Construction", "Manufacturing",
        "Wholesale Trade", "Retail Trade", "Transportation", "Information",
        "Finance & Insurance", "Real Estate", "Professional Services", "Management",
        "Administrative Services", "Education", "Healthcare",
        "Arts & Entertainment", "Accommodation & Food", "Other Services",
        "Public Administration", "Military"
    ]
    return pd.cut(ind_series, bins=bins, labels=labels, right=True)

# ── 5. load_employment() — Julie ────────────────
def load_employment() -> pd.DataFrame:
    df = _base_load(PROCESSED / "employed_only.parquet")
    df = df[df["INCWAGE"] > 0]          # drop zero wages
    df = df.dropna(subset=["SEX_LABEL", "STATE_NAME"])
    df["INDUSTRY"] = map_ind_to_industry(df["IND"])
    df = df.dropna(subset=["INDUSTRY"])
    return df

# ── 6. load_job_postings() — Julie ──────────────
def load_job_postings() -> pd.DataFrame:
    df = _base_load(PROCESSED / "lightcast_job_postings_relational.parquet")
    df["POSTED"] = pd.to_datetime(df["POSTED"], errors="coerce")
    df = df.dropna(subset=["POSTED"])
    df["REMOTE_TYPE_NAME"] = df["REMOTE_TYPE_NAME"].replace("[None]", "Not Specified")
    df["EMPLOYMENT_TYPE_NAME"] = df["EMPLOYMENT_TYPE_NAME"].str.replace("â\u2030¤", "≤", regex=False)
    df["POSTED_MONTH"] = df["POSTED"].dt.to_period("M").astype(str)
    return df

# ── 7. load_location() — Julie ──────────────────
def load_location() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "lightcast_location.parquet")
    df["STATE_CODE"] = (df["MSA_NAME_INCOMING"]
                        .str.split(",").str[-1]
                        .str.strip()
                        .str.split("-").str[0]
                        .str.strip())
    return df

# ── 8. load_education(), load_salary() etc ────
# Teammates add their own load_*() here
# Follow same pattern: _base_load() + specific cleaning


# ── 10. build_cross_state() — Julie ───────────────────
_STATE_ABBREV = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","District of Columbia":"DC",
    "Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL",
    "Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA",
    "Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN",
    "Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV",
    "New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY",
    "North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR",
    "Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD",
    "Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA",
    "Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY"
}

def build_cross_state(df_emp: pd.DataFrame, df_jobs_state: pd.DataFrame) -> pd.DataFrame:
    """Merge IPUMS wage by state+sex with Lightcast posting count by state."""
    wage = (df_emp.groupby(["STATE_NAME", "SEX_LABEL"])["INCWAGE"]
            .mean().reset_index()
            .rename(columns={"INCWAGE": "AVG_WAGE"}))
    wage["STATE_CODE"] = wage["STATE_NAME"].map(_STATE_ABBREV)

    posting_count = (df_jobs_state.groupby("STATE_CODE")
                     .size().reset_index(name="POSTING_COUNT"))

    return wage.merge(posting_count, on="STATE_CODE").dropna()

# ── 9. save_fig() ─────────────────────────────
def save_fig(fig, filename_stem: str) -> None:
    apply_theme(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIGURES / f"{filename_stem}.html")
    fig.write_image(FIGURES / f"{filename_stem}.png")
    print(f"Saved: {filename_stem}.html + .png")
