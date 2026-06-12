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

# ── 5. load_employment() — Julie ────────────────
def load_employment() -> pd.DataFrame:
    df = _base_load(PROCESSED / "employed_only.parquet")
    df = df[df["INCWAGE"] > 0]          # drop zero wages
    df = df.dropna(subset=["SEX_LABEL", "STATE_NAME"])
    return df

# ── 6. load_job_postings() — Julie ──────────────
def load_job_postings() -> pd.DataFrame:
    df = _base_load(PROCESSED / "lightcast_job_postings_relational.parquet")
    df["POSTED"] = pd.to_datetime(df["POSTED"], errors="coerce")
    df = df.dropna(subset=["POSTED"])
    df["REMOTE_TYPE_NAME"] = df["REMOTE_TYPE_NAME"].replace("[None]", "Not Specified")
    df["POSTED_MONTH"] = df["POSTED"].dt.to_period("M").astype(str)
    return df

# ── 7. load_location() — Julie ──────────────────
def load_location() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "lightcast_location.csv")
    df["STATE_CODE"] = (df["MSA_NAME_INCOMING"]
                        .str.split(",").str[-1]
                        .str.strip()
                        .str.split("-").str[0]
                        .str.strip())
    return df

# ── 8. load_education(), load_salary() etc ────
# Teammates add their own load_*() here
# Follow same pattern: _base_load() + specific cleaning

# ── 9. save_fig() ─────────────────────────────
def save_fig(fig, filename_stem: str) -> None:
    apply_theme(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIGURES / f"{filename_stem}.html")
    fig.write_image(FIGURES / f"{filename_stem}.png")
    print(f"Saved: {filename_stem}.html + .png")
