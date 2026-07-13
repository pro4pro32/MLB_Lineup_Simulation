"""
data.py
=======
Data loading and processing for Baseball Savant CSV exports.
load_savant_csv() is decorated with @st.cache_data so the CSV is read
and processed only once per Streamlit session, regardless of how many
widget interactions trigger reruns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from constants import PLAYER_TEAM, PROJECTION_PA_THRESHOLD


@st.cache_data(show_spinner=False)
def load_savant_csv(path: str) -> pd.DataFrame:
    """
    Load and process a Baseball Savant CSV export.

    Key steps
    ---------
    - encoding='utf-8-sig' strips the BOM that causes a 2-column shift bug
    - k_percent / bb_percent are divided by 100 (Savant exports as 21.5, not 0.215)
    - HR_pct and HBP_pct are computed as count / PA
    - Hit-type split rates derived from single / double / triple counts
    - xStats projection columns derived from xba / xslg / xobp
    - Each player tagged with their MLB team via PLAYER_TEAM mapping
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    df["Name"] = df["last_name, first_name"].astype(str).str.strip()

    numeric_cols = [
        "pa", "ab", "single", "double", "triple", "home_run",
        "k_percent", "bb_percent", "babip", "b_hit_by_pitch",
        "batting_avg", "on_base_percent", "slg_percent",
        "xba", "xslg", "xobp", "xwoba",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["PA"] = df["pa"].clip(lower=1)

    # Savant exports percentage columns in 0-100 scale (e.g. 21.5 for 21.5%)
    df["K_pct"]  = df["k_percent"]  / 100.0
    df["BB_pct"] = df["bb_percent"] / 100.0

    pa = df["PA"]
    df["HR_pct"]  = df["home_run"]       / pa
    df["HBP_pct"] = df["b_hit_by_pitch"] / pa
    df["BABIP"]   = df["babip"]

    non_hr = (df["single"] + df["double"] + df["triple"]).clip(lower=1)
    df["1B_rate"] = df["single"] / non_hr
    df["2B_rate"] = df["double"] / non_hr
    df["3B_rate"] = df["triple"] / non_hr

    # xStats projection fallback for players with PA < PROJECTION_PA_THRESHOLD
    if all(c in df.columns for c in ("xba", "xobp", "xslg")):
        xba  = df["xba"].clip(lower=0.10, upper=0.40)
        xobp = df["xobp"].clip(lower=0.15, upper=0.55)
        xslg = df["xslg"].clip(lower=0.20, upper=0.80)
        xiso = (xslg - xba).clip(lower=0.0)

        df["xK_pct"]  = df["K_pct"]                                    # no direct xK in Savant
        df["xBB_pct"] = (xobp - xba).clip(lower=0.02, upper=0.25)
        df["xHR_pct"] = (xiso * 0.35).clip(upper=0.12)
        df["xBABIP"]  = (xba / (1.0 - df["K_pct"]).clip(lower=0.40)).clip(0.18, 0.42)
    else:
        df["xK_pct"]  = df["K_pct"]
        df["xBB_pct"] = df["BB_pct"]
        df["xHR_pct"] = df["HR_pct"]
        df["xBABIP"]  = df["BABIP"]

    # Attach team; unrecognised players → "FA" (free agent / unknown)
    df["Team"] = df["Name"].map(PLAYER_TEAM).fillna("FA")

    return (
        df.sort_values("PA", ascending=False)
          .drop_duplicates("Name")
          .reset_index(drop=True)
    )


# Fallback sample data — shown when CSV is not present (e.g. first-time setup)
SAMPLE_DATA = pd.DataFrame({
    "Name":    ["Judge, Aaron", "Ohtani, Shohei", "Freeman, Freddie",
                "Ramírez, José", "Harper, Bryce", "Acuña Jr., Ronald",
                "Alvarez, Yordan", "Lindor, Francisco", "Betts, Mookie"],
    "PA":      [650, 640, 680, 660, 620, 500, 630, 660, 650],
    "K_pct":   [0.236, 0.257, 0.098, 0.148, 0.210, 0.175, 0.200, 0.198, 0.136],
    "BB_pct":  [0.183, 0.150, 0.102, 0.099, 0.168, 0.095, 0.118, 0.086, 0.110],
    "HR_pct":  [0.078, 0.076, 0.038, 0.060, 0.057, 0.063, 0.065, 0.045, 0.050],
    "BABIP":   [0.298, 0.290, 0.305, 0.322, 0.295, 0.340, 0.290, 0.288, 0.299],
    "HBP_pct": [0.012, 0.015, 0.006, 0.012, 0.020, 0.014, 0.010, 0.008, 0.012],
    "1B_rate": [0.68, 0.65, 0.74, 0.68, 0.67, 0.66, 0.68, 0.70, 0.68],
    "2B_rate": [0.22, 0.25, 0.19, 0.22, 0.22, 0.23, 0.22, 0.21, 0.22],
    "3B_rate": [0.05, 0.05, 0.04, 0.05, 0.04, 0.06, 0.04, 0.04, 0.04],
    "xK_pct":  [0.236, 0.257, 0.098, 0.148, 0.210, 0.175, 0.200, 0.198, 0.136],
    "xBB_pct": [0.183, 0.150, 0.102, 0.099, 0.168, 0.095, 0.118, 0.086, 0.110],
    "xHR_pct": [0.078, 0.076, 0.038, 0.060, 0.057, 0.063, 0.065, 0.045, 0.050],
    "xBABIP":  [0.298, 0.290, 0.305, 0.322, 0.295, 0.340, 0.290, 0.288, 0.299],
    "Team":    ["NYY", "LAD", "LAD", "CLE", "PHI", "ATL", "HOU", "NYM", "LAD"],
})
