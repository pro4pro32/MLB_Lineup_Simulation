"""
download_stats.py
=================
Fetches the full 2025 MLB batting dataset and saves it as
``Batters_Savant_stats.csv`` ready for Monte Carlo Baseball 2026.

Run this script LOCALLY (not on a server):
    pip install pybaseball pandas requests
    python download_stats.py

Two data sources are merged:
  1. Baseball Savant — xStats, K%, BB%, BABIP, Statcast metrics
  2. FanGraphs        — season totals (PA, HR, singles, doubles …)

The merged file contains ~700–900 qualified batters (min. 1 PA).
"""

from __future__ import annotations

import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import pandas as pd
    import requests
    from pybaseball import (
        batting_stats,
        statcast_batter_expected_stats,
        statcast_batter_exitvelo_barrels,
    )
except ImportError as exc:
    print(f"\n❌  Missing dependency: {exc}")
    print("    Run:  pip install pybaseball pandas requests\n")
    sys.exit(1)

OUTPUT_FILE = "Batters_Savant_stats.csv"
YEAR        = 2025
MIN_PA      = 1       # include everyone; app filters by PA at runtime


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1 — Baseball Savant custom leaderboard (browser-like request)
# ═══════════════════════════════════════════════════════════════════════════════

SAVANT_COLS = ",".join([
    "b_ab", "b_pa", "b_hit", "b_single", "b_double", "b_triple",
    "b_home_run", "b_strikeout", "b_walk", "b_k_percent", "b_bb_percent",
    "batting_avg", "slg_percent", "on_base_percent", "on_base_plus_slg",
    "isolated_power", "babip", "xba", "xslg", "xobp", "xwoba", "xiso",
    "b_rbi", "b_hit_by_pitch", "exit_velocity_avg", "launch_angle_avg",
    "sweet_spot_percent", "barrel_batted_rate",
])

SAVANT_URL = (
    f"https://baseballsavant.mlb.com/leaderboard/custom"
    f"?year={YEAR}&type=batter&filter=&sort=4&sortDir=desc&min={MIN_PA}"
    f"&selections={SAVANT_COLS}&csv=true"
)


def _fetch_savant() -> pd.DataFrame | None:
    """
    Download Savant leaderboard CSV.
    Uses a browser User-Agent + Referer header to pass the bot check.
    """
    print("⬇  Downloading from Baseball Savant …")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer":    "https://baseballsavant.mlb.com/leaderboard/custom",
        "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    session = requests.Session()
    # Prime a cookie by visiting the leaderboard page first
    try:
        session.get("https://baseballsavant.mlb.com/leaderboard/custom",
                    headers=headers, timeout=20)
        time.sleep(1)
        r = session.get(SAVANT_URL, headers=headers, timeout=60)
        if r.status_code != 200:
            print(f"   ⚠  Savant returned HTTP {r.status_code}")
            return None
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        df.columns = df.columns.str.strip().str.lower()
        print(f"   ✅  Savant: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as exc:
        print(f"   ⚠  Savant request failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 — FanGraphs via pybaseball (batting_stats)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_fangraphs() -> pd.DataFrame | None:
    """
    Fetch season batting totals from FanGraphs via pybaseball.
    Returns a normalised DataFrame with columns matching Savant naming.
    """
    print("⬇  Downloading from FanGraphs via pybaseball …")
    try:
        fg = batting_stats(YEAR, qual=MIN_PA)
        fg.columns = fg.columns.str.strip()
        print(f"   ✅  FanGraphs: {len(fg)} rows")

        # ── Column mapping: FanGraphs name → our standard name ───────────────
        rename = {
            "Name":  "fg_name",
            "PA":    "pa",
            "AB":    "ab",
            "H":     "hit",
            "1B":    "single",
            "2B":    "double",
            "3B":    "triple",
            "HR":    "home_run",
            "SO":    "strikeout",
            "BB":    "walk",
            "HBP":   "b_hit_by_pitch",
            "AVG":   "batting_avg",
            "OBP":   "on_base_percent",
            "SLG":   "slg_percent",
            "OPS":   "on_base_plus_slg",
            "ISO":   "isolated_power",
            "BABIP": "babip",
            "K%":    "k_percent_fg",   # rename to avoid conflict; will use for k_percent
            "BB%":   "bb_percent_fg",
            "RBI":   "b_rbi",
            "Team":  "fg_team",
        }
        fg = fg.rename(columns={k: v for k, v in rename.items() if k in fg.columns})

        # FanGraphs K%/BB% already in decimal (0.215) — convert to Savant percent format
        for col, out in [("k_percent_fg", "k_percent"), ("bb_percent_fg", "bb_percent")]:
            if col in fg.columns:
                vals = pd.to_numeric(fg[col], errors="coerce").fillna(0)
                fg[out] = vals * 100.0 if vals.median() < 1.0 else vals

        return fg

    except Exception as exc:
        print(f"   ⚠  FanGraphs request failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3 — pybaseball statcast expected stats (xba, xslg, xobp …)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_expected() -> pd.DataFrame | None:
    print("⬇  Downloading Statcast expected stats via pybaseball …")
    try:
        ex = statcast_batter_expected_stats(YEAR, minPA=MIN_PA)
        print(f"   ✅  Expected stats: {len(ex)} rows")
        ex.columns = ex.columns.str.strip().str.lower()
        return ex
    except Exception as exc:
        print(f"   ⚠  Expected stats failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE & NORMALISE
# ═══════════════════════════════════════════════════════════════════════════════

def _build_name_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a unified 'last_name, first_name' column exists.
    FanGraphs provides 'Name' as 'First Last'; Savant provides 'last_name, first_name'.
    """
    if "last_name, first_name" in df.columns:
        return df
    if "last_name" in df.columns and "first_name" in df.columns:
        df["last_name, first_name"] = (
            df["last_name"].str.strip() + ", " + df["first_name"].str.strip()
        )
    elif "fg_name" in df.columns:
        # Convert "First Last" → "Last, First"
        def _flip(n: str) -> str:
            parts = str(n).strip().split(" ", 1)
            return f"{parts[1]}, {parts[0]}" if len(parts) == 2 else n
        df["last_name, first_name"] = df["fg_name"].apply(_flip)
    return df


def _merge_sources(
    savant: pd.DataFrame | None,
    fg: pd.DataFrame | None,
    expected: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge available DataFrames, preferring Savant values where present.
    Falls back to FanGraphs when Savant is unavailable.
    """
    # ── Case 1: Savant available — primary source ─────────────────────────────
    if savant is not None:
        savant = _build_name_col(savant)
        df = savant.copy()

        # Enrich with expected stats if available
        if expected is not None:
            expected = _build_name_col(expected)
            xmap = {
                "xba": "xba", "xslg": "xslg", "xobp": "xobp",
                "xwoba": "xwoba", "xiso": "xiso",
            }
            xcols = ["last_name, first_name"] + [c for c in xmap if c in expected.columns]
            if len(xcols) > 1:
                df = df.merge(expected[xcols], on="last_name, first_name", how="left",
                              suffixes=("", "_x"))
                for old, new in xmap.items():
                    dup = f"{new}_x"
                    if dup in df.columns:
                        df[new] = df[new].fillna(df[dup])
                        df.drop(columns=[dup], inplace=True)

        return df

    # ── Case 2: Savant failed — build from FanGraphs + expected stats ─────────
    print("   ℹ  Building dataset from FanGraphs only (Savant unavailable)")

    if fg is None and expected is None:
        print("❌  All data sources failed. Cannot build dataset.")
        sys.exit(1)

    base = fg if fg is not None else expected
    base = _build_name_col(base)

    if fg is not None and expected is not None:
        expected = _build_name_col(expected)
        xcols = ["last_name, first_name"] + [
            c for c in ("xba", "xslg", "xobp", "xwoba", "xiso") if c in expected.columns
        ]
        if len(xcols) > 1:
            base = base.merge(expected[xcols], on="last_name, first_name",
                              how="left", suffixes=("", "_dup"))
            base.drop(columns=[c for c in base.columns if c.endswith("_dup")], inplace=True)

    # Ensure required Savant-style columns exist (fill zeros if missing)
    required = {
        "k_percent": 22.0, "bb_percent": 8.0, "babip": 0.290,
        "xba": 0.0, "xslg": 0.0, "xobp": 0.0, "xwoba": 0.0,
        "exit_velocity_avg": 0.0, "launch_angle_avg": 0.0,
        "sweet_spot_percent": 0.0, "barrel_batted_rate": 0.0,
    }
    for col, default in required.items():
        if col not in base.columns:
            base[col] = default

    return base


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{'='*60}")
    print(f"  Monte Carlo Baseball 2026 — Data Downloader")
    print(f"  Season: {YEAR}  |  Min PA: {MIN_PA}")
    print(f"{'='*60}\n")

    savant   = _fetch_savant()
    fg       = _fetch_fangraphs()
    expected = _fetch_expected()

    print("\n⚙  Merging sources …")
    df = _merge_sources(savant, fg, expected)
    df = _build_name_col(df)

    # ── Final clean-up ────────────────────────────────────────────────────────
    # Drop completely empty rows and obvious test rows
    df = df[df["last_name, first_name"].notna()]
    df = df[~df["last_name, first_name"].str.lower().str.contains("totals|average", na=False)]

    # Numeric coercion on every column that should be numeric
    numeric_hints = [
        "pa", "ab", "hit", "single", "double", "triple", "home_run",
        "strikeout", "walk", "b_hit_by_pitch", "b_rbi",
        "k_percent", "bb_percent", "batting_avg", "slg_percent",
        "on_base_percent", "isolated_power", "babip",
        "xba", "xslg", "xobp", "xwoba",
        "exit_velocity_avg", "launch_angle_avg",
        "sweet_spot_percent", "barrel_batted_rate",
    ]
    for col in numeric_hints:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Sort by PA descending so top hitters appear first in app dropdowns
    if "pa" in df.columns:
        df = df.sort_values("pa", ascending=False).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅  Saved → {OUTPUT_FILE}")
    print(f"   Rows  : {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    print(f"\n   Top 5 players by PA:")
    preview_cols = ["last_name, first_name", "pa", "k_percent", "bb_percent",
                    "batting_avg", "on_base_percent", "babip"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(5).to_string(index=False))
    print(f"\n   Place '{OUTPUT_FILE}' in the same folder as app.py and run:")
    print("   streamlit run app.py\n")


def verify(path: str) -> None:
    """Quick sanity-check on an existing CSV file."""
    print(f"\n🔍  Verifying: {path}\n")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", nrows=5000)
        df.columns = df.columns.str.strip().str.lower()
    except Exception as exc:
        print(f"❌  Cannot read file: {exc}")
        sys.exit(1)

    checks = {
        "'last_name, first_name' column": "last_name, first_name" in df.columns,
        "'pa' column":                    "pa" in df.columns,
        "'k_percent' column":             "k_percent" in df.columns,
        "'bb_percent' column":            "bb_percent" in df.columns,
        "'babip' column":                 "babip" in df.columns,
        "xStats present":                 "xba" in df.columns,
        "PA values numeric":              pd.to_numeric(df.get("pa", pd.Series([0])),
                                                        errors="coerce").notna().all(),
        "≥ 100 players":                  len(df) >= 100,
    }
    pa_col = pd.to_numeric(df.get("pa", pd.Series([0])), errors="coerce").fillna(0)
    qualified = int((pa_col >= 100).sum())

    all_ok = True
    for desc, ok in checks.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon}  {desc}")
        if not ok:
            all_ok = False

    print(f"\n  Total rows   : {len(df)}")
    print(f"  PA ≥ 100     : {qualified}")
    if "k_percent" in df.columns:
        sample = pd.to_numeric(df["k_percent"], errors="coerce").dropna()
        fmt    = "percent (e.g. 21.5)" if sample.median() > 1.0 else "decimal (e.g. 0.215)"
        print(f"  k_percent fmt: {fmt}")

    print(f"\n{'✅  File looks good!' if all_ok else '⚠  Some checks failed — see above.'}\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--verify":
        verify(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--verify":
        verify(OUTPUT_FILE)
    else:
        main()
