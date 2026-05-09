"""
Monte Carlo Baseball 2025
Stochastic run-scoring simulator for MLB batting lineups.
Uses Baseball Savant CSV export data and a 9-inning plate-appearance model.
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── i18n ──────────────────────────────────────────────────────────────────────

LANGUAGES = {"English": "en", "Polski": "pl", "Español": "es", "Français": "fr", "日本語": "ja"}

_TR: dict[str, dict[str, str]] = {
    "app_title":       {"en": "Monte Carlo Baseball 2025", "pl": "Monte Carlo Baseball 2025",
                        "es": "Monte Carlo Béisbol 2025",  "fr": "Monte Carlo Baseball 2025",
                        "ja": "モンテカルロ野球 2025"},
    "lang_label":      {"en": "Language", "pl": "Język", "es": "Idioma", "fr": "Langue", "ja": "言語"},
    "era_plus":        {"en": "Opposing ERA+", "pl": "ERA+ przeciwnika", "es": "ERA+ rival",
                        "fr": "ERA+ adverse",              "ja": "相手ERA+"},
    "min_pa":          {"en": "Minimum PA",    "pl": "Minimum PA",       "es": "PA mínimo",
                        "fr": "PA minimum",                "ja": "最低PA"},
    "ultra_fast":      {"en": "Ultra-fast mode (fewer games)", "pl": "Tryb szybki (mniej gier)",
                        "es": "Modo ultra-rápido",          "fr": "Mode ultra-rapide",
                        "ja": "超高速モード"},
    "tab_lineup":      {"en": "Mixed Lineup",      "pl": "Mieszany Lineup",
                        "es": "Alineación Mixta",  "fr": "Alignement Mixte", "ja": "混合打線"},
    "tab_same":        {"en": "9× Same Batter",    "pl": "9× Ten Sam",
                        "es": "9× Mismo Bateador", "fr": "9× Même Frappeur", "ja": "9×同一打者"},
    "tab_sensitivity": {"en": "Sensitivity Analysis",   "pl": "Analiza Wrażliwości",
                        "es": "Análisis de Sensibilidad", "fr": "Analyse de Sensibilité",
                        "ja": "感度分析"},
    "add_player":      {"en": "Add Custom Player",          "pl": "Dodaj własnego gracza",
                        "es": "Añadir jugador personalizado","fr": "Ajouter un joueur",
                        "ja": "カスタム選手を追加"},
    "player_name":     {"en": "Name", "pl": "Imię i nazwisko", "es": "Nombre", "fr": "Nom", "ja": "名前"},
    "stat_level":      {"en": "Stat input level",         "pl": "Poziom szczegółowości statystyk",
                        "es": "Nivel de estadísticas",    "fr": "Niveau des statistiques",
                        "ja": "統計入力レベル"},
    "level_basic":     {"en": "Basic  ·  BA / OBP / SLG", "pl": "Podstawowy  ·  BA / OBP / SLG",
                        "es": "Básico  ·  BA / OBP / SLG","fr": "Basique  ·  BA / OBP / SLG",
                        "ja": "基本  ·  BA / OBP / SLG"},
    "level_mid":       {"en": "Intermediate  ·  + K% / HR%", "pl": "Średni  ·  + K% / HR%",
                        "es": "Intermedio  ·  + K% / HR%",   "fr": "Intermédiaire  ·  + K% / HR%",
                        "ja": "中級  ·  + K% / HR%"},
    "level_adv":       {"en": "Advanced  ·  full stat set",    "pl": "Zaawansowany  ·  pełny zestaw",
                        "es": "Avanzado  ·  estadísticas completas","fr": "Avancé  ·  ensemble complet",
                        "ja": "上級  ·  完全な統計セット"},
    "btn_add":         {"en": "Add player",  "pl": "Dodaj gracza",    "es": "Añadir",
                        "fr": "Ajouter",     "ja": "追加"},
    "btn_simulate":    {"en": "▶  Simulate lineup",        "pl": "▶  Symuluj lineup",
                        "es": "▶  Simular alineación",    "fr": "▶  Simuler l'alignement",
                        "ja": "▶  シミュレーション開始"},
    "btn_compare":     {"en": "⚔  Compare", "pl": "⚔  Porównaj", "es": "⚔  Comparar",
                        "fr": "⚔  Comparer", "ja": "⚔  比較"},
    "btn_run_sens":    {"en": "▶  Run analysis",   "pl": "▶  Uruchom analizę",
                        "es": "▶  Ejecutar análisis","fr": "▶  Lancer l'analyse",
                        "ja": "▶  分析実行"},
    "spinner":         {"en": "Simulating…",      "pl": "Symulacja…",
                        "es": "Simulando…",       "fr": "Simulation en cours…",
                        "ja": "シミュレーション中…"},
    "rpg":             {"en": "runs / game",         "pl": "runów / mecz",
                        "es": "carreras / juego",   "fr": "points / match",
                        "ja": "得点 / 試合"},
    "rps":             {"en": "projected runs / 162-game season",
                        "pl": "runów w sezonie (162 mecze)",
                        "es": "carreras proyectadas (162 juegos)",
                        "fr": "points projetés sur 162 matches",
                        "ja": "投影得点 / 162試合シーズン"},
    "elapsed":         {"en": "Elapsed", "pl": "Czas", "es": "Tiempo", "fr": "Durée", "ja": "所要時間"},
    "prob_header":     {"en": "Outcome probabilities",     "pl": "Rozkład prawdopodobieństwa",
                        "es": "Probabilidades por resultado","fr": "Probabilités des résultats",
                        "ja": "結果別確率"},
    "slot":            {"en": "Slot", "pl": "Slot", "es": "Pos.", "fr": "Pos.", "ja": "打順"},
    "player_a":        {"en": "Player A",   "pl": "Gracz A",    "es": "Jugador A",
                        "fr": "Joueur A",   "ja": "選手A"},
    "player_b":        {"en": "Player B",   "pl": "Gracz B",    "es": "Jugador B",
                        "fr": "Joueur B",   "ja": "選手B"},
    "sens_player":     {"en": "Player for analysis",   "pl": "Gracz do analizy",
                        "es": "Jugador para análisis", "fr": "Joueur à analyser",
                        "ja": "分析対象選手"},
    "sens_desc":       {"en": "RPG impact of each stat — 9× same batter, all others held constant.",
                        "pl": "Wpływ każdej statystyki na RPG — 9× ten sam pałkarz, pozostałe stałe.",
                        "es": "Impacto de cada estadística en RPG — 9× mismo bateador, resto fijos.",
                        "fr": "Impact de chaque stat sur les points/match — 9× même frappeur.",
                        "ja": "各統計がRPGに与える影響 — 9×同一打者、他固定。"},
    "tornado_title":   {"en": "Sensitivity Tornado — RPG swing across realistic stat range",
                        "pl": "Tornado wrażliwości — zmiana RPG w realistycznym zakresie",
                        "es": "Tornado de sensibilidad — variación de RPG en rango realista",
                        "fr": "Tornade de sensibilité — variation de points/match",
                        "ja": "感度トルネード — 現実的な統計範囲でのRPG変化"},
    "obp_slg_title":   {"en": "OBP vs SLG — equal-increment comparison (+Δ each metric, same scale)",
                        "pl": "OBP vs SLG — porównanie przy równym przyroście (+Δ każdej metryki)",
                        "es": "OBP vs SLG — comparación con igual incremento (+Δ cada métrica)",
                        "fr": "OBP vs SLG — comparaison à incrément égal (+Δ chaque métrique)",
                        "ja": "OBP vs SLG — 同一増分での比較（+Δ各指標、同スケール）"},
    "rpg_swing":       {"en": "RPG swing",      "pl": "Zmiana RPG",
                        "es": "Var. carr./juego","fr": "Var. pts/match", "ja": "RPG変化幅"},
    "obp_line":        {"en": "OBP  (via BB%)", "pl": "OBP  (przez BB%)",
                        "es": "OBP  (vía BB%)", "fr": "OBP  (via BB%)",  "ja": "OBP（BB%経由）"},
    "slg_line":        {"en": "SLG power  (via HR%)", "pl": "Moc SLG  (przez HR%)",
                        "es": "Potencia SLG  (vía HR%)","fr": "Puissance SLG  (via HR%)",
                        "ja": "SLGパワー（HR%経由）"},
    "delta_label":     {"en": "+Δ stat value", "pl": "+Δ wartość statystyki",
                        "es": "+Δ valor est.", "fr": "+Δ valeur stat.",  "ja": "+Δ統計値"},
    "no_csv":          {"en": "CSV not found — sample data loaded.",
                        "pl": "Brak pliku CSV — załadowano dane przykładowe.",
                        "es": "CSV no encontrado — cargados datos de muestra.",
                        "fr": "CSV introuvable — données exemple chargées.",
                        "ja": "CSV未検出 — サンプルデータを使用。"},
    "added":           {"en": "Added", "pl": "Dodano", "es": "Añadido", "fr": "Ajouté", "ja": "追加済み"},
    "sens_lineup_note":{"en": "Analysis uses 9× the selected player against the sidebar ERA+.",
                        "pl": "Analiza używa 9× wybranego gracza przy ERA+ z paska bocznego.",
                        "es": "El análisis usa 9× el jugador seleccionado con el ERA+ del panel.",
                        "fr": "L'analyse utilise 9× le joueur sélectionné avec l'ERA+ de la barre.",
                        "ja": "分析はサイドバーのERA+で選択選手9人を使用します。"},
}


def t(key: str) -> str:
    lang = st.session_state.get("lang", "en")
    return _TR.get(key, {}).get(lang) or _TR.get(key, {}).get("en", key)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_savant_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    df["Name"] = df["last_name, first_name"].astype(str).str.strip()

    for col in ["pa", "ab", "single", "double", "triple", "home_run",
                "k_percent", "bb_percent", "babip", "b_hit_by_pitch"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["PA"] = df["pa"].clip(lower=1)

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

    return (df.sort_values("PA", ascending=False)
              .drop_duplicates("Name")
              .reset_index(drop=True))


_SAMPLE_DATA = pd.DataFrame({
    "Name":    ["Aaron Judge", "Bobby Witt Jr.", "Luis Arráez",
                "Shohei Ohtani", "Ozzie Albies", "Willy Adames"],
    "PA":      [650,   680,   620,   640,   600,   650],
    "K_pct":   [0.236, 0.198, 0.085, 0.257, 0.140, 0.260],
    "BB_pct":  [0.183, 0.092, 0.072, 0.150, 0.080, 0.110],
    "HR_pct":  [0.078, 0.046, 0.018, 0.076, 0.027, 0.045],
    "BABIP":   [0.298, 0.312, 0.325, 0.290, 0.290, 0.310],
    "HBP_pct": [0.012, 0.008, 0.005, 0.015, 0.010, 0.010],
    "1B_rate": [0.68,  0.72,  0.78,  0.65,  0.65,  0.60],
    "2B_rate": [0.22,  0.20,  0.18,  0.25,  0.25,  0.25],
    "3B_rate": [0.05,  0.08,  0.04,  0.05,  0.05,  0.05],
})


# ── Stat derivation from basic / intermediate inputs ─────────────────────────

def _derive_hit_split(iso: float) -> tuple[float, float, float]:
    xbh = min(iso * 2.0, 0.45)
    r1 = max(0.50, 1.0 - xbh)
    r2 = xbh * 0.85
    r3 = xbh * 0.15
    tot = r1 + r2 + r3
    return r1 / tot, r2 / tot, r3 / tot


def derive_from_basic(ba: float, obp: float, slg: float) -> dict:
    bb_pct = max(0.02, min(obp - ba, 0.25))
    iso    = max(0.0, slg - ba)
    k_pct  = 0.220
    hr_pct = max(0.0, min(iso * 0.35, 0.12))
    denom  = max(0.01, 1.0 - k_pct - hr_pct)
    babip  = min(max((ba - hr_pct) / denom, 0.18), 0.42)
    r1, r2, r3 = _derive_hit_split(iso)
    return dict(K_pct=k_pct, BB_pct=bb_pct, HBP_pct=0.008,
                HR_pct=hr_pct, BABIP=babip,
                r1=r1, r2=r2, r3=r3)


def derive_from_intermediate(ba: float, obp: float, slg: float,
                              k_pct: float, hr_pct: float) -> dict:
    bb_pct = max(0.02, min(obp - ba, 0.25))
    iso    = max(0.0, slg - ba)
    denom  = max(0.01, 1.0 - k_pct - hr_pct)
    babip  = min(max((ba - hr_pct) / denom, 0.18), 0.42)
    r1, r2, r3 = _derive_hit_split(iso)
    return dict(K_pct=k_pct, BB_pct=bb_pct, HBP_pct=0.008,
                HR_pct=hr_pct, BABIP=babip,
                r1=r1, r2=r2, r3=r3)


# ── Probability model ─────────────────────────────────────────────────────────

@st.cache_data
def get_batter_probs(K: float, BB: float, HBP: float, HR: float,
                     babip: float, r1: float, r2: float, r3: float,
                     era_plus: int) -> np.ndarray:
    K     = min(max(K,     0.0), 0.60)
    BB    = min(max(BB,    0.0), 0.30)
    HBP   = min(max(HBP,   0.0), 0.05)
    HR    = min(max(HR,    0.0), 0.15)
    babip = min(max(babip, 0.10), 0.45)

    q     = era_plus / 100.0
    pK    = max(0.0, min(K   * (1.0 + 0.5  * (q - 1.0)), 0.55))
    pBB   = max(0.0,     BB  * (1.0 - 0.4  * (q - 1.0)))
    pHBP  = max(0.0,     HBP * (1.0 - 0.4  * (q - 1.0)))
    pHR   = max(0.0,     HR  * (1.0 - 0.4  * (q - 1.0)))
    adj_b = max(0.15, min(babip * (1.0 - 0.25 * (q - 1.0)), 0.40))

    total_non = pK + pBB + pHBP + pHR
    if total_non > 0.65:
        s = 0.65 / total_non
        pK *= s; pBB *= s; pHBP *= s; pHR *= s

    p_c   = max(0.0, 1.0 - pK - pBB - pHBP - pHR)
    hits  = adj_b * p_c
    probs = np.array([p_c - hits + pK, pBB + pHBP,
                      hits * r1, hits * r2, hits * r3, pHR], dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    return probs / s if s > 0 else np.array([0.68, 0.09, 0.13, 0.05, 0.01, 0.04])


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_games(lineup_probs: np.ndarray, n_games: int) -> float:
    rng       = np.random.default_rng()
    n_batters = lineup_probs.shape[0]
    cumprobs  = np.cumsum(lineup_probs, axis=1)

    MAX_PA   = 270
    total_pa = n_games * MAX_PA

    batter_seq = np.arange(total_pa) % n_batters
    rand_pa    = rng.random(total_pa)
    outcomes   = (cumprobs[batter_seq] < rand_pa[:, None]).sum(axis=1)
    outcomes   = np.clip(outcomes, 0, 5).astype(np.int32).tolist()

    total_runs = 0

    for g in range(n_games):
        base      = g * MAX_PA
        pa_cursor = 0
        game_runs = 0

        for _ in range(9):
            bases = 0
            outs  = 0

            while outs < 3 and pa_cursor < MAX_PA:
                res = outcomes[base + pa_cursor]
                pa_cursor += 1

                if res == 0:
                    outs += 1
                    continue

                on1 = bases & 1
                on2 = (bases >> 1) & 1
                on3 = (bases >> 2) & 1

                if res == 5:
                    game_runs += on1 + on2 + on3 + 1
                    bases = 0
                elif res == 4:
                    game_runs += on1 + on2 + on3
                    bases = 4
                elif res == 3:
                    game_runs += on2 + on3
                    bases = (on1 << 2) | 2
                elif res == 2:
                    game_runs += on3
                    bases = 1 | (on1 << 1) | (on2 << 2)
                else:
                    if on1 and on2 and on3:
                        game_runs += 1
                    elif on1 and on2:
                        bases = 7
                    elif on1:
                        bases = 3
                    else:
                        bases |= 1

        total_runs += game_runs

    return total_runs / n_games


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_get(row, col: str, default: float) -> float:
    val = row.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def build_probs(row, era_plus: int) -> np.ndarray:
    return get_batter_probs(
        safe_get(row, "K_pct",   0.22), safe_get(row, "BB_pct",  0.08),
        safe_get(row, "HBP_pct", 0.01), safe_get(row, "HR_pct",  0.03),
        safe_get(row, "BABIP",   0.290),
        safe_get(row, "1B_rate", 0.70), safe_get(row, "2B_rate", 0.20),
        safe_get(row, "3B_rate", 0.05), era_plus,
    )


def prob_label(row, era_plus: int) -> str:
    p = build_probs(row, era_plus)
    return (f"out {p[0]*100:.1f}% · walk/HBP {p[1]*100:.1f}% · "
            f"1B {p[2]*100:.1f}% · 2B {p[3]*100:.1f}% · "
            f"3B {p[4]*100:.1f}% · HR {p[5]*100:.1f}%")


def _base_stats(row) -> dict:
    return {
        "K_pct":   safe_get(row, "K_pct",   0.22),
        "BB_pct":  safe_get(row, "BB_pct",  0.08),
        "HBP_pct": safe_get(row, "HBP_pct", 0.01),
        "HR_pct":  safe_get(row, "HR_pct",  0.03),
        "BABIP":   safe_get(row, "BABIP",   0.290),
        "1B_rate": safe_get(row, "1B_rate", 0.70),
        "2B_rate": safe_get(row, "2B_rate", 0.20),
        "3B_rate": safe_get(row, "3B_rate", 0.05),
    }


def _rpg_override(stats: dict, era_plus: int, n_games: int, **kw) -> float:
    s = {**stats, **kw}
    r1, r2, r3 = s["1B_rate"], s["2B_rate"], s["3B_rate"]
    tot = r1 + r2 + r3
    p = get_batter_probs(s["K_pct"], s["BB_pct"], s["HBP_pct"], s["HR_pct"],
                         s["BABIP"], r1 / tot, r2 / tot, r3 / tot, era_plus)
    return simulate_games(np.array([p] * 9, dtype=np.float64), n_games)


# ── Sensitivity computation ───────────────────────────────────────────────────

_SENS_RANGES: dict[str, tuple[str, float, float]] = {
    "K%":      ("K_pct",   0.08, 0.38),
    "BB%":     ("BB_pct",  0.03, 0.18),
    "HR%":     ("HR_pct",  0.005, 0.09),
    "BABIP":   ("BABIP",   0.220, 0.370),
    "2B rate": ("2B_rate", 0.08,  0.30),
}

_N_SENS = 14
_N_GAMES_SENS = 150


def compute_sensitivity(row, era_plus: int):
    base    = _base_stats(row)
    base_rpg = _rpg_override(base, era_plus, _N_GAMES_SENS)

    tornado: dict[str, tuple] = {}
    for label, (col, lo, hi) in _SENS_RANGES.items():
        lo_c = min(lo, base[col])
        hi_c = max(hi, base[col])
        rpg_lo = _rpg_override(base, era_plus, _N_GAMES_SENS, **{col: lo_c})
        rpg_hi = _rpg_override(base, era_plus, _N_GAMES_SENS, **{col: hi_c})
        tornado[label] = (rpg_lo, base_rpg, rpg_hi, lo_c, base[col], hi_c)

    deltas      = np.linspace(0.0, 0.08, _N_SENS)
    obp_curve   = [_rpg_override(base, era_plus, _N_GAMES_SENS,
                                 BB_pct=min(base["BB_pct"] + d, 0.30)) for d in deltas]
    slg_curve   = [_rpg_override(base, era_plus, _N_GAMES_SENS,
                                 HR_pct=min(base["HR_pct"] + d / 3.0, 0.15)) for d in deltas]

    detail: dict[str, tuple] = {}
    for label, (col, lo, hi) in _SENS_RANGES.items():
        lo_c = min(lo, base[col])
        hi_c = max(hi, base[col])
        xs   = np.linspace(lo_c, hi_c, _N_SENS)
        rpgs = [_rpg_override(base, era_plus, _N_GAMES_SENS, **{col: float(x)}) for x in xs]
        detail[label] = (xs.tolist(), rpgs)

    return base_rpg, tornado, deltas.tolist(), obp_curve, slg_curve, detail


# ── Plotly charts ─────────────────────────────────────────────────────────────

def _tornado_chart(tornado: dict, base_rpg: float) -> go.Figure:
    items = sorted(tornado.items(), key=lambda kv: abs(kv[1][2] - kv[1][0]))
    labels  = [k for k, _ in items]
    swings  = [abs(v[2] - v[0]) for _, v in items]
    lo_rpgs = [v[0] for _, v in items]
    hi_rpgs = [v[2] for _, v in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[r - base_rpg for r in lo_rpgs],
        base=[base_rpg] * len(labels),
        orientation="h", name="Low end",
        marker_color="#4C8AC6", hovertemplate="%{x:.2f} RPG<extra>%{y} low</extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[r - base_rpg for r in hi_rpgs],
        base=[base_rpg] * len(labels),
        orientation="h", name="High end",
        marker_color="#E07B54", hovertemplate="%{x:.2f} RPG<extra>%{y} high</extra>",
    ))
    fig.add_vline(x=base_rpg, line_dash="dot", line_color="white", line_width=1.5)
    fig.update_layout(
        barmode="overlay",
        xaxis_title=t("rpg"), yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=340, margin=dict(l=80, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


def _obp_slg_chart(deltas: list, obp_curve: list, slg_curve: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deltas, y=obp_curve, mode="lines+markers",
        name=t("obp_line"), line=dict(color="#4C8AC6", width=2),
        hovertemplate="Δ=+%{x:.3f}  →  %{y:.2f} RPG<extra>OBP</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=deltas, y=slg_curve, mode="lines+markers",
        name=t("slg_line"), line=dict(color="#E07B54", width=2, dash="dash"),
        hovertemplate="Δ=+%{x:.3f}  →  %{y:.2f} RPG<extra>SLG</extra>",
    ))
    fig.update_layout(
        xaxis_title=t("delta_label"), yaxis_title=t("rpg"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320, margin=dict(l=60, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


def _detail_chart(detail: dict) -> go.Figure:
    colors = ["#4C8AC6", "#E07B54", "#5DBB8A", "#C67BB5", "#E0C454"]
    fig = go.Figure()
    for i, (label, (xs, rpgs)) in enumerate(detail.items()):
        fig.add_trace(go.Scatter(
            x=xs, y=rpgs, mode="lines+markers", name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f"{label}=%{{x:.3f}}  →  %{{y:.2f}} RPG<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Stat value", yaxis_title=t("rpg"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=360, margin=dict(l=60, r=20, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Monte Carlo Baseball 2025", layout="wide")

# Language selector (top of sidebar, persists via session state)
lang_names = list(LANGUAGES.keys())
lang_idx   = lang_names.index(
    next(k for k, v in LANGUAGES.items() if v == st.session_state.get("lang", "en"))
)
chosen_lang = st.sidebar.selectbox(t("lang_label"), lang_names, index=lang_idx)
st.session_state["lang"] = LANGUAGES[chosen_lang]

st.title(t("app_title"))

# Data
if "players_df" not in st.session_state:
    try:
        st.session_state.players_df = load_savant_csv("Batters_Savant_stats.csv")
    except FileNotFoundError:
        st.info(t("no_csv"))
        st.session_state.players_df = _SAMPLE_DATA.copy()

# Sidebar controls
era_plus   = st.sidebar.slider(t("era_plus"), 70, 150, 100)
min_pa     = st.sidebar.slider(t("min_pa"),   0,  700, 100)
ultra_fast = st.sidebar.checkbox(t("ultra_fast"), value=True)
n_mixed    = 300  if ultra_fast else 1000
n_same     = 200  if ultra_fast else 600

# ── Add custom player ─────────────────────────────────────────────────────────
with st.sidebar.expander(t("add_player")):
    new_name  = st.text_input(t("player_name"), "Custom Player")
    level_map = {t("level_basic"): "basic", t("level_mid"): "mid", t("level_adv"): "adv"}
    level_key = st.radio(t("stat_level"), list(level_map.keys()), horizontal=False)
    level     = level_map[level_key]

    if level == "basic":
        ba  = st.slider("BA",  0.150, 0.400, 0.260, 0.001)
        obp = st.slider("OBP", 0.200, 0.500, 0.330, 0.001)
        slg = st.slider("SLG", 0.250, 0.700, 0.420, 0.001)
        derived = derive_from_basic(ba, obp, slg)
    elif level == "mid":
        ba    = st.slider("BA",  0.150, 0.400, 0.260, 0.001)
        obp   = st.slider("OBP", 0.200, 0.500, 0.330, 0.001)
        slg   = st.slider("SLG", 0.250, 0.700, 0.420, 0.001)
        k_pct = st.slider("K%", 0.05, 0.50, 0.22, 0.001)
        hr_pct= st.slider("HR%",0.00, 0.12, 0.03, 0.001)
        derived = derive_from_intermediate(ba, obp, slg, k_pct, hr_pct)
    else:
        derived = {}
        derived["K_pct"]   = st.slider("K%",      0.0,  0.50, 0.22, 0.001)
        derived["BB_pct"]  = st.slider("BB%",      0.0,  0.30, 0.08, 0.001)
        derived["HR_pct"]  = st.slider("HR%",      0.0,  0.15, 0.03, 0.001)
        derived["BABIP"]   = st.slider("BABIP",  0.200, 0.400, 0.300, 0.001)
        derived["HBP_pct"] = st.slider("HBP%",    0.0,  0.05, 0.01, 0.001)
        derived["r1"]      = st.slider("1B rate", 0.50,  0.90, 0.70, 0.01)
        derived["r2"]      = st.slider("2B rate", 0.05,  0.35, 0.20, 0.01)
        derived["r3"]      = st.slider("3B rate", 0.00,  0.15, 0.05, 0.01)

    if st.button(t("btn_add")):
        r1, r2, r3 = derived.get("r1", 0.70), derived.get("r2", 0.20), derived.get("r3", 0.05)
        tot = r1 + r2 + r3
        new_row = pd.DataFrame([{
            "Name":    new_name,       "PA":      600,
            "K_pct":   derived["K_pct"],  "BB_pct":  derived["BB_pct"],
            "HR_pct":  derived["HR_pct"], "HBP_pct": derived.get("HBP_pct", 0.008),
            "BABIP":   derived["BABIP"],
            "1B_rate": r1 / tot, "2B_rate": r2 / tot, "3B_rate": r3 / tot,
        }])
        st.session_state.players_df = pd.concat(
            [st.session_state.players_df, new_row], ignore_index=True
        )
        st.success(f"{t('added')}: {new_name}")
        st.rerun()

df      = st.session_state.players_df[
    st.session_state.players_df["PA"] >= min_pa].copy()
players = sorted(df["Name"].unique())

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([t("tab_lineup"), t("tab_same"), t("tab_sensitivity")])

# ── Tab 1: Mixed lineup ───────────────────────────────────────────────────────
with tab1:
    lineup_names = [
        st.selectbox(f"{t('slot')} {i + 1}", players, key=f"slot{i}")
        for i in range(9)
    ]
    if st.button(t("btn_simulate"), type="primary"):
        probs_list, debug_lines = [], []
        for name in lineup_names:
            row = df[df["Name"] == name].iloc[0]
            probs_list.append(build_probs(row, era_plus))
            debug_lines.append(f"**{name}** — {prob_label(row, era_plus)}")

        with st.spinner(t("spinner")):
            start = time.time()
            rpg   = simulate_games(np.array(probs_list, dtype=np.float64), n_mixed)
            elapsed = time.time() - start

        st.success(f"**{rpg:.2f} {t('rpg')}** → **{int(rpg * 162)} {t('rps')}**")
        st.caption(f"{t('elapsed')}: {elapsed:.2f} s")
        with st.expander(t("prob_header")):
            for line in debug_lines:
                st.markdown(line)

# ── Tab 2: 9× same batter ─────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox(t("player_a"), players, key="player_a")
    with c2:
        b = st.selectbox(t("player_b"), players, key="player_b")

    if st.button(t("btn_compare"), type="primary"):
        row_a, row_b = df[df["Name"] == a].iloc[0], df[df["Name"] == b].iloc[0]
        la = np.array([build_probs(row_a, era_plus)] * 9, dtype=np.float64)
        lb = np.array([build_probs(row_b, era_plus)] * 9, dtype=np.float64)

        with st.spinner(t("spinner")):
            start   = time.time()
            r_a     = simulate_games(la, n_same)
            r_b     = simulate_games(lb, n_same)
            elapsed = time.time() - start

        delta = r_a - r_b
        st.metric(f"9× {a}", f"{r_a:.2f}", f"{delta:+.2f}")
        st.metric(f"9× {b}", f"{r_b:.2f}")
        st.caption(f"{t('elapsed')}: {elapsed:.2f} s")
        with st.expander(t("prob_header")):
            st.markdown(f"**{a}** — {prob_label(row_a, era_plus)}")
            st.markdown(f"**{b}** — {prob_label(row_b, era_plus)}")

# ── Tab 3: Sensitivity analysis ───────────────────────────────────────────────
with tab3:
    sens_player = st.selectbox(t("sens_player"), players, key="sens_player")
    st.caption(t("sens_lineup_note"))
    st.caption(t("sens_desc"))

    if st.button(t("btn_run_sens"), type="primary"):
        sens_row = df[df["Name"] == sens_player].iloc[0]
        with st.spinner(t("spinner")):
            base_rpg, tornado, deltas, obp_curve, slg_curve, detail = \
                compute_sensitivity(sens_row, era_plus)

        st.markdown(f"**Base RPG ({sens_player}, ERA+ {era_plus}): {base_rpg:.2f}**")

        st.subheader(t("tornado_title"))
        st.plotly_chart(_tornado_chart(tornado, base_rpg),
                        use_container_width=True, key="tornado")

        st.subheader(t("obp_slg_title"))
        st.plotly_chart(_obp_slg_chart(deltas, obp_curve, slg_curve),
                        use_container_width=True, key="obp_slg")

        with st.expander("Per-stat detail curves"):
            st.plotly_chart(_detail_chart(detail),
                            use_container_width=True, key="detail")