"""
Monte Carlo Baseball 2026
=========================
Stochastic run-scoring simulator for MLB batting lineups.
Data source: Baseball Savant 2025 actuals + xStats-based 2026 projections.

Sections
--------
 1. Imports & constants
 2. Internationalization (i18n)
 3. Team rosters & default lineups
 4. Data loading & processing
 5. Stat derivation (basic / intermediate custom player input)
 6. Probability model
 7. Monte Carlo simulation engine
 8. Sensitivity analysis
 9. Chart builders
10. Helper utilities
11. App layout — sidebar
12. App layout — main tabs
"""

from __future__ import annotations

import io
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

APP_TITLE   = "Monte Carlo Baseball 2026"
APP_VERSION = "2.0.0"
DATA_SOURCE = "Baseball Savant 2025 actuals · xStats-based 2026 projections"
CSV_PATH    = "Batters_Savant_stats.csv"

# Minimum PA to treat a player's stats as "actual"; below this, use xStats
PROJECTION_PA_THRESHOLD = 100

# Simulation defaults
_N_GAMES_DEFAULT_MIXED = 300
_N_GAMES_DEFAULT_SAME  = 200
_N_GAMES_DEFAULT_FULL  = 1000

# Sensitivity analysis
_N_SENS_POINTS  = 14
_N_GAMES_SENS   = 150

# Platoon: approximate share of LHP starts in MLB
_LHP_RATE = 0.38

# Chart palette
_PALETTE = ["#4C8AC6", "#E07B54", "#5DBB8A", "#C67BB5", "#E0C454", "#7BC6C6"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INTERNATIONALIZATION (i18n)
# ═══════════════════════════════════════════════════════════════════════════════

LANGUAGES: dict[str, str] = {
    "English": "en", "Polski": "pl", "Español": "es",
    "Français": "fr", "日本語": "ja",
}

_TR: dict[str, dict[str, str]] = {
    # ── App shell ──────────────────────────────────────────────────────────────
    "app_subtitle":   {"en": "2025 stats · 2026 projections · Baseball Savant",
                       "pl": "Statystyki 2025 · Projekcje 2026 · Baseball Savant",
                       "es": "Stats 2025 · Proyecciones 2026 · Baseball Savant",
                       "fr": "Stats 2025 · Projections 2026 · Baseball Savant",
                       "ja": "2025実績・2026予測・Baseball Savant"},
    "lang_label":     {"en": "Language", "pl": "Język", "es": "Idioma",
                       "fr": "Langue",   "ja": "言語"},
    # ── Sidebar sections ───────────────────────────────────────────────────────
    "sec_sim":        {"en": "⚙ Simulation", "pl": "⚙ Symulacja", "es": "⚙ Simulación",
                       "fr": "⚙ Simulation", "ja": "⚙ シミュレーション"},
    "sec_team":       {"en": "🏟 Team Mode", "pl": "🏟 Tryb Drużynowy", "es": "🏟 Modo Equipo",
                       "fr": "🏟 Mode Équipe","ja": "🏟 チームモード"},
    "sec_player":     {"en": "➕ Custom Player", "pl": "➕ Własny Gracz",
                       "es": "➕ Jugador Personalizado","fr": "➕ Joueur Personnalisé",
                       "ja": "➕ カスタム選手"},
    "sec_lineup_io":  {"en": "💾 Save / Load Lineup", "pl": "💾 Zapisz / Wczytaj Lineup",
                       "es": "💾 Guardar / Cargar Alineación","fr": "💾 Sauvegarder / Charger",
                       "ja": "💾 保存 / 読込"},
    # ── Simulation controls ────────────────────────────────────────────────────
    "era_plus":       {"en": "Opposing ERA+", "pl": "ERA+ przeciwnika", "es": "ERA+ rival",
                       "fr": "ERA+ adverse",  "ja": "相手ERA+"},
    "era_plus_help":  {"en": "100 = league-average pitcher. 150 = ace. 70 = replacement level.",
                       "pl": "100 = przeciętny miotacz. 150 = as. 70 = miotacz zastępczy.",
                       "es": "100 = lanzador promedio. 150 = as. 70 = nivel de sustitución.",
                       "fr": "100 = lanceur moyen. 150 = as. 70 = remplaçant.",
                       "ja": "100=平均投手、150=エース、70=控え投手レベル"},
    "min_pa":         {"en": "Minimum PA filter", "pl": "Filtr min. PA",
                       "es": "Filtro PA mínimo","fr": "Filtre PA minimum","ja": "最低PA フィルター"},
    "ultra_fast":     {"en": "Ultra-fast mode (fewer simulated games)",
                       "pl": "Tryb szybki (mniej gier)",
                       "es": "Modo ultra-rápido (menos partidos simulados)",
                       "fr": "Mode ultra-rapide (moins de matchs simulés)",
                       "ja": "超高速モード（試合数削減）"},
    "use_platoon":    {"en": "Use platoon splits (38 % LHP)", "pl": "Platoon splits (38% LHP)",
                       "es": "Usar divisiones platoon (38 % LHP)",
                       "fr": "Activer les splits platoon (38 % LHP)",
                       "ja": "プラトーンスプリット使用（LHP 38%）"},
    "platoon_help":   {"en": "Simulates ~38% of PAs vs LHP (generic RHH adjustment — "
                             "per-player handedness not available in CSV).",
                       "pl": "Symuluje ~38% PA przeciw LHP (ogólna korekta dla RHH — "
                             "strona odbicia nie jest dostępna w CSV).",
                       "es": "Simula ~38% de PAs vs LHP (ajuste genérico LHP — "
                             "lateralidad no disponible en CSV).",
                       "fr": "Simule ~38% des PAs vs LHP (ajustement générique — "
                             "latéralité non disponible dans le CSV).",
                       "ja": "PA約38%をLHP対戦でシミュレート（CSV内打席側データなし、汎用補正）"},
    # ── Team mode ──────────────────────────────────────────────────────────────
    "team_mode":      {"en": "Restrict to one MLB team",
                       "pl": "Ogranicz do jednej drużyny MLB",
                       "es": "Restringir a un equipo MLB",
                       "fr": "Restreindre à une équipe MLB",
                       "ja": "1チームに限定"},
    "select_team":    {"en": "Select team", "pl": "Wybierz drużynę",
                       "es": "Seleccionar equipo","fr": "Choisir l'équipe","ja": "チーム選択"},
    "load_lineup":    {"en": "Load example lineup", "pl": "Wczytaj przykładowy lineup",
                       "es": "Cargar alineación ejemplo","fr": "Charger un alignement exemple",
                       "ja": "サンプル打順を読み込む"},
    "btn_load":       {"en": "Load", "pl": "Wczytaj", "es": "Cargar",
                       "fr": "Charger","ja": "読込"},
    "btn_reset":      {"en": "Reset lineup", "pl": "Resetuj lineup",
                       "es": "Reiniciar alineación","fr": "Réinitialiser","ja": "リセット"},
    # ── Custom player ──────────────────────────────────────────────────────────
    "player_name":    {"en": "Player name", "pl": "Imię i nazwisko",
                       "es": "Nombre","fr": "Nom","ja": "名前"},
    "stat_level":     {"en": "Stat input level", "pl": "Poziom szczegółowości statystyk",
                       "es": "Nivel de estadísticas","fr": "Niveau des statistiques",
                       "ja": "統計入力レベル"},
    "level_basic":    {"en": "Basic — BA / OBP / SLG",  "pl": "Podstawowy — BA / OBP / SLG",
                       "es": "Básico — BA / OBP / SLG", "fr": "Basique — BA / OBP / SLG",
                       "ja": "基本 — BA / OBP / SLG"},
    "level_mid":      {"en": "Intermediate — + K% / HR%","pl": "Średni — + K% / HR%",
                       "es": "Intermedio — + K% / HR%",  "fr": "Intermédiaire — + K% / HR%",
                       "ja": "中級 — + K% / HR%"},
    "level_adv":      {"en": "Advanced — full stat set",     "pl": "Zaawansowany — pełny zestaw",
                       "es": "Avanzado — estadísticas completas","fr": "Avancé — ensemble complet",
                       "ja": "上級 — 完全統計セット"},
    "btn_add":        {"en": "Add player","pl": "Dodaj gracza","es": "Añadir",
                       "fr": "Ajouter","ja": "追加"},
    # ── Lineup I/O ─────────────────────────────────────────────────────────────
    "btn_save_json":  {"en": "Download lineup (JSON)","pl": "Pobierz lineup (JSON)",
                       "es": "Descargar alineación (JSON)","fr": "Télécharger l'alignement (JSON)",
                       "ja": "打順をダウンロード (JSON)"},
    "upload_json":    {"en": "Upload lineup JSON","pl": "Wgraj plik lineup JSON",
                       "es": "Subir JSON de alineación","fr": "Importer JSON alignement",
                       "ja": "打順JSONをアップロード"},
    # ── Tabs ───────────────────────────────────────────────────────────────────
    "tab_lineup":     {"en": "📋 Mixed Lineup",    "pl": "📋 Mieszany Lineup",
                       "es": "📋 Alineación Mixta", "fr": "📋 Alignement Mixte","ja": "📋 混合打線"},
    "tab_same":       {"en": "⚔ 9× Same Batter",  "pl": "⚔ 9× Ten Sam",
                       "es": "⚔ 9× Mismo Bateador","fr": "⚔ 9× Même Frappeur","ja": "⚔ 9×同一打者"},
    "tab_sensitivity":{"en": "📊 Sensitivity",     "pl": "📊 Wrażliwość",
                       "es": "📊 Sensibilidad",    "fr": "📊 Sensibilité","ja": "📊 感度分析"},
    # ── Buttons ────────────────────────────────────────────────────────────────
    "btn_simulate":   {"en": "▶  Simulate lineup","pl": "▶  Symuluj lineup",
                       "es": "▶  Simular","fr": "▶  Simuler","ja": "▶  シミュレーション"},
    "btn_compare":    {"en": "⚔  Compare","pl": "⚔  Porównaj","es": "⚔  Comparar",
                       "fr": "⚔  Comparer","ja": "⚔  比較"},
    "btn_run_sens":   {"en": "▶  Run analysis","pl": "▶  Uruchom analizę",
                       "es": "▶  Ejecutar análisis","fr": "▶  Lancer l'analyse",
                       "ja": "▶  分析実行"},
    # ── Labels & metrics ───────────────────────────────────────────────────────
    "slot":           {"en": "Slot","pl": "Slot","es": "Pos.","fr": "Pos.","ja": "打順"},
    "player_a":       {"en": "Player A","pl": "Gracz A","es": "Jugador A",
                       "fr": "Joueur A","ja": "選手A"},
    "player_b":       {"en": "Player B","pl": "Gracz B","es": "Jugador B",
                       "fr": "Joueur B","ja": "選手B"},
    "sens_player":    {"en": "Player for analysis","pl": "Gracz do analizy",
                       "es": "Jugador para análisis","fr": "Joueur à analyser",
                       "ja": "分析対象選手"},
    "spinner":        {"en": "Simulating…","pl": "Symulacja…","es": "Simulando…",
                       "fr": "Simulation en cours…","ja": "シミュレーション中…"},
    "rpg":            {"en": "runs / game","pl": "runów / mecz","es": "carreras / juego",
                       "fr": "points / match","ja": "得点 / 試合"},
    "rps":            {"en": "projected runs / 162-game season",
                       "pl": "runów w sezonie (162 mecze)",
                       "es": "carreras proyectadas / temporada 162",
                       "fr": "points projetés / saison 162 matchs",
                       "ja": "162試合シーズン投影得点"},
    "elapsed":        {"en": "Elapsed","pl": "Czas","es": "Tiempo","fr": "Durée","ja": "所要時間"},
    "prob_header":    {"en": "Outcome probabilities","pl": "Rozkład prawdopodobieństwa",
                       "es": "Probabilidades","fr": "Probabilités","ja": "結果別確率"},
    "data_badge_act": {"en": "actual","pl": "aktualne","es": "real","fr": "réel","ja": "実績"},
    "data_badge_proj":{"en": "proj","pl": "proj","es": "proy","fr": "proj","ja": "予測"},
    # ── Sensitivity ────────────────────────────────────────────────────────────
    "sens_note":      {"en": "9× same batter, ERA+ from sidebar, all other stats held constant.",
                       "pl": "9× ten sam pałkarz, ERA+ z paska bocznego, pozostałe stałe.",
                       "es": "9× mismo bateador, ERA+ del panel, resto de stats fijos.",
                       "fr": "9× même frappeur, ERA+ barre latérale, autres stats fixes.",
                       "ja": "9×同一打者、サイドバーのERA+、他統計固定。"},
    "computing":      {"en": "Computing…","pl": "Obliczanie…","es": "Calculando…",
                       "fr": "Calcul en cours…","ja": "計算中…"},
    "tornado_title":  {"en": "Sensitivity Tornado — RPG swing across realistic stat range",
                       "pl": "Tornado wrażliwości — zmiana RPG w realistycznym zakresie",
                       "es": "Tornado de sensibilidad — variación de RPG",
                       "fr": "Tornade de sensibilité","ja": "感度トルネード"},
    "obp_slg_title":  {"en": "OBP vs SLG — equal-increment comparison (same Δ scale)",
                       "pl": "OBP vs SLG — równy przyrost (ta sama skala Δ)",
                       "es": "OBP vs SLG — incremento igual (misma escala Δ)",
                       "fr": "OBP vs SLG — incrément identique","ja": "OBP vs SLG 同一増分比較"},
    "obp_line":       {"en": "OBP (via BB%)","pl": "OBP (przez BB%)","es": "OBP (vía BB%)",
                       "fr": "OBP (via BB%)","ja": "OBP（BB%経由）"},
    "slg_line":       {"en": "SLG power (via HR%)","pl": "Moc SLG (przez HR%)",
                       "es": "Potencia SLG (vía HR%)","fr": "Puissance SLG (via HR%)",
                       "ja": "SLGパワー（HR%経由）"},
    "delta_label":    {"en": "+Δ stat value","pl": "+Δ wartość statystyki",
                       "es": "+Δ valor","fr": "+Δ valeur","ja": "+Δ統計値"},
    "detail_curves":  {"en": "Per-stat detail curves","pl": "Krzywe szczegółowe per statystyka",
                       "es": "Curvas detalladas por estadística",
                       "fr": "Courbes détaillées par statistique","ja": "統計別詳細曲線"},
    "base_rpg":       {"en": "Base RPG","pl": "Bazowe RPG","es": "RPG base",
                       "fr": "Points/match base","ja": "ベースRPG"},
    # ── Notices ────────────────────────────────────────────────────────────────
    "no_csv":         {"en": "CSV not found — sample data loaded.",
                       "pl": "Brak pliku CSV — załadowano dane przykładowe.",
                       "es": "CSV no encontrado — datos de muestra cargados.",
                       "fr": "CSV introuvable — données exemple chargées.",
                       "ja": "CSV未検出 — サンプルデータを使用。"},
    "added":          {"en": "Added","pl": "Dodano","es": "Añadido","fr": "Ajouté","ja": "追加済み"},
    "lineup_loaded":  {"en": "Lineup loaded","pl": "Lineup wczytany","es": "Alineación cargada",
                       "fr": "Alignement chargé","ja": "打順読込完了"},
    "lineup_reset":   {"en": "Lineup reset","pl": "Lineup zresetowany","es": "Alineación reiniciada",
                       "fr": "Alignement réinitialisé","ja": "打順リセット完了"},
    "team_not_found": {"en": "Some players from this lineup are not in the current dataset.",
                       "pl": "Niektórzy zawodnicy z tego lineupu nie są w bieżącym zbiorze danych.",
                       "es": "Algunos jugadores de esta alineación no están en el dataset.",
                       "fr": "Certains joueurs de cet alignement ne sont pas dans le dataset.",
                       "ja": "この打順の一部選手が現在のデータセットに存在しません。"},
    "proj_note":      {"en": "🔮 Using xStats projection (PA < 100)",
                       "pl": "🔮 Używam projekcji xStats (PA < 100)",
                       "es": "🔮 Usando proyección xStats (PA < 100)",
                       "fr": "🔮 Utilisation projection xStats (PA < 100)",
                       "ja": "🔮 xStats予測使用（PA < 100）"},
    "upload_ok":      {"en": "Lineup imported successfully.",
                       "pl": "Lineup zaimportowany pomyślnie.",
                       "es": "Alineación importada correctamente.",
                       "fr": "Alignement importé avec succès.",
                       "ja": "打順のインポートに成功しました。"},
    "upload_err":     {"en": "Could not parse lineup file.",
                       "pl": "Nie można wczytać pliku lineupu.",
                       "es": "No se pudo analizar el archivo.",
                       "fr": "Impossible d'analyser le fichier.",
                       "ja": "ファイルを解析できませんでした。"},
}


def t(key: str) -> str:
    """Return translated string for the active language."""
    lang = st.session_state.get("lang", "en")
    return _TR.get(key, {}).get(lang) or _TR.get(key, {}).get("en", key)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TEAM ROSTERS & DEFAULT LINEUPS
# ═══════════════════════════════════════════════════════════════════════════════

# Player-name → team abbreviation mapping (names as stored in CSV).
# Used to build the "Team" column in the loaded DataFrame.
_PLAYER_TEAM: dict[str, str] = {
    # Yankees
    "Judge, Aaron": "NYY", "Soto, Juan": "NYY", "Stanton, Giancarlo": "NYY",
    "Torres, Gleyber": "NYY", "Volpe, Anthony": "NYY", "Cabrera, Oswaldo": "NYY",
    "Wells, Austin": "NYY", "Trevino, Jose": "NYY", "Verdugo, Alex": "NYY",
    # Dodgers
    "Ohtani, Shohei": "LAD", "Betts, Mookie": "LAD", "Freeman, Freddie": "LAD",
    "Muncy, Max": "LAD", "Smith, Will": "LAD", "Lux, Gavin": "LAD",
    "Hernández, Teoscar": "LAD", "Outman, James": "LAD", "Edman, Tommy": "LAD",
    # Mets
    "Lindor, Francisco": "NYM", "Alonso, Pete": "NYM", "Nimmo, Brandon": "NYM",
    "Vientos, Mark": "NYM", "McNeil, Jeff": "NYM", "Alvarez, Francisco": "NYM",
    "Iglesias, Jose": "NYM", "Marte, Starling": "NYM", "Stewart, DJ": "NYM",
    # Phillies
    "Harper, Bryce": "PHI", "Turner, Trea": "PHI", "Schwarber, Kyle": "PHI",
    "Castellanos, Nick": "PHI", "Bohm, Alec": "PHI", "Stott, Bryson": "PHI",
    "Realmuto, J.T.": "PHI", "Marsh, Brandon": "PHI", "Rojas, Josh": "PHI",
    # Braves
    "Acuña Jr., Ronald": "ATL", "Riley, Austin": "ATL", "Albies, Ozzie": "ATL",
    "Murphy, Sean": "ATL", "Harris II, Michael": "ATL", "Olson, Matt": "ATL",
    "Ozuna, Marcell": "ATL", "Rosario, Amed": "ATL", "d'Arnaud, Travis": "ATL",
    # Astros
    "Alvarez, Yordan": "HOU", "Bregman, Alex": "HOU", "Tucker, Kyle": "HOU",
    "Peña, Jeremy": "HOU", "Meyers, Jake": "HOU", "Diaz, Yainer": "HOU",
    "McCormick, Chas": "HOU", "Dubón, Mauricio": "HOU", "Abreu, Wilyer": "HOU",
    # Rangers
    "Seager, Corey": "TEX", "Lowe, Nathaniel": "TEX", "García, Adolis": "TEX",
    "Langford, Wyatt": "TEX", "Jung, Josh": "TEX", "Carter, Evan": "TEX",
    "Heim, Jonah": "TEX", "Taveras, Leody": "TEX", "Smith, Dominic": "TEX",
    # Guardians
    "Ramírez, José": "CLE", "Naylor, Josh": "CLE", "Freeman, Tyler": "CLE",
    "Brennan, Will": "CLE", "Kwan, Steven": "CLE", "Giménez, Andrés": "CLE",
    "Miller, Owen": "CLE", "Straw, Myles": "CLE", "Hedges, Austin": "CLE",
    # Cardinals
    "Arenado, Nolan": "STL", "Goldschmidt, Paul": "STL", "Gorman, Nolan": "STL",
    "Walker, Jordan": "STL", "Carlson, Dylan": "STL", "Donovan, Brendan": "STL",
    "Contreras, Willson": "STL", "Burleson, Alec": "STL", "Nootbaar, Lars": "STL",
    # Cubs
    "Suzuki, Seiya": "CHC", "Swanson, Dansby": "CHC", "Happ, Ian": "CHC",
    "Morel, Christopher": "CHC", "Amaya, Miguel": "CHC", "Hoerner, Nico": "CHC",
    "Bellinger, Cody": "CHC", "Busch, Michael": "CHC", "Tauchman, Mike": "CHC",
    # Orioles
    "Henderson, Gunnar": "BAL", "Mullins, Cedric": "BAL", "Santander, Anthony": "BAL",
    "Mountcastle, Ryan": "BAL", "Rutschman, Adley": "BAL", "O'Hearn, Ryan": "BAL",
    "Westburg, Jordan": "BAL", "Hays, Austin": "BAL", "Cowser, Colton": "BAL",
    # Blue Jays
    "Guerrero Jr., Vladimir": "TOR", "Bichette, Bo": "TOR", "Kirk, Alejandro": "TOR",
    "Biggio, Cavan": "TOR", "Springer, George": "TOR", "Jansen, Danny": "TOR",
    "Varsho, Daulton": "TOR", "Barger, Addison": "TOR", "Kiermaier, Kevin": "TOR",
    # Red Sox
    "Devers, Rafael": "BOS", "Yoshida, Masataka": "BOS", "Casas, Triston": "BOS",
    "Turner, Justin": "BOS", "Duran, Jarren": "BOS", "McGuire, Reese": "BOS",
    "Hamilton, David": "BOS", "Refsnyder, Rob": "BOS", "Dalbec, Bobby": "BOS",
    # Padres
    "Bogaerts, Xander": "SD", "Tatis Jr., Fernando": "SD", "Machado, Manny": "SD",
    "Profar, Jurickson": "SD", "Cronenworth, Jake": "SD", "Campusano, Luis": "SD",
    "Soto, Juan": "SD",  # note: may have moved; included for roster completeness
    # Giants
    "Bailey, Patrick": "SF", "Flores, Wilmer": "SF", "Conforto, Michael": "SF",
    "Slater, Austin": "SF", "Estrada, Thairo": "SF", "Davis, JD": "SF",
    # Angels
    "Trout, Mike": "LAA", "Rengifo, Luis": "LAA", "Moniak, Mickey": "LAA",
    "Ward, Taylor": "LAA", "Neto, Zach": "LAA", "O'Hoppe, Logan": "LAA",
}

# Full team name ↔ abbreviation
TEAMS: dict[str, str] = {
    "New York Yankees":    "NYY", "Los Angeles Dodgers": "LAD",
    "New York Mets":       "NYM", "Philadelphia Phillies":"PHI",
    "Atlanta Braves":      "ATL", "Houston Astros":      "HOU",
    "Texas Rangers":       "TEX", "Cleveland Guardians": "CLE",
    "St. Louis Cardinals": "STL", "Chicago Cubs":        "CHC",
    "Baltimore Orioles":   "BAL", "Toronto Blue Jays":   "TOR",
    "Boston Red Sox":      "BOS", "San Diego Padres":    "SD",
    "San Francisco Giants":"SF",  "Los Angeles Angels":  "LAA",
}

# Default lineups: team abbrev → ordered list of 9 player names (as in CSV)
_DEFAULT_LINEUPS: dict[str, list[str]] = {
    "NYY": ["Volpe, Anthony", "Soto, Juan", "Judge, Aaron", "Stanton, Giancarlo",
            "Torres, Gleyber", "Verdugo, Alex", "Cabrera, Oswaldo", "Wells, Austin",
            "Trevino, Jose"],
    "LAD": ["Betts, Mookie", "Ohtani, Shohei", "Freeman, Freddie", "Hernández, Teoscar",
            "Muncy, Max", "Smith, Will", "Edman, Tommy", "Lux, Gavin", "Outman, James"],
    "NYM": ["Nimmo, Brandon", "Lindor, Francisco", "Alonso, Pete", "Marte, Starling",
            "Vientos, Mark", "McNeil, Jeff", "Iglesias, Jose", "Alvarez, Francisco",
            "Stewart, DJ"],
    "PHI": ["Schwarber, Kyle", "Turner, Trea", "Harper, Bryce", "Castellanos, Nick",
            "Bohm, Alec", "Marsh, Brandon", "Stott, Bryson", "Realmuto, J.T.",
            "Rojas, Josh"],
    "ATL": ["Acuña Jr., Ronald", "Albies, Ozzie", "Riley, Austin", "Olson, Matt",
            "Ozuna, Marcell", "Harris II, Michael", "Murphy, Sean", "Rosario, Amed",
            "d'Arnaud, Travis"],
    "HOU": ["Alvarez, Yordan", "Bregman, Alex", "Tucker, Kyle", "Peña, Jeremy",
            "McCormick, Chas", "Diaz, Yainer", "Meyers, Jake", "Abreu, Wilyer",
            "Dubón, Mauricio"],
    "TEX": ["Seager, Corey", "Langford, Wyatt", "García, Adolis", "Lowe, Nathaniel",
            "Jung, Josh", "Carter, Evan", "Taveras, Leody", "Heim, Jonah",
            "Smith, Dominic"],
    "CLE": ["Kwan, Steven", "Giménez, Andrés", "Ramírez, José", "Naylor, Josh",
            "Freeman, Tyler", "Brennan, Will", "Miller, Owen", "Straw, Myles",
            "Hedges, Austin"],
    "STL": ["Donovan, Brendan", "Gorman, Nolan", "Arenado, Nolan", "Goldschmidt, Paul",
            "Walker, Jordan", "Contreras, Willson", "Carlson, Dylan", "Burleson, Alec",
            "Nootbaar, Lars"],
    "CHC": ["Hoerner, Nico", "Suzuki, Seiya", "Happ, Ian", "Bellinger, Cody",
            "Busch, Michael", "Morel, Christopher", "Swanson, Dansby", "Amaya, Miguel",
            "Tauchman, Mike"],
    "BAL": ["Mullins, Cedric", "Henderson, Gunnar", "Santander, Anthony",
            "Mountcastle, Ryan", "Rutschman, Adley", "Westburg, Jordan",
            "O'Hearn, Ryan", "Hays, Austin", "Cowser, Colton"],
    "TOR": ["Springer, George", "Guerrero Jr., Vladimir", "Bichette, Bo",
            "Varsho, Daulton", "Kirk, Alejandro", "Biggio, Cavan",
            "Jansen, Danny", "Barger, Addison", "Kiermaier, Kevin"],
    "BOS": ["Duran, Jarren", "Devers, Rafael", "Yoshida, Masataka", "Casas, Triston",
            "Turner, Justin", "Refsnyder, Rob", "Hamilton, David",
            "McGuire, Reese", "Dalbec, Bobby"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_savant_csv(path: str) -> pd.DataFrame:
    """
    Load and process a Baseball Savant CSV export.

    Key steps
    ---------
    - Strip BOM with encoding='utf-8-sig' (prevents column-shift bug)
    - Derive K_pct / BB_pct from percent-format Savant columns
    - Compute HR_pct / HBP_pct as count / PA
    - Derive hit-type splits from single/double/triple counts
    - Attach xStats-based projection columns for small-sample players
    - Tag each player with their MLB team (via _PLAYER_TEAM mapping)
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()

    df["Name"] = df["last_name, first_name"].astype(str).str.strip()

    numeric_cols = ["pa", "ab", "single", "double", "triple", "home_run",
                    "k_percent", "bb_percent", "babip", "b_hit_by_pitch",
                    "batting_avg", "on_base_percent", "slg_percent",
                    "xba", "xslg", "xobp", "xwoba"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["PA"] = df["pa"].clip(lower=1)

    # Savant exports k_percent / bb_percent as percentages (21.5, not 0.215)
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

    # xStats columns for projection fallback (small sample / injured players)
    # Derive proxy K%/BB%/HR% from xba/xslg/xobp when available
    if "xba" in df.columns and "xobp" in df.columns and "xslg" in df.columns:
        xba  = df["xba"].clip(lower=0.10, upper=0.40)
        xobp = df["xobp"].clip(lower=0.15, upper=0.55)
        xslg = df["xslg"].clip(lower=0.20, upper=0.80)
        xiso = (xslg - xba).clip(lower=0.0)

        df["xK_pct"]  = df["K_pct"]   # no direct xK from Savant; keep actual
        df["xBB_pct"] = (xobp - xba).clip(lower=0.02, upper=0.25)
        df["xHR_pct"] = (xiso * 0.35).clip(upper=0.12)
        df["xBABIP"]  = xba / ((1.0 - df["K_pct"]).clip(lower=0.40))
        df["xBABIP"]  = df["xBABIP"].clip(lower=0.18, upper=0.42)
    else:
        df["xK_pct"]  = df["K_pct"]
        df["xBB_pct"] = df["BB_pct"]
        df["xHR_pct"] = df["HR_pct"]
        df["xBABIP"]  = df["BABIP"]

    # Assign teams via lookup; unrecognised players → "FA" (free agent / unknown)
    df["Team"] = df["Name"].map(_PLAYER_TEAM).fillna("FA")

    return (df.sort_values("PA", ascending=False)
              .drop_duplicates("Name")
              .reset_index(drop=True))


_SAMPLE_DATA = pd.DataFrame({
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


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STAT DERIVATION (BASIC / INTERMEDIATE CUSTOM PLAYER INPUT)
# ═══════════════════════════════════════════════════════════════════════════════

def _derive_hit_split(iso: float) -> tuple[float, float, float]:
    """Estimate 1B/2B/3B rate shares from ISO (SLG − BA)."""
    xbh = min(iso * 2.0, 0.45)
    r1  = max(0.50, 1.0 - xbh)
    r2  = xbh * 0.85
    r3  = xbh * 0.15
    tot = r1 + r2 + r3
    return r1 / tot, r2 / tot, r3 / tot


def derive_from_basic(ba: float, obp: float, slg: float) -> dict[str, Any]:
    bb_pct = max(0.02, min(obp - ba, 0.25))
    iso    = max(0.0, slg - ba)
    k_pct  = 0.220
    hr_pct = max(0.0, min(iso * 0.35, 0.12))
    denom  = max(0.01, 1.0 - k_pct - hr_pct)
    babip  = float(np.clip((ba - hr_pct) / denom, 0.18, 0.42))
    r1, r2, r3 = _derive_hit_split(iso)
    return dict(K_pct=k_pct, BB_pct=bb_pct, HBP_pct=0.008,
                HR_pct=hr_pct, BABIP=babip, r1=r1, r2=r2, r3=r3)


def derive_from_intermediate(ba: float, obp: float, slg: float,
                              k_pct: float, hr_pct: float) -> dict[str, Any]:
    bb_pct = max(0.02, min(obp - ba, 0.25))
    iso    = max(0.0, slg - ba)
    denom  = max(0.01, 1.0 - k_pct - hr_pct)
    babip  = float(np.clip((ba - hr_pct) / denom, 0.18, 0.42))
    r1, r2, r3 = _derive_hit_split(iso)
    return dict(K_pct=k_pct, BB_pct=bb_pct, HBP_pct=0.008,
                HR_pct=hr_pct, BABIP=babip, r1=r1, r2=r2, r3=r3)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PROBABILITY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def get_batter_probs(K: float, BB: float, HBP: float, HR: float,
                     babip: float, r1: float, r2: float, r3: float,
                     era_plus: int, lhp_mode: bool = False) -> np.ndarray:
    """
    Convert per-batter stats into a 6-outcome probability vector.

    ERA+ adjustment (linear around ERA+100)
    ----------------------------------------
    Better pitcher (ERA+ > 100): more Ks, fewer walks / HRs, lower BABIP
    Worse  pitcher (ERA+ < 100): fewer Ks, more walks / HRs, higher BABIP

    Platoon adjustment (lhp_mode = True)
    --------------------------------------
    Generic right-handed-batter-vs-LHP approximation:
      K%  × 1.08,  BB% × 0.95,  HR% × 0.90,  BABIP × 0.97

    Outcomes: 0=out  1=walk/HBP  2=1B  3=2B  4=3B  5=HR
    """
    K     = float(np.clip(K,     0.0, 0.60))
    BB    = float(np.clip(BB,    0.0, 0.30))
    HBP   = float(np.clip(HBP,  0.0, 0.05))
    HR    = float(np.clip(HR,    0.0, 0.15))
    babip = float(np.clip(babip, 0.10, 0.45))

    if lhp_mode:
        K     *= 1.08
        BB    *= 0.95
        HR    *= 0.90
        babip *= 0.97

    q     = era_plus / 100.0
    pK    = float(np.clip(K   * (1.0 + 0.5  * (q - 1.0)), 0.0, 0.55))
    pBB   = max(0.0, BB  * (1.0 - 0.4  * (q - 1.0)))
    pHBP  = max(0.0, HBP * (1.0 - 0.4  * (q - 1.0)))
    pHR   = max(0.0, HR  * (1.0 - 0.4  * (q - 1.0)))
    adj_b = float(np.clip(babip * (1.0 - 0.25 * (q - 1.0)), 0.15, 0.40))

    total_non = pK + pBB + pHBP + pHR
    if total_non > 0.65:
        s = 0.65 / total_non
        pK *= s; pBB *= s; pHBP *= s; pHR *= s

    p_c   = max(0.0, 1.0 - pK - pBB - pHBP - pHR)
    hits  = adj_b * p_c
    probs = np.array([p_c - hits + pK, pBB + pHBP,
                      hits * r1, hits * r2, hits * r3, pHR], dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s     = probs.sum()
    return probs / s if s > 0 else np.array([0.68, 0.09, 0.13, 0.05, 0.01, 0.04])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MONTE CARLO SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_games(lineup_probs: np.ndarray,
                   n_games: int,
                   lhp_lineup_probs: np.ndarray | None = None) -> float:
    """
    9-inning Monte Carlo baseball simulation.

    Parameters
    ----------
    lineup_probs      : (n_batters, 6) probability matrix for RHP plate appearances
    n_games           : number of games to simulate
    lhp_lineup_probs  : optional (n_batters, 6) matrix for LHP PAs (platoon mode);
                        if provided, each PA is randomly assigned RHP (~62%) or LHP (~38%)

    Key design decisions
    --------------------
    - bases resets to 0 at the start of every inning (stranded runners don't carry over)
    - batter index is continuous across innings (lineup cycles through the full game)
    - all random numbers are pre-generated with a single vectorised NumPy call;
      the inner hot loop contains only Python integer arithmetic and list indexing

    Outcome codes
    -------------
    0 = out (K or field out)   1 = walk / HBP
    2 = single                 3 = double
    4 = triple                 5 = home run
    """
    rng        = np.random.default_rng()
    n_batters  = lineup_probs.shape[0]
    cumprobs_r = np.cumsum(lineup_probs, axis=1)
    use_platoon = lhp_lineup_probs is not None
    if use_platoon:
        cumprobs_l = np.cumsum(lhp_lineup_probs, axis=1)

    MAX_PA   = 270  # 9 innings × 30 PA budget — physically impossible to exceed
    total_pa = n_games * MAX_PA

    batter_seq = np.arange(total_pa) % n_batters
    rand_pa    = rng.random(total_pa)
    platoon_pa = rng.random(total_pa) if use_platoon else None

    # Vectorised outcome computation (no NumPy inside hot loop)
    if use_platoon:
        is_lhp = (platoon_pa < _LHP_RATE)
        cp_all = np.where(
            is_lhp[:, None],
            cumprobs_l[batter_seq],
            cumprobs_r[batter_seq],
        )
    else:
        cp_all = cumprobs_r[batter_seq]

    outcomes = (cp_all < rand_pa[:, None]).sum(axis=1)
    outcomes = np.clip(outcomes, 0, 5).astype(np.int32).tolist()

    total_runs = 0

    for g in range(n_games):
        base      = g * MAX_PA
        pa_cursor = 0
        game_runs = 0

        for _ in range(9):
            bases = 0  # bitmask: bit0=1B, bit1=2B, bit2=3B
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

                if res == 5:                          # HR: all score
                    game_runs += on1 + on2 + on3 + 1
                    bases = 0
                elif res == 4:                        # 3B: all runners score
                    game_runs += on1 + on2 + on3
                    bases = 4
                elif res == 3:                        # 2B: 2nd+3rd score, 1st→3rd
                    game_runs += on2 + on3
                    bases = (on1 << 2) | 2
                elif res == 2:                        # 1B: 3rd scores, runners +1 base
                    game_runs += on3
                    bases = 1 | (on1 << 1) | (on2 << 2)
                else:                                 # BB/HBP: force-advance only
                    if on1 and on2 and on3:           # loaded → run scores
                        game_runs += 1
                    elif on1 and on2:                 # fill bases
                        bases = 7
                    elif on1:                         # batter→1st, runner→2nd
                        bases = 3
                    else:                             # batter takes 1st only
                        bases |= 1

        total_runs += game_runs

    return total_runs / n_games


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# Stat ranges to sweep (label → (DataFrame column, low, high))
_SENS_RANGES: dict[str, tuple[str, float, float]] = {
    "K%":      ("K_pct",   0.08, 0.38),
    "BB%":     ("BB_pct",  0.03, 0.18),
    "HR%":     ("HR_pct",  0.005, 0.09),
    "BABIP":   ("BABIP",   0.220, 0.370),
    "2B rate": ("2B_rate", 0.08,  0.30),
}


def _base_stats(row: pd.Series) -> dict[str, float]:
    return {
        "K_pct":   _sg(row, "K_pct",   0.22),
        "BB_pct":  _sg(row, "BB_pct",  0.08),
        "HBP_pct": _sg(row, "HBP_pct", 0.01),
        "HR_pct":  _sg(row, "HR_pct",  0.03),
        "BABIP":   _sg(row, "BABIP",   0.290),
        "1B_rate": _sg(row, "1B_rate", 0.70),
        "2B_rate": _sg(row, "2B_rate", 0.20),
        "3B_rate": _sg(row, "3B_rate", 0.05),
    }


def _rpg_override(stats: dict, era_plus: int, n_games: int, **kw) -> float:
    s   = {**stats, **kw}
    r1, r2, r3 = s["1B_rate"], s["2B_rate"], s["3B_rate"]
    tot = max(r1 + r2 + r3, 1e-9)
    p   = get_batter_probs(s["K_pct"], s["BB_pct"], s["HBP_pct"], s["HR_pct"],
                           s["BABIP"], r1 / tot, r2 / tot, r3 / tot, era_plus)
    return simulate_games(np.array([p] * 9, dtype=np.float64), n_games)


def compute_sensitivity(row: pd.Series, era_plus: int, progress_cb=None):
    """
    Returns (base_rpg, tornado, deltas, obp_curve, slg_curve, detail).

    progress_cb : optional callable(fraction: float) for progress bar updates
    """
    base = _base_stats(row)

    # Total computation steps for progress tracking
    total_steps = (
        1                              # base RPG
        + 2 * len(_SENS_RANGES)        # tornado lo / hi per stat
        + _N_SENS_POINTS * 2           # OBP + SLG curves
        + _N_SENS_POINTS * len(_SENS_RANGES)  # detail curves
    )
    step = [0]

    def _tick():
        step[0] += 1
        if progress_cb:
            progress_cb(step[0] / total_steps)

    base_rpg = _rpg_override(base, era_plus, _N_GAMES_SENS);  _tick()

    tornado: dict[str, tuple] = {}
    for label, (col, lo, hi) in _SENS_RANGES.items():
        lo_c = min(lo, base[col])
        hi_c = max(hi, base[col])
        rpg_lo = _rpg_override(base, era_plus, _N_GAMES_SENS, **{col: lo_c}); _tick()
        rpg_hi = _rpg_override(base, era_plus, _N_GAMES_SENS, **{col: hi_c}); _tick()
        tornado[label] = (rpg_lo, base_rpg, rpg_hi, lo_c, base[col], hi_c)

    deltas    = np.linspace(0.0, 0.08, _N_SENS_POINTS)
    obp_curve = []
    for d in deltas:
        obp_curve.append(_rpg_override(base, era_plus, _N_GAMES_SENS,
                                       BB_pct=min(base["BB_pct"] + d, 0.30)));  _tick()
    slg_curve = []
    for d in deltas:
        slg_curve.append(_rpg_override(base, era_plus, _N_GAMES_SENS,
                                       HR_pct=min(base["HR_pct"] + d / 3.0, 0.15))); _tick()

    detail: dict[str, tuple] = {}
    for label, (col, lo, hi) in _SENS_RANGES.items():
        lo_c = min(lo, base[col])
        hi_c = max(hi, base[col])
        xs   = np.linspace(lo_c, hi_c, _N_SENS_POINTS)
        rpgs = []
        for x in xs:
            rpgs.append(_rpg_override(base, era_plus, _N_GAMES_SENS, **{col: float(x)}));  _tick()
        detail[label] = (xs.tolist(), rpgs)

    return base_rpg, tornado, deltas.tolist(), obp_curve, slg_curve, detail


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter, Arial, sans-serif"),
    margin=dict(l=80, r=20, t=28, b=44),
)


def _tornado_chart(tornado: dict, base_rpg: float) -> go.Figure:
    items   = sorted(tornado.items(), key=lambda kv: abs(kv[1][2] - kv[1][0]))
    labels  = [k for k, _ in items]
    lo_rpgs = [v[0] for _, v in items]
    hi_rpgs = [v[2] for _, v in items]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[r - base_rpg for r in lo_rpgs], base=[base_rpg] * len(labels),
        orientation="h", name="Low",
        marker_color=_PALETTE[0],
        hovertemplate="%{base:.2f} RPG (low end)<extra>%{y}</extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[r - base_rpg for r in hi_rpgs], base=[base_rpg] * len(labels),
        orientation="h", name="High",
        marker_color=_PALETTE[1],
        hovertemplate="%{base:.2f} RPG (high end)<extra>%{y}</extra>",
    ))
    fig.add_vline(x=base_rpg, line_dash="dot", line_color="rgba(255,255,255,0.6)",
                  line_width=1.5)
    fig.update_layout(**_CHART_LAYOUT, barmode="overlay",
                      xaxis_title=t("rpg"), yaxis_title="", height=340,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _obp_slg_chart(deltas: list, obp_curve: list, slg_curve: list) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=deltas, y=obp_curve, mode="lines+markers", name=t("obp_line"),
        line=dict(color=_PALETTE[0], width=2),
        hovertemplate="Δ=+%{x:.3f} → %{y:.2f} RPG<extra>OBP</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=deltas, y=slg_curve, mode="lines+markers", name=t("slg_line"),
        line=dict(color=_PALETTE[1], width=2, dash="dash"),
        hovertemplate="Δ=+%{x:.3f} → %{y:.2f} RPG<extra>SLG</extra>",
    ))
    fig.update_layout(**_CHART_LAYOUT, xaxis_title=t("delta_label"),
                      yaxis_title=t("rpg"), height=320,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _detail_chart(detail: dict) -> go.Figure:
    fig = go.Figure()
    for i, (label, (xs, rpgs)) in enumerate(detail.items()):
        fig.add_trace(go.Scatter(
            x=xs, y=rpgs, mode="lines+markers", name=label,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            hovertemplate=f"{label}=%{{x:.3f}} → %{{y:.2f}} RPG<extra></extra>",
        ))
    fig.update_layout(**_CHART_LAYOUT, xaxis_title="Stat value",
                      yaxis_title=t("rpg"), height=360,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _sg(row: pd.Series, col: str, default: float) -> float:
    """Safe-get: returns `default` if value is missing or NaN."""
    val = row.get(col)
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    return float(val)


def _use_projection(row: pd.Series) -> bool:
    """Return True when actual PA is too small to trust raw stats."""
    return float(_sg(row, "PA", 0)) < PROJECTION_PA_THRESHOLD


def _stat_cols(row: pd.Series) -> tuple[float, ...]:
    """Return (K, BB, HBP, HR, babip, r1, r2, r3) using projection when needed."""
    proj = _use_projection(row)
    k    = _sg(row, "xK_pct"  if proj else "K_pct",   0.22)
    bb   = _sg(row, "xBB_pct" if proj else "BB_pct",  0.08)
    hbp  = _sg(row, "HBP_pct",                          0.01)
    hr   = _sg(row, "xHR_pct" if proj else "HR_pct",  0.03)
    bab  = _sg(row, "xBABIP"  if proj else "BABIP",    0.290)
    r1   = _sg(row, "1B_rate", 0.70)
    r2   = _sg(row, "2B_rate", 0.20)
    r3   = _sg(row, "3B_rate", 0.05)
    return k, bb, hbp, hr, bab, r1, r2, r3


def build_probs(row: pd.Series, era_plus: int,
                lhp_mode: bool = False) -> np.ndarray:
    k, bb, hbp, hr, bab, r1, r2, r3 = _stat_cols(row)
    return get_batter_probs(k, bb, hbp, hr, bab, r1, r2, r3, era_plus, lhp_mode)


def prob_label(row: pd.Series, era_plus: int) -> str:
    p    = build_probs(row, era_plus)
    badge = f"🔮 {t('data_badge_proj')}" if _use_projection(row) else f"✅ {t('data_badge_act')}"
    return (f"{badge}  ·  out {p[0]*100:.1f}% · walk/HBP {p[1]*100:.1f}% · "
            f"1B {p[2]*100:.1f}% · 2B {p[3]*100:.1f}% · "
            f"3B {p[4]*100:.1f}% · HR {p[5]*100:.1f}%")


def _lineup_to_json(lineup: list[str]) -> str:
    return json.dumps({"version": "2026", "lineup": lineup}, ensure_ascii=False, indent=2)


def _json_to_lineup(raw: bytes) -> list[str] | None:
    try:
        data = json.loads(raw.decode("utf-8"))
        lineup = data.get("lineup", [])
        if isinstance(lineup, list) and len(lineup) == 9:
            return lineup
    except Exception:
        pass
    return None


def _apply_lineup(names: list[str], all_players: list[str]) -> None:
    """Write a 9-player list into session state slot keys."""
    for i, name in enumerate(names):
        if name in all_players:
            st.session_state[f"slot{i}"] = name


# ═══════════════════════════════════════════════════════════════════════════════
# 11. APP LAYOUT — PAGE CONFIG & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Language selector (must come before any t() call in UI) ──────────────────
lang_names = list(LANGUAGES.keys())
lang_idx   = lang_names.index(
    next(k for k, v in LANGUAGES.items() if v == st.session_state.get("lang", "en"))
)
chosen_lang = st.sidebar.selectbox(
    t("lang_label"), lang_names, index=lang_idx, key="lang_selector"
)
st.session_state["lang"] = LANGUAGES[chosen_lang]

# ── Page header ──────────────────────────────────────────────────────────────
st.title(f"⚾ {APP_TITLE}")
st.caption(t("app_subtitle"))

# ── Data loading ──────────────────────────────────────────────────────────────
if "players_df" not in st.session_state:
    try:
        st.session_state.players_df = load_savant_csv(CSV_PATH)
    except FileNotFoundError:
        st.info(t("no_csv"))
        st.session_state.players_df = _SAMPLE_DATA.copy()

_df_all: pd.DataFrame = st.session_state.players_df

# ── Sidebar: Simulation controls ─────────────────────────────────────────────
st.sidebar.markdown(f"### {t('sec_sim')}")

era_plus = st.sidebar.slider(
    t("era_plus"), 70, 150, 100,
    help=t("era_plus_help"),
)
min_pa = st.sidebar.slider(t("min_pa"), 0, 700, 100)
ultra_fast   = st.sidebar.checkbox(t("ultra_fast"), value=True)
use_platoon  = st.sidebar.checkbox(t("use_platoon"), value=False,
                                    help=t("platoon_help"))

n_mixed = _N_GAMES_DEFAULT_MIXED if ultra_fast else _N_GAMES_DEFAULT_FULL
n_same  = _N_GAMES_DEFAULT_SAME  if ultra_fast else _N_GAMES_DEFAULT_FULL // 2

# ── Sidebar: Team Mode ───────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t('sec_team')}")

team_mode = st.sidebar.checkbox(t("team_mode"), value=False)
selected_team_abbr: str | None = None

if team_mode:
    team_names   = ["—"] + list(TEAMS.keys())
    chosen_team  = st.sidebar.selectbox(t("select_team"), team_names, key="chosen_team")
    if chosen_team != "—":
        selected_team_abbr = TEAMS[chosen_team]

# ── Sidebar: Load example lineup ─────────────────────────────────────────────
lineup_options = {"—": None}
for full_name, abbr in TEAMS.items():
    if abbr in _DEFAULT_LINEUPS:
        lineup_options[full_name] = abbr

example_choice = st.sidebar.selectbox(
    t("load_lineup"), list(lineup_options.keys()), key="example_lineup_sel"
)

if st.sidebar.button(t("btn_load"), key="btn_load_lineup"):
    abbr = lineup_options.get(example_choice)
    if abbr and abbr in _DEFAULT_LINEUPS:
        all_names = sorted(_df_all[_df_all["PA"] >= min_pa]["Name"].unique())
        names     = _DEFAULT_LINEUPS[abbr]
        missing   = [n for n in names if n not in all_names]
        _apply_lineup(names, all_names)
        if missing:
            st.sidebar.warning(f"{t('team_not_found')}\n\nMissing: {', '.join(missing)}")
        else:
            st.sidebar.success(t("lineup_loaded"))
        st.rerun()

if st.sidebar.button(t("btn_reset"), key="btn_reset_lineup"):
    for i in range(9):
        if f"slot{i}" in st.session_state:
            del st.session_state[f"slot{i}"]
    st.sidebar.info(t("lineup_reset"))
    st.rerun()

# ── Sidebar: Data source note ─────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption(f"📂 {DATA_SOURCE}")
st.sidebar.caption(f"v{APP_VERSION}")

# ── Sidebar: Add custom player ────────────────────────────────────────────────
with st.sidebar.expander(t("sec_player")):
    new_name  = st.text_input(t("player_name"), "Custom Player", key="new_player_name")
    level_map = {
        t("level_basic"): "basic",
        t("level_mid"):   "mid",
        t("level_adv"):   "adv",
    }
    level_key = st.radio(t("stat_level"), list(level_map.keys()),
                         horizontal=False, key="custom_level")
    level     = level_map[level_key]

    if level == "basic":
        ba  = st.slider("BA",  0.150, 0.400, 0.260, 0.001, key="c_ba")
        obp = st.slider("OBP", 0.200, 0.500, 0.330, 0.001, key="c_obp")
        slg = st.slider("SLG", 0.250, 0.700, 0.420, 0.001, key="c_slg")
        derived = derive_from_basic(ba, obp, slg)

    elif level == "mid":
        ba     = st.slider("BA",  0.150, 0.400, 0.260, 0.001, key="c_ba2")
        obp    = st.slider("OBP", 0.200, 0.500, 0.330, 0.001, key="c_obp2")
        slg    = st.slider("SLG", 0.250, 0.700, 0.420, 0.001, key="c_slg2")
        k_pct  = st.slider("K%",  0.05, 0.50, 0.22, 0.001, key="c_k2")
        hr_pct = st.slider("HR%", 0.00, 0.12, 0.03, 0.001, key="c_hr2")
        derived = derive_from_intermediate(ba, obp, slg, k_pct, hr_pct)

    else:
        derived = {
            "K_pct":   st.slider("K%",      0.00, 0.50, 0.22, 0.001, key="c_k3"),
            "BB_pct":  st.slider("BB%",     0.00, 0.30, 0.08, 0.001, key="c_bb3"),
            "HR_pct":  st.slider("HR%",     0.00, 0.15, 0.03, 0.001, key="c_hr3"),
            "BABIP":   st.slider("BABIP", 0.200, 0.400, 0.300, 0.001, key="c_bab3"),
            "HBP_pct": st.slider("HBP%",   0.00, 0.05, 0.01, 0.001, key="c_hbp3"),
            "r1":      st.slider("1B rate", 0.50, 0.90, 0.70, 0.01,  key="c_r1"),
            "r2":      st.slider("2B rate", 0.05, 0.35, 0.20, 0.01,  key="c_r2"),
            "r3":      st.slider("3B rate", 0.00, 0.15, 0.05, 0.01,  key="c_r3"),
        }

    if st.button(t("btn_add"), key="btn_add_player"):
        r1, r2, r3 = derived.get("r1", 0.70), derived.get("r2", 0.20), derived.get("r3", 0.05)
        tot = r1 + r2 + r3
        new_row = pd.DataFrame([{
            "Name":    new_name,              "PA":      600,
            "K_pct":   derived["K_pct"],      "BB_pct":  derived["BB_pct"],
            "HR_pct":  derived["HR_pct"],     "HBP_pct": derived.get("HBP_pct", 0.008),
            "BABIP":   derived["BABIP"],      "Team":    "FA",
            "1B_rate": r1 / tot,
            "2B_rate": r2 / tot,
            "3B_rate": r3 / tot,
            "xK_pct":  derived["K_pct"],      "xBB_pct": derived["BB_pct"],
            "xHR_pct": derived["HR_pct"],     "xBABIP":  derived["BABIP"],
        }])
        st.session_state.players_df = pd.concat(
            [st.session_state.players_df, new_row], ignore_index=True
        )
        st.success(f"{t('added')}: {new_name}")
        st.rerun()

# ── Sidebar: Save / Load lineup JSON ──────────────────────────────────────────
with st.sidebar.expander(t("sec_lineup_io")):
    # Build current slot names for export
    _all_for_export = sorted(
        _df_all[_df_all["PA"] >= min_pa]["Name"].unique().tolist()
    )
    _current_lineup = [
        st.session_state.get(f"slot{i}", _all_for_export[0] if _all_for_export else "")
        for i in range(9)
    ]

    st.download_button(
        label=t("btn_save_json"),
        data=_lineup_to_json(_current_lineup).encode("utf-8"),
        file_name="lineup_2026.json",
        mime="application/json",
        key="dl_lineup",
    )

    uploaded = st.file_uploader(t("upload_json"), type="json", key="upload_lineup")
    if uploaded is not None:
        raw    = uploaded.read()
        parsed = _json_to_lineup(raw)
        if parsed:
            _apply_lineup(parsed, _all_for_export)
            st.success(t("upload_ok"))
            st.rerun()
        else:
            st.error(t("upload_err"))


# ═══════════════════════════════════════════════════════════════════════════════
# 12. APP LAYOUT — MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Player pool resolution ────────────────────────────────────────────────────
df_pool = _df_all[_df_all["PA"] >= min_pa].copy()

if team_mode and selected_team_abbr:
    df_pool = df_pool[df_pool["Team"] == selected_team_abbr]

all_players = sorted(df_pool["Name"].unique().tolist())

if not all_players:
    st.warning("No players match current filters. Adjust min PA or team selection.")
    st.stop()

# ── LHP probability matrices (pre-build once if platoon is on) ────────────────
def _get_lhp_probs(row: pd.Series) -> np.ndarray:
    k, bb, hbp, hr, bab, r1, r2, r3 = _stat_cols(row)
    return get_batter_probs(k, bb, hbp, hr, bab, r1, r2, r3, era_plus, lhp_mode=True)

# ── Tab definitions ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([t("tab_lineup"), t("tab_same"), t("tab_sensitivity")])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: MIXED LINEUP
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f"#### {APP_TITLE} · {t('tab_lineup')}")

    if team_mode and selected_team_abbr:
        st.info(f"🏟 Team mode: **{next(k for k,v in TEAMS.items() if v==selected_team_abbr)}** "
                f"— {len(all_players)} players available")

    # Build 9 selectboxes; in team mode each player can only appear once
    lineup_names: list[str] = []
    cols_per_row = st.columns([1, 1, 1])

    for i in range(9):
        # Remove already-chosen players from the options for this slot
        already_chosen = set(lineup_names)
        available      = [p for p in all_players if p not in already_chosen]

        # Preserve previous selection if still valid
        prev = st.session_state.get(f"slot{i}")
        default_idx = available.index(prev) if prev in available else 0

        with cols_per_row[i % 3]:
            chosen = st.selectbox(
                f"{t('slot')} {i + 1}",
                available,
                index=default_idx,
                key=f"slot{i}",
            )
        lineup_names.append(chosen)

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        run_sim = st.button(t("btn_simulate"), type="primary", key="btn_sim_lineup")

    if run_sim:
        probs_list:     list[np.ndarray] = []
        lhp_probs_list: list[np.ndarray] = []
        debug_lines:    list[str]        = []

        for name in lineup_names:
            row = df_pool[df_pool["Name"] == name].iloc[0]
            probs_list.append(build_probs(row, era_plus))
            lhp_probs_list.append(_get_lhp_probs(row))
            debug_lines.append(f"**{name}** — {prob_label(row, era_plus)}")

        lineup_arr = np.array(probs_list,     dtype=np.float64)
        lhp_arr    = np.array(lhp_probs_list, dtype=np.float64) if use_platoon else None

        with st.spinner(t("spinner")):
            start   = time.time()
            rpg     = simulate_games(lineup_arr, n_mixed, lhp_arr)
            elapsed = time.time() - start

        c1, c2, c3 = st.columns(3)
        c1.metric(t("rpg"),   f"{rpg:.2f}")
        c2.metric(t("rps"),   f"{int(rpg * 162)}")
        c3.metric(t("elapsed"), f"{elapsed:.2f}s")

        if use_platoon:
            st.caption("⚔ Platoon mode active — 38% of PAs vs LHP (generic RHH adjustment)")

        with st.expander(t("prob_header")):
            for line in debug_lines:
                st.markdown(line)


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: 9× SAME BATTER
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f"#### {APP_TITLE} · {t('tab_same')}")

    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox(t("player_a"), all_players, key="player_a")
    with c2:
        b = st.selectbox(t("player_b"), all_players, key="player_b")

    if st.button(t("btn_compare"), type="primary", key="btn_compare"):
        row_a = df_pool[df_pool["Name"] == a].iloc[0]
        row_b = df_pool[df_pool["Name"] == b].iloc[0]

        pa   = build_probs(row_a, era_plus)
        pb   = build_probs(row_b, era_plus)
        la   = np.array([pa] * 9, dtype=np.float64)
        lb   = np.array([pb] * 9, dtype=np.float64)

        lhp_a = np.array([_get_lhp_probs(row_a)] * 9, dtype=np.float64) if use_platoon else None
        lhp_b = np.array([_get_lhp_probs(row_b)] * 9, dtype=np.float64) if use_platoon else None

        with st.spinner(t("spinner")):
            start   = time.time()
            r_a     = simulate_games(la, n_same, lhp_a)
            r_b     = simulate_games(lb, n_same, lhp_b)
            elapsed = time.time() - start

        delta = r_a - r_b
        m1, m2, m3 = st.columns(3)
        m1.metric(f"9× {a}", f"{r_a:.2f}", f"{delta:+.2f}")
        m2.metric(f"9× {b}", f"{r_b:.2f}")
        m3.metric(t("elapsed"), f"{elapsed:.2f}s")

        with st.expander(t("prob_header")):
            st.markdown(f"**{a}** — {prob_label(row_a, era_plus)}")
            st.markdown(f"**{b}** — {prob_label(row_b, era_plus)}")

            if _use_projection(row_a):
                st.info(f"**{a}**: {t('proj_note')}")
            if _use_projection(row_b):
                st.info(f"**{b}**: {t('proj_note')}")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: SENSITIVITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"#### {APP_TITLE} · {t('tab_sensitivity')}")
    st.caption(t("sens_note"))

    sens_player = st.selectbox(t("sens_player"), all_players, key="sens_player")

    if st.button(t("btn_run_sens"), type="primary", key="btn_sens"):
        sens_row = df_pool[df_pool["Name"] == sens_player].iloc[0]

        prog_bar   = st.progress(0.0, text=t("computing"))
        prog_label = st.empty()

        def _progress(frac: float):
            prog_bar.progress(min(frac, 1.0), text=f"{t('computing')} {frac*100:.0f}%")

        start = time.time()
        base_rpg, tornado, deltas, obp_curve, slg_curve, detail = \
            compute_sensitivity(sens_row, era_plus, progress_cb=_progress)
        elapsed = time.time() - start

        prog_bar.empty()
        prog_label.empty()

        c1, c2 = st.columns(2)
        c1.metric(t("base_rpg"), f"{base_rpg:.2f}")
        c2.metric(t("elapsed"),  f"{elapsed:.1f}s")

        if _use_projection(sens_row):
            st.info(t("proj_note"))

        st.subheader(t("tornado_title"))
        st.plotly_chart(_tornado_chart(tornado, base_rpg),
                        use_container_width=True, key="tornado")

        st.subheader(t("obp_slg_title"))
        st.plotly_chart(_obp_slg_chart(deltas, obp_curve, slg_curve),
                        use_container_width=True, key="obp_slg")

        with st.expander(t("detail_curves")):
            st.plotly_chart(_detail_chart(detail),
                            use_container_width=True, key="detail")
