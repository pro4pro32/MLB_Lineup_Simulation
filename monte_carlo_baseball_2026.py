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

# Column alias map: b_* prefix (Savant URL export) → standard names (Savant UI export)
_SAVANT_ALIASES: dict[str, str] = {
    "b_pa": "pa", "b_ab": "ab", "b_hit": "hit",
    "b_single": "single", "b_double": "double", "b_triple": "triple",
    "b_home_run": "home_run", "b_strikeout": "strikeout", "b_walk": "walk",
    "b_k_percent": "k_percent", "b_bb_percent": "bb_percent",
    "b_rbi": "rbi",
}



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

# Player-name → team abbreviation mapping (exact names from Baseball Savant CSV)
_PLAYER_TEAM: dict[str, str] = {
    # New York Yankees
    "Judge, Aaron": "NYY", "Wells, Austin": "NYY", "Volpe, Anthony": "NYY",
    "Trevino, Jose": "NYY", "Stanton, Giancarlo": "NYY", "Verdugo, Alex": "NYY",
    "Chisholm Jr., Jazz": "NYY", "Kepler, Max": "NYY", "Arenado, Nolan": "NYY",
    "Grisham, Trent": "NYY", "Rice, Ben": "NYY", "Domínguez, Jasson": "NYY",
    "Higashioka, Kyle": "NYY", "Kiner-Falefa, Isiah": "NYY", "Sweeney, Trey": "NYY",
    "Goldschmidt, Paul": "NYY",
    # New York Mets
    "Soto, Juan": "NYM", "Lindor, Francisco": "NYM", "Alonso, Pete": "NYM",
    "Nimmo, Brandon": "NYM", "Vientos, Mark": "NYM", "McNeil, Jeff": "NYM",
    "Alvarez, Francisco": "NYM", "Marte, Starling": "NYM", "Iglesias, Jose": "NYM",
    "Baty, Brett": "NYM", "Vargas, Miguel": "NYM",
    # Los Angeles Dodgers
    "Ohtani, Shohei": "LAD", "Betts, Mookie": "LAD", "Freeman, Freddie": "LAD",
    "Hernández, Teoscar": "LAD", "Muncy, Max": "LAD", "Smith, Will": "LAD",
    "Edman, Tommy": "LAD", "Lux, Gavin": "LAD", "Pages, Andy": "LAD",
    "Hernández, Enrique": "LAD",
    # Philadelphia Phillies
    "Harper, Bryce": "PHI", "Turner, Trea": "PHI", "Schwarber, Kyle": "PHI",
    "Castellanos, Nick": "PHI", "Bohm, Alec": "PHI", "Stott, Bryson": "PHI",
    "Realmuto, J.T.": "PHI", "Marsh, Brandon": "PHI", "Rojas, Josh": "PHI",
    "Sosa, Edmundo": "PHI",
    # Atlanta Braves
    "Acuña Jr., Ronald": "ATL", "Riley, Austin": "ATL", "Albies, Ozzie": "ATL",
    "Olson, Matt": "ATL", "Murphy, Sean": "ATL", "Harris II, Michael": "ATL",
    "Ozuna, Marcell": "ATL", "d'Arnaud, Travis": "ATL", "Baldwin, Drake": "ATL",
    "Arcia, Orlando": "ATL",
    # Houston Astros
    "Alvarez, Yordan": "HOU", "Peña, Jeremy": "HOU", "Diaz, Yainer": "HOU",
    "Abreu, Wilyer": "HOU", "Meyers, Jake": "HOU", "Dubón, Mauricio": "HOU",
    "Smith, Cam": "HOU", "Altuve, Jose": "HOU", "Garver, Mitch": "HOU",
    # Texas Rangers
    "Seager, Corey": "TEX", "Lowe, Nathaniel": "TEX", "García, Adolis": "TEX",
    "Langford, Wyatt": "TEX", "Jung, Josh": "TEX", "Heim, Jonah": "TEX",
    "Smith, Josh": "TEX", "Semien, Marcus": "TEX", "Carter, Evan": "TEX",
    # Cleveland Guardians
    "Ramírez, José": "CLE", "Naylor, Josh": "CLE", "Freeman, Tyler": "CLE",
    "Kwan, Steven": "CLE", "Giménez, Andrés": "CLE", "Naylor, Bo": "CLE",
    "Manzardo, Kyle": "CLE", "Martínez, Angel": "CLE", "Arias, Gabriel": "CLE",
    "Rocchio, Brayan": "CLE", "Straw, Myles": "CLE",
    # St. Louis Cardinals
    "Gorman, Nolan": "STL", "Walker, Jordan": "STL", "Donovan, Brendan": "STL",
    "Contreras, Willson": "STL", "Nootbaar, Lars": "STL", "Herrera, Iván": "STL",
    "Scott II, Victor": "STL", "Pagés, Pedro": "STL", "Winn, Masyn": "STL",
    "Saggese, Thomas": "STL",
    # Chicago Cubs
    "Suzuki, Seiya": "CHC", "Bellinger, Cody": "CHC", "Hoerner, Nico": "CHC",
    "Happ, Ian": "CHC", "Busch, Michael": "CHC", "Tucker, Kyle": "CHC",
    "Morel, Christopher": "CHC", "Shaw, Matt": "CHC", "Swanson, Dansby": "CHC",
    "Tauchman, Mike": "CHC", "Paredes, Isaac": "CHC", "Crow-Armstrong, Pete": "CHC",
    "Narváez, Carlos": "CHC",
    # Baltimore Orioles
    "Henderson, Gunnar": "BAL", "Mullins, Cedric": "BAL", "Rutschman, Adley": "BAL",
    "Mountcastle, Ryan": "BAL", "Westburg, Jordan": "BAL", "Cowser, Colton": "BAL",
    "Ortiz, Joey": "BAL", "Urías, Ramón": "BAL", "Hays, Austin": "BAL",
    "Holliday, Jackson": "BAL", "Norby, Connor": "BAL", "Mayo, Coby": "BAL",
    # Toronto Blue Jays
    "Guerrero Jr., Vladimir": "TOR", "Bichette, Bo": "TOR", "Kirk, Alejandro": "TOR",
    "Springer, George": "TOR", "Barger, Addison": "TOR", "Jansen, Danny": "TOR",
    "Horwitz, Spencer": "TOR", "Varsho, Daulton": "TOR", "Lopez, Otto": "TOR",
    "Schneider, Davis": "TOR",
    # Boston Red Sox
    "Devers, Rafael": "BOS", "Bregman, Alex": "BOS", "Duran, Jarren": "BOS",
    "Rafaela, Ceddanne": "BOS", "Anthony, Roman": "BOS", "Story, Trevor": "BOS",
    "Casas, Triston": "BOS", "Yoshida, Masataka": "BOS", "O'Neill, Tyler": "BOS",
    "Campbell, Kristian": "BOS",
    # San Diego Padres
    "Tatis Jr., Fernando": "SD", "Machado, Manny": "SD", "Bogaerts, Xander": "SD",
    "Cronenworth, Jake": "SD", "Merrill, Jackson": "SD", "Arraez, Luis": "SD",
    "Profar, Jurickson": "SD", "Gurriel Jr., Lourdes": "SD",
    # Seattle Mariners
    "Rodríguez, Julio": "SEA", "Raleigh, Cal": "SEA", "Crawford, J.P.": "SEA",
    "Caballero, José": "SEA", "Polanco, Jorge": "SEA", "Moore, Dylan": "SEA",
    "Arozarena, Randy": "SEA", "Canzone, Dominic": "SEA", "Ramos, Heliot": "SEA",
    # Minnesota Twins
    "Buxton, Byron": "MIN", "Correa, Carlos": "MIN", "Larnach, Trevor": "MIN",
    "Jeffers, Ryan": "MIN", "Lewis, Royce": "MIN", "Wallner, Matt": "MIN",
    "Castro, Willi": "MIN", "Lee, Brooks": "MIN", "Julien, Edouard": "MIN",
    # Detroit Tigers
    "Greene, Riley": "DET", "Torres, Gleyber": "DET", "Torkelson, Spencer": "DET",
    "Keith, Colt": "DET", "Carpenter, Kerry": "DET", "Dingler, Dillon": "DET",
    "Pérez, Wenceel": "DET", "Meadows, Parker": "DET", "McKinstry, Zach": "DET",
    # Kansas City Royals
    "Witt Jr., Bobby": "KC", "Perez, Salvador": "KC", "Garcia, Maikel": "KC",
    "Pasquantino, Vinnie": "KC", "Isbel, Kyle": "KC", "Fermin, Freddy": "KC",
    "Massey, Michael": "KC", "Young, Cole": "KC",
    # Tampa Bay Rays
    "Lowe, Brandon": "TB", "Walls, Taylor": "TB", "Aranda, Jonathan": "TB",
    "Simpson, Chandler": "TB", "Sanoja, Javier": "TB", "Díaz, Yandy": "TB",
    "Mead, Curtis": "TB", "Caglianone, Jac": "TB", "Lowe, Josh": "TB",
    # Oakland Athletics
    "Rooker, Brent": "OAK", "Butler, Lawrence": "OAK", "Langeliers, Shea": "OAK",
    "Kurtz, Nick": "OAK", "Wilson, Jacob": "OAK", "Soderstrom, Tyler": "OAK",
    "Bleday, JJ": "OAK",
    # Miami Marlins
    "Edwards, Xavier": "MIA", "Sánchez, Jesús": "MIA", "Soler, Jorge": "MIA",
    "Burger, Jake": "MIA", "Fortes, Nick": "MIA", "Benson, Will": "MIA",
    "Lile, Daylen": "MIA",
    # Colorado Rockies
    "McMahon, Ryan": "COL", "Tovar, Ezequiel": "COL", "Goodman, Hunter": "COL",
    "Doyle, Brenton": "COL", "Beck, Jordan": "COL", "Toglia, Michael": "COL",
    "Jones, Nolan": "COL", "Canario, Alexander": "COL",
    # Arizona Diamondbacks
    "Carroll, Corbin": "ARI", "Thomas, Alek": "ARI", "Walker, Christian": "ARI",
    "Marte, Ketel": "ARI", "Moreno, Gabriel": "ARI", "McCarthy, Jake": "ARI",
    "Perdomo, Geraldo": "ARI", "Bell, Josh": "ARI", "Suárez, Eugenio": "ARI",
    # Cincinnati Reds
    "De La Cruz, Elly": "CIN", "India, Jonathan": "CIN", "Friedl, TJ": "CIN",
    "Stephenson, Tyler": "CIN", "McLain, Matt": "CIN", "Steer, Spencer": "CIN",
    "Marte, Noelvi": "CIN", "France, Ty": "CIN", "Fraley, Jake": "CIN",
    # Pittsburgh Pirates
    "Cruz, Oneil": "PIT", "Reynolds, Bryan": "PIT", "Hayes, Ke'Bryan": "PIT",
    "Gonzales, Nick": "PIT", "Davis, Henry": "PIT", "Triolo, Jared": "PIT",
    "Bart, Joey": "PIT", "Gonzalez, Romy": "PIT",
    # Milwaukee Brewers
    "Yelich, Christian": "MIL", "Chourio, Jackson": "MIL", "Frelick, Sal": "MIL",
    "Hoskins, Rhys": "MIL", "Contreras, William": "MIL", "Turang, Brice": "MIL",
    "Adames, Willy": "MIL", "Collins, Isaac": "MIL",
    # Washington Nationals
    "Abrams, CJ": "WSH", "Young, Jacob": "WSH", "Wood, James": "WSH",
    "Crews, Dylan": "WSH", "García Jr., Luis": "WSH", "Ruiz, Keibert": "WSH",
    "House, Brady": "WSH", "Hassell III, Robert": "WSH",
    # San Francisco Giants
    "Chapman, Matt": "SF", "Yastrzemski, Mike": "SF", "Lee, Jung Hoo": "SF",
    "Bailey, Patrick": "SF", "Flores, Wilmer": "SF", "Conforto, Michael": "SF",
    "Schmitt, Casey": "SF", "Fitzgerald, Tyler": "SF", "Wade Jr., LaMonte": "SF",
    # Los Angeles Angels
    "Trout, Mike": "LAA", "Neto, Zach": "LAA", "Ward, Taylor": "LAA",
    "Rengifo, Luis": "LAA", "O'Hoppe, Logan": "LAA", "Moniak, Mickey": "LAA",
    "Schanuel, Nolan": "LAA", "Adell, Jo": "LAA",
    # Chicago White Sox
    "Robert Jr., Luis": "CWS", "Vaughn, Andrew": "CWS", "Sosa, Lenyn": "CWS",
    "Montgomery, Colson": "CWS", "Quero, Edgar": "CWS", "Sheets, Gavin": "CWS",
    "Benintendi, Andrew": "CWS", "Moncada, Yoán": "CWS", "Teel, Kyle": "CWS",
}

# Full team name ↔ abbreviation
TEAMS: dict[str, str] = {
    "New York Yankees":     "NYY", "New York Mets":          "NYM",
    "Los Angeles Dodgers":  "LAD", "Philadelphia Phillies":  "PHI",
    "Atlanta Braves":       "ATL", "Houston Astros":         "HOU",
    "Texas Rangers":        "TEX", "Cleveland Guardians":    "CLE",
    "St. Louis Cardinals":  "STL", "Chicago Cubs":           "CHC",
    "Baltimore Orioles":    "BAL", "Toronto Blue Jays":      "TOR",
    "Boston Red Sox":       "BOS", "San Diego Padres":       "SD",
    "Seattle Mariners":     "SEA", "Minnesota Twins":        "MIN",
    "Detroit Tigers":       "DET", "Kansas City Royals":     "KC",
    "Tampa Bay Rays":       "TB",  "Oakland Athletics":      "OAK",
    "Miami Marlins":        "MIA", "Colorado Rockies":       "COL",
    "Arizona Diamondbacks": "ARI", "Cincinnati Reds":        "CIN",
    "Pittsburgh Pirates":   "PIT", "Milwaukee Brewers":      "MIL",
    "Washington Nationals": "WSH", "San Francisco Giants":   "SF",
    "Los Angeles Angels":   "LAA", "Chicago White Sox":      "CWS",
}

# Default lineups: team abbrev → ordered list of 9 player names (as in CSV)
_DEFAULT_LINEUPS: dict[str, list[str]] = {
    "NYY": ["Volpe, Anthony", "Judge, Aaron", "Chisholm Jr., Jazz", "Kepler, Max",
            "Arenado, Nolan", "Wells, Austin", "Rice, Ben", "Trevino, Jose",
            "Grisham, Trent"],
    "NYM": ["Nimmo, Brandon", "Lindor, Francisco", "Soto, Juan", "Alonso, Pete",
            "Vientos, Mark", "Marte, Starling", "McNeil, Jeff", "Alvarez, Francisco",
            "Iglesias, Jose"],
    "LAD": ["Betts, Mookie", "Ohtani, Shohei", "Freeman, Freddie", "Hernández, Teoscar",
            "Muncy, Max", "Smith, Will", "Edman, Tommy", "Lux, Gavin", "Pages, Andy"],
    "PHI": ["Schwarber, Kyle", "Turner, Trea", "Harper, Bryce", "Castellanos, Nick",
            "Bohm, Alec", "Marsh, Brandon", "Stott, Bryson", "Realmuto, J.T.",
            "Rojas, Josh"],
    "ATL": ["Acuña Jr., Ronald", "Albies, Ozzie", "Riley, Austin", "Olson, Matt",
            "Ozuna, Marcell", "Harris II, Michael", "Murphy, Sean", "Arcia, Orlando",
            "d'Arnaud, Travis"],
    "HOU": ["Alvarez, Yordan", "Altuve, Jose", "Peña, Jeremy", "Diaz, Yainer",
            "Abreu, Wilyer", "Meyers, Jake", "Smith, Cam", "Dubón, Mauricio",
            "Garver, Mitch"],
    "TEX": ["Seager, Corey", "Semien, Marcus", "Langford, Wyatt", "García, Adolis",
            "Lowe, Nathaniel", "Jung, Josh", "Carter, Evan", "Heim, Jonah",
            "Smith, Josh"],
    "CLE": ["Kwan, Steven", "Giménez, Andrés", "Ramírez, José", "Naylor, Josh",
            "Freeman, Tyler", "Manzardo, Kyle", "Naylor, Bo", "Rocchio, Brayan",
            "Martínez, Angel"],
    "STL": ["Donovan, Brendan", "Gorman, Nolan", "Walker, Jordan", "Contreras, Willson",
            "Nootbaar, Lars", "Scott II, Victor", "Herrera, Iván", "Winn, Masyn",
            "Pagés, Pedro"],
    "CHC": ["Hoerner, Nico", "Suzuki, Seiya", "Tucker, Kyle", "Bellinger, Cody",
            "Happ, Ian", "Busch, Michael", "Morel, Christopher", "Shaw, Matt",
            "Crow-Armstrong, Pete"],
    "BAL": ["Holliday, Jackson", "Henderson, Gunnar", "Mountcastle, Ryan",
            "Rutschman, Adley", "Westburg, Jordan", "Cowser, Colton",
            "Ortiz, Joey", "Norby, Connor", "Mullins, Cedric"],
    "TOR": ["Springer, George", "Guerrero Jr., Vladimir", "Bichette, Bo",
            "Horwitz, Spencer", "Kirk, Alejandro", "Varsho, Daulton",
            "Lopez, Otto", "Barger, Addison", "Jansen, Danny"],
    "BOS": ["Duran, Jarren", "Anthony, Roman", "Devers, Rafael", "Bregman, Alex",
            "Yoshida, Masataka", "Story, Trevor", "Casas, Triston",
            "Rafaela, Ceddanne", "Campbell, Kristian"],
    "SEA": ["Rodríguez, Julio", "Raleigh, Cal", "Arozarena, Randy", "Crawford, J.P.",
            "Caballero, José", "Polanco, Jorge", "Moore, Dylan",
            "Ramos, Heliot", "Canzone, Dominic"],
    "MIN": ["Buxton, Byron", "Correa, Carlos", "Lewis, Royce", "Larnach, Trevor",
            "Wallner, Matt", "Jeffers, Ryan", "Castro, Willi", "Lee, Brooks",
            "Julien, Edouard"],
    "DET": ["Greene, Riley", "Torres, Gleyber", "Torkelson, Spencer", "Keith, Colt",
            "Carpenter, Kerry", "Dingler, Dillon", "Pérez, Wenceel",
            "Meadows, Parker", "McKinstry, Zach"],
    "KC":  ["Witt Jr., Bobby", "Pasquantino, Vinnie", "Perez, Salvador",
            "Garcia, Maikel", "Massey, Michael", "Young, Cole",
            "Fermin, Freddy", "Isbel, Kyle", "Massey, Michael"],
    "TB":  ["Díaz, Yandy", "Lowe, Brandon", "Aranda, Jonathan", "Mead, Curtis",
            "Simpson, Chandler", "Walls, Taylor", "Sanoja, Javier",
            "Caglianone, Jac", "Lowe, Josh"],
    "MIL": ["Yelich, Christian", "Chourio, Jackson", "Frelick, Sal",
            "Hoskins, Rhys", "Contreras, William", "Turang, Brice",
            "Adames, Willy", "Collins, Isaac", "Narváez, Carlos"],
    "WSH": ["Wood, James", "Abrams, CJ", "Young, Jacob", "Crews, Dylan",
            "García Jr., Luis", "Ruiz, Keibert", "House, Brady",
            "Hassell III, Robert", "Alexander, Blaze"],
    "SF":  ["Chapman, Matt", "Yastrzemski, Mike", "Lee, Jung Hoo",
            "Conforto, Michael", "Bailey, Patrick", "Flores, Wilmer",
            "Fitzgerald, Tyler", "Schmitt, Casey", "Wade Jr., LaMonte"],
    "SD":  ["Tatis Jr., Fernando", "Machado, Manny", "Bogaerts, Xander",
            "Arraez, Luis", "Merrill, Jackson", "Cronenworth, Jake",
            "Profar, Jurickson", "Gurriel Jr., Lourdes", "Vázquez, Christian"],
    "ARI": ["Carroll, Corbin", "Marte, Ketel", "Walker, Christian",
            "Perdomo, Geraldo", "Thomas, Alek", "Moreno, Gabriel",
            "McCarthy, Jake", "Bell, Josh", "Suárez, Eugenio"],
    "CIN": ["De La Cruz, Elly", "India, Jonathan", "McLain, Matt",
            "Friedl, TJ", "Stephenson, Tyler", "Steer, Spencer",
            "Marte, Noelvi", "France, Ty", "Fraley, Jake"],
    "PIT": ["Cruz, Oneil", "Reynolds, Bryan", "Hayes, Ke'Bryan",
            "Gonzales, Nick", "Davis, Henry", "Triolo, Jared",
            "Bart, Joey", "Gonzalez, Romy", "Bader, Harrison"],
    "OAK": ["Butler, Lawrence", "Rooker, Brent", "Kurtz, Nick",
            "Langeliers, Shea", "Wilson, Jacob", "Soderstrom, Tyler",
            "Bleday, JJ", "Allen, Nick", "Kemp, Otto"],
    "MIA": ["Edwards, Xavier", "Sánchez, Jesús", "Soler, Jorge",
            "Burger, Jake", "Fortes, Nick", "Benson, Will",
            "Lile, Daylen", "Pham, Tommy", "Durbin, Caleb"],
    "COL": ["Tovar, Ezequiel", "McMahon, Ryan", "Goodman, Hunter",
            "Doyle, Brenton", "Beck, Jordan", "Toglia, Michael",
            "Jones, Nolan", "Canario, Alexander", "Toro, Abraham"],
    "LAA": ["Trout, Mike", "Neto, Zach", "Ward, Taylor",
            "Schanuel, Nolan", "O'Hoppe, Logan", "Rengifo, Luis",
            "Adell, Jo", "Moniak, Mickey", "Grichuk, Randal"],
    "CWS": ["Robert Jr., Luis", "Vaughn, Andrew", "Sosa, Lenyn",
            "Montgomery, Colson", "Quero, Edgar", "Sheets, Gavin",
            "Benintendi, Andrew", "Moncada, Yoán", "Teel, Kyle"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA LOADING & PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

from players_2025 import PLAYERS_2025  # 631 embedded players (2025 season)

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

    # ── Column alias remapping (see module-level _SAVANT_ALIASES) ─────────────
    for old, new in _SAVANT_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    # ── Name column ───────────────────────────────────────────────────────────
    # Savant exports a single "last_name, first_name" column; some exports split
    # it or use "player_name" / "name" instead.
    if "last_name, first_name" in df.columns:
        df["Name"] = df["last_name, first_name"].astype(str).str.strip()
    elif "player_name" in df.columns:
        df["Name"] = df["player_name"].astype(str).str.strip()
    elif "last_name" in df.columns and "first_name" in df.columns:
        df["Name"] = (df["last_name"].str.strip()
                      + ", " + df["first_name"].str.strip())
    else:
        # Last resort: use first text column
        df["Name"] = df.iloc[:, 0].astype(str).str.strip()

    numeric_cols = ["pa", "ab", "single", "double", "triple", "home_run",
                    "k_percent", "bb_percent", "babip", "b_hit_by_pitch",
                    "batting_avg", "on_base_percent", "slg_percent",
                    "xba", "xslg", "xobp", "xwoba"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── PA ────────────────────────────────────────────────────────────────────
    if "pa" not in df.columns:
        # Reconstruct from AB if PA column is missing entirely
        df["pa"] = df.get("ab", pd.Series(0, index=df.index))
    df["PA"] = df["pa"].clip(lower=1)

    # ── Rate stats ────────────────────────────────────────────────────────────
    # k_percent / bb_percent: Savant exports as percentage (21.5, not 0.215).
    # Detect format by median and convert accordingly.
    for raw, out in [("k_percent", "K_pct"), ("bb_percent", "BB_pct")]:
        if raw in df.columns:
            vals = df[raw]
            df[out] = vals / 100.0 if vals.median() > 1.0 else vals
        else:
            df[out] = 0.22 if out == "K_pct" else 0.08

    pa = df["PA"]
    df["HR_pct"]  = df["home_run"].clip(lower=0)       / pa if "home_run"       in df.columns else 0.03
    df["HBP_pct"] = df["b_hit_by_pitch"].clip(lower=0) / pa if "b_hit_by_pitch" in df.columns else 0.01
    df["BABIP"]   = df["babip"] if "babip" in df.columns else 0.290

    singles  = df["single"].clip(lower=0) if "single" in df.columns else pd.Series(0, index=df.index)
    doubles  = df["double"].clip(lower=0) if "double" in df.columns else pd.Series(0, index=df.index)
    triples  = df["triple"].clip(lower=0) if "triple" in df.columns else pd.Series(0, index=df.index)
    non_hr   = (singles + doubles + triples).clip(lower=1)
    df["1B_rate"] = singles / non_hr
    df["2B_rate"] = doubles / non_hr
    df["3B_rate"] = triples / non_hr

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



# Fallback: 631 MLB batters (2025) embedded directly — no CSV required.
_SAMPLE_DATA = pd.DataFrame(PLAYERS_2025)

def _build_sample_df() -> pd.DataFrame:
    return _SAMPLE_DATA.copy()


def load_savant_csv_bytes(raw: bytes) -> "pd.DataFrame":
    """Process a user-uploaded CSV (from st.file_uploader) and return a player DataFrame."""
    import io as _io
    import pandas as _pd
    df = _pd.read_csv(_io.BytesIO(raw), encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.lower()
    for old, new in _SAVANT_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    if "last_name, first_name" in df.columns:
        df["Name"] = df["last_name, first_name"].astype(str).str.strip()
    elif "player_name" in df.columns:
        df["Name"] = df["player_name"].astype(str).str.strip()
    else:
        df["Name"] = df.iloc[:, 0].astype(str).str.strip()
    for col in ["pa","single","double","triple","home_run","k_percent",
                "bb_percent","babip","b_hit_by_pitch","xba","xslg","xobp"]:
        if col in df.columns:
            df[col] = _pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["PA"] = (df["pa"] if "pa" in df.columns else _pd.Series(1, index=df.index)).clip(lower=1)
    for rc, oc in [("k_percent","K_pct"),("bb_percent","BB_pct")]:
        if rc in df.columns:
            v = df[rc]; df[oc] = v / 100.0 if v.median() > 1.0 else v
        else:
            df[oc] = 0.22 if oc == "K_pct" else 0.08
    pa = df["PA"]
    df["HR_pct"]  = df["home_run"].clip(lower=0) / pa if "home_run" in df.columns else 0.03
    df["HBP_pct"] = df["b_hit_by_pitch"].clip(lower=0) / pa if "b_hit_by_pitch" in df.columns else 0.01
    df["BABIP"]   = df["babip"] if "babip" in df.columns else 0.290
    s = df["single"].clip(lower=0) if "single" in df.columns else _pd.Series(0, index=df.index)
    d = df["double"].clip(lower=0) if "double" in df.columns else _pd.Series(0, index=df.index)
    t3 = df["triple"].clip(lower=0) if "triple" in df.columns else _pd.Series(0, index=df.index)
    nhr = (s + d + t3).clip(lower=1)
    df["1B_rate"] = s / nhr; df["2B_rate"] = d / nhr; df["3B_rate"] = t3 / nhr
    if all(c in df.columns for c in ("xba","xobp","xslg")):
        xba = df["xba"].clip(0.10, 0.40); xobp = df["xobp"].clip(0.15, 0.55)
        xslg = df["xslg"].clip(0.20, 0.80); xiso = (xslg - xba).clip(0.0)
        df["xK_pct"]  = df["K_pct"]
        df["xBB_pct"] = (xobp - xba).clip(0.02, 0.25)
        df["xHR_pct"] = (xiso * 0.35).clip(upper=0.12)
        df["xBABIP"]  = (xba / (1.0 - df["K_pct"]).clip(lower=0.40)).clip(0.18, 0.42)
    else:
        df["xK_pct"] = df["K_pct"]; df["xBB_pct"] = df["BB_pct"]
        df["xHR_pct"] = df["HR_pct"]; df["xBABIP"] = df["BABIP"]
    df["Team"] = df["Name"].map(_PLAYER_TEAM).fillna("FA")
    return df.sort_values("PA", ascending=False).drop_duplicates("Name").reset_index(drop=True)




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
    except Exception as _csv_err:
        st.warning(
            f"⚠️ Could not parse `{CSV_PATH}` "
            f"({type(_csv_err).__name__}: {_csv_err}). "
            "Check **MANUAL_DOWNLOAD.md** for the correct export format. "
            "Sample data loaded."
        )
        st.session_state.players_df = _SAMPLE_DATA.copy()

_df_all: pd.DataFrame = st.session_state.players_df

# ── Sidebar: Simulation controls ─────────────────────────────────────────────
st.sidebar.markdown(f"### {t('sec_sim')}")

era_plus = st.sidebar.slider(
    t("era_plus"), 70, 150, 100,
    help=t("era_plus_help"),
)
min_pa = st.sidebar.slider(t("min_pa"), 0, 700, 0)
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

    st.markdown("##### 📤 Update player data (optional)")
    uploaded_csv = st.file_uploader(
        "Upload Batters_Savant_stats.csv",
        type="csv", key="upload_csv",
        help="Upload a fresh Baseball Savant export. Overwrites embedded 2025 data for this session.",
    )
    if uploaded_csv is not None:
        try:
            fresh = load_savant_csv_bytes(uploaded_csv.read())
            st.session_state.players_df = fresh
            st.success(f"Loaded {len(fresh):,} players from uploaded CSV.")
            st.rerun()
        except Exception as _e:
            st.error(f"CSV parse error: {_e}")
    st.markdown("---")
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
    team_label = selected_team_abbr or "all teams"
    st.warning(
        f"No players match: min PA={min_pa}, team={team_label}. "
        f"Dataset has {len(_df_all)} players total. "
        "Try setting **min PA = 0** or disabling Team Mode."
    )
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

        probs_a = build_probs(row_a, era_plus)
        probs_b = build_probs(row_b, era_plus)
        la      = np.array([probs_a] * 9, dtype=np.float64)
        lb      = np.array([probs_b] * 9, dtype=np.float64)

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

        prog_bar = st.progress(0.0, text=t("computing"))

        def _progress(frac: float):
            prog_bar.progress(min(frac, 1.0), text=f"{t('computing')} {frac*100:.0f}%")

        start = time.time()
        base_rpg, tornado, deltas, obp_curve, slg_curve, detail = \
            compute_sensitivity(sens_row, era_plus, progress_cb=_progress)
        elapsed = time.time() - start

        prog_bar.empty()

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
