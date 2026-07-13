"""
constants.py
============
Application-level constants, team rosters and default lineups.
No Streamlit dependency — imported once and cached by Python's module system.
"""

# ── App metadata ──────────────────────────────────────────────────────────────
APP_TITLE   = "Monte Carlo Baseball 2026"
APP_VERSION = "2.0.0"
DATA_SOURCE = "Baseball Savant 2025 actuals · xStats-based 2026 projections"
CSV_PATH    = "Batters_Savant_stats.csv"

# ── Simulation defaults ───────────────────────────────────────────────────────
PROJECTION_PA_THRESHOLD = 100   # below this PA → use xStats projections
N_GAMES_MIXED           = 300
N_GAMES_SAME            = 200
N_GAMES_FULL            = 1000
N_SENS_POINTS           = 14
N_GAMES_SENS            = 150
LHP_RATE                = 0.38  # share of LHP starts in MLB

# ── Visual palette ────────────────────────────────────────────────────────────
PALETTE = ["#4C8AC6", "#E07B54", "#5DBB8A", "#C67BB5", "#E0C454", "#7BC6C6"]

# ── Team dictionary: full name → abbreviation ─────────────────────────────────
TEAMS: dict[str, str] = {
    "New York Yankees":     "NYY", "Los Angeles Dodgers":  "LAD",
    "New York Mets":        "NYM", "Philadelphia Phillies": "PHI",
    "Atlanta Braves":       "ATL", "Houston Astros":       "HOU",
    "Texas Rangers":        "TEX", "Cleveland Guardians":  "CLE",
    "St. Louis Cardinals":  "STL", "Chicago Cubs":         "CHC",
    "Baltimore Orioles":    "BAL", "Toronto Blue Jays":    "TOR",
    "Boston Red Sox":       "BOS", "San Diego Padres":     "SD",
    "San Francisco Giants": "SF",  "Los Angeles Angels":   "LAA",
}

# ── Player → team mapping (names as stored in Savant CSV) ────────────────────
PLAYER_TEAM: dict[str, str] = {
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
    # Giants
    "Bailey, Patrick": "SF", "Flores, Wilmer": "SF", "Conforto, Michael": "SF",
    "Slater, Austin": "SF", "Estrada, Thairo": "SF", "Davis, JD": "SF",
    # Angels
    "Trout, Mike": "LAA", "Rengifo, Luis": "LAA", "Moniak, Mickey": "LAA",
    "Ward, Taylor": "LAA", "Neto, Zach": "LAA", "O'Hoppe, Logan": "LAA",
}

# ── Default lineups: abbrev → 9 ordered player names ─────────────────────────
DEFAULT_LINEUPS: dict[str, list[str]] = {
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
