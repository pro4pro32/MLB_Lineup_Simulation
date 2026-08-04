# Jak pobrać pełne dane 2025 z Baseball Savant

Baseball Savant blokuje automatyczne requesty z serwerów (Cloudflare bot protection).
Poniżej dwie opcje — wybierz tę która Ci odpowiada.

---

## Opcja A — Przeglądarka (najszybsza, 2 minuty)

Otwórz ten URL w przeglądarce i poczekaj aż strona się załaduje,
następnie kliknij **Export CSV** (prawy górny róg tabeli):

```
https://baseballsavant.mlb.com/leaderboard/custom?year=2025&type=batter&filter=&sort=4&sortDir=desc&min=1&selections=b_ab,b_pa,b_hit,b_single,b_double,b_triple,b_home_run,b_strikeout,b_walk,b_k_percent,b_bb_percent,batting_avg,slg_percent,on_base_percent,on_base_plus_slg,isolated_power,babip,xba,xslg,xobp,xwoba,xiso,b_rbi,b_hit_by_pitch,exit_velocity_avg,launch_angle_avg,sweet_spot_percent,barrel_batted_rate&csv=true
```

Lub wejdź bezpośrednio po CSV (otwiera plik w przeglądarce / pobiera):

```
https://baseballsavant.mlb.com/leaderboard/custom?year=2025&type=batter&filter=&sort=4&sortDir=desc&min=1&selections=b_ab,b_pa,b_hit,b_single,b_double,b_triple,b_home_run,b_strikeout,b_walk,b_k_percent,b_bb_percent,batting_avg,slg_percent,on_base_percent,on_base_plus_slg,isolated_power,babip,xba,xslg,xobp,xwoba,xiso,b_rbi,b_hit_by_pitch,exit_velocity_avg,launch_angle_avg,sweet_spot_percent,barrel_batted_rate&csv=true
```

Zapisz plik jako: **`Batters_Savant_stats.csv`**
Umieść go w tym samym folderze co `app.py`.

---

## Opcja B — Skrypt Python (lokalnie na Twoim komputerze)

```bash
pip install pybaseball pandas requests
python download_stats.py
```

Skrypt próbuje pobrać dane z Baseball Savant z pełnymi nagłówkami
przeglądarki + plik z FanGraphs jako backup. Działa lokalnie,
nie działa na serwerze (403 bot protection).

---

## Opcja C — pybaseball CLI (tylko FanGraphs)

Jeśli Savant dalej blokuje, możesz użyć samego FanGraphs:

```python
from pybaseball import batting_stats
import pandas as pd

df = batting_stats(2025, qual=1)
df.to_csv("Batters_Savant_stats.csv", index=False, encoding="utf-8-sig")
print(f"Pobrano {len(df)} zawodników")
```

⚠️ FanGraphs nie ma xStats (xBA, xSLG, xOBP) — projekcje 2026
   będą używać rzeczywistych statystyk zamiast modelu Statcast.
   Wszystkie inne funkcje aplikacji działają normalnie.

---

## Jakie kolumny są potrzebne

Minimalne wymaganie (app.py sam obsługuje brakujące kolumny):

| Kolumna              | Opis                        |
|----------------------|-----------------------------|
| `last_name, first_name` | Imię i nazwisko          |
| `pa`                 | Plate appearances           |
| `single/double/triple/home_run` | Liczby uderzeń   |
| `k_percent`          | Strikeout % (format 21.5)   |
| `bb_percent`         | Walk % (format 8.3)         |
| `babip`              | BABIP (format 0.296)        |
| `b_hit_by_pitch`     | Hit by pitch count          |

Opcjonalne (używane do projekcji xStats):

| Kolumna | Opis               |
|---------|--------------------|
| `xba`   | Expected batting avg |
| `xslg`  | Expected SLG       |
| `xobp`  | Expected OBP       |
| `xwoba` | Expected wOBA      |

---

## Sprawdzenie poprawności pliku

Po pobraniu uruchom:

```bash
python download_stats.py --verify Batters_Savant_stats.csv
```

Lub ręcznie w Python:

```python
import pandas as pd
df = pd.read_csv("Batters_Savant_stats.csv", encoding="utf-8-sig")
print(f"Zawodnicy: {len(df)}")
print(f"Kolumny:   {list(df.columns[:8])}")
# Powinno wyświetlić ~700-900 zawodników i kolumnę 'last_name, first_name'
```
