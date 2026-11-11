"""Konstanten: Vehikel "Tagesaufträge eines Depots" (wie in den Stücken 1 bis 11 der Linie), Regler-Grenzen, Sprungformen, Kostenannahmen, Presets."""

EPS = 1e-9
SEED_MAX = 999999

# --- Vehikel (identisch zu naive-forecast-demo: drei Jahre täglich, Ursprünge im letzten Jahr) -----------------------------------------------
N_DAYS = 1095
FIRST_TEST = 730
LEVEL = 100.0
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
HOLIDAY_DROP = 0.5
HOLIDAY_REBOUND = 0.15
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3
DEFAULT_TREND, DEFAULT_WEEKLY, DEFAULT_YEARLY, DEFAULT_EVENTS = 10, 1.0, 0.2, 0.5
HORIZON = 14                                         # Prognosehorizont der MASE
FIT_FROM = 28                                        # die Regression nutzt die Tage 28 bis 729
REF_LEN = 91                                         # Referenzfenster des Detektors: die 13 Wochen vor dem Testjahr
FOURIER_PAIRS = 2

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 2, 40, 2, 14                  # Prozent (Streuung des log-normalen Tagesrauschens)
SHIFT_MIN, SHIFT_MAX, SHIFT_STEP, DEFAULT_SHIFT = 5, 100, 5, 30                 # Prozent: Größe des Sprungs
AR_MIN, AR_MAX, AR_STEP, DEFAULT_AR = 0.0, 0.8, 0.1, 0.0                        # Autokorrelation des Log-Rauschens (AR(1))
H_MIN, H_MAX, H_STEP, DEFAULT_H = 2.0, 20.0, 0.5, 6.0                           # Schwelle des CUSUM (in Standardabweichungen)
K_MIN, K_MAX, K_STEP, DEFAULT_K = 0.25, 2.0, 0.25, 0.5                          # Zulage des CUSUM
ALPHA_LOW_MIN, ALPHA_LOW_MAX, ALPHA_LOW_STEP, DEFAULT_ALPHA_LOW = 0.01, 0.30, 0.01, 0.06   # langsame Nachführung zwischen den Meldungen
WMIN_MIN, WMIN_MAX, DEFAULT_WMIN = 1, 14, 5                                      # Mindestfenster beim Zurücksetzen (Tage)
EWMA_MIN, EWMA_MAX, EWMA_STEP, DEFAULT_EWMA = 0.02, 0.50, 0.01, 0.10            # Gegner 1: EWMA-Nachführung (α)
MA_MIN, MA_MAX, DEFAULT_MA = 7, 56, 14                                           # Gegner 2: Mittel der letzten k Tage
TRIGG_LAM_MIN, TRIGG_LAM_MAX, TRIGG_LAM_STEP, DEFAULT_TRIGG_LAM = 0.05, 0.30, 0.05, 0.05   # Gegner 3: Trigg-Leach, Glättung der Fehlersignale
TRIGG_AMIN_MIN, TRIGG_AMIN_MAX, TRIGG_AMIN_STEP, DEFAULT_TRIGG_AMIN = 0.02, 0.20, 0.01, 0.08  # Trigg-Leach: kleinstes α
TRIGG_AMAX = 0.5
# Bestätigung: Wartetage c und Schwelle theta in einer Auswahl, damit jeder Wert wirkt (theta ohne Wartezeit wäre ein toter Regler)
CONFIRM_RULES = {"off": (0, 1.0), "c3": (3, 1.0), "c7": (7, 1.0), "c10": (10, 1.0), "c14": (14, 1.0), "c21": (21, 1.0), "c10t15": (10, 1.5), "c14t15": (14, 1.5)}
CONFIRM_NAMES = {
    "off": "aus (jede Meldung gilt sofort)", "c3": "nach 3 Tagen, θ = 1,0", "c7": "nach 7 Tagen, θ = 1,0", "c10": "nach 10 Tagen, θ = 1,0", "c14": "nach 14 Tagen, θ = 1,0",
    "c21": "nach 21 Tagen, θ = 1,0", "c10t15": "nach 10 Tagen, θ = 1,5", "c14t15": "nach 14 Tagen, θ = 1,5",
}
DEFAULT_CONFIRM_RULE = "off"
CONFIRM_DEFAULT_TEST = 10                                                        # Wartezeit des Bestätigungsversuchs, wenn die Auswahl auf "aus" steht
TAU_OPTIONS = (0.8, 0.9, 0.95, 0.99)
DEFAULT_TAU = 0.95
LEAD_MIN, LEAD_MAX, DEFAULT_LEAD = 0, 7, 3                                       # Wiederbeschaffungszeit in Tagen
RULES = ("empirical", "conformal")
RULE_NAMES = {"empirical": "empirisch (Quantil der Trainingsfehler)", "conformal": "konform (letzte 90 realisierte Fehler)"}
DEFAULT_RULE = "empirical"
RUNS_MIN, RUNS_MAX, RUNS_STEP, DEFAULT_RUNS = 12, 60, 6, 30                      # Zahl der Reihen je Auswertung
CONFORMAL_WINDOW = 90

MODELS = ("weekday", "fourier_trend", "calendar")
MODEL_NAMES = {
    "weekday": "nur Wochentag",
    "fourier_trend": "Wochentag, Jahresmuster, Trend",
    "calendar": "zusätzlich Feiertage und Aktionsplan",
}
DEFAULT_MODEL = "calendar"

FORMS = ("step", "step_back", "step_two", "ramp", "promo", "spike")
FORM_NAMES = {
    "step": "Stufe: dauerhafter Sprung",
    "step_back": "Stufe, nach 90 Tagen zurück",
    "step_two": "Stufe, nach 120 Tagen zweiter Sprung (zwei Drittel der Größe)",
    "ramp": "Rampe über 60 Tage",
    "promo": "Aktionswoche (7 Tage)",
    "spike": "Ausreißer (1 Tag)",
}
DEFAULT_FORM = "step"
WINDOW_SHORT, WINDOW_LONG = 60, 180                  # Fenster für Gewinne: ab dem Sprung (bei Folgen von Sprüngen länger)
BACK_AFTER, SECOND_AFTER, RAMP_DAYS, PROMO_DAYS = 90, 120, 60, 7
SECOND_FRACTION = 2.0 / 3.0

# --- Gegner und Referenzverfahren ------------------------------------------------------------------------------------------------------------
ORACLE_ALPHA = 0.03                                  # Nachführung des Orakels zwischen den Sprüngen (Abstimmung der Vormessung, Zelle Rauschen 14 %, +30 %)
HW_ALPHA, HW_GAMMA = 0.10, 0.02                      # feste saisonale Glättung (multiplikativ)
METHODS = ("snaive", "wmean", "hw", "none", "ewma", "ma", "trigg", "detector", "oracle", "oracle_delayed")
METHOD_NAMES = {
    "snaive": "Saisonal naiv (Stück 1 der Linie)",
    "wmean": "Wochenmittel, 4 Wochen (Stück 1 der Linie)",
    "hw": "Saisonale Glättung (feste Parameter)",
    "none": "Regression ohne Nachführung",
    "ewma": "Regression + EWMA-Nachführung",
    "ma": "Regression + Mittel der letzten k Tage",
    "trigg": "Regression + Trigg-Leach (adaptives α)",
    "detector": "Regression + CUSUM-Zurücksetzen",
    "oracle": "Orakel: kennt Tag und Größe",
    "oracle_delayed": "Orakel, um die Verzögerung des Detektors verspätet",
}
OPPONENTS = ("ewma", "ma", "trigg")

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = tuple(range(10000, 10020))             # feste Reihen für Sweeps (nie ein einzelner Seed)
MEASURE_SEEDS = tuple(range(10000, 10300))           # Messreihen der Karte (300 Reihen)
TUNE_SEEDS = tuple(range(12000, 12100))              # Abstimm-Reihen (getrennt von den Messreihen)
TUNE_SEEDS_B = tuple(range(22000, 22100))            # Wiederholung der Karte: andere Abstimm- und Messreihen
MEASURE_SEEDS_B = tuple(range(20000, 20300))
CONFIRM_SEEDS = tuple(range(14000, 14600))           # Reihen der Bestätigungsmessung (README)
HEAD_SEEDS = tuple(range(16000, 16600))              # Standardszenario (README)
KARTE_NOISE = (4, 8, 14, 24)
KARTE_SHIFT = (10, 20, 30, 50)
KARTE_AR = (0.0, 0.6)
H_GRID = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)  # Schwellen der Kennlinie
KENNLINIE_RUNS = 60

COLOR_GEGNER = "#4c78a8"
COLOR_DETEKTOR = "#2F6B65"
COLOR_ORAKEL = "#9467bd"
COLOR_ALARM = "#d62728"
COLOR_RAW = "rgba(150,150,150,0.55)"
COLOR_JUMP = "#e69f00"

# --- Presets ----------------------------------------------------------------------------------------------------------------------------------
# Die Presets setzen die Regler des Szenarios und der Bestätigung. Die Voreinstellungen von Detektor und Gegnern sind die Abstimmung der Karten-Zelle Rauschen 14 %, +30 %, keine Autokorrelation (Ziel Kosten);
# die Presets mit anderem Rauschen oder Autokorrelation übernehmen stattdessen die Abstimmung ihrer eigenen Zelle (Karte, `data/karte.json`; `tests/test_claims.py` prüft die Werte gegen die Karte).
TUNED_NOISE_4 = {"h": 4.0, "k": 1.0, "alpha_low": 0.03, "wmin": 2, "ewma": 0.25, "ma": 7, "trigg_lam": 0.05, "trigg_amin": 0.08}
TUNED_NOISE_24 = {"h": 8.0, "k": 1.0, "alpha_low": 0.06, "wmin": 2, "ewma": 0.06, "ma": 28, "trigg_lam": 0.05, "trigg_amin": 0.08}
TUNED_AR_06 = {"h": 12.0, "k": 1.0, "alpha_low": 0.1, "wmin": 2, "ewma": 0.15, "ma": 21, "trigg_lam": 0.05, "trigg_amin": 0.08}
_DEFAULT = {
    "noise": DEFAULT_NOISE, "shift": DEFAULT_SHIFT, "form": DEFAULT_FORM, "ar": DEFAULT_AR, "model": DEFAULT_MODEL, "h": DEFAULT_H, "k": DEFAULT_K,
    "alpha_low": DEFAULT_ALPHA_LOW, "wmin": DEFAULT_WMIN, "ewma": DEFAULT_EWMA, "ma": DEFAULT_MA, "trigg_lam": DEFAULT_TRIGG_LAM, "trigg_amin": DEFAULT_TRIGG_AMIN,
    "confirm_rule": DEFAULT_CONFIRM_RULE, "tau": DEFAULT_TAU, "lead": DEFAULT_LEAD, "rule": DEFAULT_RULE, "runs": DEFAULT_RUNS, "seed": 3,
}
PRESETS = {
    "Standardszenario (+30 %, Rauschen 14 %)": dict(_DEFAULT),
    "Ruhige Reihe (Rauschen 4 %)": {**_DEFAULT, "noise": 4, **TUNED_NOISE_4},
    "Laute Reihe (Rauschen 24 %)": {**_DEFAULT, "noise": 24, **TUNED_NOISE_24},
    "Aktionswoche +50 %, ohne Bestätigung": {**_DEFAULT, "shift": 50, "form": "promo"},
    "Aktionswoche +50 %, Bestätigung nach 10 Tagen": {**_DEFAULT, "shift": 50, "form": "promo", "confirm_rule": "c10"},
    "Autokorrelierte Fehler (0,6)": {**_DEFAULT, "ar": 0.6, **TUNED_AR_06},
    "Sprung hoch und zurück": {**_DEFAULT, "form": "step_back"},
}
DEFAULT_PRESET = next(iter(PRESETS))
PRESET_HELP = {
    "Standardszenario (+30 %, Rauschen 14 %)": "Der Standardfall: Sprung von +30 % an einem zufälligen Tag im Testjahr, Rauschen 14 %, keine Autokorrelation, Regression mit Kalender.",
    "Ruhige Reihe (Rauschen 4 %)": "Wenig Rauschen, Sprung +30 %: der Sprung hebt sich klar ab. Detektor und Gegner wie in der Abstimmung dieser Karten-Zelle.",
    "Laute Reihe (Rauschen 24 %)": "Viel Rauschen, Sprung +30 %: der auf Kosten abgestimmte Detektor meldet den Sprung fast nie (Schwelle und Zulage sind dafür zu hoch). Detektor und Gegner wie in der Abstimmung dieser Karten-Zelle; mit den Standardwerten wird der Sprung öfter gemeldet.",
    "Aktionswoche +50 %, ohne Bestätigung": "Eine einwöchige Aktion mit +50 % statt eines dauerhaften Sprungs: der Detektor meldet sie als Sprung und setzt das Niveau fälschlich hoch.",
    "Aktionswoche +50 %, Bestätigung nach 10 Tagen": "Dieselbe Aktionswoche, aber eine Meldung gilt erst, wenn die letzten Tage zehn Tage nach dem Alarm noch erhöht sind.",
    "Autokorrelierte Fehler (0,6)": "Die Tagesfehler hängen zusammen (AR(1), Koeffizient 0,6), Rauschen 14 %, Sprung +30 %. Detektor und Gegner wie in der Abstimmung dieser Karten-Zelle.",
    "Sprung hoch und zurück": "+30 % und nach 90 Tagen wieder zurück auf das alte Niveau: das Zurücksetzen muss zweimal richtig melden.",
}
