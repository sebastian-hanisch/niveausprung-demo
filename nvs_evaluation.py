"""Auswertung: Reihen erzeugen, Regression anpassen, alle Nachführer nebeneinander in MASE und Bestandskosten messen, gepaarte Gewinne, Kennlinie der Schwelle, Bestätigung
gegen die Formverwechslung. Die Verdikte der App sind reine Funktionen am Ende der Datei (Art, Text)."""

from dataclasses import dataclass, fields, replace

import numpy as np

import nvs_algorithm as A
import nvs_constants as C
import nvs_inventory as INV
import nvs_scenario as S
from nvs_scenario import T, T0

ORG_TEST = np.arange(T0, T - C.HORIZON + 1)              # Ursprünge des Testjahres für die MASE
TEST_IDX = ORG_TEST - INV.ORG_ALL[0]


@dataclass(frozen=True)
class Settings:
    noise: int = C.DEFAULT_NOISE
    shift: int = C.DEFAULT_SHIFT
    form: str = C.DEFAULT_FORM
    ar: float = C.DEFAULT_AR
    model: str = C.DEFAULT_MODEL
    h: float = C.DEFAULT_H
    k: float = C.DEFAULT_K
    alpha_low: float = C.DEFAULT_ALPHA_LOW
    wmin: int = C.DEFAULT_WMIN
    ewma: float = C.DEFAULT_EWMA
    ma: int = C.DEFAULT_MA
    trigg_lam: float = C.DEFAULT_TRIGG_LAM
    trigg_amin: float = C.DEFAULT_TRIGG_AMIN
    confirm_rule: str = C.DEFAULT_CONFIRM_RULE
    tau: float = C.DEFAULT_TAU
    lead: int = C.DEFAULT_LEAD
    rule: str = C.DEFAULT_RULE
    runs: int = C.DEFAULT_RUNS
    seed: int = 3


SETTING_NAMES = tuple(f.name for f in fields(Settings))


def _confirm(s):
    return C.CONFIRM_RULES[s.confirm_rule]


Settings.confirm = property(lambda s: _confirm(s)[0])          # Wartetage c (0 = aus)
Settings.theta = property(lambda s: _confirm(s)[1])            # Bestätigungsschwelle theta


def detector_params(s: Settings) -> dict:
    return {"alpha": s.alpha_low, "h": s.h, "k": s.k, "wmin": s.wmin, "confirm": s.confirm, "theta": s.theta}


def tracker_params(s: Settings) -> dict:
    """Parameter der Nachführer: {Name: (Art, Parameter)}."""
    return {
        "none": ("none", {}),
        "ewma": ("ewma", {"alpha": s.ewma}),
        "ma": ("ma", {"k": s.ma}),
        "trigg": ("trigg", {"lam": s.trigg_lam, "amin": s.trigg_amin, "amax": C.TRIGG_AMAX}),
        "detector": ("cusum", detector_params(s)),
        "oracle": ("oracle", {"alpha": C.ORACLE_ALPHA}),
    }


def mean_se(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def paired_gain(ref, other):
    """Gewinn von `other` gegen `ref` in Prozent des Mittels von `ref` (positiv: other ist besser, also kleiner), mit Standardfehler in Prozent; gepaart über die Reihen."""
    ref = np.asarray(ref, float)
    d = ref - np.asarray(other, float)
    m = float(ref.mean())
    if abs(m) < C.EPS:
        return 0.0, 0.0
    return float(d.mean() / m * 100.0), float(d.std(ddof=1) / np.sqrt(len(d)) / abs(m) * 100.0) if len(d) > 1 else 0.0


@dataclass
class Prepared:
    """Reihen, Regression und Residuen einer Auswertung (ohne Nachführer)."""
    batch: S.Batch
    fit: np.ndarray
    R: np.ndarray
    s2: np.ndarray
    scale: np.ndarray


def prepare(batch: S.Batch, model: str) -> Prepared:
    X = A.design(batch.promo, batch.holiday, batch.after, model)
    fit, R, s2 = A.reg_fit(batch.y, X)
    return Prepared(batch, fit, R, s2, S.mase_scale(batch.y))


def first_meldung(meldung, J, win, after_extra=0):
    """Erste Meldung im Fenster [J, J + win + after_extra) je Reihe: (gefunden (N,) bool, Verzögerung in Tagen (N,), -1 wenn keine)."""
    N, Tn = meldung.shape
    t = np.arange(Tn)[None, :]
    inwin = meldung & (t >= J[:, None]) & (t < J[:, None] + win + after_extra)
    found = inwin.any(axis=1)
    ft = np.argmax(inwin, axis=1)
    return found, np.where(found, ft - J, -1)


def evaluate_forecasts(batch, P, forecasts, lead, rule, tau, win):
    """MASE (alle Ursprünge, Fenster) und Bestandskosten (Testjahr, Fenster) je Verfahren: {Name: {Maß: (N,)}}. forecasts: {Name: (N, len(ORG_ALL), 14)}."""
    A_act = A.targets(batch.y, ORG_TEST)
    mask = (ORG_TEST[None, :] >= batch.J[:, None]) & (ORG_TEST[None, :] < batch.J[:, None] + win)
    names = [m for m in forecasts]
    stocks, metrics = [], {}
    for m in names:
        F = forecasts[m]
        metrics[m] = {"mase_alle": A.mase(F[:, TEST_IDX], A_act, P.scale), "mase_fenster": A.mase(F[:, TEST_IDX], A_act, P.scale, mask)}
        Fm = INV.interval_sum(F, lead)
        stocks.append(INV.stock_levels(Fm, INV.scores(batch.y, Fm, lead), lead, rule, tau))
    c = INV.costs(np.stack(stocks), batch.y, batch.J, lead, tau, win)
    for b, m in enumerate(names):
        metrics[m]["kosten_alle"] = c["alle"][b]
        metrics[m]["kosten_fenster"] = c["fenster"][b]
    return metrics


@dataclass
class Analysis:
    settings: Settings
    prepared: Prepared
    tracks: dict
    delay_days: object                    # Verzögerung des Orakels (ganze Tage) oder None
    metrics: dict                         # {Verfahren: {"mase_alle": (N,), "mase_fenster", "kosten_alle", "kosten_fenster"}}
    found: np.ndarray                     # (N,) Meldung im Fenster
    delay: np.ndarray                     # (N,) Verzögerung (-1: keine)
    meldungen: np.ndarray                 # (N,) Meldungen im Testjahr
    vor_sprung: np.ndarray                # (N,) Meldungen im Testjahr vor dem Sprungtag
    window: int


def analyse(s: Settings, seeds=None) -> Analysis:
    """Alle Verfahren auf denselben Reihen (Seeds seed bis seed + runs - 1). Zähleinheiten: MASE je Reihe (Horizont 1 bis 14 Tage, Ursprünge Tag 730 bis 1081, Nenner: saisonal naiver Fehler auf den Tagen 7 bis 729), Kosten je Tag in Tagesbedarfen."""
    seeds = tuple(range(s.seed, s.seed + s.runs)) if seeds is None else tuple(seeds)
    batch = S.batch(seeds, s.form, s.shift / 100.0, s.noise / 100.0, s.ar)
    P = prepare(batch, s.model)
    tracks = {}
    for name, (kind, p) in tracker_params(s).items():
        tracks[name] = A.track(P.R, kind, p, jumps=A.oracle_jumps(batch.J, batch.jumps, batch.has_jump) if kind == "oracle" else None)
    win = batch.window
    found, delay = first_meldung(tracks["detector"]["meldung"], batch.J, C.WINDOW_SHORT, s.confirm)
    # die Verzögerung für das verspätete Orakel gibt es nur für dauerhafte Änderungen (bei Aktionswoche und Ausreißer ist "Verzögerung bis zum Sprung" nicht definiert)
    delay_days = int(round(float(delay[found].mean()))) if found.any() and s.form not in ("promo", "spike") else None
    if delay_days is not None:
        tracks["oracle_delayed"] = A.track(P.R, "oracle", {"alpha": C.ORACLE_ALPHA}, jumps=A.oracle_jumps(batch.J, batch.jumps, batch.has_jump, delay=delay_days))
    org = INV.ORG_ALL
    forecasts = {}
    Lh, Sh = A.holt_winters(batch.y, C.HW_ALPHA, C.HW_GAMMA)
    forecasts["hw"] = A.hw_forecasts(Lh, Sh, org)
    forecasts["snaive"] = A.seasonal_naive_forecasts(batch.y, org)
    forecasts["wmean"] = A.weekly_mean_forecasts(batch.y, org)
    for name, tr in tracks.items():
        forecasts[name] = A.forecasts_from_offsets(P.fit, tr["O"], P.s2, org)
    metrics = evaluate_forecasts(batch, P, forecasts, s.lead, s.rule, s.tau, win)
    meld = tracks["detector"]["meldung"][:, T0:]
    day = np.arange(T0, T)[None, :]
    return Analysis(s, P, tracks, delay_days, metrics, found, delay, meld.sum(axis=1), (meld & (day < batch.J[:, None])).sum(axis=1), win)


MEASURES = ("mase_alle", "mase_fenster", "kosten_alle", "kosten_fenster")
MEASURE_NAMES = {"mase_alle": "MASE, alle Ursprünge", "mase_fenster": "MASE, Fenster", "kosten_alle": "Kosten, Testjahr", "kosten_fenster": "Kosten, Fenster"}


def strongest_opponent(a: Analysis, measure: str) -> str:
    """Der Gegner (EWMA, Mittel, Trigg-Leach) mit dem kleinsten Mittel des Maßes: der stärkste unter den eingestellten Parametern."""
    return min(C.OPPONENTS, key=lambda m: float(a.metrics[m][measure].mean()))


def summary(a: Analysis) -> dict:
    """Je Maß: Mittel und Standardfehler jedes Verfahrens, stärkster Gegner, Gewinn des Detektors und des Orakels gegen ihn (Prozent ± Standardfehler)."""
    out = {}
    for ms in MEASURES:
        opp = strongest_opponent(a, ms)
        row = {"opponent": opp, "means": {m: mean_se(v[ms]) for m, v in a.metrics.items()}}
        row["gain_detector"] = paired_gain(a.metrics[opp][ms], a.metrics["detector"][ms])
        row["gain_oracle"] = paired_gain(a.metrics[opp][ms], a.metrics["oracle"][ms])
        row["gain_regression"] = paired_gain(a.metrics["hw"][ms], a.metrics[opp][ms])      # Vorsprung der Regression + Nachführung gegen die Glättung
        row["gain_total"] = paired_gain(a.metrics["hw"][ms], a.metrics["detector"][ms])
        if "oracle_delayed" in a.metrics:
            row["gain_oracle_delayed"] = paired_gain(a.metrics[opp][ms], a.metrics["oracle_delayed"][ms])
        out[ms] = row
    return out


def reserve_share(gain_detector: float, gain_oracle: float):
    """Anteil der Orakel-Reserve, den der Detektor holt: Gewinn Detektor / Gewinn Orakel (beide in Prozent gegen denselben Gegner). None, wenn das Orakel nichts gewinnt."""
    if gain_oracle <= 0.05:
        return None
    return gain_detector / gain_oracle


# --------------------------------------------------------------------------------------------------------------------------------------------
# Kennlinie der Schwelle: Fehlmeldungen je Jahr gegen Verzögerung
def kennlinie(s: Settings, hs=C.H_GRID, runs=C.KENNLINIE_RUNS, offset=500):
    """Je Schwelle h: Fehlmeldungen je 365 Tage (Reihen ohne Sprung), Anteil gefundener Sprünge und mittlere Verzögerung in Tagen (Reihen mit Sprung). Gleiche Zufallszahlen mit und ohne Sprung.
    Bestätigung (c, theta) und alle übrigen Detektorparameter aus den Einstellungen."""
    seeds = tuple(range(s.seed + offset, s.seed + offset + runs))
    form_batch = S.batch(seeds, s.form, s.shift / 100.0, s.noise / 100.0, s.ar)
    null_batch = S.batch(seeds, s.form, 0.0, s.noise / 100.0, s.ar)
    Pj, Pn = prepare(form_batch, s.model), prepare(null_batch, s.model)
    rows = []
    for h in hs:
        p = {**detector_params(s), "h": h}
        trj = A.track(Pj.R, "cusum", p)
        trn = A.track(Pn.R, "cusum", p)
        found, delay = first_meldung(trj["meldung"], form_batch.J, C.WINDOW_SHORT, s.confirm)
        fehl = float(trn["meldung"][:, T0:].sum(axis=1).mean())
        rows.append({"h": float(h), "fehlmeldungen": fehl, "gefunden": float(found.mean()), "verzoegerung": float(delay[found].mean()) if found.any() else None})
    return rows


# --------------------------------------------------------------------------------------------------------------------------------------------
# Bestätigung gegen die Formverwechslung
CONFIRM_FORMS = ("step", "ramp", "promo", "spike")


def confirm_experiment(s: Settings, c=None, forms=CONFIRM_FORMS, runs=C.KENNLINIE_RUNS, offset=700):
    """Je Form (in der Größe der Einstellungen) und ohne Form: Anteil der Reihen mit einer Meldung im Fenster [J, J + 60 + c) und mittlere Verzögerung, ohne Bestätigung (c = 0) und mit c Tagen
    Wartezeit (Standard: c der Einstellungen, bei 0: 10 Tage). Zeile "keine": Fehlmeldungen je Jahr. Rückgabe: (c, Zeilen)."""
    c = s.confirm if c is None and s.confirm > 0 else (C.CONFIRM_DEFAULT_TEST if c is None else c)
    seeds = tuple(range(s.seed + offset, s.seed + offset + runs))
    rows = []
    for form in tuple(forms) + ("none",):
        batch = S.batch(seeds, "step" if form == "none" else form, 0.0 if form == "none" else s.shift / 100.0, s.noise / 100.0, s.ar)
        P = prepare(batch, s.model)
        row = {"form": form}
        for tag, cc in (("ohne", 0), ("mit", c)):
            tr = A.track(P.R, "cusum", {**detector_params(s), "confirm": cc})
            if form == "none":
                row[tag] = {"fehlmeldungen": float(tr["meldung"][:, T0:].sum(axis=1).mean())}
            else:
                found, delay = first_meldung(tr["meldung"], batch.J, C.WINDOW_SHORT, cc)
                row[tag] = {"anteil": float(found.mean()), "verzoegerung": float(delay[found].mean()) if found.any() else None}
        rows.append(row)
    return c, rows


# --------------------------------------------------------------------------------------------------------------------------------------------
# Verdikte (reine Funktionen: Art, Text)
def fmt(x, digits=1):
    """Zahl mit deutschem Dezimalkomma; eine auf null gerundete Zahl hat kein Vorzeichen."""
    s = f"{x:.{digits}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{digits}f}"
    return s.replace(".", ",")


def _pct(x, digits=1):
    return fmt(x, digits)


def verdict_main(summ: dict, delay_days, found_share: float, measure="kosten_alle"):
    """Kernfrage 4: Wie viel vom Gewinn des Orakels holt der Detektor, und wohin geht der Rest?"""
    row = summ[measure]
    gd, sd = row["gain_detector"]
    go, so = row["gain_oracle"]
    opp = C.METHOD_NAMES[row["opponent"]]
    share = reserve_share(gd, go)
    if go <= 0.05:
        return "info", f"Das Orakel spart gegen den stärksten Gegner ({opp}) nichts (**{_pct(go)} %**): hier gibt es nichts zu erkennen."
    text = f"Gegen den stärksten Gegner ({opp}) spart das Orakel **{_pct(go)} %**, der Detektor **{_pct(gd)} %** (± {_pct(sd)})"
    text += f", das sind {_pct(share * 100, 0)} % der Orakel-Reserve." if gd > 0 else ", das heißt: der Detektor holt nichts von der Orakel-Reserve."
    if "gain_oracle_delayed" in row and delay_days is not None and delay_days >= 1:
        gv = row["gain_oracle_delayed"][0]
        text += f" Ein Orakel, das den Sprung erst nach {delay_days} Tagen kennt (mittlere Verzögerung des Detektors), spart noch **{_pct(gv)} %**."
    if found_share < 0.5:
        text += f" Der Detektor meldet die Änderung nur in {_pct(found_share * 100, 0)} % der Reihen."
    if gd >= 2 * sd and gd > 0.5:
        return "success", text
    if gd < -2 * sd:
        return "warning", text + " Das Zurücksetzen ist hier schlechter als der Gegner."
    return "info", text


def verdict_regression_share(summ: dict, measure="kosten_alle"):
    """Wie viel des Vorsprungs gegen die Glättung kommt von der Regression, wie viel vom Detektor?"""
    row = summ[measure]
    total, _ = row["gain_total"]
    reg, _ = row["gain_regression"]
    det = total - reg
    if total <= 0.05:
        return "info", f"Die Regression mit Zurücksetzen liegt hier nicht vor der Glättung (Gesamtvorsprung **{_pct(total)} %**)."
    return "info", (f"Gegen die feste Glättung liegt die Regression mit Zurücksetzen um **{_pct(total)} %** vorn; **{_pct(reg)} Prozentpunkte** davon bringt schon die Regression mit dem "
                    f"stärksten Nachführer, **{_pct(det)} Prozentpunkte** bleiben dem Zurücksetzen.")


def with_settings(s: Settings, **kw) -> Settings:
    return replace(s, **kw)


def verdict_kennlinie(rows, runs=C.KENNLINIE_RUNS):
    """Kernfrage 1: Welche Verzögerung kostet der Betriebspunkt "höchstens eine Fehlmeldung je Jahr"? rows aus `kennlinie` (aufsteigende Schwellen)."""
    ok = [r for r in rows if r["fehlmeldungen"] <= 1.0]
    if not ok:
        r = rows[-1]
        return "warning", (f"Auch bei der größten Schwelle (h = {r['h']:g}) gibt es {_pct(r['fehlmeldungen'], 2)} Fehlmeldungen je Jahr (Mittel über {runs} Reihen ohne Sprung): "
                           "der Betriebspunkt \"höchstens eine Fehlmeldung je Jahr\" ist mit dieser Bereinigung nicht erreichbar.")
    r = ok[0]
    if r["verzoegerung"] is None:
        return "warning", f"Ab h = {r['h']:g} gibt es höchstens eine Fehlmeldung je Jahr ({_pct(r['fehlmeldungen'], 2)}), aber der Sprung wird in keiner der {runs} Reihen gemeldet."
    text = (f"Höchstens eine Fehlmeldung je Jahr ({_pct(r['fehlmeldungen'], 2)} im Mittel über {runs} Reihen ohne Sprung) gibt es ab **h = {r['h']:g}**; der Sprung wird dann in "
            f"**{_pct(r['gefunden'] * 100, 0)} %** der Reihen nach im Mittel **{_pct(r['verzoegerung'])} Tagen** gemeldet.")
    return ("success" if r["gefunden"] >= 0.9 else "info"), text


def karte_cell_label(zellen, noise, shift, ar):
    """Nächstgelegene Zelle der Karte zur Einstellung: (Schlüssel, genau?)."""
    n = min(C.KARTE_NOISE, key=lambda v: abs(v - noise))
    sh = min(C.KARTE_SHIFT, key=lambda v: abs(v - shift))
    a = min(C.KARTE_AR, key=lambda v: abs(v - ar))
    return f"{n}|{sh}|{a}", (n == noise and sh == shift and a == ar)


def karte_counts(zellen, ar, objective="kosten", measure="kosten_alle"):
    """Zählung über die Karte (eine Autokorrelation): (Zellen mit Gewinn über zwei Standardfehlern und mindestens 0,5 Punkten, Zellen mit Verlust unter zwei Standardfehlern, alle Zellen)."""
    plus = minus = total = 0
    for n in C.KARTE_NOISE:
        for sh in C.KARTE_SHIFT:
            g, se = zellen[f"{n}|{sh}|{ar}"]["messung"][objective]["gewinn_detektor"][measure]
            total += 1
            plus += int(g > 2 * se and g >= 0.5)
            minus += int(g < -2 * se and g <= -0.5)
    return plus, minus, total


def verdict_karte(zellen, noise, shift, ar):
    """Kernfrage 2: Lohnt Erkennen an der gewählten Stelle der Karte? Aussage über die nächstgelegene Zelle (Stufe, tau = 0,95, L = 3, empirische Regel)."""
    key, exact = karte_cell_label(zellen, noise, shift, ar)
    n, sh, a = key.split("|")
    m = zellen[key]["messung"]["kosten"]
    g, se = m["gewinn_detektor"]["kosten_alle"]
    gw, sew = m["gewinn_detektor"]["kosten_fenster"]
    orc = m["gewinn_orakel"]["kosten_alle"][0]
    where = "In dieser Zelle" if exact else f"In der nächstgelegenen Zelle (Rauschen {n} %, Sprung +{sh} %, Autokorrelation {_pct(float(a), 1)})"
    text = (f"{where} spart das Zurücksetzen gegen den stärksten abgestimmten Gegner im Testjahr **{_pct(g)} %** der Bestandskosten (± {_pct(se)}), im Sprungfenster {_pct(gw)} % (± {_pct(sew)}); "
            f"das Orakel würde im Testjahr {_pct(orc)} % sparen.")
    if g > 2 * se and g >= 0.5:
        return "success", text
    if g < -2 * se and g <= -0.5:
        return "warning", text + " Hier kostet das Zurücksetzen nachweisbar mehr, als es spart."
    return "info", text + " Das ist innerhalb von zwei Standardfehlern null oder unter 0,5 Prozentpunkten: hier ist kein Gewinn nachweisbar."


def verdict_confirm(c, rows):
    """Kernfrage 3: Was kostet das Trennen von Aktion und Sprung? rows aus `confirm_experiment`."""
    by = {r["form"]: r for r in rows}
    promo, step = by["promo"], by["step"]
    if step["mit"]["verzoegerung"] is None or step["ohne"]["verzoegerung"] is None:
        return "warning", "Der Sprung wird hier in keiner Reihe gemeldet; die Bestätigung lässt sich nicht beurteilen."
    extra = step["mit"]["verzoegerung"] - step["ohne"]["verzoegerung"]
    text = (f"Die Aktionswoche wird ohne Bestätigung in **{_pct(promo['ohne']['anteil'] * 100, 0)} %** der Reihen als Sprung gemeldet, mit Bestätigung nach {c} Tagen in "
            f"**{_pct(promo['mit']['anteil'] * 100, 0)} %**. Der dauerhafte Sprung wird dann in {_pct(step['mit']['anteil'] * 100, 0)} % der Reihen gemeldet "
            f"statt in {_pct(step['ohne']['anteil'] * 100, 0)} %, im Mittel **{_pct(extra)} Tage** später.")
    return ("success" if promo["mit"]["anteil"] < promo["ohne"]["anteil"] / 2 else "info"), text
