"""Messläufe für die README und die Karte (große Stichproben): Standardszenario mit allen Verfahren, Bestätigungsregel gegen die Formverwechslung, Kennlinie der Schwelle je
Bereinigungsmodell, Kostenannahmen. Aufruf über `tools/messlauf.py`; die Ergebnisse liegen in `data/*.json`, die Tests prüfen jede README-Zahl dagegen und rechnen kleine Stichproben neu.

Zähleinheiten: MASE je Reihe (Horizont 1 bis 14 Tage, Ursprünge Tag 730 bis 1081, Nenner: saisonal naiver Fehler auf den Tagen 7 bis 729); Kosten je Tag in Tagesbedarfen (Lager 1 je Stück und Tag, Rückstand p = tau / (1 - tau));
Fenster = 60 Tage ab dem Sprung; Meldung = Zurücksetzen des Niveaus nach einem CUSUM-Alarm (mit Bestätigung: der bestätigte Alarm)."""

import numpy as np

import nvs_algorithm as A
import nvs_constants as C
import nvs_evaluation as ev
import nvs_inventory as INV
import nvs_karte as K
import nvs_scenario as S
from nvs_scenario import T0


# --------------------------------------------------------------------------------------------------------------------------------------------
def head(best, seeds=C.HEAD_SEEDS, noise=14, shift=30, lead=C.DEFAULT_LEAD, tau=C.DEFAULT_TAU, model="calendar"):
    """Standardszenario: alle Verfahren nebeneinander (Gegner und Detektor je Ziel MASE / Kosten abgestimmt, `best` aus nvs_karte.choose), beide Regeln des Sicherheitsbestands.
    Rückgabe: {Regel: {Verfahren: {Maß: (Mittel, SE)}}} und gepaarte Gewinne des CUSUM-Zurücksetzens (Ziel Kosten) und des Orakels gegen jedes Verfahren."""
    batch = S.batch(tuple(seeds), "step", shift / 100.0, noise / 100.0, 0.0)
    P = ev.prepare(batch, model)
    cell = K.Cell(batch, model)
    org = INV.ORG_ALL
    F = {"snaive": A.seasonal_naive_forecasts(batch.y, org), "wmean": A.weekly_mean_forecasts(batch.y, org), "erwartung": A.expectation_forecasts(batch.mu, org)}
    F["hw"] = cell.forecasts("hw_fest", {"alpha": C.HW_ALPHA, "gamma": C.HW_GAMMA})[0]
    F["none"] = cell.forecasts("ewma", {"alpha": 0.0})[0]
    for obj in ("mase", "kosten"):
        b = best[obj]
        for fam in K.FAMILIES:
            F[f"{fam}[{obj}]"] = cell.forecasts(fam, b["abgestimmt"][fam][0])[0]
        F[f"cusum[{obj}]"] = cell.forecasts("cusum", b["detektor_params"])[0]
        F[f"oracle[{obj}]"] = cell.forecasts("oracle", b["orakel_params"])[0]
    tr = A.track(P.R, "cusum", best["kosten"]["detektor_params"])
    found, delay = ev.first_meldung(tr["meldung"], batch.J, C.WINDOW_SHORT)
    dd = int(round(float(delay[found].mean())))
    trd = A.track(P.R, "oracle", best["kosten"]["orakel_params"], jumps=A.oracle_jumps(batch.J, batch.jumps, batch.has_jump, delay=dd))
    F["oracle_delayed"] = A.forecasts_from_offsets(P.fit, trd["O"], P.s2, org)
    out = {"parameter": {o: {f: v[0] for f, v in best[o]["abgestimmt"].items()} for o in ("mase", "kosten")}, "n": len(seeds), "verzoegerung_tage": dd, "gefunden": float(found.mean()),
           "verzoegerung_mittel": float(delay[found].mean())}
    for rule in C.RULES:
        met = ev.evaluate_forecasts(batch, P, F, lead, rule, tau, batch.window)
        out[rule] = {m: {k: ev.mean_se(v) for k, v in d.items()} for m, d in met.items()}
        out[rule + "_gewinn"] = {}
        for ref in ("cusum[kosten]", "oracle[kosten]", "cusum[mase]", "oracle[mase]", "oracle_delayed"):
            out[rule + "_gewinn"][ref] = {m: {k: ev.paired_gain(d[k], met[ref][k]) for k in d} for m, d in met.items()}
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------
CONFIRM_FORMS = {
    "Stufe +28 %": ("step", 0.28), "Stufe +14 %": ("step", 0.14), "Aktionswoche +25 %": ("promo", 0.25), "Aktionswoche +50 %": ("promo", 0.50),
    "Ausreißer +100 %": ("spike", 1.0), "Rampe auf +28 %": ("ramp", 0.28), "keine Form": ("step", 0.0),
}


def confirm(det_params, seeds=C.CONFIRM_SEEDS, cs=(0, 3, 7, 10, 14, 21), thetas=(1.0, 1.5), noise=14, model="calendar"):
    """Je Form, Wartezeit c und Schwelle theta: Anteil der Reihen mit Meldung im Fenster [J, J + 60 + c), mittlere Verzögerung ab Beginn der Form; bei "keine Form" die Meldungen je Jahr (Testjahr)."""
    out = {}
    for name, (form, shift) in CONFIRM_FORMS.items():
        batch = S.batch(tuple(seeds), form, shift, noise / 100.0, 0.0)
        P = ev.prepare(batch, model)
        for theta in thetas:
            for c in cs:
                if c == 0 and theta != thetas[0]:
                    continue
                tr = A.track(P.R, "cusum", {**det_params, "confirm": c, "theta": theta})
                if shift == 0.0:
                    n = tr["meldung"][:, T0:].sum(axis=1).astype(float)
                    out[f"{name}|{theta}|{c}"] = {"meldungen_je_jahr": ev.mean_se(n)}
                else:
                    found, delay = ev.first_meldung(tr["meldung"], batch.J, C.WINDOW_SHORT, c)
                    out[f"{name}|{theta}|{c}"] = {"anteil": float(found.mean()), "verzoegerung": float(delay[found].mean()) if found.any() else None,
                                                  "verzoegerung_se": float(delay[found].std(ddof=1) / np.sqrt(found.sum())) if found.sum() > 1 else None}
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------
def kennlinie_modelle(det_params, seeds=C.CONFIRM_SEEDS, models=C.MODELS, hs=(3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0), shifts=(0.07, 0.14, 0.28, 0.42), noise=14):
    """Je Bereinigungsmodell und Schwelle h: Meldungen je Jahr auf Reihen ohne Sprung, Anteil gefundener Sprünge (Stufe, 60 Tage) und mittlere Verzögerung je Sprunggröße."""
    out = {}
    for model in models:
        nulls = ev.prepare(S.batch(tuple(seeds), "step", 0.0, noise / 100.0, 0.0), model)
        jumped = {sh: ev.prepare(S.batch(tuple(seeds), "step", sh, noise / 100.0, 0.0), model) for sh in shifts}
        for h in hs:
            p = {**det_params, "h": h, "confirm": 0}
            row = {"fehlmeldungen": ev.mean_se(A.track(nulls.R, "cusum", p)["meldung"][:, T0:].sum(axis=1).astype(float))}
            for sh, P in jumped.items():
                found, delay = ev.first_meldung(A.track(P.R, "cusum", p)["meldung"], P.batch.J, C.WINDOW_SHORT)
                row[f"sprung_{sh}"] = {"anteil": float(found.mean()), "verzoegerung": float(delay[found].mean()) if found.any() else None}
            out[f"{model}|{h}"] = row
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------
def confirm_ruhig(det_params, seeds=C.CONFIRM_SEEDS):
    """Bestätigung bei Rauschen 4 % (mit der Abstimmung der Zelle Rauschen 4 %, +30 %): Anteil gemeldeter Aktionswochen und Stufen je Wartezeit und theta."""
    return confirm(det_params, seeds=seeds, cs=(0, 10, 14, 21), thetas=(1.0, 1.5), noise=4)


def kostenannahmen(best, noise, seeds=C.MEASURE_SEEDS, taus=(0.8, 0.95, 0.99), leads=(0, 3, 7), shift=30):
    """Kostenannahmen als Regler: Gewinn des Detektors gegen den Gegner (abgestimmt für tau = 0,95 und L = 3, nicht neu abgestimmt) je tau und Wiederbeschaffungszeit L."""
    batch = S.batch(tuple(seeds), "step", shift / 100.0, noise / 100.0, 0.0)
    cell = K.Cell(batch)
    b = best["kosten"]
    F = {"gegner": cell.forecasts(b["gegner"], b["gegner_params"])[0], "detektor": cell.forecasts("cusum", b["detektor_params"])[0], "orakel": cell.forecasts("oracle", b["orakel_params"])[0]}
    out = {"gegner_familie": b["gegner"]}
    for tau in taus:
        for lead in leads:
            met = ev.evaluate_forecasts(batch, cell.P, F, lead, "empirical", tau, batch.window)
            out[f"{tau}|{lead}"] = {k: {"gegner": ev.mean_se(met["gegner"][k]), "detektor": ev.mean_se(met["detektor"][k]), "orakel": ev.mean_se(met["orakel"][k]),
                                        "gewinn_detektor": ev.paired_gain(met["gegner"][k], met["detektor"][k]), "gewinn_orakel": ev.paired_gain(met["gegner"][k], met["orakel"][k])}
                                    for k in ("kosten_alle", "kosten_fenster")}
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------
def nachfuehrung(det_params, seeds=C.CONFIRM_SEEDS, alphas=(0.0, 0.03, 0.06), noise=14, model="calendar", hs_gleich=(6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)):
    """Wie die langsame Nachführung zwischen den Meldungen kleine und langsame Änderungen verschluckt: Anteil gemeldeter Sprünge (Fenster 60 Tage) und mittlere Verzögerung je alpha, ohne Bestätigung,
    bei gleicher Schwelle h. Dazu die Meldungen je Jahr auf Reihen ohne Sprung je alpha (bei gleichem h sehr verschieden) und ein Vergleich bei gleicher Meldungsrate: das kleinste h des Rasters, mit dem
    alpha = 0 höchstens so viele Meldungen je Jahr macht wie der Standard-alpha (der größte der alphas) bei der Schwelle der Einstellung."""
    forms = {"Stufe +14 %": ("step", 0.14), "Rampe auf +28 %": ("ramp", 0.28), "Stufe +28 %": ("step", 0.28)}
    prepared = {name: ev.prepare(S.batch(tuple(seeds), form, shift, noise / 100.0, 0.0), model) for name, (form, shift) in forms.items()}
    null = ev.prepare(S.batch(tuple(seeds), "step", 0.0, noise / 100.0, 0.0), model)

    def run(alpha, h):
        p = {**det_params, "alpha": alpha, "h": h, "confirm": 0}
        row = {"fehlmeldungen": float(A.track(null.R, "cusum", p)["meldung"][:, T0:].sum(axis=1).mean())}
        for name, P in prepared.items():
            found, delay = ev.first_meldung(A.track(P.R, "cusum", p)["meldung"], P.batch.J, C.WINDOW_SHORT)
            row[name] = {"anteil": float(found.mean()), "verzoegerung": float(delay[found].mean()) if found.any() else None}
        return row

    out = {}
    for al in alphas:
        r = run(al, det_params["h"])
        out[f"fehlmeldungen|{al}"] = r["fehlmeldungen"]
        for name in forms:
            out[f"{name}|{al}"] = r[name]
    ziel = out[f"fehlmeldungen|{max(alphas)}"]
    for h in hs_gleich:
        r = run(0.0, h)
        if r["fehlmeldungen"] <= ziel:
            out["gleiche_rate"] = {"h": h, "ziel_fehlmeldungen": ziel, **r}
            break
    return out


def zweiter_sprung(det_params, noise, seeds=C.MEASURE_SEEDS, model="calendar"):
    """Wird der zweite Sprung (+20 % nach +30 %, 120 Tage später) gemeldet? Anteil der Reihen mit Meldung in den 60 Tagen ab dem zweiten Sprung, nur Reihen, in denen dieses Fenster vor dem Reihenende
    liegt, gegen den Einzelsprung +20 % allein (Meldung in den 60 Tagen ab dem Sprung, gleiche Reihen)."""
    two = S.batch(tuple(seeds), "step_two", 0.30, noise / 100.0, 0.0)
    one = S.batch(tuple(seeds), "step", 0.20, noise / 100.0, 0.0)
    P2, P1 = ev.prepare(two, model), ev.prepare(one, model)
    p = {**det_params, "confirm": 0}
    m2 = A.track(P2.R, "cusum", p)["meldung"]
    m1 = A.track(P1.R, "cusum", p)["meldung"]
    voll = two.J + C.SECOND_AFTER + C.WINDOW_SHORT <= C.N_DAYS
    t = np.arange(C.N_DAYS)[None, :]
    in2 = (m2 & (t >= (two.J + C.SECOND_AFTER)[:, None]) & (t < (two.J + C.SECOND_AFTER + C.WINDOW_SHORT)[:, None])).any(axis=1)
    in1 = (m1 & (t >= one.J[:, None]) & (t < (one.J + C.WINDOW_SHORT)[:, None])).any(axis=1)
    return {"reihen_mit_vollem_fenster": int(voll.sum()), "zweiter_sprung_gemeldet": float(in2[voll].mean()), "einzelsprung_20_gemeldet": float(in1[voll].mean())}


def laute_reihe(best_by_shift, seeds=C.CONFIRM_SEEDS, noise=24, shifts=(0.30, 0.50), model="calendar"):
    """Rauschen 24 %: wird der Sprung gemeldet? Mit den Standardwerten des Detektors (Einstellung der App) und mit den auf Kosten abgestimmten Werten der Zelle: Anteil der Reihen mit Meldung in den 60 Tagen
    ab dem Sprung, Verzögerung und Meldungen je Jahr auf Reihen ohne Sprung. best_by_shift: {Sprung in Prozent: Detektorparameter der Zelle}."""
    std = {"alpha": C.DEFAULT_ALPHA_LOW, "h": C.DEFAULT_H, "k": C.DEFAULT_K, "wmin": C.DEFAULT_WMIN}
    null = ev.prepare(S.batch(tuple(seeds), "step", 0.0, noise / 100.0, 0.0), model)
    out = {}
    for sh in shifts:
        P = ev.prepare(S.batch(tuple(seeds), "step", sh, noise / 100.0, 0.0), model)
        for name, p in (("standard", std), ("abgestimmt", best_by_shift[str(int(round(sh * 100)))])):
            p = {**p, "confirm": 0}
            found, delay = ev.first_meldung(A.track(P.R, "cusum", p)["meldung"], P.batch.J, C.WINDOW_SHORT)
            fehl = float(A.track(null.R, "cusum", p)["meldung"][:, T0:].sum(axis=1).mean())
            out[f"{int(round(sh * 100))}|{name}"] = {"parameter": p, "anteil": float(found.mean()), "verzoegerung": float(delay[found].mean()) if found.any() else None, "fehlmeldungen_je_jahr": fehl}
    return out




def autokorrelation_fehlmeldungen(det_params, seeds=C.CONFIRM_SEEDS, noise=14, ars=(0.0, 0.6), model="calendar"):
    """Meldungen je Jahr auf Reihen ohne Sprung bei Autokorrelation 0 und 0,6, mit denselben Detektorwerten (kein Neuabstimmen): trennt die Wirkung der Autokorrelation auf den Detektor von der Abstimmung."""
    out = {}
    for ar in ars:
        P = ev.prepare(S.batch(tuple(seeds), "step", 0.0, noise / 100.0, ar), model)
        n = A.track(P.R, "cusum", {**det_params, "confirm": 0})["meldung"][:, T0:].sum(axis=1).astype(float)
        out[str(ar)] = {"meldungen_je_jahr": ev.mean_se(n)}
    return out
