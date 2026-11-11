"""Karte "Wo lohnt Erkennen?": je Zelle (Rauschen x Sprung x Autokorrelation) werden alle Verfahren auf getrennten Abstimm-Reihen abgestimmt (je Ziel MASE und Bestandskosten), der stärkste
Gegner gewählt und auf den Messreihen mit dem CUSUM-Zurücksetzen und dem Orakel verglichen. Alle Verfahren teilen dieselbe Regression und unterscheiden sich nur in der Niveau-Nachführung der Residuen
(Ausnahme: die feste Glättung). Das Ergebnis liegt in `data/karte.json` (erzeugt von `tools/messlauf.py`); die App liest nur diese Datei. Bestandsannahmen der Karte: tau = 0,95, Wiederbeschaffungszeit 3, Regel empirisch."""

import itertools
import json
from pathlib import Path

import numpy as np

import nvs_algorithm as A
import nvs_constants as C
import nvs_evaluation as ev
import nvs_inventory as INV
import nvs_scenario as S

DATA = Path(__file__).resolve().parent / "data"
LEAD, TAU, RULE = C.DEFAULT_LEAD, C.DEFAULT_TAU, "empirical"
FAMILIES = ("ewma", "ma", "trigg", "hw_fest")
GRID = {
    "ewma": [{"alpha": a} for a in (0.03, 0.06, 0.1, 0.15, 0.25, 0.4)],
    "ma": [{"k": k} for k in (7, 14, 21, 28, 42, 56)],
    "trigg": [{"lam": lam, "amin": amin, "amax": C.TRIGG_AMAX} for lam in (0.05, 0.1, 0.2) for amin in (0.03, 0.08)],
    "hw_fest": [{"alpha": a, "gamma": C.HW_GAMMA} for a in (0.03, 0.06, 0.1, 0.15, 0.25, 0.4)],
    "cusum": [{"alpha": a, "h": h, "k": k, "wmin": w} for a, h, k, w in itertools.product((0.03, 0.06, 0.1), (4, 6, 8, 12), (0.5, 1.0), (2, 5))],
    "oracle": [{"alpha": a} for a in (0.03, 0.06, 0.1, 0.15)],
}
KEYS = ("mase_alle", "mase_fenster", "kosten_alle", "kosten_fenster")


class Cell:
    """Reihen einer Zelle mit Regression; Prognosen je Familie und Parameter."""

    def __init__(self, batch, model="calendar", win=None):
        self.batch = batch
        self.P = ev.prepare(batch, model)
        self.win = batch.window if win is None else win
        self._hw = {}

    def forecasts(self, fam, p):
        b, P = self.batch, self.P
        if fam == "hw_fest":
            key = p["alpha"]
            if key not in self._hw:
                self._hw[key] = A.hw_forecasts(*A.holt_winters(b.y, p["alpha"], p["gamma"]), INV.ORG_ALL)
            return self._hw[key], None
        kind = "cusum" if fam == "cusum" else fam
        tr = A.track(P.R, kind, p, jumps=A.oracle_jumps(b.J, b.jumps, b.has_jump) if fam == "oracle" else None)
        return A.forecasts_from_offsets(P.fit, tr["O"], P.s2, INV.ORG_ALL), tr

    def evaluate(self, fam, p):
        F, tr = self.forecasts(fam, p)
        m = ev.evaluate_forecasts(self.batch, self.P, {"x": F}, LEAD, RULE, TAU, self.win)["x"]
        if tr is not None and fam == "cusum":
            m["meldungen"] = tr["meldung"][:, C.FIRST_TEST:].sum(axis=1).astype(float)
        return m


def _mixed(shift, noise, ar, form, seeds):
    half = len(seeds) // 2
    return S.concat(S.batch(seeds[:half], form, shift, noise, ar), S.batch(seeds[half:], form, 0.0, noise, ar))


def tune(noise, shift, ar, form="step", seeds=C.TUNE_SEEDS):
    """Abstimmung je Familie und Ziel auf Reihen, die zur Hälfte den Sprung enthalten: {Familie: [(Parameter, mase_alle_mittel, kosten_alle_mittel), ...]}."""
    cell = Cell(_mixed(shift, noise, ar, form, tuple(seeds)))
    out = {}
    for fam in FAMILIES + ("cusum", "oracle"):
        out[fam] = []
        for p in GRID[fam]:
            m = cell.evaluate(fam, p)
            out[fam].append((p, float(m["mase_alle"].mean()), float(m["kosten_alle"].mean())))
    return out


def choose(tuned):
    """Je Ziel: bester Parameter jeder Familie, stärkster Gegner (Familie mit dem kleinsten abgestimmten Zielwert), Parameter von Detektor und Orakel."""
    best = {}
    for obj, idx in (("mase", 1), ("kosten", 2)):
        bf = {fam: min(rows, key=lambda r: r[idx]) for fam, rows in tuned.items()}
        opp = min(FAMILIES, key=lambda f: bf[f][idx])
        best[obj] = {"gegner": opp, "gegner_params": bf[opp][0], "detektor_params": bf["cusum"][0], "orakel_params": bf["oracle"][0],
                     "abgestimmt": {f: [bf[f][0], bf[f][idx]] for f in bf}}
    return best


def hinter_ende(batch, win):
    """Anteil der Reihen, in denen der letzte Sprung der Folge hinter dem Reihenende liegt (Tag J + Abstand >= 1095), und Anteil, in dem das Fenster [J, J + win) über das Reihenende hinausragt."""
    last = max((off for off, _ in batch.jumps), default=0)
    return {"letzter_sprung": float((batch.J + last >= C.N_DAYS).mean()), "fenster_verkuerzt": float((batch.J + win > C.N_DAYS).mean())}


def measure(noise, shift, ar, best, form="step", seeds=C.MEASURE_SEEDS, win=None):
    """Messung auf den Messreihen mit Sprung: je Ziel Gewinn von Detektor und Orakel gegen den gewählten Gegner (Prozent des Gegners, Standardfehler in Prozent, gepaart).
    win: Fenster der Gewinne in Tagen (Standard: 60, bei Folgen von Sprüngen 180)."""
    cell = Cell(S.batch(tuple(seeds), form, shift, noise, ar), win=win)
    res = {"fenster_tage": cell.win, "hinter_ende": hinter_ende(cell.batch, cell.win)}
    for obj in ("mase", "kosten"):
        b = best[obj]
        opp = cell.evaluate(b["gegner"], b["gegner_params"])
        det = cell.evaluate("cusum", b["detektor_params"])
        orc = cell.evaluate("oracle", b["orakel_params"])
        res[obj] = {"meldungen_je_reihe": float(det["meldungen"].mean()),
                    "gegner": {k: ev.mean_se(opp[k]) for k in KEYS}, "detektor": {k: ev.mean_se(det[k]) for k in KEYS}, "orakel": {k: ev.mean_se(orc[k]) for k in KEYS},
                    "gewinn_detektor": {k: ev.paired_gain(opp[k], det[k]) for k in KEYS}, "gewinn_orakel": {k: ev.paired_gain(opp[k], orc[k]) for k in KEYS}}
    return res


def karte_zelle(noise, shift, ar, form="step", tuned=None, tune_seeds=C.TUNE_SEEDS, measure_seeds=C.MEASURE_SEEDS, win=None):
    """Eine Zelle (Rauschen und Sprung in Prozent). Mit `tuned` wird nicht neu abgestimmt (Folgen von Sprüngen nutzen die Abstimmung des Einzelsprungs)."""
    if tuned is None:
        tuned = tune(noise / 100.0, shift / 100.0, ar, "step", tune_seeds)
    best = choose(tuned)
    m = measure(noise / 100.0, shift / 100.0, ar, best, form, measure_seeds, win)
    return {"noise": noise, "shift": shift, "ar": ar, "form": form, "best": best, "messung": m}, tuned


def _job(args):
    noise, shift, ar, tune_seeds, measure_seeds = args
    r, _ = karte_zelle(noise, shift, ar, tune_seeds=tune_seeds, measure_seeds=measure_seeds)
    return f"{noise}|{shift}|{ar}", r


def _job_multi(args):
    """Folgen von Sprüngen: Abstimmung des Einzelsprungs, in allen drei Zeilen dasselbe Fenster (180 Tage), damit die Zeilen vergleichbar sind."""
    noise, tune_seeds, measure_seeds = args
    _, tuned = karte_zelle(noise, 30, 0.0, tune_seeds=tune_seeds, measure_seeds=tuple(measure_seeds)[:20])      # nur die Abstimmung
    out = {}
    for form in ("step", "step_back", "step_two"):
        r, _ = karte_zelle(noise, 30, 0.0, form=form, tuned=tuned, measure_seeds=measure_seeds, win=C.WINDOW_LONG)
        out[form] = r
    return noise, out


def build(path=DATA / "karte.json", workers=8, tune_seeds=C.TUNE_SEEDS, measure_seeds=C.MEASURE_SEEDS, multi=True):
    """Karte und Folgen von Sprüngen berechnen und als JSON schreiben (mehrere Minuten). Andere Seeds: Wiederholung (`tools/messlauf.py wiederholung`)."""
    from concurrent.futures import ProcessPoolExecutor
    cells = [(n, s, a, tuple(tune_seeds), tuple(measure_seeds)) for n in C.KARTE_NOISE for s in C.KARTE_SHIFT for a in C.KARTE_AR]
    with ProcessPoolExecutor(workers) as ex:
        zellen = dict(ex.map(_job, cells))
        multi_res = dict(ex.map(_job_multi, [(n, tuple(tune_seeds), tuple(measure_seeds)) for n in C.KARTE_NOISE])) if multi else {}
    DATA.mkdir(exist_ok=True)
    out = {"zellen": zellen, "multi": {str(k): v for k, v in multi_res.items()}, "annahmen": {"tau": TAU, "lead": LEAD, "regel": RULE, "messreihen": len(measure_seeds), "abstimmreihen": len(tune_seeds),
                                                                                                "abstimm_seeds": [min(tune_seeds), max(tune_seeds)], "mess_seeds": [min(measure_seeds), max(measure_seeds)]}}
    Path(path).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return out


def load(path=DATA / "karte.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))
