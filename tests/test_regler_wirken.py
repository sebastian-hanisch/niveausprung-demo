"""Keine toten Regler (DEMO-PLAYBOOK §3): jeder Regler der Seitenleiste ändert im Standardszenario die Kennzahlen seines Verfahrens, und die übrigen Verfahren bleiben unverändert (im Gewinn gegen den stärksten Gegner wirkt ein Gegner-Regler nur, solange dieser Gegner der stärkste ist)."""

import numpy as np
import pytest

import nvs_evaluation as ev

BASE = ev.Settings(runs=18)
ALLE = ("snaive", "wmean", "hw", "none", "ewma", "ma", "trigg", "detector", "oracle")
KOSTEN = ("kosten_alle", "kosten_fenster")
MASE = ("mase_alle", "mase_fenster")

# Regler -> (neuer Wert, Verfahren, deren Kennzahlen sich ändern müssen, Kennzahlen, die sich ändern müssen)
WIRKUNG = {
    "h": (3.0, ("detector",), KOSTEN + MASE),
    "k": (1.0, ("detector",), KOSTEN + MASE),
    "alpha_low": (0.2, ("detector",), KOSTEN + MASE),
    "wmin": (14, ("detector",), KOSTEN + MASE),
    "confirm_rule": ("c10", ("detector",), KOSTEN + MASE),
    "ewma": (0.3, ("ewma",), KOSTEN + MASE),
    "ma": (28, ("ma",), KOSTEN + MASE),
    "trigg_lam": (0.3, ("trigg",), KOSTEN + MASE),
    "trigg_amin": (0.2, ("trigg",), KOSTEN + MASE),
    "tau": (0.8, ALLE, KOSTEN),
    "lead": (7, ALLE, KOSTEN),
    "rule": ("conformal", ALLE, KOSTEN),
}


def _differs(a, b, methods, measures):
    return all(not np.allclose(a.metrics[m][ms], b.metrics[m][ms]) for m in methods for ms in measures)


def _same(a, b, methods, measures):
    return all(np.allclose(a.metrics[m][ms], b.metrics[m][ms]) for m in methods for ms in measures)


@pytest.fixture(scope="module")
def base():
    return ev.analyse(BASE)


@pytest.mark.parametrize("name", WIRKUNG)
def test_regler_wirkt_dort_wo_er_hingehoert(name, base):
    value, methods, measures = WIRKUNG[name]
    other = ev.analyse(ev.Settings(**{**BASE.__dict__, name: value}))
    assert _differs(base, other, methods, measures), name
    unaffected = [m for m in ALLE if m not in methods]
    assert _same(base, other, unaffected, measures + MASE), name              # alle anderen Verfahren bleiben bit-gleich


def test_bestaetigungsschwelle_theta_wirkt_in_der_auswahl():
    c10 = ev.analyse(ev.Settings(**{**BASE.__dict__, "confirm_rule": "c10"}))
    c10t15 = ev.analyse(ev.Settings(**{**BASE.__dict__, "confirm_rule": "c10t15"}))
    assert ev.Settings(confirm_rule="c10t15").theta == 1.5 and ev.Settings(confirm_rule="c10t15").confirm == 10
    assert _differs(c10, c10t15, ("detector",), KOSTEN + MASE)


@pytest.mark.parametrize("name,value", [("noise", 30), ("shift", 60), ("form", "promo"), ("ar", 0.6), ("model", "weekday"), ("seed", 50)])
def test_szenarioregler_aendern_die_regressionsverfahren(name, value, base):
    other = ev.analyse(ev.Settings(**{**BASE.__dict__, name: value}))
    assert _differs(base, other, ("ewma", "detector", "oracle"), KOSTEN + MASE), name


def test_zahl_der_reihen_aendert_den_umfang_und_den_standardfehler():
    a, b = ev.analyse(ev.Settings(runs=12)), ev.analyse(ev.Settings(runs=24))
    assert a.metrics["ewma"]["kosten_alle"].shape == (12,) and b.metrics["ewma"]["kosten_alle"].shape == (24,)
    assert ev.summary(a)["kosten_alle"]["means"]["ewma"][1] != ev.summary(b)["kosten_alle"]["means"]["ewma"][1]


def test_bereinigungsmodell_aendert_die_regression_aber_nicht_die_glaettung(base):
    other = ev.analyse(ev.Settings(**{**BASE.__dict__, "model": "weekday"}))
    assert _same(base, other, ("snaive", "wmean", "hw"), KOSTEN + MASE)
    assert _differs(base, other, ("none", "ewma", "ma", "trigg"), MASE)


def test_kennlinie_und_bestaetigungsversuch_reagieren_auf_ihre_regler():
    s = ev.Settings()
    k0 = ev.kennlinie(s, hs=(4.0, 8.0), runs=12)
    assert k0 != ev.kennlinie(ev.Settings(k=1.0), hs=(4.0, 8.0), runs=12)
    assert k0 != ev.kennlinie(ev.Settings(model="weekday"), hs=(4.0, 8.0), runs=12)
    c0 = ev.confirm_experiment(s, runs=12)
    assert c0 != ev.confirm_experiment(ev.Settings(confirm_rule="c14"), runs=12)
    assert c0 != ev.confirm_experiment(ev.Settings(shift=50), runs=12)
