"""Orakel Bestand: die Fortschreibung (Order-up-to, Wiederbeschaffungszeit L, Rückstand) gegen die geschlossene Formel bei konstantem Bestellbestand und gegen eine Tagesschleife je Reihe,
die Regeln des Sicherheitsbestands gegen Schleifen mit Sortieren von Hand, die Kosten gegen die Definition. Handrechnung: Nachfrage 10, S = 50, L = 3 gibt Lagerbestand 10 (50 - 4 * 10)."""

import math

import numpy as np
import pytest

import nvs_constants as C
import nvs_inventory as INV
import nvs_scenario as S
from nvs_scenario import T, T0


def day_loop(S_t, y, L, tau):
    """Eine Reihe, ein Bestellbestandsverlauf: Tagesschleife mit Liste der Lieferungen (nur Pythonzahlen). Rückgabe: Lagerbestand, Rückstand, Fehlmenge je Tag."""
    n = len(y)
    arrivals = [0.0] * (n + L + 2)
    ni = S_t[0]
    oh, bo, sh = [], [], []
    for t in range(n):
        ni += arrivals[t]
        pipeline = sum(arrivals[t + 1:t + L + 1])
        q = max(S_t[t] - (ni + pipeline), 0.0)
        if L == 0:
            ni += q
        else:
            arrivals[t + L] += q
        sh.append(max(y[t] - max(ni, 0.0), 0.0))
        ni -= y[t]
        oh.append(max(ni, 0.0))
        bo.append(max(-ni, 0.0))
    return oh, bo, sh


def test_handrechnung_konstante_nachfrage():
    y = np.full((1, 30), 10.0)
    r = INV.simulate(np.full((1, 1, 30), 50.0), y, 3, 0.95)
    assert r["on_hand"][0, 0, 10:].tolist() == [10.0] * 20 and not r["backorder"][0, 0].any() and not r["short"][0, 0].any()


def test_handrechnung_zu_knapper_bestand_gibt_rueckstand():
    y = np.full((1, 30), 10.0)
    r = INV.simulate(np.full((1, 1, 30), 20.0), y, 3, 0.95)                  # 20 - 4 * 10 = -20
    assert r["backorder"][0, 0, 10:].tolist() == [20.0] * 20 and not r["on_hand"][0, 0, 10:].any()


@pytest.mark.parametrize("lead", [0, 1, 3, 7])
def test_konstanter_bestand_folgt_der_geschlossenen_formel(lead):
    rng = np.random.default_rng(lead)
    y = rng.integers(1, 20, size=(1, 80)).astype(float)
    Sx = np.full((1, 1, 80), 90.0)
    r = INV.simulate(Sx, y, lead, 0.95)
    for t in range(lead, 80):
        ni = 90.0 - y[0, t - lead:t + 1].sum()                                # Nettobestand am Tagesende: S minus Nachfrage der Tage t - L bis t
        assert r["on_hand"][0, 0, t] == pytest.approx(max(ni, 0.0)) and r["backorder"][0, 0, t] == pytest.approx(max(-ni, 0.0)), (lead, t)


@pytest.mark.parametrize("lead", [0, 2, 3, 7])
def test_fortschreibung_stimmt_mit_tagesschleife_bei_wechselndem_bestand(lead):
    rng = np.random.default_rng(10 + lead)
    y = rng.integers(0, 30, size=(3, 120)).astype(float)
    Sx = rng.uniform(20, 120, size=(2, 3, 120))                              # zwei Regeln nebeneinander
    r = INV.simulate(Sx, y, lead, 0.9)
    for b in range(2):
        for i in range(3):
            oh, bo, sh = day_loop(Sx[b, i].tolist(), y[i].tolist(), lead, 0.9)
            assert r["on_hand"][b, i] == pytest.approx(oh) and r["backorder"][b, i] == pytest.approx(bo) and r["short"][b, i] == pytest.approx(sh)


def _quantile_by_hand(values, tau):
    v = sorted(values)
    pos = (len(v) - 1) * tau
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def _fm(y, lead):
    """Eine feste Summenprognose je Ursprung: 100 mal Schutzintervall mit sanfter Schwankung (nur um Scores zu erzeugen)."""
    m = lead + 1
    return 100.0 * m * (1.0 + 0.1 * np.sin(np.arange(len(INV.ORG_ALL)) / 17.0))[None, :] * np.ones((y.shape[0], 1))


@pytest.mark.parametrize("lead", [0, 3, 7])
def test_fehler_der_regeln_gegen_schleife(lead):
    b = S.batch((5, 6), "step", 0.3)
    Fm = _fm(b.y, lead)
    Sc = INV.scores(b.y, Fm, lead)
    m = lead + 1
    for i in range(2):
        for c, o in enumerate(INV.ORG_ALL):
            if o + m <= T:
                d = sum(b.y[i, o:o + m])
                assert Sc[i, c] == pytest.approx(math.log((d + 1.0) / (Fm[i, c] + 1.0)))
            else:
                assert math.isnan(Sc[i, c])


@pytest.mark.parametrize("lead,tau", [(3, 0.95), (0, 0.8), (7, 0.99)])
def test_empirische_regel_gegen_quantil_von_hand(lead, tau):
    b = S.batch((5, 6), "step", 0.3)
    Fm = _fm(b.y, lead)
    Sc = INV.scores(b.y, Fm, lead)
    St = INV.stock_levels(Fm, Sc, lead, "empirical", tau)
    m = lead + 1
    for i in range(2):
        train = [Sc[i, c] for c, o in enumerate(INV.ORG_ALL) if o + m <= T0]
        q = _quantile_by_hand(train, tau)
        for c, t in enumerate(range(T0, T)):
            assert St[i, c] == pytest.approx(max((Fm[i, t - INV.ORG_ALL[0]] + 1.0) * math.exp(q) - 1.0, 0.0))


@pytest.mark.parametrize("lead,tau", [(3, 0.95), (0, 0.8)])
def test_konforme_regel_gegen_schleife(lead, tau):
    b = S.batch((5,), "step", 0.3)
    Fm = _fm(b.y, lead)
    Sc = INV.scores(b.y, Fm, lead)
    St = INV.stock_levels(Fm, Sc, lead, "conformal", tau)
    m = lead + 1
    for c, t in enumerate(range(T0, T)):
        last = t - m - INV.ORG_ALL[0]                                        # letzter Ursprung, dessen Zieltage vor t liegen
        win = sorted(Sc[0, max(0, last - C.CONFORMAL_WINDOW + 1):last + 1])
        k = len(win)
        rank = min(max(math.ceil((k + 1) * tau), 1), k)
        q = win[rank - 1]
        assert St[0, c] == pytest.approx(max((Fm[0, t - INV.ORG_ALL[0]] + 1.0) * math.exp(q) - 1.0, 0.0))


def test_kosten_gegen_definition_mit_fenster():
    b = S.batch((5, 6), "step", 0.3)
    Fm = _fm(b.y, 3)
    St = INV.stock_levels(Fm, INV.scores(b.y, Fm, 3), 3, "empirical", 0.95)
    c = INV.costs(St[None], b.y, b.J, 3, 0.95, 60)
    p = 0.95 / 0.05
    for i in range(2):
        oh, bo, _ = day_loop(St[i].tolist(), b.y[i, T0:].tolist(), 3, 0.95)
        cost = [o + p * r for o, r in zip(oh, bo)]
        dm = b.y[i, T0:].mean()
        assert c["alle"][0, i] == pytest.approx(sum(cost) / len(cost) / dm)
        win = [cost[t - T0] for t in range(T0, T) if b.J[i] <= t < b.J[i] + 60]
        assert c["fenster"][0, i] == pytest.approx(sum(win) / len(win) / dm)


def test_rueckstandskosten_aus_dem_servicegrad():
    # kritisches Verhältnis p / (p + 1) = tau: p = tau / (1 - tau)
    for tau in C.TAU_OPTIONS:
        p = tau / (1 - tau)
        assert p / (p + 1.0) == pytest.approx(tau)
    assert round(0.95 / 0.05) == 19 and round(0.99 / 0.01) == 99 and round(0.8 / 0.2) == 4        # p = 19, 99, 4 wie in der README
