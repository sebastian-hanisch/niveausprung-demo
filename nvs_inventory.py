"""Bestandsentscheidung nach der Prognose (Regeln und Fortschreibung wie in forecast-inventory-demo, hier für Prognosen mit Offsets).

Ablauf eines Tages t: Wareneingang der Bestellung von Tag t - L; bestellt wird bis zum Bestellbestand S_t (Order-up-to, bei L = 0 kommt die Bestellung sofort);
danach die Nachfrage y_t, Fehlmengen werden nachgeliefert (Rückstand). Am Tagesende kostet Lagerbestand 1 je Stück, Rückstand p = tau / (1 - tau) je Stück.
Der Bestellbestand S_t deckt das Schutzintervall von m = L + 1 Tagen: S = (F_m + 1) exp(q) - 1 mit F_m = Summe der Punktprognosen der Horizonte 1 bis m und q = tau-Quantil der
Fehler s = log((D_m + 1) / (F_m + 1)); D_m ist die tatsächliche Nachfrage. Regel "empirical": q aus den Trainingsfehlern (Ursprünge mit Zieltagen vor dem Testjahr);
Regel "conformal": q aus den letzten W realisierten Fehlern (Rang ceil((k + 1) tau))."""

import numpy as np

import nvs_constants as C
from nvs_scenario import T, T0

ORG_ALL = np.arange(91, T)                  # Ursprünge, für die Prognosen und Fehler vorliegen


def simulate(S, y, lead, tau):
    """S: (B, N, Tage) Bestellbestand je Tag (B Regeln nebeneinander), y: (N, Tage) Nachfrage. Rückgabe on_hand, backorder, short (je (B, N, Tage))."""
    B, N, Tn = S.shape
    arr = np.zeros((B, N, Tn + lead + 1))
    NI = S[:, :, 0].copy()
    on_hand = np.zeros((B, N, Tn))
    backorder = np.zeros((B, N, Tn))
    short = np.zeros((B, N, Tn))
    for t in range(Tn):
        NI = NI + arr[:, :, t]
        pipeline = arr[:, :, t + 1:t + lead + 1].sum(axis=2) if lead > 0 else 0.0
        q = np.maximum(S[:, :, t] - (NI + pipeline), 0.0)
        if lead == 0:
            NI = NI + q
        else:
            arr[:, :, t + lead] += q
        d = y[:, t][None, :]
        short[:, :, t] = np.maximum(d - np.maximum(NI, 0.0), 0.0)
        NI = NI - d
        on_hand[:, :, t] = np.maximum(NI, 0.0)
        backorder[:, :, t] = np.maximum(-NI, 0.0)
    return {"on_hand": on_hand, "backorder": backorder, "short": short}


def interval_sum(F, lead):
    """Summe der Punktprognosen über das Schutzintervall: (N, O, h) -> (N, O)."""
    return F[:, :, :lead + 1].sum(axis=2)


def scores(y, Fm, lead):
    """Fehler s_o = log((D_m + 1) / (F_m + 1)) je Ursprung o (ORG_ALL) mit m = lead + 1; NaN, wenn die Zieltage über das Ende hinausgehen. (N, len(ORG_ALL))"""
    m = lead + 1
    cs = np.concatenate([np.zeros((y.shape[0], 1)), np.cumsum(y, axis=1)], axis=1)
    ok = ORG_ALL + m <= T
    D = np.full(Fm.shape, np.nan)
    D[:, ok] = cs[:, ORG_ALL[ok] + m] - cs[:, ORG_ALL[ok]]
    return np.log((D + 1.0) / (Fm + 1.0))


def stock_levels(Fm, Sc, lead, rule, tau, window=C.CONFORMAL_WINDOW):
    """Bestellbestand S_t für die Testtage 730 bis 1094: (N, 365)."""
    m = lead + 1
    t_test = np.arange(T0, T)
    i_test = t_test - ORG_ALL[0]
    Fm_t = Fm[:, i_test]
    if rule == "empirical":
        train = ORG_ALL + m <= T0
        q = np.quantile(Sc[:, train], tau, axis=1)[:, None]
    elif rule == "conformal":
        q = np.empty_like(Fm_t)
        for c, t in enumerate(t_test):
            last = t - m - ORG_ALL[0]                         # Index des letzten realisierten Ursprungs
            win = Sc[:, max(0, last - window + 1):last + 1]
            k = win.shape[1]
            rank = int(np.clip(np.ceil((k + 1) * tau), 1, k))
            q[:, c] = np.sort(win, axis=1)[:, rank - 1]
    else:
        raise ValueError(rule)
    return np.maximum((Fm_t + 1.0) * np.exp(q) - 1.0, 0.0)


def costs(S_stack, y, J, lead, tau, win):
    """Kosten je Tag in Tagesbedarfen (Mittel des Testjahres der Reihe) für jede Regel: ganzes Testjahr und Fenster [J, J + win). Rückgabe dict mit (B, N)-Feldern."""
    yt = y[:, T0:]
    r = simulate(S_stack, yt, lead, tau)
    p = tau / (1.0 - tau)
    cost = r["on_hand"] + p * r["backorder"]
    dm = yt.mean(axis=1)[None, :]
    day = np.arange(T0, T)[None, None, :]
    w = (day >= J[None, :, None]) & (day < J[None, :, None] + win)
    n_w = np.maximum(w.sum(axis=2), 1)
    return {"alle": cost.mean(axis=2) / dm, "fenster": (cost * w).sum(axis=2) / n_w / dm}
