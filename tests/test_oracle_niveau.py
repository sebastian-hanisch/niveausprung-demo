"""Orakel Nachführer: die vektorisierten Nachführer (EWMA, Mittel, Trigg-Leach, CUSUM mit und ohne Bestätigung, Orakel) gegen eine skalare Schleife je Reihe, die nur Pythonzahlen benutzt;
die Regression gegen die Normalgleichungen. Die Referenz ist aus der Beschreibung in `nvs_algorithm.py` neu geschrieben, nicht aus dem Code kopiert."""

import math

import numpy as np
import pytest

import nvs_algorithm as A
import nvs_constants as C
import nvs_evaluation as ev
import nvs_scenario as S

REF = (639, 730)


def _std(v):
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def scalar_track(r, kind, p, jump_days=(), ref=REF):
    """Eine Reihe r (Liste), Schritt für Schritt. Rückgabe: O, alarm, meldung, stat (Listen)."""
    n = len(r)
    o = 0.0
    O, alarm, meldung, stat = [], [False] * n, [False] * n, [0.0] * n
    E = []
    Es, Ms = 0.0, 1e-3
    hi = lo = 0.0
    st_hi = st_lo = 0
    pend = None                                    # (Alarmtag, Seite, Beginn)
    sig = r0 = s0 = None
    for t in range(n):
        e = r[t] - o
        E.append(e)
        if kind == "none":
            nxt = o
        elif kind in ("ewma", "cusum", "oracle"):
            nxt = o + p["alpha"] * e
        elif kind == "ma":
            lo_i = max(0, t + 1 - p["k"])
            nxt = sum(r[lo_i:t + 1]) / (t + 1 - lo_i)
        elif kind == "trigg":
            Es = p["lam"] * e + (1 - p["lam"]) * Es
            Ms = p["lam"] * abs(e) + (1 - p["lam"]) * Ms
            a = min(max(abs(Es) / max(Ms, 1e-9), p["amin"]), p["amax"])
            nxt = o + a * e
        if kind == "cusum":
            if t == ref[1] - 1:
                sig = max(_std(E[ref[0]:ref[1]]), 1e-9)
                r0 = sum(r[ref[0]:ref[1]]) / (ref[1] - ref[0])
                s0 = max(_std(r[ref[0]:ref[1]]), 1e-9)
            if t >= ref[1]:
                z = e / sig
                if hi == 0.0:
                    st_hi = t
                if lo == 0.0:
                    st_lo = t
                hi = max(0.0, hi + z - p["k"])
                lo = max(0.0, lo - z - p["k"])
                up, down = hi > p["h"], lo > p["h"]
                stat[t] = hi if hi >= lo else -lo
                if up or down:
                    alarm[t] = True
                    side, start = (1.0, st_hi) if up else (-1.0, st_lo)
                    if p.get("confirm", 0) == 0:
                        w = min(max(t - start + 1, p["wmin"]), t + 1)
                        nxt = sum(r[t + 1 - w:t + 1]) / w
                        meldung[t] = True
                    elif pend is None:
                        pend = (t, side, start)
                    hi = lo = 0.0
                if pend is not None and t == pend[0] + p.get("confirm", 0) and p.get("confirm", 0) > 0:
                    w7 = min(p["confirm"], 7)
                    seg = sum(r[t + 1 - w7:t + 1]) / w7
                    if pend[1] * (seg - r0) / s0 >= p.get("theta", 1.0):
                        w = min(max(t - pend[2] + 1, p["wmin"]), t + 1)
                        nxt = sum(r[t + 1 - w:t + 1]) / w
                        meldung[t] = True
                    pend = None
        if kind == "oracle":
            for day, dlog in jump_days:
                if day == t:
                    nxt += dlog
        o = nxt
        O.append(o)
    return O, alarm, meldung, stat


def _residuals(form, shift, seeds, noise=0.14, model="calendar"):
    b = S.batch(seeds, form, shift, noise, 0.0)
    return ev.prepare(b, model)


CASES = [
    ("ewma", {"alpha": 0.1}), ("ewma", {"alpha": 0.4}), ("ma", {"k": 14}), ("ma", {"k": 7}), ("trigg", {"lam": 0.05, "amin": 0.08, "amax": 0.5}), ("trigg", {"lam": 0.3, "amin": 0.02, "amax": 0.5}),
    ("none", {}),
    ("cusum", {"alpha": 0.06, "h": 6.0, "k": 0.5, "wmin": 5}),
    ("cusum", {"alpha": 0.03, "h": 2.0, "k": 0.25, "wmin": 1}),                                        # viele Alarme
    ("cusum", {"alpha": 0.06, "h": 6.0, "k": 0.5, "wmin": 5, "confirm": 10, "theta": 1.0}),
    ("cusum", {"alpha": 0.06, "h": 3.0, "k": 0.5, "wmin": 3, "confirm": 3, "theta": 0.5}),
    ("cusum", {"alpha": 0.1, "h": 4.0, "k": 1.0, "wmin": 14, "confirm": 21, "theta": 2.0}),
]


@pytest.mark.parametrize("kind,p", CASES)
def test_nachfuehrer_stimmt_mit_skalarer_schleife_ueberein(kind, p):
    P = _residuals("promo", 0.5, range(6), noise=0.14)
    out = A.track(P.R, kind, p)
    for i in range(P.R.shape[0]):
        O, alarm, meldung, stat = scalar_track(P.R[i].tolist(), kind, p)
        assert out["O"][i] == pytest.approx(O, abs=1e-12), (kind, p, i)
        assert out["alarm"][i].tolist() == alarm and out["meldung"][i].tolist() == meldung, (kind, p, i)
        assert out["stat"][i] == pytest.approx(stat, abs=1e-12)


def test_orakel_stimmt_mit_skalarer_schleife_ueberein():
    P = _residuals("step_back", 0.3, range(5))
    b = P.batch
    jumps = A.oracle_jumps(b.J, b.jumps, b.has_jump, delay=3)
    out = A.track(P.R, "oracle", {"alpha": 0.03}, jumps=jumps)
    for i in range(5):
        O, *_ = scalar_track(P.R[i].tolist(), "oracle", {"alpha": 0.03}, jump_days=[(int(d[i]), dl) for d, dl in jumps])
        assert out["O"][i] == pytest.approx(O, abs=1e-12)


def test_zweig_jede_variante_weicht_von_der_nachbarvariante_ab():
    # Zweig-Test (DEMO-PLAYBOOK §0): keine Variante ist ein toter Zweig; auf mindestens einer Reihe unterscheiden sich die Niveaus
    P = _residuals("promo", 0.5, range(40))                                  # genug Reihen, damit auch die Bestätigungsschwelle theta einen Unterschied macht
    base = {"alpha": 0.06, "h": 6.0, "k": 0.5, "wmin": 5}
    variants = {
        "ewma": A.track(P.R, "ewma", {"alpha": 0.1})["O"], "ma": A.track(P.R, "ma", {"k": 14})["O"], "trigg": A.track(P.R, "trigg", {"lam": 0.05, "amin": 0.08, "amax": 0.5})["O"],
        "cusum": A.track(P.R, "cusum", base)["O"], "cusum_c10": A.track(P.R, "cusum", {**base, "confirm": 10})["O"],
        "cusum_theta": A.track(P.R, "cusum", {**base, "confirm": 7, "theta": 1.5})["O"], "cusum_c7": A.track(P.R, "cusum", {**base, "confirm": 7})["O"],
        "cusum_wmin": A.track(P.R, "cusum", {**base, "wmin": 14})["O"], "cusum_k": A.track(P.R, "cusum", {**base, "k": 1.0})["O"], "cusum_h": A.track(P.R, "cusum", {**base, "h": 4.0})["O"],
        "cusum_alpha": A.track(P.R, "cusum", {**base, "alpha": 0.2})["O"],
    }
    pairs = [("ewma", "ma"), ("ewma", "trigg"), ("ma", "trigg"), ("ewma", "cusum"), ("cusum", "cusum_c10"), ("cusum_c7", "cusum_theta"), ("cusum", "cusum_wmin"), ("cusum", "cusum_k"),
             ("cusum", "cusum_h"), ("cusum", "cusum_alpha"), ("cusum_c7", "cusum_c10")]
    for a, b in pairs:
        assert not np.allclose(variants[a], variants[b]), (a, b)


# --- Regression -----------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("model", C.MODELS)
def test_regression_stimmt_mit_den_normalgleichungen_ueberein(model):
    b = S.batch((4, 9), "step", 0.3)
    X = A.design(b.promo, b.holiday, b.after, model)
    fit, R, s2 = A.reg_fit(b.y, X)
    for i in range(2):
        Xi = X[i, C.FIT_FROM:C.FIRST_TEST]
        yi = np.log(b.y[i, C.FIT_FROM:C.FIRST_TEST] + 1.0)
        beta = np.linalg.solve(Xi.T @ Xi, Xi.T @ yi)                         # Normalgleichungen statt lstsq
        assert fit[i] == pytest.approx(X[i] @ beta, abs=1e-8)
        res = yi - Xi @ beta
        assert s2[i] == pytest.approx(res @ res / (len(yi) - X.shape[2]), rel=1e-8)          # Restvarianz mit Freiheitsgraden n - p
        assert R[i] == pytest.approx(np.log(b.y[i] + 1.0) - X[i] @ beta, abs=1e-8)


def test_regression_modelle_unterscheiden_sich_und_mehr_regressoren_passen_besser():
    b = S.batch((4, 9, 13), "step", 0.0)
    s2 = {m: A.reg_fit(b.y, A.design(b.promo, b.holiday, b.after, m))[2] for m in C.MODELS}
    assert (s2["fourier_trend"] < s2["weekday"]).all() and (s2["calendar"] < s2["fourier_trend"]).all()
