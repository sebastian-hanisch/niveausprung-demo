"""Kern: Regression (Wochentag, Jahresmuster, Trend, Kalender) im Log, Niveau-Nachführer auf den Residuen (EWMA, Mittel der letzten k Tage, Trigg-Leach, CUSUM-Zurücksetzen mit
optionaler Bestätigung, Orakel), Prognosen und MASE sowie eine feste saisonale Glättung als Vergleich.

Alle Funktionen rechnen über viele Reihen zugleich (Achse 0). Ein Ursprung t heißt: bekannt sind die Tage 0 bis t - 1.

Aufbau der Prognose: log(y + 1) = Regression (auf den Tagen 28 bis 729 angepasst) + Niveau o_t + Rauschen. Das Niveau o_t (Offset) ist das, was die Nachführer unterscheidet:
  none      o = 0
  ewma      o_t = o_(t-1) + alpha e_t                                  e_t = Residuum minus o_(t-1)  (Ein-Schritt-Fehler)
  ma        o_t = Mittel der Residuen der letzten k Tage
  trigg     wie ewma, aber alpha_t = |geglätteter Fehler| / geglättetes |Fehler| (Trigg-Leach), auf [alpha_min, alpha_max] geklemmt
  cusum     ewma mit kleinem alpha_low, dazu ein zweiseitiger CUSUM auf z_t = e_t / sigma (sigma aus dem Referenzfenster); bei hi oder lo > h wird gemeldet und o auf das Mittel der Residuen
            seit Beginn der Auslenkung gesetzt (mindestens wmin Tage); Beginn = erster Tag der laufenden Auslenkung, der Tag nach dem letzten Tag mit Statistik 0. Mit Bestätigung (c > 0, theta) wird erst c Tage nach dem Alarm gesetzt, und nur, wenn das Mittel der letzten
            min(c, 7) Residuen in Alarmrichtung mindestens theta Standardabweichungen über dem Referenzmittel liegt; sonst wird der Alarm verworfen.
  oracle    ewma mit alpha, dazu kennt es Tag und Größe jedes Sprungs und setzt o genau dort (ein Maßstab, keine strenge Obergrenze: kleinstes alpha des Abstimmungsrasters, siehe README)."""

import numpy as np

import nvs_constants as C
from nvs_scenario import T, T0


# --------------------------------------------------------------------------------------------------------------------------------------------
# Regression
def design(promo, holiday, after, model):
    """Entwurfsmatrix (N, T, p). Spalten: Konstante; je nach Modell Wochentage (Tage 1 bis 6), Trend und Fourier-Paare des Jahresmusters, Feiertag, Tag nach dem Feiertag, Aktionstag."""
    N = promo.shape[0]
    t = np.arange(T)
    cols = [np.ones(T)]
    cols += [(t % 7 == d).astype(float) for d in range(1, 7)]
    if model in ("fourier_trend", "calendar"):
        cols += [t / 365.0]
        for k in range(1, C.FOURIER_PAIRS + 1):
            cols += [np.sin(2 * np.pi * k * (t % 365) / 365.0), np.cos(2 * np.pi * k * (t % 365) / 365.0)]
    X0 = np.stack(cols, axis=1)
    X = np.broadcast_to(X0, (N,) + X0.shape).copy()
    if model == "calendar":
        X = np.concatenate([X, holiday[:, :, None], after[:, :, None], promo[:, :, None]], axis=2)
    return X


def reg_fit(y, X, train=T0):
    """OLS je Reihe auf log(y + 1) der Tage 28 bis train - 1. Rückgabe: Fit (N, T), Residuen (N, T) = log(y + 1) - Fit und die Restvarianz je Reihe (N,)."""
    Lg = np.log(y + 1.0)
    fit = np.empty_like(Lg)
    s2 = np.empty(len(y))
    for i in range(len(y)):
        b, *_ = np.linalg.lstsq(X[i, C.FIT_FROM:train], Lg[i, C.FIT_FROM:train], rcond=None)
        fit[i] = X[i] @ b
        r = Lg[i, C.FIT_FROM:train] - fit[i, C.FIT_FROM:train]
        s2[i] = r.var(ddof=X.shape[2])
    return fit, Lg - fit, s2


# --------------------------------------------------------------------------------------------------------------------------------------------
# Nachführer
def track(R, kind, p=None, jumps=None, ref=(T0 - C.REF_LEN, T0)):
    """Niveau (Offset) nach jedem Tag. R: Residuen (N, T). kind: none | ewma | ma | trigg | cusum | oracle.
    p: Parameter (ewma: alpha; ma: k; trigg: lam, amin, amax; cusum: alpha, h, k, wmin und optional confirm, theta; oracle: alpha).
    jumps (nur oracle): Liste (Tage (N,), Änderung im Log) ; ein Eintrag mit Tag < 0 wirkt nicht.
    Rückgabe: dict mit O (N, T) = Offset nach Tag t (gilt für Prognosen ab Tag t + 1), E (N, T) Ein-Schritt-Fehler,
    alarm (N, T) bool: Tage mit CUSUM-Alarm, meldung (N, T) bool: Tage, an denen das Niveau zurückgesetzt wurde (ohne Bestätigung gleich alarm), stat (N, T): CUSUM-Statistik
    (max(hi, lo) vor dem Zurücksetzen, positiv bei Aufwärts-, negativ bei Abwärtsalarm; 0 vor dem Testjahr)."""
    p = p or {}
    N, Tn = R.shape
    O = np.zeros((N, Tn))
    E = np.zeros((N, Tn))
    alarm = np.zeros((N, Tn), dtype=bool)
    meldung = np.zeros((N, Tn), dtype=bool)
    stat = np.zeros((N, Tn))
    o = np.zeros(N)
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(R, axis=1)], axis=1)
    if kind == "trigg":
        Es = np.zeros(N)
        Ms = np.full(N, 1e-3)
    if kind == "cusum":
        t_det = ref[1]
        hi = np.zeros(N)
        lo = np.zeros(N)
        st_hi = np.zeros(N, dtype=int)
        st_lo = np.zeros(N, dtype=int)
        pend_t = np.full(N, -1)
        pend_side = np.zeros(N)
        pend_start = np.zeros(N, dtype=int)
        conf_c = int(p.get("confirm", 0))
        theta = float(p.get("theta", 1.0))
        sig = None
        r0 = s0 = None
    idx = np.arange(N)
    for t in range(Tn):
        e = R[:, t] - o
        E[:, t] = e
        if kind == "none":
            o_new = o
        elif kind in ("ewma", "cusum", "oracle"):
            o_new = o + p["alpha"] * e
        elif kind == "ma":
            lo_i = max(0, t + 1 - int(p["k"]))
            o_new = (cs[:, t + 1] - cs[:, lo_i]) / (t + 1 - lo_i)
        elif kind == "trigg":
            Es = p["lam"] * e + (1 - p["lam"]) * Es
            Ms = p["lam"] * np.abs(e) + (1 - p["lam"]) * Ms
            a = np.clip(np.abs(Es) / np.maximum(Ms, 1e-9), p["amin"], p["amax"])
            o_new = o + a * e
        else:
            raise ValueError(kind)
        if kind == "cusum":
            if t == t_det - 1:
                sig = np.maximum(E[:, ref[0]:ref[1]].std(axis=1, ddof=1), 1e-9)
                r0 = R[:, ref[0]:ref[1]].mean(axis=1)
                s0 = np.maximum(R[:, ref[0]:ref[1]].std(axis=1, ddof=1), 1e-9)
            if t >= t_det:
                z = e / sig
                st_hi = np.where(hi == 0.0, t, st_hi)
                st_lo = np.where(lo == 0.0, t, st_lo)
                hi = np.maximum(0.0, hi + z - p["k"])
                lo = np.maximum(0.0, lo - z - p["k"])
                hit_hi, hit_lo = hi > p["h"], lo > p["h"]
                hit = hit_hi | hit_lo
                stat[:, t] = np.where(hi >= lo, hi, -lo)
                alarm[hit, t] = True
                for i in idx[hit]:
                    side = 1.0 if hit_hi[i] else -1.0
                    start = st_hi[i] if hit_hi[i] else st_lo[i]
                    if conf_c == 0:
                        w = min(max(t - start + 1, int(p["wmin"])), t + 1)
                        o_new[i] = (cs[i, t + 1] - cs[i, t + 1 - w]) / w
                        meldung[i, t] = True
                    elif pend_t[i] < 0:
                        pend_t[i], pend_side[i], pend_start[i] = t, side, start
                    hi[i] = 0.0
                    lo[i] = 0.0
                if conf_c > 0:
                    due = idx[(pend_t >= 0) & (t == pend_t + conf_c)]
                    for i in due:
                        w7 = max(min(conf_c, 7), 1)
                        seg_mean = (cs[i, t + 1] - cs[i, t + 1 - w7]) / w7
                        if pend_side[i] * (seg_mean - r0[i]) / s0[i] >= theta:
                            w = min(max(t - pend_start[i] + 1, int(p["wmin"])), t + 1)
                            o_new[i] = (cs[i, t + 1] - cs[i, t + 1 - w]) / w
                            meldung[i, t] = True
                        pend_t[i] = -1
        if kind == "oracle":
            for days, dlog in jumps or ():
                o_new = np.where(days == t, o_new + dlog, o_new)
        o = o_new
        O[:, t] = o
    return {"O": O, "E": E, "alarm": alarm, "meldung": meldung, "stat": stat}


def oracle_jumps(J, jumps, has_jump=None, delay=0):
    """Sprünge für das Orakel: je Eintrag (Tage (N,), Änderung im Log). delay: Orakel um so viele Tage verspätet. has_jump (N,) bool: Reihen ohne Sprung bekommen Tag -1."""
    N = len(J)
    on = np.ones(N, dtype=bool) if has_jump is None else np.asarray(has_jump, dtype=bool)
    return [(np.where(on, J + off + delay, -1), float(np.log1p(rel))) for off, rel in jumps]


# --------------------------------------------------------------------------------------------------------------------------------------------
# Prognosen und MASE
def forecasts_from_offsets(fit, O, s2, org, h=C.HORIZON):
    """Prognosen (N, len(org), h) im Original-Maßstab: exp(Fit[t + j] + O[t - 1] + s2 / 2) - 1 (mittelwerttreu), nicht negativ."""
    j = np.arange(h)
    F = np.exp(fit[:, np.minimum(org[:, None] + j[None, :], T - 1)] + O[:, org - 1][:, :, None] + 0.5 * s2[:, None, None]) - 1.0
    return np.maximum(F, 0.0)


def targets(y, org, h=C.HORIZON):
    return np.stack([y[:, t:t + h] for t in org], axis=1)


def mase(F, A, scale, mask=None):
    """MASE je Reihe: mittlerer absoluter Fehler über Horizont (1 bis 14 Tage) und Ursprünge (mit Maske nur über die Ursprünge der Maske), geteilt durch den Nenner `scale` (mittlerer absoluter Fehler der saisonal naiven Prognose auf den Tagen 7 bis 729)."""
    ae = np.abs(F - A).mean(axis=2)
    if mask is None:
        return ae.mean(axis=1) / scale
    return (ae * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1) / scale


# --------------------------------------------------------------------------------------------------------------------------------------------
# Feste saisonale Glättung (multiplikativ), Vergleichsverfahren
def holt_winters(y, alpha, gamma):
    """Niveau und multiplikative Wochenfaktoren mit festen Parametern, gestartet auf den ersten 28 Tagen. Rückgabe: Niveau (N, T) und Faktoren (N, T, 7) nach jedem Tag."""
    N, Tn = y.shape
    Lv = y[:, :28].mean(axis=1)
    S = np.stack([y[:, d:28:7].mean(axis=1) for d in range(7)], axis=1) / Lv[:, None]
    Lh = np.zeros((N, Tn))
    Sh = np.zeros((N, Tn, 7))
    for t in range(Tn):
        d = t % 7
        yt = y[:, t]
        Ln = alpha * yt / S[:, d] + (1 - alpha) * Lv
        S[:, d] = gamma * yt / Ln + (1 - gamma) * S[:, d]
        Lv = Ln
        Lh[:, t] = Lv
        Sh[:, t, :] = S
    return Lh, Sh


def hw_forecasts(Lh, Sh, org, h=C.HORIZON):
    j = np.arange(h)
    out = np.empty((Lh.shape[0], len(org), h))
    for o, t in enumerate(org):
        out[:, o, :] = Lh[:, t - 1][:, None] * Sh[:, t - 1, :][:, (t + j) % 7]
    return out


# --------------------------------------------------------------------------------------------------------------------------------------------
# Verfahren der Stücke 1 bis 3 der Linie (naive-forecast-demo): saisonal naiv und Mittel der letzten 4 Wochen; Orakel = wahrer Erwartungswert
def seasonal_naive_forecasts(y, org, h=C.HORIZON):
    """Derselbe Wochentag der letzten Woche: Prognose der Tage t + j ist y[t - 7 + (j mod 7)]."""
    j = np.arange(h)
    return y[:, org[:, None] - 7 + (j % 7)[None, :]]


def weekly_mean_forecasts(y, org, h=C.HORIZON, weeks=4):
    """Mittel desselben Wochentags über die letzten `weeks` Wochen."""
    j = np.arange(h)
    return np.mean([y[:, org[:, None] - 7 * (i + 1) + (j % 7)[None, :]] for i in range(weeks)], axis=0)


def expectation_forecasts(mu, org, h=C.HORIZON):
    """Orakel der Linie: der wahre Erwartungswert (ohne Rauschen)."""
    j = np.arange(h)
    return mu[:, np.minimum(org[:, None] + j[None, :], T - 1)]
