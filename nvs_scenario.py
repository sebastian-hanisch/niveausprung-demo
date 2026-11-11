"""Vehikel "Tagesaufträge eines Depots" (Stück 1 der Linie) mit Sprungformen und Autokorrelation.

Erwartungswert = Niveau x Trend x Wochenmuster x Jahresmuster x Sprungfolge x Aktionen x Feiertage; beobachtet wird ein Zählwert mit log-normalem Rauschen (mittelwerttreu).
Der Zufallsstrom ist der von `nf_scenario.generate` (naive-forecast-demo): Phase, Sprungtag, Aktionstage, Rauschen in dieser Reihenfolge. Anders als dort wird der Sprungtag auch ohne Sprung
gezogen, damit Reihen mit und ohne Sprung bis auf den Sprung dieselben Zufallszahlen benutzen (gepaarter Vergleich). Bei Sprung ungleich 0 und Autokorrelation 0 ist die Reihe bit-gleich zu `nf_scenario`
(`tests/test_oracle_vehikel.py`).

Ein Ursprung t heißt in allen Modulen: bekannt sind die Tage 0 bis t - 1."""

from dataclasses import dataclass

import numpy as np

import nvs_constants as C

T = C.N_DAYS
T0 = C.FIRST_TEST


@dataclass(frozen=True)
class Batch:
    y: np.ndarray            # (N, T) beobachtete Tagesaufträge
    mu: np.ndarray           # (N, T) Erwartungswert ohne Rauschen
    J: np.ndarray            # (N,) Tag des ersten Sprungs (auch ohne Sprung gezogen)
    promo: np.ndarray        # (N, T) Aktionstage (dem Modell bekannt)
    holiday: np.ndarray      # (N, T) Feiertage
    after: np.ndarray        # (N, T) Tag nach dem Feiertag
    jumps: tuple             # ((Tage nach J, relative Änderung), ...) für die Reihen mit Sprung
    seeds: tuple
    has_jump: np.ndarray = None   # (N,) bool: Reihe enthält die Sprungfolge (Standard: alle, wenn jumps nicht leer)

    @property
    def n(self):
        return len(self.seeds)

    @property
    def window(self):
        """Fenster für die Gewinne ab dem Sprung: bei Folgen von Sprüngen länger."""
        return C.WINDOW_LONG if len(self.jumps) and max(off for off, _ in self.jumps) >= C.BACK_AFTER else C.WINDOW_SHORT


def jumps_for(form, shift):
    """Sprungfolge einer Form als Liste (Tage nach dem Sprungtag J, relative Änderung des Niveaus); Faktoren werden multipliziert.
    shift: relative Größe (0,3 = +30 %). Größe 0: keine Änderung."""
    if shift == 0:
        return []
    up = 1.0 + shift
    if form == "step":
        return [(0, shift)]
    if form == "step_back":
        return [(0, shift), (C.BACK_AFTER, 1.0 / up - 1.0)]
    if form == "step_two":
        return [(0, shift), (C.SECOND_AFTER, C.SECOND_FRACTION * shift)]
    if form == "ramp":
        per_day = up ** (1.0 / C.RAMP_DAYS) - 1.0
        return [(d, per_day) for d in range(C.RAMP_DAYS)]
    if form == "promo":
        return [(0, shift), (C.PROMO_DAYS, 1.0 / up - 1.0)]
    if form == "spike":
        return [(0, shift), (1, 1.0 / up - 1.0)]
    raise ValueError(form)


def generate(seed, jumps=(), noise=C.DEFAULT_NOISE / 100.0, ar=0.0, trend=C.DEFAULT_TREND, weekly=C.DEFAULT_WEEKLY, yearly=C.DEFAULT_YEARLY, events=C.DEFAULT_EVENTS):
    """Eine Reihe: (y, mu, J, promo, holiday, after). noise: Streuung des Log-Rauschens (0,14 = 14 %); ar: AR(1)-Koeffizient des Log-Rauschens bei Einheitsvarianz."""
    rng = np.random.default_rng(int(seed))
    t = np.arange(T)
    dow = t % 7
    doy = t % 365
    pattern = np.array(C.WEEKLY_PATTERN)
    pattern = pattern / pattern.mean()
    week_f = 1.0 + weekly * (pattern[dow] - 1.0)
    phase = rng.uniform(0, 2 * np.pi)
    year_f = 1.0 + yearly * np.sin(2 * np.pi * doy / 365.0 + phase)
    trend_f = 1.0 + (trend / 100.0) * t / 365.0
    J = int(rng.integers(C.FIRST_TEST + 30, T - 60))
    shift_f = np.ones(T)
    for off, rel in jumps:
        shift_f = shift_f * np.where(t >= J + off, 1.0 + rel, 1.0)
    holiday = np.isin(doy, C.HOLIDAY_DOY).astype(float)
    after = np.roll(holiday, 1)
    after[0] = 0.0
    holiday_f = 1.0 - events * C.HOLIDAY_DROP * holiday + events * C.HOLIDAY_REBOUND * after
    promo = np.zeros(T)
    starts = rng.choice(np.arange(30, T - C.PROMO_LENGTH), size=C.PROMO_PER_YEAR * (T // 365), replace=False)
    for s in starts:
        promo[s:s + C.PROMO_LENGTH] = 1.0
    promo_f = 1.0 + events * 0.5 * promo
    mu = np.maximum(C.LEVEL * trend_f * week_f * year_f * shift_f * holiday_f * promo_f, 1.0)
    z = rng.normal(size=T)
    if ar > 0:
        zz = np.empty(T)
        zz[0] = z[0]
        for i in range(1, T):
            zz[i] = ar * zz[i - 1] + np.sqrt(1.0 - ar ** 2) * z[i]
        z = zz
    y = np.maximum(np.rint(mu * np.exp(noise * z - 0.5 * noise ** 2)), 0.0)
    return y, mu, J, promo, holiday, after


def batch(seeds, form=C.DEFAULT_FORM, shift=0.0, noise=C.DEFAULT_NOISE / 100.0, ar=0.0, **kw):
    """Mehrere Reihen mit derselben Sprungfolge (shift = 0: keine Änderung). Rückgabe: Batch."""
    jumps = tuple(jumps_for(form, shift))
    out = [generate(int(s), jumps=jumps, noise=noise, ar=ar, **kw) for s in seeds]
    return Batch(
        y=np.array([o[0] for o in out]), mu=np.array([o[1] for o in out]), J=np.array([o[2] for o in out]), promo=np.array([o[3] for o in out]),
        holiday=np.array([o[4] for o in out]), after=np.array([o[5] for o in out]), jumps=jumps, seeds=tuple(int(s) for s in seeds),
        has_jump=np.full(len(out), len(jumps) > 0),
    )


def mase_scale(y):
    """Nenner der MASE: mittlerer absoluter Fehler der saisonal naiven Prognose (Periode 7) auf den Tagen < 730; (N,)."""
    return np.abs(y[:, 7:T0] - y[:, :T0 - 7]).mean(axis=1)


def concat(with_jump: Batch, without_jump: Batch) -> Batch:
    """Zwei Stapel zu einem (z. B. Abstimm-Reihen: halb mit, halb ohne Sprung). Die Sprungfolge ist die des ersten; has_jump trennt die Reihen."""
    cat = lambda a, b: np.concatenate([a, b], axis=0)  # noqa: E731
    return Batch(cat(with_jump.y, without_jump.y), cat(with_jump.mu, without_jump.mu), cat(with_jump.J, without_jump.J), cat(with_jump.promo, without_jump.promo),
                 cat(with_jump.holiday, without_jump.holiday), cat(with_jump.after, without_jump.after), with_jump.jumps, with_jump.seeds + without_jump.seeds,
                 cat(with_jump.has_jump, np.zeros(without_jump.n, dtype=bool)))
