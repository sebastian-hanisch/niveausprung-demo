"""Orakel Vehikel: der Erzeuger gegen eine unabhängig geschriebene Kopie der Erzeugung der naive-forecast-demo (Stück 1), und die MASE-Werte der Stücke 1 bis 3 am Preset der Linie
(Seed 3, +30 % an Tag 809: saisonal naiv 1,36, Wochenmittel 1,08, Orakel 0,90 laut README der Linie; die Werte stehen dort in zwei Stellen).

Die Kopie unten folgt `nf_scenario.generate` Zeile für Zeile (Quelle: naive-forecast-demo), die Prognosen und die MASE sind Schleifen über Ursprünge wie in `nf_forecast.rolling_origin`."""

import numpy as np
import pytest

import nvs_algorithm as A
import nvs_constants as C
import nvs_inventory as INV
import nvs_scenario as S


def _nf_generate(shift, noise, seed, n_days=1095):
    """Kopie von nf_scenario.generate (trend 10, weekly 1, yearly 0,2, events 0,5)."""
    trend, weekly, yearly, events = 10, 1.0, 0.2, 0.5
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dow = t % 7
    doy = t % 365
    pattern = np.array((1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35))
    pattern = pattern / pattern.mean()
    week_f = 1.0 + weekly * (pattern[dow] - 1.0)
    phase = rng.uniform(0, 2 * np.pi)
    year_f = 1.0 + yearly * np.sin(2 * np.pi * doy / 365.0 + phase)
    trend_f = 1.0 + (trend / 100.0) * t / 365.0
    shift_day = int(rng.integers(730 + 30, n_days - 60)) if shift != 0 else -1
    shift_f = np.where((t >= shift_day) & (shift_day >= 0), 1.0 + shift / 100.0, 1.0)
    holiday = np.isin(doy, (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)).astype(float)
    holiday_f = 1.0 - events * 0.5 * holiday
    after = np.roll(holiday, 1)
    after[0] = 0.0
    holiday_f = holiday_f + events * 0.15 * after
    promo = np.zeros(n_days)
    starts = rng.choice(np.arange(30, n_days - 7), size=3 * (n_days // 365), replace=False)
    for s in starts:
        promo[s:s + 7] = 1.0
    promo_f = 1.0 + events * 0.5 * promo
    mu = np.maximum(100.0 * trend_f * week_f * year_f * shift_f * holiday_f * promo_f, 1.0)
    z = rng.normal(size=n_days)
    y = np.maximum(np.rint(mu * np.exp(noise * z - 0.5 * noise ** 2)), 0.0)
    return y, mu, shift_day


@pytest.mark.parametrize("seed", range(8))
def test_erzeuger_ist_bitgleich_zu_nf_scenario(seed):
    y_ref, mu_ref, day_ref = _nf_generate(30, 0.14, seed)
    b = S.batch((seed,), "step", 0.30, 0.14, 0.0)
    assert b.J[0] == day_ref
    assert np.array_equal(b.y[0], y_ref) and np.array_equal(b.mu[0], mu_ref)


def test_erzeuger_ohne_sprung_stimmt_bis_auf_den_sprung_ueberein():
    # nf_scenario zieht ohne Sprung keinen Sprungtag, der Strom ist dann verschoben; die Demo zieht immer. Deshalb: gleiche Reihe, wenn der Sprung später wegfällt.
    a, b = S.batch((11,), "step", 0.30), S.batch((11,), "step", 0.0)
    y_ref, mu_ref, day_ref = _nf_generate(30, 0.14, 11)
    assert np.array_equal(a.y[0], y_ref)
    assert np.array_equal(b.mu[0, :day_ref], mu_ref[:day_ref])


def _rolling_mase_reference(y, mu):
    """MASE der Verfahren der Linie als Schleife über Ursprünge (nf_forecast.rolling_origin / summarize): Horizont 14, Ursprünge 730 bis 1081."""
    h = 14
    scale = np.abs(y[7:730] - y[:723]).mean()
    out = {"snaive": [], "wmean4": [], "orakel": []}
    for t in range(730, len(y) - h + 1):
        actual = y[t:t + h]
        hist = y[:t]
        m = len(hist)
        j = np.arange(h)
        out["snaive"].append(np.abs(hist[m - 7 + (j % 7)] - actual).mean())
        out["wmean4"].append(np.abs(np.mean([hist[m - 7 * (i + 1) + (j % 7)] for i in range(4)], axis=0) - actual).mean())
        out["orakel"].append(np.abs(mu[t:t + h] - actual).mean())
    return {k: float(np.mean(v) / scale) for k, v in out.items()}


def test_mase_der_linie_am_preset_seed_3():
    b = S.batch((3,), "step", 0.30, 0.14, 0.0)
    assert b.J[0] == 809                                                     # Preset der Linie: +30 % an Tag 809
    ref = _rolling_mase_reference(b.y[0], b.mu[0])
    # Literaturwerte (README der Linie, zwei Stellen): 1,36 / 1,08 / 0,90
    assert round(ref["snaive"], 2) == 1.36 and round(ref["wmean4"], 2) == 1.08 and round(ref["orakel"], 2) == 0.90
    org = INV.ORG_ALL
    scale = S.mase_scale(b.y)
    idx = np.arange(730, 1082) - org[0]
    Act = A.targets(b.y, np.arange(730, 1082))
    got = {
        "snaive": A.mase(A.seasonal_naive_forecasts(b.y, org)[:, idx], Act, scale)[0],
        "wmean4": A.mase(A.weekly_mean_forecasts(b.y, org)[:, idx], Act, scale)[0],
        "orakel": A.mase(A.expectation_forecasts(b.mu, org)[:, idx], Act, scale)[0],
    }
    for k in got:
        assert got[k] == pytest.approx(ref[k], rel=1e-12), k


def test_mase_nenner_ist_der_der_linie():
    b = S.batch((3,), "step", 0.30)
    assert S.mase_scale(b.y)[0] == pytest.approx(np.abs(b.y[0, 7:730] - b.y[0, :723]).mean())


def test_konstanten_des_vehikels_wie_in_stueck_1():
    assert (C.N_DAYS, C.FIRST_TEST, C.LEVEL, C.WEEKLY_PATTERN) == (1095, 730, 100.0, (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35))
    assert C.HOLIDAY_DOY == (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)
