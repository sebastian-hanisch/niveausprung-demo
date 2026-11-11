"""Kern gegen Handrechnung auf Mini-Reihen: Nachführer, CUSUM mit und ohne Bestätigung, Regression, Prognose, MASE, Glättung, Sprungformen (DEMO-PLAYBOOK §1b)."""

import numpy as np
import pytest

import nvs_algorithm as A
import nvs_constants as C
import nvs_scenario as S

SIG = np.sqrt(10 / 9)           # Standardabweichung der Referenztage der Fixture ref_ten (ddof = 1)
REF = (0, 10)


def test_ewma_ist_die_handrechnung():
    R = np.ones((1, 4))
    out = A.track(R, "ewma", {"alpha": 0.5})
    assert out["O"][0] == pytest.approx([0.5, 0.75, 0.875, 0.9375])
    assert out["E"][0] == pytest.approx([1.0, 0.5, 0.25, 0.125])           # Ein-Schritt-Fehler = Residuum minus Niveau vom Vortag


def test_mittel_der_letzten_k_tage():
    R = np.array([[1.0, 3.0, 5.0, 7.0]])
    assert A.track(R, "ma", {"k": 2})["O"][0] == pytest.approx([1.0, 2.0, 4.0, 6.0])


def test_trigg_leach_handrechnung_bei_konstantem_residuum():
    # lam = 0,5: Tag 0: e = 1, Es = 0,5, Ms = 0,5005, alpha = min(0,999, amax) = 0,5, o = 0,5; Tag 1: e = 0,5, Es = 0,5, Ms = 0,50025, alpha = 0,5, o = 0,75
    out = A.track(np.ones((1, 2)), "trigg", {"lam": 0.5, "amin": 0.1, "amax": 0.5})
    assert out["O"][0] == pytest.approx([0.5, 0.75])


def test_trigg_leach_untergrenze_gilt_bei_verschwindendem_signal():
    # abwechselnd +1 / -1: der geglättete Fehler ist fast 0, alpha fällt auf amin
    R = np.array([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]])
    out = A.track(R, "trigg", {"lam": 0.5, "amin": 0.2, "amax": 0.5})
    steps = np.diff(np.concatenate([[0.0], out["O"][0]]))
    assert np.all(np.abs(steps) <= 0.5 * np.abs(out["E"][0]) + 1e-12)
    assert np.all(np.abs(steps) >= 0.2 * np.abs(out["E"][0]) - 1e-12)


def test_none_haelt_das_niveau_bei_null():
    assert A.track(np.ones((2, 5)), "none")["O"].tolist() == [[0.0] * 5] * 2


def test_cusum_aufwaertsalarm_und_zuruecksetzen_von_hand(ref_ten):
    R = ref_ten(3.0)
    out = A.track(R, "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1}, ref=REF)
    z = 3.0 / SIG
    assert out["stat"][0, 10] == pytest.approx(z - 0.5)                      # erste Auslenkung
    assert out["stat"][0, 11] == pytest.approx(2 * (z - 0.5))                # 4,69 > h = 4
    assert np.nonzero(out["alarm"][0])[0].tolist() == [11]                  # danach ist das Niveau 3, die Ein-Schritt-Fehler sind 0
    assert np.nonzero(out["meldung"][0])[0].tolist() == [11]
    assert out["O"][0, 10] == 0.0 and out["O"][0, 11] == pytest.approx(3.0)   # Mittel der Residuen seit Beginn der Auslenkung (Tage 10 bis 11)


def test_cusum_abwaertsalarm_hat_negatives_vorzeichen(ref_ten):
    out = A.track(ref_ten(-3.0), "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1}, ref=REF)
    assert out["alarm"][0, 11] and out["stat"][0, 11] < 0
    assert out["O"][0, 11] == pytest.approx(-3.0)


def test_cusum_mindestfenster_mittelt_auch_tage_vor_der_auslenkung(ref_ten):
    R = ref_ten(3.0)
    out = A.track(R, "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 4}, ref=REF)
    assert out["O"][0, 11] == pytest.approx(R[0, 8:12].mean())              # Tage 8 bis 11 (w = max(2, 4))


def test_cusum_ohne_ueberschreitung_meldet_nichts(ref_ten):
    out = A.track(ref_ten(0.4), "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1}, ref=REF)      # 0,4 / 1,054 < k: die Statistik bleibt 0
    assert not out["alarm"].any() and not out["stat"].any()


def test_bestaetigung_setzt_erst_c_tage_nach_dem_alarm(ref_ten):
    out = A.track(ref_ten(3.0), "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1, "confirm": 3, "theta": 1.0}, ref=REF)
    assert out["alarm"][0, 11]
    assert np.nonzero(out["meldung"][0])[0][0] == 14                         # Alarm an Tag 11 plus 3 Tage
    assert out["O"][0, 13] == 0.0 and out["O"][0, 14] == pytest.approx(3.0)  # Mittel der Tage 10 bis 14


def test_bestaetigung_verwirft_bei_zu_hoher_schwelle(ref_ten):
    # Mittel der letzten 3 Residuen ist 3, Abstand 3 / 1,054 = 2,85 Standardabweichungen: theta = 3 verwirft jeden Alarm
    out = A.track(ref_ten(3.0), "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1, "confirm": 3, "theta": 3.0}, ref=REF)
    assert out["alarm"].any() and not out["meldung"].any() and not out["O"].any()


def test_bestaetigung_prueft_die_alarmrichtung(ref_ten):
    out = A.track(ref_ten(-3.0), "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1, "confirm": 3, "theta": 1.0}, ref=REF)
    assert np.nonzero(out["meldung"][0])[0][0] == 14 and out["O"][0, 14] == pytest.approx(-3.0)


def test_bestaetigung_verwirft_eine_kurze_aktion(ref_ten):
    R = ref_ten(0.0)
    R[0, 10:13] = 3.0                                                        # dreitägiger Ausschlag, danach wieder normal
    out = A.track(R, "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1, "confirm": 5, "theta": 1.0}, ref=REF)
    assert out["alarm"][0, 11]                                              # Alarm ...
    assert not out["meldung"].any()                                          # ... aber nicht bestätigt: Mittel der Tage 12 bis 16 = 0,6, Abstand 0,57 < 1


def test_orakel_setzt_das_niveau_am_sprungtag():
    R = np.zeros((2, 10))
    out = A.track(R, "oracle", {"alpha": 0.0}, jumps=[(np.array([4, -1]), 0.3)])
    assert out["O"][0].tolist() == [0.0] * 4 + [0.3] * 6                     # Reihe 0 springt an Tag 4
    assert not out["O"][1].any()                                             # Tag -1: Reihe ohne Sprung


def test_oracle_jumps_verschiebt_um_die_verzoegerung():
    jumps = A.oracle_jumps(np.array([100, 200]), ((0, 0.3), (10, -0.2)), has_jump=np.array([True, False]), delay=5)
    assert [d.tolist() for d, _ in jumps] == [[105, -1], [115, -1]]
    assert [dl for _, dl in jumps] == pytest.approx([np.log(1.3), np.log(0.8)])


def test_unbekannter_nachfuehrer_wird_abgelehnt():
    with pytest.raises(ValueError):
        A.track(np.zeros((1, 3)), "magie", {})


# --- Regression, Prognose, MASE ---------------------------------------------------------------------------------------------------------------
def test_entwurfsmatrix_spalten_je_modell():
    z = np.zeros((2, C.N_DAYS))
    assert A.design(z, z, z, "weekday").shape == (2, C.N_DAYS, 7)             # Konstante + 6 Wochentage
    assert A.design(z, z, z, "fourier_trend").shape == (2, C.N_DAYS, 12)      # + Trend + 2 Fourier-Paare
    assert A.design(z, z, z, "calendar").shape == (2, C.N_DAYS, 15)           # + Feiertag, Tag danach, Aktion
    X = A.design(z, z, z, "weekday")
    assert np.all(X[0, :, 0] == 1) and X[0, 0, 1:].sum() == 0 and X[0, 1, 1] == 1 and X[0, 8, 1] == 1   # Tag 1 und Tag 8 sind Wochentag 1


def test_regression_findet_exakte_koeffizienten_ohne_rauschen():
    rng = np.random.default_rng(1)
    z = np.zeros((1, C.N_DAYS))
    X = A.design(z, z, z, "fourier_trend")
    beta = rng.normal(size=X.shape[2]) * 0.1
    beta[0] = 4.0
    y = np.exp(X[0] @ beta)[None, :] - 1.0                                   # log(y + 1) = X beta exakt
    fit, R, s2 = A.reg_fit(y, X)
    assert np.abs(R).max() < 1e-8 and s2[0] < 1e-16
    assert fit[0] == pytest.approx(X[0] @ beta, abs=1e-8)


def test_prognose_aus_offsets_von_hand():
    fit = np.full((1, C.N_DAYS), np.log(11.0))
    O = np.zeros((1, C.N_DAYS))
    F = A.forecasts_from_offsets(fit, O, np.zeros(1), np.array([800]), h=3)
    assert F[0, 0] == pytest.approx([10.0, 10.0, 10.0])                      # exp(ln 11) - 1
    O[:, 799] = np.log(2.0)                                                   # Niveau nach Tag 799 gilt für Prognosen ab Tag 800
    assert A.forecasts_from_offsets(fit, O, np.zeros(1), np.array([800]), h=1)[0, 0, 0] == pytest.approx(21.0)
    assert A.forecasts_from_offsets(fit, O, np.full(1, 2 * np.log(2.0)), np.array([800]), h=1)[0, 0, 0] == pytest.approx(2.0 * 11.0 * 2.0 - 1.0)   # + s2 / 2 = ln 2 im Exponenten


def test_prognose_ist_nie_negativ():
    fit = np.full((1, C.N_DAYS), -5.0)
    assert (A.forecasts_from_offsets(fit, np.zeros((1, C.N_DAYS)), np.zeros(1), np.array([800]), h=2) == 0.0).all()


def test_mase_von_hand_mit_und_ohne_maske():
    F = np.array([[[1.0, 2.0], [3.0, 3.0]]])
    Act = np.array([[[2.0, 4.0], [3.0, 5.0]]])
    # mittlere absolute Fehler je Ursprung: 1,5 und 1,0
    assert A.mase(F, Act, np.array([2.0]))[0] == pytest.approx((1.5 + 1.0) / 2 / 2.0)
    assert A.mase(F, Act, np.array([2.0]), mask=np.array([[False, True]]))[0] == pytest.approx(1.0 / 2.0)
    assert A.mase(F, Act, np.array([2.0]), mask=np.array([[False, False]]))[0] == 0.0         # leeres Fenster: kein Teilen durch 0


def test_glaettung_haelt_ein_reines_wochenmuster_exakt():
    y = np.tile(np.arange(1, 8) * 10.0, 5)[None, :]                          # 35 Tage, Muster 10, 20, ..., 70
    Lh, Sh = A.holt_winters(y, 0.3, 0.1)
    assert Lh[0] == pytest.approx(np.full(35, 40.0))
    F = A.hw_forecasts(Lh, Sh, np.array([30]), h=7)
    assert F[0, 0] == pytest.approx([10.0 * ((30 + j) % 7 + 1) for j in range(7)])                      # das Muster setzt sich fort


def test_saisonal_naiv_und_wochenmittel_von_hand():
    y = (np.arange(60.0) ** 2)[None, :]
    org = np.array([40])
    assert A.seasonal_naive_forecasts(y, org, h=3)[0, 0] == pytest.approx([33 ** 2, 34 ** 2, 35 ** 2])
    wm = A.weekly_mean_forecasts(y, org, h=2, weeks=2)[0, 0]
    assert wm == pytest.approx([(33 ** 2 + 26 ** 2) / 2, (34 ** 2 + 27 ** 2) / 2])
    assert A.expectation_forecasts(y, org, h=2)[0, 0] == pytest.approx([40 ** 2, 41 ** 2])


# --- Sprungformen und Erzeuger ------------------------------------------------------------------------------------------------------------------
def _factor(form, shift, day_after):
    """Niveaufaktor `day_after` Tage nach dem Sprungtag aus der Sprungfolge (Faktoren werden multipliziert)."""
    f = 1.0
    for off, rel in S.jumps_for(form, shift):
        if day_after >= off:
            f *= 1.0 + rel
    return f


@pytest.mark.parametrize("form,expect", [
    ("step", {0: 1.3, 500: 1.3}),
    ("step_back", {0: 1.3, 89: 1.3, 90: 1.0, 500: 1.0}),
    ("step_two", {0: 1.3, 119: 1.3, 120: 1.3 * (1 + 0.3 * 2 / 3)}),
    ("promo", {0: 1.3, 6: 1.3, 7: 1.0, 300: 1.0}),
    ("spike", {0: 1.3, 1: 1.0, 300: 1.0}),
    ("ramp", {0: 1.3 ** (1 / 60), 59: 1.3, 300: 1.3}),
])
def test_sprungformen_haben_den_erwarteten_verlauf(form, expect):
    for day, f in expect.items():
        assert _factor(form, 0.3, day) == pytest.approx(f), (form, day)


def test_ohne_groesse_gibt_es_keine_sprungfolge():
    assert S.jumps_for("step", 0) == [] and S.jumps_for("ramp", 0) == []
    with pytest.raises(ValueError):
        S.jumps_for("zickzack", 0.3)


def test_fenster_ist_lang_bei_folgen_von_spruengen():
    assert S.batch((1,), "step", 0.3).window == C.WINDOW_SHORT
    assert S.batch((1,), "ramp", 0.3).window == C.WINDOW_SHORT
    assert S.batch((1,), "step_back", 0.3).window == C.WINDOW_LONG
    assert S.batch((1,), "step_two", 0.3).window == C.WINDOW_LONG


def test_reihen_mit_und_ohne_sprung_teilen_alle_zufallszahlen():
    a, b = S.batch((5,), "step", 0.3), S.batch((5,), "step", 0.0)
    assert a.J[0] == b.J[0] and (a.promo == b.promo).all()
    ratio = a.mu[0, a.J[0]:] / b.mu[0, a.J[0]:]
    assert ratio == pytest.approx(np.full(len(ratio), 1.3))
    assert (a.mu[0, :a.J[0]] == b.mu[0, :a.J[0]]).all()
    assert a.has_jump.tolist() == [True] and b.has_jump.tolist() == [False]


def test_sprungtag_liegt_im_testjahr():
    b = S.batch(range(200), "step", 0.3)
    assert b.J.min() >= C.FIRST_TEST + 30 and b.J.max() < C.N_DAYS - 60


def test_autokorrelation_wird_erzeugt_und_hat_einheitsvarianz():
    z = []
    for seed in range(30):
        y, mu, *_ = S.generate(seed, noise=0.14, ar=0.6)
        z.append(np.log((y + 0.0) / mu))
    assert np.corrcoef(np.concatenate([w[1:] for w in z]), np.concatenate([w[:-1] for w in z]))[0, 1] == pytest.approx(0.6, abs=0.05)


def test_concat_trennt_reihen_mit_und_ohne_sprung():
    c = S.concat(S.batch((1, 2), "step", 0.3), S.batch((3, 4), "step", 0.0))
    assert c.n == 4 and c.has_jump.tolist() == [True, True, False, False] and c.seeds == (1, 2, 3, 4)


def test_auslenkung_beginnt_am_tag_nach_dem_letzten_nullstand(ref_ten):
    """Beginn der Auslenkung (t0) = erster Tag der laufenden Auslenkung: der Tag nach dem letzten Tag, an dem die Statistik 0 war (README, Modell)."""
    R = ref_ten(0.0)
    R[0, 15:] = 3.0                                                          # Tage 10 bis 14: Residuum 0 (Statistik bleibt 0), Sprung ab Tag 15
    out = A.track(R, "cusum", {"alpha": 0.0, "h": 4.0, "k": 0.5, "wmin": 1}, ref=REF)
    assert out["stat"][0, 14] == 0.0 and out["stat"][0, 15] > 0                # letzter Nullstand Tag 14, erste Auslenkung Tag 15
    assert np.nonzero(out["alarm"][0])[0].tolist() == [16]
    assert out["O"][0, 16] == pytest.approx(3.0)                              # Mittel der Tage 15 bis 16; begänne das Fenster am Nullstand (Tag 14), wäre es (0 + 3 + 3) / 3 = 2
