"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): jede Zeile prüft einen Rand der eigenen Routinen (eine Reihe, kein Sprung, kein Alarm, alle Regler am Rand)."""

import numpy as np
import pytest

import nvs_algorithm as A
import nvs_constants as C
import nvs_evaluation as ev
import nvs_scenario as S


def _finite(a: ev.Analysis):
    for m, d in a.metrics.items():
        for k, v in d.items():
            assert np.isfinite(v).all(), (m, k)
    assert np.isfinite(a.meldungen).all()


def test_genau_eine_reihe_hat_keinen_standardfehler_aber_keinen_absturz():
    a = ev.analyse(ev.Settings(runs=1))
    _finite(a)
    summ = ev.summary(a)
    assert summ["kosten_alle"]["means"]["detector"][1] == 0.0                  # eine Reihe: Standardfehler 0 statt NaN
    assert summ["kosten_alle"]["gain_detector"][1] == 0.0


def test_genau_zwei_reihen():
    a = ev.analyse(ev.Settings(runs=2))
    _finite(a)
    assert ev.summary(a)["kosten_alle"]["means"]["detector"][1] > 0.0


def test_ohne_sprungfolge_ist_das_orakel_die_langsame_ewma():
    b = S.batch(range(3, 9), "step", 0.0)
    P = ev.prepare(b, "calendar")
    tr = A.track(P.R, "oracle", {"alpha": C.ORACLE_ALPHA}, jumps=A.oracle_jumps(b.J, b.jumps, b.has_jump))
    assert tr["O"].tolist() == A.track(P.R, "ewma", {"alpha": C.ORACLE_ALPHA})["O"].tolist()      # ohne Sprungfolge ist das Orakel die langsame EWMA


def test_detektor_ohne_alarm_laesst_verzoegerung_leer():
    a = ev.analyse(ev.Settings(h=C.H_MAX, noise=C.NOISE_MAX, shift=C.SHIFT_MIN, runs=12))
    assert not a.found.any() and a.delay_days is None and "oracle_delayed" not in a.metrics
    _finite(a)
    kind, text = ev.verdict_main(ev.summary(a), a.delay_days, float(a.found.mean()))
    assert "erst nach" not in text and "nur in 0 %" in text and "Änderung" in text


def test_extreme_regler_an_den_raendern():
    for kw in ({"noise": C.NOISE_MIN}, {"noise": C.NOISE_MAX, "shift": C.SHIFT_MAX}, {"ar": C.AR_MAX}, {"h": C.H_MIN, "k": C.K_MIN}, {"h": C.H_MAX, "k": C.K_MAX},
               {"alpha_low": C.ALPHA_LOW_MAX, "wmin": C.WMIN_MAX}, {"ewma": C.EWMA_MAX, "ma": C.MA_MAX}, {"trigg_lam": C.TRIGG_LAM_MAX, "trigg_amin": C.TRIGG_AMIN_MAX},
               {"confirm_rule": "c21"}, {"confirm_rule": "c14t15"}, {"lead": C.LEAD_MAX, "tau": 0.99, "rule": "conformal"}, {"lead": 0, "tau": 0.8}, {"model": "weekday"}):
        a = ev.analyse(ev.Settings(runs=6, **kw))
        _finite(a)
        ev.summary(a)


@pytest.mark.parametrize("form", C.FORMS)
def test_jede_form_laeuft_durch_und_gibt_endliche_zahlen(form):
    a = ev.analyse(ev.Settings(form=form, shift=C.SHIFT_MAX, runs=6))
    _finite(a)
    assert a.window == (C.WINDOW_LONG if form in ("step_back", "step_two") else C.WINDOW_SHORT)


def test_mase_nenner_ist_nie_null():
    assert (S.mase_scale(S.batch(range(20), "step", 0.3).y) > 0).all()


def test_paired_gain_ohne_bezugswert():
    assert ev.paired_gain([0.0, 0.0], [0.0, 0.0]) == (0.0, 0.0)                   # Mittel 0: kein Teilen durch 0
    assert ev.mean_se([5.0]) == (5.0, 0.0)


def test_kennlinie_und_bestaetigung_laufen_bei_extremen_einstellungen():
    rows = ev.kennlinie(ev.Settings(noise=C.NOISE_MAX, shift=C.SHIFT_MIN), hs=(2.0, 20.0), runs=6)
    assert [r["h"] for r in rows] == [2.0, 20.0] and rows[1]["gefunden"] == 0.0 and rows[1]["verzoegerung"] is None
    c, rows = ev.confirm_experiment(ev.Settings(confirm_rule="c21"), runs=6)
    assert c == 21 and {r["form"] for r in rows} == set(ev.CONFIRM_FORMS) | {"none"}
    assert ev.confirm_experiment(ev.Settings(confirm_rule="off"), runs=6)[0] == 10        # Wartezeit 0 in der Seitenleiste: der Versuch vergleicht mit 10 Tagen
