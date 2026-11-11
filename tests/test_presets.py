"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink-Begrenzung; Orakel für die statistischen Hilfsgrößen."""

import numpy as np
import pytest

import nvs_constants as C
import nvs_evaluation as ev
import nvs_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_or_choices():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[C.DEFAULT_PRESET]) == ev.Settings()
    assert ev.Settings().seed == 3                                              # Seed 3: das Preset der Linie (+30 % an Tag 809)


def test_setting_specs_cover_every_setting_and_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(ev.SETTING_NAMES)


def test_setting_defaults_gleichen_den_standardwerten():
    for key, spec in P.SETTING_SPECS.items():
        assert spec.default == getattr(ev.Settings(), key), key


def test_presets_unterscheiden_sich_und_nur_die_szenarioregler_wechseln():
    base = C.PRESETS[C.DEFAULT_PRESET]
    differing = {name: {k for k in p if p[k] != base[k]} for name, p in C.PRESETS.items() if name != C.DEFAULT_PRESET}
    assert all(diff for diff in differing.values())                              # jedes Preset ändert etwas
    assert set().union(*differing.values()) <= {"noise", "shift", "form", "ar", "confirm_rule", "h", "k", "alpha_low", "wmin", "ewma", "ma", "trigg_lam", "trigg_amin"}
    assert len({tuple(sorted(p.items())) for p in C.PRESETS.values()}) == len(C.PRESETS)


def test_permalink_begrenzt_auf_die_reglergrenzen_und_prueft_auswahlwerte():
    spec = P.SETTING_SPECS["noise"]
    assert P.coerce(spec, "999") == C.NOISE_MAX and P.coerce(spec, "-3") == C.NOISE_MIN and P.coerce(spec, "x") is None
    assert P.coerce(P.SETTING_SPECS["form"], "ramp") == "ramp" and P.coerce(P.SETTING_SPECS["form"], "zickzack") is None
    assert P.coerce(P.SETTING_SPECS["tau"], "0.99") == 0.99 and P.coerce(P.SETTING_SPECS["tau"], "0.97") is None
    assert P.coerce(P.SETTING_SPECS["rule"], "conformal") == "conformal"


# --- Orakel für die Hilfsgrößen ----------------------------------------------------------------------------------------------------------------
def test_paired_gain_gegen_die_formel_von_hand():
    ref = [10.0, 12.0, 8.0, 10.0]
    other = [9.0, 10.0, 8.0, 9.0]                                                # Differenzen 1, 2, 0, 1: Mittel 1, Standardabweichung (ddof 1) sqrt(2/3)
    pct, se = ev.paired_gain(ref, other)
    assert pct == pytest.approx(10.0)                                            # 1 / 10 in Prozent
    assert se == pytest.approx(np.sqrt(2 / 3) / 2 / 10 * 100)                    # sd / sqrt(4) / Mittel(ref) in Prozent


def test_mean_se_gegen_die_formel_von_hand():
    m, se = ev.mean_se([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and se == pytest.approx(np.sqrt(5 / 3) / 2)


def test_staerkster_gegner_ist_der_mit_dem_kleinsten_mittel():
    a = ev.analyse(ev.Settings(runs=12))
    for ms in ev.MEASURES:
        best = ev.strongest_opponent(a, ms)
        assert all(a.metrics[best][ms].mean() <= a.metrics[o][ms].mean() for o in C.OPPONENTS)
