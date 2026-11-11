"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import nvs_constants as C
import nvs_evaluation as ev


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None       # Auswahlregler: erlaubte Werte (statt Grenzen)


_D = ev.Settings()
SETTING_SPECS = {
    "noise": SettingSpec("rauschen", int, _D.noise, C.NOISE_MIN, C.NOISE_MAX),
    "shift": SettingSpec("sprung", int, _D.shift, C.SHIFT_MIN, C.SHIFT_MAX),
    "form": SettingSpec("form", str, _D.form, choices=C.FORMS),
    "ar": SettingSpec("ar", float, _D.ar, C.AR_MIN, C.AR_MAX),
    "model": SettingSpec("modell", str, _D.model, choices=C.MODELS),
    "h": SettingSpec("h", float, _D.h, C.H_MIN, C.H_MAX),
    "k": SettingSpec("k", float, _D.k, C.K_MIN, C.K_MAX),
    "alpha_low": SettingSpec("alpha_low", float, _D.alpha_low, C.ALPHA_LOW_MIN, C.ALPHA_LOW_MAX),
    "wmin": SettingSpec("wmin", int, _D.wmin, C.WMIN_MIN, C.WMIN_MAX),
    "ewma": SettingSpec("ewma", float, _D.ewma, C.EWMA_MIN, C.EWMA_MAX),
    "ma": SettingSpec("ma", int, _D.ma, C.MA_MIN, C.MA_MAX),
    "trigg_lam": SettingSpec("trigg_lam", float, _D.trigg_lam, C.TRIGG_LAM_MIN, C.TRIGG_LAM_MAX),
    "trigg_amin": SettingSpec("trigg_amin", float, _D.trigg_amin, C.TRIGG_AMIN_MIN, C.TRIGG_AMIN_MAX),
    "confirm_rule": SettingSpec("bestaetigung", str, _D.confirm_rule, choices=tuple(C.CONFIRM_RULES)),
    "tau": SettingSpec("tau", float, _D.tau, choices=C.TAU_OPTIONS),
    "lead": SettingSpec("lead", int, _D.lead, C.LEAD_MIN, C.LEAD_MAX),
    "rule": SettingSpec("regel", str, _D.rule, choices=C.RULES),
    "runs": SettingSpec("reihen", int, _D.runs, C.RUNS_MIN, C.RUNS_MAX),
    "seed": SettingSpec("seed", int, _D.seed, 0, C.SEED_MAX),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def coerce(spec, raw):
    """Wert aus der Adresszeile: umwandeln, auf die Grenzen begrenzen bzw. auf erlaubte Werte prüfen; ungültig -> None."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        match = [c for c in spec.choices if c == value]
        return match[0] if match else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = coerce(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    `tests/test_layout.py` prüft, dass jeder Regler der Seitenleiste in SETTING_SPECS steht und seine Änderung die Markierung aufhebt."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed"] = random.randint(0, C.SEED_MAX - C.RUNS_MAX)
