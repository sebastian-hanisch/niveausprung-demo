"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusätze für diese Demo: Auswahlregler (Seitenleiste) und die Auswahl im Hauptbereich ---------------------------------------------------------

import nvs_constants as C  # noqa: E402
import nvs_evaluation as ev  # noqa: E402

SELECTBOXES = {"form": C.FORMS, "model": C.MODELS, "confirm_rule": tuple(C.CONFIRM_RULES), "rule": C.RULES}
RADIOS = {"main_measure": ev.MEASURES, "karte_measure": ("kosten_alle", "kosten_fenster", "mase_alle", "mase_fenster"), "karte_ar": C.KARTE_AR}


def test_every_choice_of_every_selectbox_does_not_raise():
    assert {b.key for b in _fresh_app().selectbox} == set(SELECTBOXES)
    for key, choices in SELECTBOXES.items():
        for choice in choices:
            at = _fresh_app()
            at.selectbox(key=key).select(choice).run(timeout=TIMEOUT)
            assert not at.exception, (key, choice, [str(e) for e in at.exception])


def test_every_choice_of_the_main_radios_and_the_tau_slider_does_not_raise():
    assert {r.key for r in _fresh_app().radio} == set(RADIOS)
    for key, choices in RADIOS.items():
        for choice in choices:
            at = _fresh_app()
            at.radio(key=key).set_value(choice).run(timeout=TIMEOUT)
            assert not at.exception, (key, choice, [str(e) for e in at.exception])
    for tau in C.TAU_OPTIONS:
        at = _fresh_app()
        at.select_slider(key="tau").set_value(tau).run(timeout=TIMEOUT)
        assert not at.exception, tau


def test_expander_stehen_nach_titel_und_die_formeln_haben_genau_eine_lesehilfe():
    at = _fresh_app()
    labels = [e.label for e in at.expander]
    assert "💬 In einfachen Worten" in labels and "📐 Mathematische Formulierung" in labels
    assert labels.index("💬 In einfachen Worten") < labels.index("So funktioniert Niveausprung erkennen") < labels.index("📐 Mathematische Formulierung")
    mathe = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")
    text = " ".join(m.value for m in mathe.markdown)
    assert text.count("**Lesehilfe.**") == 1 and "| Symbol | Bedeutung |" in text and "$$" in text      # eine Lesehilfe, Formeln erhalten
    assert "Direkt darunter steht der Expander **💬 In einfachen Worten**" in " ".join(m.value for m in at.markdown)
