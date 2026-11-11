"""Jede Zahl der README steht hier (DEMO-PLAYBOOK §4, §11a): Tabellen exakt gegen die gespeicherten Messungen (`data/*.json`), Sätze Zahl für Zahl als formatierte Fundstelle, die Messungen
selbst mit dem Code der Demo neu gerechnet (Band 0,05 Prozentpunkte), Zählungen über die Karte, Voreinstellungen gegen die Abstimmung der Zelle, Verdikte jedes Presets.

Die Messungen (`data/*.json`) stammen aus `tools/messlauf.py`; die Zufalls-Seeds stehen in `nvs_constants.py`. Was die Tests nicht prüfen: die Vormessung (Spalte „Vormessung“ der Vergleichstabelle) ist die Messung eines
anderen Zufallsstroms und steht in der README als Zahl der Vormessung, nicht als Zahl dieser Demo."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import nvs_constants as C  # noqa: E402
import nvs_evaluation as ev  # noqa: E402
import nvs_karte as K  # noqa: E402
import nvs_messung as M  # noqa: E402
import readme_tabellen as RT  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
f, pm = RT.f, RT.pm


def _load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


HEAD, KARTE, CONFIRM = _load("head.json"), _load("karte.json"), _load("confirm.json")["ergebnis"]
Z = KARTE["zellen"]
WDH = _load("karte_wiederholung.json")["zellen"]
LAUT = _load("laute_reihe.json")
al = _load("alpha.json")["ergebnis"]


def test_tabellen_der_readme_sind_erzeugt():
    assert RT.render(README) == README                                       # kein Block weicht von der Messung ab
    for name in RT.BLOCKS:
        assert f"<!-- BEGIN:{name} -->" in README and f"<!-- END:{name} -->" in README, name


def _g(rule, ref, gegner, key):
    return HEAD[rule + "_gewinn"][ref][gegner][key]


def _satz_zahlen():
    """Alle formatierten Zahlen der Sätze der README mit ihrer Quelle in der Messung: Liste (Beschreibung, Fundstelle)."""
    out = []
    # Befund 1: Kennlinie
    for model, h_txt in (("calendar", "**h = 6**"), ("fourier_trend", "h = 8"), ("weekday", "h = 10")):
        h, row = RT.kennlinie_betrieb(model)
        r28 = row["sprung_0.28"]
        assert h_txt == (f"**h = {f(h, 0)}**" if model == "calendar" else f"h = {f(h, 0)}"), model
        out.append((f"Kennlinie {model}", f"{f(r28['anteil'] * 100, 0)} % der Reihen nach {f(r28['verzoegerung'], 1)} Tagen" if model != "weekday" else f"{f(r28['anteil'] * 100, 0)} % nach {f(r28['verzoegerung'], 1)} Tagen"))
    h, row = RT.kennlinie_betrieb("calendar")
    out.append(("Kennlinie Meldungen", f"{f(row['fehlmeldungen'][0], 2)} Meldungen je Jahr"))
    # Befund 2: Standardszenario
    e = "empirical"
    out += [
        ("MASE ewma", f"{f(_g(e, 'cusum[mase]', 'ewma[mase]', 'mase_alle')[0], 1, True)} % MASE"),
        ("Kosten ewma", f"{f(_g(e, 'cusum[kosten]', 'ewma[kosten]', 'kosten_alle')[0], 1, True)} % Kosten gegen die EWMA"),
        ("Kosten MA", f"gegen das Mittel der letzten k Tage {f(_g(e, 'cusum[kosten]', 'ma[kosten]', 'kosten_alle')[0], 1, True)} %"),
        ("Kosten Trigg", f"gegen Trigg-Leach {f(_g(e, 'cusum[kosten]', 'trigg[kosten]', 'kosten_alle')[0], 1, True)} %"),
        ("Fenster ewma", f"gegen die EWMA {pm(_g(e, 'cusum[kosten]', 'ewma[kosten]', 'kosten_fenster'))} %"),
        ("Fenster trigg", f"{pm(_g(e, 'cusum[kosten]', 'trigg[kosten]', 'kosten_fenster'))} % Kosten (konforme Regel {pm(_g('conformal', 'cusum[kosten]', 'trigg[kosten]', 'kosten_fenster'))} %)"),
        ("MASE Regression gegen Glättung", f"MASE von {f(HEAD[e]['ewma[mase]']['mase_alle'][0], 3)} gegen {f(HEAD[e]['hw']['mase_alle'][0], 3)} der festen Glättung; das Zurücksetzen bringt dazu {f(HEAD[e]['cusum[mase]']['mase_alle'][0], 3)}"),
        ("Orakel Jahr", f"{pm(_g(e, 'oracle[kosten]', 'ewma[kosten]', 'kosten_alle'))} % Kosten im Testjahr und {f(_g(e, 'oracle[kosten]', 'ewma[kosten]', 'kosten_fenster')[0], 1, True)} % im Fenster"),
        ("Anteil Reserve", f"holt davon im Testjahr {f(_g(e, 'cusum[kosten]', 'ewma[kosten]', 'kosten_alle')[0] / _g(e, 'oracle[kosten]', 'ewma[kosten]', 'kosten_alle')[0] * 100, 0)} % ({f(_g(e, 'cusum[kosten]', 'ewma[kosten]', 'kosten_alle')[0], 1)} von {f(_g(e, 'oracle[kosten]', 'ewma[kosten]', 'kosten_alle')[0], 1)} Prozentpunkten, Quotient der ungerundeten Werte)"),
        ("Verzögerung", f"({HEAD['verzoegerung_tage']} Tage, im Mittel über die {f(HEAD['gefunden'] * 100, 0)} % gemeldeten Sprünge)"),
        ("verspätetes Orakel Jahr", f"{f(HEAD[e]['oracle_delayed']['kosten_alle'][0], 3)} gegen {f(HEAD[e]['cusum[kosten]']['kosten_alle'][0], 3)} des Detektors"),
        ("verspätetes Orakel Fenster", f"{f(HEAD[e]['oracle_delayed']['kosten_fenster'][0], 3)} gegen {f(HEAD[e]['cusum[kosten]']['kosten_fenster'][0], 3)}"),
        ("konform ohne Nachführung", f"{f(HEAD['conformal']['none']['kosten_alle'][0], 3)} gegen {f(HEAD['conformal']['cusum[kosten]']['kosten_alle'][0], 3)} des Zurücksetzens ({pm(_g('conformal', 'cusum[kosten]', 'none', 'kosten_alle'))} %)"),
        ("konform ewma", f"({pm(_g('conformal', 'cusum[kosten]', 'ewma[kosten]', 'kosten_alle'))} %)"),
    ]
    # Befund 3: Karte
    z = Z
    out += [
        ("Karte 4 %, +50 %", f"Größter Wert der Zeile Rauschen 4 %: {f(z['4|50|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'][0], 1, True)} % (Sprung +50 %)"),
        ("Karte 14 %, +30 %", f"{pm(z['14|30|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'])} % bei Rauschen 14 % und +30 %"),
        ("Karte 24 %, +50 %", f"bei +50 % {pm(z['24|50|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'])}"),
    ]
    multi = KARTE["multi"]["14"]
    out += [
        ("Rücksprung", f"+{f(multi['step_back']['messung']['kosten']['gewinn_detektor']['kosten_alle'][0], 1)} ± {f(multi['step_back']['messung']['kosten']['gewinn_detektor']['kosten_alle'][1], 1)} % Kosten im Testjahr gegen {pm(multi['step']['messung']['kosten']['gewinn_detektor']['kosten_alle'])} % beim Einzelsprung"),
        ("zweiter Sprung 14 %", f"im Testjahr keinen Gewinn ({pm(multi['step_two']['messung']['kosten']['gewinn_detektor']['kosten_alle'])} %)"),
        ("zweiter Sprung 8 %", f"Fensterwert bei Rauschen 8 % ({pm(KARTE['multi']['8']['step_two']['messung']['kosten']['gewinn_detektor']['kosten_fenster'])} %)"),
    ]
    # Befund 4: Bestätigung
    def a(form, theta, c, key="anteil"):
        return CONFIRM[f"{form}|{theta}|{c}"][key]
    out += [
        ("Aktion +50 %", f"der einwöchigen +50-%-Aktion von {f(a('Aktionswoche +50 %', '1.0', 0) * 100, 1)} % auf {f(a('Aktionswoche +50 %', '1.0', 10) * 100, 1)} %"),
        ("Aktion +25 %", f"die der +25-%-Aktion von {f(a('Aktionswoche +25 %', '1.0', 0) * 100, 1)} % auf {f(a('Aktionswoche +25 %', '1.0', 10) * 100, 1)} %"),
        ("Ausreißer", f"des Ausreißers von {f(a('Ausreißer +100 %', '1.0', 0) * 100, 1)} % auf {f(a('Ausreißer +100 %', '1.0', 10) * 100, 1)} %"),
        ("Stufe bleibt", f"zu {f(a('Stufe +28 %', '1.0', 10) * 100, 1)} % bestätigt, mit {f(a('Stufe +28 %', '1.0', 10, 'verzoegerung'), 1)} statt {f(a('Stufe +28 %', '1.0', 0, 'verzoegerung'), 1)} Tagen Verzögerung (+{f(a('Stufe +28 %', '1.0', 10, 'verzoegerung') - a('Stufe +28 %', '1.0', 0, 'verzoegerung'), 1)} Tage)"),
        ("c = 7", f"Mit c = 7 bleiben {f(a('Aktionswoche +50 %', '1.0', 7) * 100, 1)} %"),
        ("c = 14", f"mit c = 14 sind es {f(a('Aktionswoche +50 %', '1.0', 14) * 100, 1)} %"),
        ("Stufe +14 %", f"{f(a('Stufe +14 %', '1.0', 10) * 100, 1)} % statt {f(a('Stufe +14 %', '1.0', 0) * 100, 1)} % bestätigt"),
        ("Rampe", f"auf +28 % zu {f(a('Rampe auf +28 %', '1.0', 10) * 100, 1)} % statt {f(a('Rampe auf +28 %', '1.0', 0) * 100, 1)} %"),
        ("theta 1,5", f"nur noch zu {f(a('Stufe +28 %', '1.5', 10) * 100, 1)} % bestätigt"),
    ]
    cr = _load("confirm_ruhig.json")["ergebnis"]
    out += [
        ("ruhig c=10", f"mit c = 10 zu {f(cr['Aktionswoche +50 %|1.0|10']['anteil'] * 100, 1)} % bestätigt"),
        ("ruhig c=14", f"mit c = 14 nur zu {f(cr['Aktionswoche +50 %|1.0|14']['anteil'] * 100, 1)} %"),
        ("ruhig Verzögerung", f"Verzögerung {f(cr['Aktionswoche +50 %|1.0|0']['verzoegerung'], 1)} Tage"),
    ]
    # Befund 5: Kosten
    kk = _load("kosten.json")
    def kg(noise, tau, lead, key="kosten_alle"):
        return pm(kk[noise][f"{tau}|{lead}"][key]["gewinn_detektor"])
    out += [
        ("Kosten 14 %, 0,8, 7", f"τ = 0,8, L = 7: {kg('14', '0.8', 7)} %"),
        ("Kosten 14 % L = 7", f"bei Rauschen 14 % mit L = 7 ({kg('14', '0.99', 7)} %)"),
        ("Kosten 8 % L = 3 und 7", f"bei Rauschen 8 % mit L = 3 ({kg('8', '0.99', 3)} %) und L = 7 ({kg('8', '0.99', 7)} %)"),
        ("Kosten 14 % L = 3", f"Rauschen 14 % mit L = 3 ({kg('14', '0.99', 3)} %)"),
        ("Kosten 8 % L = 0", f"Rauschen 8 % mit L = 0 ({kg('8', '0.99', 0)} %)"),
    ]
    # Vergleich mit der Vormessung (Sätze)
    out += [
        ("alpha 0 Stufe +14 %", f"Stufe +14 %: {f(al['Stufe +14 %|0.0']['anteil'] * 100, 1)} % gegen 97,2 %"),
        ("alpha 0 Rampe", f"Rampe: {f(al['Rampe auf +28 %|0.0']['anteil'] * 100, 1)} % gegen 100,0 %"),
        ("Meldungen je Jahr alpha", f"({f(al['fehlmeldungen|0.0'], 2)} bei α = 0, {f(al['fehlmeldungen|0.03'], 2)} bei α = 0,03, {f(al['fehlmeldungen|0.06'], 2)} bei α = 0,06; Vormessung 0,693 bei h = 7)"),
        ("gleiche Rate", f"α = 0 mit h = {f(al['gleiche_rate']['h'], 0)} meldet {f(al['gleiche_rate']['fehlmeldungen'], 2)} Mal je Jahr (höchstens so oft wie α = 0,06 mit h = 6) und erkennt +14 % in {f(al['gleiche_rate']['Stufe +14 %']['anteil'] * 100, 1)} % der Reihen gegen {f(al['Stufe +14 %|0.06']['anteil'] * 100, 1)} %"),
        ("Aktion verworfen", f"c = 10 verwirft bei Rauschen 14 % {f(100 - a('Aktionswoche +50 %', '1.0', 10) * 100, 1)} % der Aktionswochen +50 %"),
        ("Wiederholung 24 %", f"(Wiederholung: {pm(WDH['24|50|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'])} %); die MASE {pm(Z['24|50|0.0']['messung']['mase']['gewinn_detektor']['mase_alle'])} % statt +0,7 % (Wiederholung: {pm(WDH['24|50|0.0']['messung']['mase']['gewinn_detektor']['mase_alle'])} %)"),
        ("Wiederholung 14 %", f"+9,4 ± 0,6 % statt +5,7 % (ein Abstimmungsausreißer: bei der Wiederholung mit anderen Abstimm- und Messreihen {pm(WDH['14|50|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'])} %)"),
        ("Wiederholung 4 %", f"Rauschen 4 %, +20 %: {f(Z['4|20|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'][0], 1, True)} statt {f(WDH['4|20|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'][0], 1, True)} %"),
        ("Laut standard", f"+30 % in {f(LAUT['30|standard']['anteil'] * 100, 1)} % und +50 % in {f(LAUT['50|standard']['anteil'] * 100, 1)} % der Reihen gemeldet"),
        ("Wiederholung Zaehlung", f"{f(sum(1 for k in Z if abs(Z[k]['messung']['kosten']['gewinn_detektor']['kosten_alle'][0] - WDH[k]['messung']['kosten']['gewinn_detektor']['kosten_alle'][0]) > 2 * (Z[k]['messung']['kosten']['gewinn_detektor']['kosten_alle'][1] ** 2 + WDH[k]['messung']['kosten']['gewinn_detektor']['kosten_alle'][1] ** 2) ** 0.5), 0)} von 32 Zellen um mehr als zwei kombinierte Standardfehler"),
        ("Kosten alpha Stufe", f"α = 0,01 gäbe +16,9 statt {f(_g(e, 'oracle[kosten]', 'ewma[kosten]', 'kosten_alle')[0], 1, True)} %"),
        ("verschoben Karte 14 %, +50 %", f"Kosten {pm(Z['14|50|0.0']['messung']['kosten']['gewinn_detektor']['kosten_alle'])} % statt +5,7 %"),
        ("verschoben MASE 14 %, +30 %", f"MASE {f(Z['14|30|0.0']['messung']['mase']['gewinn_detektor']['mase_alle'][0], 1, True)} % statt +1,5 %"),
        ("L = 0, 14 %", f"Rauschen 14 % bleibt es positiv: {pm(kk['14']['0.99|0']['kosten_alle']['gewinn_detektor'])} %"),
        ("24 % MASE", f"die MASE {pm(Z['24|50|0.0']['messung']['mase']['gewinn_detektor']['mase_alle'])} % statt +0,7 %"),
        ("Glättung Nachbau", f"der Nachbau gibt {f(_g(e, 'cusum[mase]', 'hw', 'mase_alle')[0], 1, True)} % und {f(_g(e, 'cusum[kosten]', 'hw', 'kosten_alle')[0], 1, True)} %"),
        ("Ausreißer Abgrenzung", f"in {f(a('Ausreißer +100 %', '1.0', 0) * 100, 1)} % der Reihen als Sprung gemeldet"),
    ]
    return out


def test_saetze_der_readme():
    fehlt = [(name, needle) for name, needle in _satz_zahlen() if needle not in README]
    assert not fehlt, f"Fundstellen fehlen in der README (Satz und Messung weichen ab): {fehlt}"


def test_standardfehler_der_mittel_hoechstens_0_12():
    mx = max(v[1] for rule in ("empirical", "conformal") for m in HEAD[rule].values() for v in m.values())
    assert mx <= 0.12 and "höchstens 0,12" in README


# --- Zählungen ------------------------------------------------------------------------------------------------------------------------------
def test_zaehlungen_ueber_die_karte():
    assert ev.karte_counts(Z, 0.0, "kosten", "kosten_alle") == (10, 0, 16)               # "10 von 16 Zellen ... in keiner Zelle schlechter"
    assert ev.karte_counts(Z, 0.6, "kosten", "kosten_alle") == (1, 6, 16)                # "6 von 16 ... in 1 von 16 spart es"
    assert ev.karte_counts(Z, 0.6, "mase", "mase_alle") == (7, 5, 16)                    # "MASE in 7 von 16, verliert in 5"
    aus = RT.karte_aus()
    assert len(aus) == 8 and len(Z) == 32                                                # "8 von 32 Zellen"
    for needle in ("in den Kosten des Testjahres in 10 von 16 Zellen", "6 von 16 Zellen kostet", "in 1 von 16 spart es", "in 7 von 16 Zellen, verliert in 5", "in 8 von 32 Zellen unter 0,05 Meldungen"):
        assert needle in README.replace("**", ""), needle


def test_rauschen_24_ohne_autokorrelation_ist_null():
    for s in C.KARTE_SHIFT:
        g, se = Z[f"24|{s}|0.0"]["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"]
        assert abs(g) < 0.15 or s == 50, s                                               # "höchstens 0,1 Prozentpunkte" (gerundet), bei +50 % +0,0 ± 0,5
    g, se = Z["24|50|0.0"]["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"]
    assert abs(g) < 2 * se


def test_orakel_reserve_in_jeder_zelle_positiv_und_groesser_als_der_detektor():
    for key, cell in Z.items():
        m = cell["messung"]["kosten"]
        assert m["gewinn_orakel"]["kosten_alle"][0] > 0, key                              # "Die Reserve ist in jeder Zelle positiv"
        assert m["gewinn_orakel"]["kosten_alle"][0] > m["gewinn_detektor"]["kosten_alle"][0], key


# --- Reproduzierbarkeit der gespeicherten Messungen -----------------------------------------------------------------------------------------
BAND = 0.05


def test_messungen_sind_reproduzierbar_karte_zelle():
    best = Z["14|30|0.0"]["best"]
    neu = K.measure(0.14, 0.30, 0.0, best)
    for obj in ("mase", "kosten"):
        for kind in ("gewinn_detektor", "gewinn_orakel"):
            for key in K.KEYS:
                assert neu[obj][kind][key][0] == pytest.approx(Z["14|30|0.0"]["messung"][obj][kind][key][0], abs=BAND), (obj, kind, key)
    assert neu["kosten"]["meldungen_je_reihe"] == pytest.approx(Z["14|30|0.0"]["messung"]["kosten"]["meldungen_je_reihe"], abs=0.02)


def test_messungen_sind_reproduzierbar_abstimmung_der_zelle():
    neu = K.choose(K.tune(0.14, 0.30, 0.0))
    assert neu["kosten"]["gegner"] == Z["14|30|0.0"]["best"]["kosten"]["gegner"] == "ewma"
    assert neu["kosten"]["detektor_params"] == Z["14|30|0.0"]["best"]["kosten"]["detektor_params"]
    assert neu["mase"]["detektor_params"] == Z["14|30|0.0"]["best"]["mase"]["detektor_params"]


def test_messungen_sind_reproduzierbar_bestaetigung_und_kennlinie():
    det = _load("confirm.json")["detektor"]
    neu = M.confirm(det, cs=(0, 10), thetas=(1.0,))
    for key, v in neu.items():
        alt = CONFIRM[key]
        if "anteil" in v:
            assert v["anteil"] == pytest.approx(alt["anteil"], abs=0.005), key
            assert v["verzoegerung"] == pytest.approx(alt["verzoegerung"], abs=0.1) if v["verzoegerung"] is not None else alt["verzoegerung"] is None, key
        else:
            assert v["meldungen_je_jahr"][0] == pytest.approx(alt["meldungen_je_jahr"][0], abs=0.01), key
    kn = M.kennlinie_modelle(det, models=("calendar",), hs=(6.0,), shifts=(0.28,))["calendar|6.0"]
    alt = _load("kennlinie.json")["ergebnis"]["calendar|6.0"]
    assert kn["fehlmeldungen"][0] == pytest.approx(alt["fehlmeldungen"][0], abs=0.01)
    assert kn["sprung_0.28"]["verzoegerung"] == pytest.approx(alt["sprung_0.28"]["verzoegerung"], abs=0.1)


def test_messungen_sind_reproduzierbar_ueberschrift_szenario():
    neu = M.head(Z["14|30|0.0"]["best"])
    for ref, gegner, key in (("cusum[kosten]", "ewma[kosten]", "kosten_alle"), ("cusum[kosten]", "trigg[kosten]", "kosten_fenster"), ("oracle[kosten]", "ewma[kosten]", "kosten_fenster"),
                             ("cusum[mase]", "hw", "mase_alle")):
        assert neu["empirical_gewinn"][ref][gegner][key][0] == pytest.approx(HEAD["empirical_gewinn"][ref][gegner][key][0], abs=BAND), (ref, gegner, key)
    assert neu["verzoegerung_tage"] == HEAD["verzoegerung_tage"]


# --- Voreinstellungen und Verdikte ----------------------------------------------------------------------------------------------------------
def test_voreinstellungen_sind_die_abstimmung_der_zelle():
    b = Z["14|30|0.0"]["best"]["kosten"]
    s = ev.Settings()
    assert (s.h, s.k, s.alpha_low, s.wmin) == (b["detektor_params"]["h"], b["detektor_params"]["k"], b["detektor_params"]["alpha"], b["detektor_params"]["wmin"])
    assert s.ewma == b["abgestimmt"]["ewma"][0]["alpha"] and s.ma == b["abgestimmt"]["ma"][0]["k"]
    assert (s.trigg_lam, s.trigg_amin) == (b["abgestimmt"]["trigg"][0]["lam"], b["abgestimmt"]["trigg"][0]["amin"])
    assert C.ORACLE_ALPHA == b["orakel_params"]["alpha"] == Z["14|30|0.0"]["best"]["mase"]["orakel_params"]["alpha"]
    assert C.HW_ALPHA == Z["14|30|0.0"]["best"]["mase"]["abgestimmt"]["hw_fest"][0]["alpha"]


@pytest.mark.parametrize("preset,cell", [("Ruhige Reihe (Rauschen 4 %)", "4|30|0.0"), ("Laute Reihe (Rauschen 24 %)", "24|30|0.0"), ("Autokorrelierte Fehler (0,6)", "14|30|0.6")])
def test_presets_mit_eigener_zelle_uebernehmen_deren_abstimmung(preset, cell):
    b = Z[cell]["best"]["kosten"]
    p = C.PRESETS[preset]
    assert (p["h"], p["k"], p["alpha_low"], p["wmin"]) == (b["detektor_params"]["h"], b["detektor_params"]["k"], b["detektor_params"]["alpha"], b["detektor_params"]["wmin"])
    assert p["ewma"] == b["abgestimmt"]["ewma"][0]["alpha"] and p["ma"] == b["abgestimmt"]["ma"][0]["k"]
    assert (p["trigg_lam"], p["trigg_amin"]) == (b["abgestimmt"]["trigg"][0]["lam"], b["abgestimmt"]["trigg"][0]["amin"])
    assert (p["noise"], p["shift"], p["ar"]) == (int(cell.split("|")[0]), 30, float(cell.split("|")[2])) and p["shift"] == C.DEFAULT_SHIFT


def _verdicts(name):
    s = ev.Settings(**C.PRESETS[name])
    a = ev.analyse(s)
    summ = ev.summary(a)
    c, rows = ev.confirm_experiment(s)
    return {"main": ev.verdict_main(summ, a.delay_days, float(a.found.mean())), "reg": ev.verdict_regression_share(summ), "kennlinie": ev.verdict_kennlinie(ev.kennlinie(s)),
            "karte": ev.verdict_karte(Z, s.noise, s.shift, s.ar), "confirm": ev.verdict_confirm(c, rows), "found": float(a.found.mean()), "settings": s}


def test_verdikte_der_presets_laute_reihe_meldet_keinen_gewinn():
    v = _verdicts("Laute Reihe (Rauschen 24 %)")
    assert v["main"][0] == "info"                                                                               # kein Gewinn des Erkennens bei Rauschen 24 %
    assert v["karte"][0] == "info" and "hier ist kein Gewinn nachweisbar" in v["karte"][1]


def test_verdikte_der_presets_ueberschrift_szenario_nennt_gewinn_und_karte():
    v = _verdicts(C.DEFAULT_PRESET)
    assert v["karte"][0] == "success" and "**2,4 %**" in v["karte"][1]                                         # Karte, Zelle Rauschen 14 %, +30 %
    assert v["kennlinie"][0] == "success"
    assert v["confirm"][0] == "success"


def test_verdikte_der_presets_autokorrelation_meldet_keinen_gewinn():
    v = _verdicts("Autokorrelierte Fehler (0,6)")
    assert v["karte"][0] == "warning" and "kostet das Zurücksetzen nachweisbar mehr" in v["karte"][1]
    assert v["main"][0] in ("info", "warning")                                                                  # nie "success"


def test_verdikte_der_presets_aktionswoche_ohne_und_mit_bestaetigung():
    ohne = _verdicts("Aktionswoche +50 %, ohne Bestätigung")
    mit = _verdicts("Aktionswoche +50 %, Bestätigung nach 10 Tagen")
    assert ohne["confirm"][0] == "success" and mit["confirm"][0] == "success"                                    # der Versuch vergleicht beide
    assert "in **100 %** der Reihen als Sprung gemeldet" in mit["confirm"][1]
    assert mit["found"] < ohne["found"] and ohne["found"] == 1.0                                                 # mit Bestätigung wird die Aktion kaum noch gemeldet


def test_verdikte_der_presets_ruhige_reihe_trennt_mit_c10_nicht():
    v = _verdicts("Ruhige Reihe (Rauschen 4 %)")
    assert v["confirm"][0] == "info" and "in **100 %** der Reihen als Sprung gemeldet" in v["confirm"][1]
    assert v["karte"][0] == "success"


# --- Wiederholung der Karte, Folgen von Sprüngen, Rauschen 24 %, Nachführung, Kostenannahmen -------------------------------------------------
def test_wiederholung_der_karte():
    import math
    assert ev.karte_counts(WDH, 0.0, "kosten", "kosten_alle") == (11, 0, 16) and ev.karte_counts(WDH, 0.6, "kosten", "kosten_alle") == (1, 6, 16)
    assert ev.karte_counts(WDH, 0.6, "mase", "mase_alle") == (9, 5, 16)
    kosten = lambda d, k: d[k]["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"]  # noqa: E731
    n2 = sum(1 for k in Z if abs(kosten(Z, k)[0] - kosten(WDH, k)[0]) > 2 * math.hypot(kosten(Z, k)[1], kosten(WDH, k)[1]))
    assert n2 == 8 and len(Z) == len(WDH) == 32                                           # "8 von 32 Zellen um mehr als zwei kombinierte Standardfehler"
    assert all(v["messung"]["kosten"]["gewinn_orakel"]["kosten_alle"][0] > v["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"][0] for v in WDH.values())      # Orakel in jeder Zelle vor dem Detektor
    assert round(kosten(WDH, "14|50|0.0")[0], 1) == 6.2 and round(kosten(Z, "14|50|0.0")[0], 1) == 9.4
    assert abs(kosten(WDH, "24|50|0.0")[0]) < 2 * kosten(WDH, "24|50|0.0")[1]
    for text in ("in jeder der 32 Zellen der Karte", "8 von 32 Zellen um mehr als zwei kombinierte Standardfehler", "10 und 11 von 16 Zellen mit Gewinn"):
        assert text in README.replace("\n  ", " "), text


def test_wiederholung_ist_reproduzierbar_in_einer_zelle():
    neu, _ = K.karte_zelle(14, 50, 0.0, tune_seeds=C.TUNE_SEEDS_B, measure_seeds=C.MEASURE_SEEDS_B)
    for obj in ("mase", "kosten"):
        for key in K.KEYS:
            assert neu["messung"][obj]["gewinn_detektor"][key][0] == pytest.approx(WDH["14|50|0.0"]["messung"][obj]["gewinn_detektor"][key][0], abs=BAND), (obj, key)


def test_orakel_alpha_ist_der_kleinste_rasterwert_in_allen_abstimmungen():
    alphas = [cell["best"][obj]["orakel_params"]["alpha"] for d in (Z, WDH) for cell in d.values() for obj in ("mase", "kosten")]
    assert len(alphas) == 128 and set(alphas) == {0.03} and min(p["alpha"] for p in K.GRID["oracle"]) == 0.03        # 64 Abstimmungen der Karte, ebenso viele der Wiederholung
    assert sum(1 for cell in Z.values() for obj in ("mase", "kosten")) == 64 and "allen 64 Abstimmungen" in README


def test_folgen_von_spruengen():
    import numpy as np
    import nvs_scenario as S
    J = S.batch(C.MEASURE_SEEDS, "step", 0.3).J
    assert J.min() >= 760 and J.max() <= 1034                                            # "Tage 760 bis 1034"
    erwartet = {"step": 0.0, "step_back": float((J + 90 >= 1095).mean()), "step_two": float((J + 120 >= 1095).mean())}
    verkuerzt = float((J + 180 > 1095).mean())
    for noise, m in KARTE["multi"].items():
        for form, r in m.items():
            e = r["messung"]["hinter_ende"]
            assert r["messung"]["fenster_tage"] == 180, (noise, form)                    # in allen drei Zeilen dasselbe Fenster
            assert e["letzter_sprung"] == pytest.approx(erwartet[form]) and e["fenster_verkuerzt"] == pytest.approx(verkuerzt), (noise, form)
    assert (round(erwartet["step_back"] * 100, 1), round(erwartet["step_two"] * 100, 1), round(verkuerzt * 100, 1)) == (7.7, 15.7, 37.7)
    for text in ("7,7 %", "15,7 %", "37,7 %", "dasselbe Fenster von 180 Tagen"):
        assert text in README.replace("\n  ", " "), text
    # Einzelsprung-Zeile und Zelle der Karte unterscheiden sich im Fenster, nicht in den Reihen: MASE alle und Kosten Testjahr sind gleich
    assert KARTE["multi"]["14"]["step"]["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"] == pytest.approx(Z["14|30|0.0"]["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"])
    assert not np.isclose(KARTE["multi"]["14"]["step"]["messung"]["kosten"]["gewinn_detektor"]["kosten_fenster"][0], Z["14|30|0.0"]["messung"]["kosten"]["gewinn_detektor"]["kosten_fenster"][0])


def test_zweiter_sprung_wird_gemeldet():
    zs = _load("zweiter_sprung.json")
    neu = M.zweiter_sprung(Z["14|30|0.0"]["best"]["kosten"]["detektor_params"], 14)
    assert neu["reihen_mit_vollem_fenster"] == zs["14"]["reihen_mit_vollem_fenster"] == 187
    assert neu["zweiter_sprung_gemeldet"] == pytest.approx(zs["14"]["zweiter_sprung_gemeldet"], abs=0.01)
    assert (round(zs["14"]["zweiter_sprung_gemeldet"] * 100, 1), round(zs["14"]["einzelsprung_20_gemeldet"] * 100, 1), zs["8"]["zweiter_sprung_gemeldet"], zs["8"]["einzelsprung_20_gemeldet"]) == (77.5, 74.3, 1.0, 1.0)
    for text in ("77,5 % der Reihen", "74,3 % der Reihen", "100,0 % (beide)", "187 Reihen"):
        assert text in README, text


def test_rauschen_24_sprung_ist_mit_standardwerten_erkennbar():
    neu = M.laute_reihe({"30": Z["24|30|0.0"]["best"]["kosten"]["detektor_params"], "50": Z["24|50|0.0"]["best"]["kosten"]["detektor_params"]})
    for key, r in LAUT.items():
        assert neu[key]["anteil"] == pytest.approx(r["anteil"], abs=0.005), key
    assert LAUT["30|standard"]["anteil"] > 0.5 > LAUT["30|abgestimmt"]["anteil"] and LAUT["50|standard"]["anteil"] > 0.9            # abgestimmt: kaum gemeldet; Standard: gemeldet
    assert LAUT["30|abgestimmt"]["parameter"]["h"] == 8 and LAUT["30|abgestimmt"]["parameter"]["k"] == 1.0                           # "h = 8, k = 1,0" der Zelle
    assert C.PRESETS["Laute Reihe (Rauschen 24 %)"]["h"] == 8.0


def test_nachfuehrung_bei_gleicher_meldungsrate_bleibt_der_abstand():
    det = _load("alpha.json")["detektor"]
    neu = M.nachfuehrung(det)
    assert neu["gleiche_rate"]["h"] == al["gleiche_rate"]["h"] == 10.0
    assert neu["fehlmeldungen|0.0"] == pytest.approx(al["fehlmeldungen|0.0"], abs=0.01)
    g = al["gleiche_rate"]
    assert g["fehlmeldungen"] <= al["fehlmeldungen|0.06"] and g["Stufe +14 %"]["anteil"] > 2 * al["Stufe +14 %|0.06"]["anteil"]


def test_kosten_kippt_nur_teilweise_ueber_zwei_standardfehler():
    kk = _load("kosten.json")
    z = lambda noise, lead: (lambda g: g[0] / g[1])(kk[noise][f"0.99|{lead}"]["kosten_alle"]["gewinn_detektor"])  # noqa: E731
    assert z("14", 7) < -2 and z("8", 3) < -2 and z("8", 7) < -2                                  # "mehr als zwei Standardfehler unter null"
    assert -2 < z("14", 3) < 0 and -2 < z("8", 0) < 0                                             # "innerhalb von zwei Standardfehlern"


def test_orakel_ist_massstab_keine_obergrenze():
    hinter = {}
    for name, d in (("karte", Z), ("wiederholung", WDH)):
        hinter[name] = sorted(k for k, v in d.items() if v["messung"]["mase"]["gewinn_orakel"]["mase_alle"][0] < 0)
        assert not [k for k, v in d.items() if v["messung"]["kosten"]["gewinn_orakel"]["kosten_alle"][0] < 0], name          # in den Kosten nie hinter dem Gegner
    assert hinter["karte"] == ["24|10|0.0", "24|10|0.6"]                                                                      # "2 von 32 Zellen (Rauschen 24 %, Sprung +10 %, ohne und mit Autokorrelation)"
    g0, g6 = (Z[k]["messung"]["mase"]["gewinn_orakel"]["mase_alle"] for k in hinter["karte"])
    for text in (f"{pm(g0, 2)} % und {pm(g6, 2)} %", "keine strenge Obergrenze", "2 von 32 Zellen", "32 Zellen mal 2 Ziele", "0,03 / 0,06 / 0,1 / 0,15"):
        assert text in README, text
    assert [p["alpha"] for p in K.GRID["oracle"]] == [0.03, 0.06, 0.1, 0.15]


def test_autokorrelation_erhoeht_die_meldungen_ohne_sprung():
    ar = _load("autokorrelation.json")
    neu = M.autokorrelation_fehlmeldungen(ar["detektor"])
    for k in ("0.0", "0.6"):
        assert neu[k]["meldungen_je_jahr"][0] == pytest.approx(ar["ergebnis"][k]["meldungen_je_jahr"][0], abs=0.02), k
    a0, a6 = ar["ergebnis"]["0.0"]["meldungen_je_jahr"], ar["ergebnis"]["0.6"]["meldungen_je_jahr"]
    assert a6[0] > 10 * a0[0]
    assert f"{f(a6[0], 2)} ± {f(a6[1], 2)} statt {f(a0[0], 2)} ± {f(a0[1], 2)} Mal je Jahr" in README
    assert "später falsch" not in README                                                      # „später“ ist nicht gemessen


def test_mase_definition_der_readme_ist_die_des_codes():
    import numpy as np
    import nvs_scenario as S
    b = S.batch((3,), "step", 0.3)
    nenner = float(np.mean([abs(b.y[0, t] - b.y[0, t - 7]) for t in range(7, 730)]))        # saisonal naiv (Wert von vor 7 Tagen), Trainingstage 7 bis 729
    assert S.mase_scale(b.y)[0] == pytest.approx(nenner)
    assert ev.ORG_TEST[0] == 730 and ev.ORG_TEST[-1] == 1081 and C.HORIZON == 14
    for text in ("Horizonte 1 bis 14 Tage", "Prognoseursprünge Tag 730 bis 1081", "saisonal naiven Prognose (Wert von vor 7 Tagen)", "Trainingstagen 7 bis 729"):
        assert text in README, text
    assert "Nenner der Linie" not in README and "Überschrift" not in README and "Standardszenario" in README


def test_beginn_der_auslenkung_ist_im_text_wie_im_code():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "letzten Tag $t_0$" not in app and "ersten Tag $t_0$ der laufenden Auslenkung" in app
    assert "Tag nach dem letzten Tag, an dem die Statistik 0 war" in README
