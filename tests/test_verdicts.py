"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, keine Seeds."""

import pytest

import nvs_constants as C
import nvs_evaluation as ev


def _summ(gd=(2.4, 0.3), go=(16.0, 0.5), gv=(2.2, 0.4), opp="ewma", total=(20.7, 0.5), reg=(18.9, 0.4)):
    row = {"opponent": opp, "gain_detector": gd, "gain_oracle": go, "gain_total": total, "gain_regression": reg}
    if gv is not None:
        row["gain_oracle_delayed"] = gv
    return {"kosten_alle": row, "mase_alle": row}


# --- Kernfrage 4: Orakel-Reserve --------------------------------------------------------------------------------------------------------------
def test_main_success_nennt_orakel_detektor_anteil_und_verzoegerung():
    kind, text = ev.verdict_main(_summ(), 5, 0.99)
    assert kind == "success"
    assert "Regression + EWMA-Nachführung" in text
    assert "spart das Orakel **16,0 %**" in text and "der Detektor **2,4 %** (± 0,3)" in text
    assert "das sind 15 % der Orakel-Reserve" in text                                   # 2,4 / 16 = 15 %
    assert "erst nach 5 Tagen kennt" in text and "spart noch **2,2 %**" in text
    assert "nur in" not in text                                                          # Fund in 99 % der Reihen: kein Zusatz


def test_main_ohne_verzoegertes_orakel_oder_ohne_verzoegerung_fehlt_der_satz():
    assert "erst nach" not in ev.verdict_main(_summ(gv=None), 5, 0.99)[1]
    assert "erst nach" not in ev.verdict_main(_summ(), None, 0.99)[1]


def test_main_schwelle_zwei_standardfehler_und_halbes_prozent():
    assert ev.verdict_main(_summ(gd=(0.6, 0.3)), 5, 0.99)[0] == "success"             # genau zwei Standardfehler und über 0,5
    assert ev.verdict_main(_summ(gd=(0.59, 0.3)), 5, 0.99)[0] == "info"                # knapp darunter
    assert ev.verdict_main(_summ(gd=(0.5, 0.1)), 5, 0.99)[0] == "info"                 # über zwei Standardfehler, aber nicht über 0,5 Punkte
    assert ev.verdict_main(_summ(gd=(0.51, 0.1)), 5, 0.99)[0] == "success"


def test_main_verlust_ist_eine_warnung():
    kind, text = ev.verdict_main(_summ(gd=(-4.0, 1.0)), 5, 0.99)
    assert kind == "warning" and "**-4,0 %**" in text and "schlechter als der Gegner" in text and "holt nichts von der Orakel-Reserve" in text
    assert "-" not in text.split("der Detektor **-4,0 %**")[1].split("Orakel-Reserve")[0].replace("-4,0", "")     # keine negative Anteilsangabe
    assert ev.verdict_main(_summ(gd=(-2.0, 1.0)), 5, 0.99)[0] == "info"                # genau zwei Standardfehler ist noch keine Warnung


def test_main_ohne_orakelgewinn_gibt_es_nichts_zu_erkennen():
    kind, text = ev.verdict_main(_summ(go=(0.05, 0.1)), None, 0.0)
    assert kind == "info" and "nichts (**0,1 %**)" in text and "nichts zu erkennen" in text


def test_main_nennt_die_gefundenen_anteile_nur_unter_der_haelfte():
    assert "meldet die Änderung nur in 49 %" in ev.verdict_main(_summ(), 5, 0.49)[1]
    assert "nur in" not in ev.verdict_main(_summ(), 5, 0.5)[1]


def test_reserve_anteil():
    assert ev.reserve_share(2.4, 16.0) == pytest.approx(0.15)
    assert ev.reserve_share(2.4, 0.05) is None and ev.reserve_share(2.4, -1.0) is None
    assert ev.reserve_share(-1.0, 10.0) == pytest.approx(-0.1)


def test_regressionsanteil_nennt_gesamt_regression_und_rest():
    kind, text = ev.verdict_regression_share(_summ())
    assert kind == "info" and "**20,7 %**" in text and "**18,9 Prozentpunkte**" in text and "**1,8 Prozentpunkte**" in text
    kind, text = ev.verdict_regression_share(_summ(total=(0.05, 0.4), reg=(0.0, 0.4)))
    assert "nicht vor der Glättung" in text and "**0,1 %**" in text


# --- Kernfrage 1: Kennlinie ---------------------------------------------------------------------------------------------------------------------
ROWS = [
    {"h": 4.0, "fehlmeldungen": 3.5, "gefunden": 1.0, "verzoegerung": 3.0},
    {"h": 5.0, "fehlmeldungen": 1.0, "gefunden": 0.95, "verzoegerung": 4.4},
    {"h": 6.0, "fehlmeldungen": 0.3, "gefunden": 0.8, "verzoegerung": 5.5},
]


def test_kennlinie_nimmt_die_kleinste_schwelle_mit_hoechstens_einer_fehlmeldung():
    kind, text = ev.verdict_kennlinie(ROWS)
    assert kind == "success" and "ab **h = 5**" in text and "(1,00 im Mittel über 60 Reihen" in text
    assert "**95 %** der Reihen nach im Mittel **4,4 Tagen**" in text


def test_kennlinie_grenze_ist_inklusive_und_info_unter_90_prozent():
    rows = [{**ROWS[1], "fehlmeldungen": 1.01}, {**ROWS[2], "gefunden": 0.89}]
    kind, text = ev.verdict_kennlinie(rows)
    assert kind == "info" and "ab **h = 6**" in text and "**89 %**" in text


def test_kennlinie_ohne_betriebspunkt_und_ohne_fund():
    kind, text = ev.verdict_kennlinie([{**ROWS[0], "fehlmeldungen": 2.0}])
    assert kind == "warning" and "(h = 4)" in text and "2,00 Fehlmeldungen je Jahr" in text and "nicht erreichbar" in text
    kind, text = ev.verdict_kennlinie([{**ROWS[1], "verzoegerung": None, "gefunden": 0.0}])
    assert kind == "warning" and "in keiner der 60 Reihen" in text


# --- Kernfrage 2: Karte -------------------------------------------------------------------------------------------------------------------------
def _zellen(fn):
    z = {}
    for n in C.KARTE_NOISE:
        for s in C.KARTE_SHIFT:
            for a in C.KARTE_AR:
                g = fn(n, s, a)
                z[f"{n}|{s}|{a}"] = {"messung": {"kosten": {"gewinn_detektor": {"kosten_alle": g, "kosten_fenster": (1.0, 0.5)}, "gewinn_orakel": {"kosten_alle": (15.5, 0.5)}},
                                                  "mase": {"gewinn_detektor": {"mase_alle": g}}}}
    return z


def test_karte_zelle_genau_und_nachbarzelle():
    z = _zellen(lambda n, s, a: (2.4, 0.3) if (n, s, a) == (14, 30, 0.0) else (0.0, 0.1))
    assert ev.karte_cell_label(z, 14, 30, 0.0) == ("14|30|0.0", True)
    assert ev.karte_cell_label(z, 16, 35, 0.0) == ("14|30|0.0", False)             # 16 liegt näher an 14 als an 24; 35 näher an 30 als an 50
    assert ev.karte_cell_label(z, 12, 40, 0.5)[1] is False
    assert ev.karte_cell_label(z, 4, 10, 0.6) == ("4|10|0.6", True)


def test_karte_gleichstand_waehlt_die_kleinere_zelle():
    z = _zellen(lambda n, s, a: (0.0, 0.1))
    assert ev.karte_cell_label(z, 9, 40, 0.2)[0] == "8|30|0.0"                      # Sprung 40 liegt gleich weit von 30 und 50: min() nimmt den ersten Wert der Reihenfolge


def test_karte_verdikt_lohnt_lohnt_nicht_und_kostet():
    z = _zellen(lambda n, s, a: (2.4, 0.3))
    kind, text = ev.verdict_karte(z, 14, 30, 0.0)
    assert kind == "success" and text.startswith("In dieser Zelle") and "**2,4 %**" in text and "(± 0,3)" in text and "im Sprungfenster 1,0 % (± 0,5)" in text and "Orakel würde im Testjahr 15,5 %" in text
    kind, text = ev.verdict_karte(_zellen(lambda n, s, a: (0.4, 0.1)), 14, 30, 0.0)
    assert kind == "info" and "unter 0,5 Prozentpunkten" in text
    kind, text = ev.verdict_karte(_zellen(lambda n, s, a: (-1.5, 0.2)), 14, 30, 0.0)
    assert kind == "warning" and "kostet das Zurücksetzen nachweisbar mehr" in text
    kind, text = ev.verdict_karte(_zellen(lambda n, s, a: (2.4, 0.3)), 16, 35, 0.0)
    assert text.startswith("In der nächstgelegenen Zelle (Rauschen 14 %, Sprung +30 %, Autokorrelation 0,0)")


def test_karte_zaehlung():
    z = _zellen(lambda n, s, a: (2.4, 0.3) if n == 4 else ((-3.0, 0.5) if n == 8 else (0.4, 0.1)))
    assert ev.karte_counts(z, 0.0) == (4, 4, 16)                                    # vier Zellen mit Gewinn, vier mit Verlust, acht neutral
    assert ev.karte_counts(z, 0.6, "mase", "mase_alle") == (4, 4, 16)


# --- Kernfrage 3: Bestätigung -------------------------------------------------------------------------------------------------------------------
def _confirm_rows(promo_mit=0.02, step_mit=(0.97, 15.0)):
    return [{"form": "step", "ohne": {"anteil": 0.98, "verzoegerung": 4.8}, "mit": {"anteil": step_mit[0], "verzoegerung": step_mit[1]}},
            {"form": "promo", "ohne": {"anteil": 0.82, "verzoegerung": 4.4}, "mit": {"anteil": promo_mit, "verzoegerung": 14.0}}]


def test_bestaetigung_nennt_anteile_und_zusatzverzoegerung():
    kind, text = ev.verdict_confirm(10, _confirm_rows())
    assert kind == "success"
    assert "in **82 %** der Reihen als Sprung gemeldet" in text and "nach 10 Tagen in **2 %**" in text
    assert "in 97 % der Reihen gemeldet statt in 98 %" in text and "**10,2 Tage** später" in text


def test_bestaetigung_halbierungsschwelle_und_fehlender_fund():
    assert ev.verdict_confirm(10, _confirm_rows(promo_mit=0.40))[0] == "success"   # 0,40 < 0,41 = 0,82 / 2
    assert ev.verdict_confirm(10, _confirm_rows(promo_mit=0.41))[0] == "info"      # genau die Hälfte reicht nicht (strikt kleiner)
    kind, text = ev.verdict_confirm(10, _confirm_rows(step_mit=(0.0, None)))
    assert kind == "warning" and "in keiner Reihe" in text


def test_fmt_hat_dezimalkomma():
    assert ev.fmt(2.25, 2) == "2,25" and ev.fmt(-0.04, 1) == "0,0" and ev.fmt(1234.0, 0) == "1234" and ev.fmt(-0.06, 1) == "-0,1"          # gerundet null: ohne Vorzeichen
