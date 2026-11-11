"""Erzeugt die Zahlentabellen der README aus `data/*.json` (Messläufe von `tools/messlauf.py`).

    python tools/readme_tabellen.py            # Blöcke ausgeben
    python tools/readme_tabellen.py --write    # README.md: Blöcke zwischen <!-- BEGIN:name --> und <!-- END:name --> ersetzen

`tests/test_claims.py` erzeugt die Blöcke neu und vergleicht sie mit der README: eine Tabellenzahl kann nicht von der Messung abweichen. Die Zahlen der Vormessung (Spalte "Vormessung")
stehen unten als Konstanten; sie stammen aus `aenderungspunkt-planung/messreihe_niveausprung/ERGEBNIS.md` (Teil A) und sind Messungen eines anderen Zufallsstroms (Seeds 0 bis 599 bzw. 0 bis 299)."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

import nvs_constants as C  # noqa: E402


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def f(x, d=1, sign=False):
    s = f"{x:.{d}f}"
    if float(s) == 0.0:
        s = f"{0.0:.{d}f}"
        s = "+" + s if sign else s
    elif sign:
        s = f"{x:+.{d}f}"
    return s.replace(".", ",").replace("-", "−")


def pm(v, d=1):
    return f"{f(v[0], d, True)} ± {f(v[1], d)}"


# --------------------------------------------------------------------------------------------------------------------------------------------
HEAD_ROWS = [
    ("snaive", "Saisonal naiv (Stück 1)"), ("wmean", "Wochenmittel, 4 Wochen (Stück 1)"), ("hw", "Saisonale Glättung, feste Parameter (α = 0,10)"), ("none", "Regression ohne Nachführung"),
    ("ewma[mase]", "Regression + EWMA, abgestimmt auf MASE"), ("ma[mase]", "Regression + Mittel der letzten k Tage, auf MASE"), ("trigg[mase]", "Regression + Trigg-Leach, auf MASE"),
    ("cusum[mase]", "Regression + CUSUM-Zurücksetzen, auf MASE"), ("ewma[kosten]", "Regression + EWMA, abgestimmt auf Kosten"), ("ma[kosten]", "Regression + Mittel der letzten k Tage, auf Kosten"),
    ("trigg[kosten]", "Regression + Trigg-Leach, auf Kosten"), ("cusum[kosten]", "Regression + CUSUM-Zurücksetzen, auf Kosten"), ("oracle[kosten]", "Orakel: kennt Tag und Größe"),
    ("oracle_delayed", "Orakel, um die mittlere Verzögerung des Detektors verspätet"), ("erwartung", "wahrer Erwartungswert (Orakel der Linie, kein Nachführer)"),
]


def block_head():
    h = load("head.json")
    lines = ["| Verfahren | MASE alle | MASE Fenster | Kosten Testjahr | Kosten Fenster | Kosten Testjahr, konform | Kosten Fenster, konform |", "|---|---|---|---|---|---|---|"]
    for key, name in HEAD_ROWS:
        e, k = h["empirical"][key], h["conformal"][key]
        lines.append(f"| {name} | {f(e['mase_alle'][0], 3)} | {f(e['mase_fenster'][0], 3)} | {f(e['kosten_alle'][0], 3)} | {f(e['kosten_fenster'][0], 3)} | {f(k['kosten_alle'][0], 3)} | {f(k['kosten_fenster'][0], 3)} |")
    return "\n".join(lines)


GEWINN_ROWS = [("hw", "Saisonale Glättung (α = 0,10)"), ("none", "Regression ohne Nachführung"), ("ewma", "Regression + EWMA"), ("ma", "Regression + Mittel der letzten k Tage"), ("trigg", "Regression + Trigg-Leach")]


def _gk(h, rule, ref, gegner, key):
    return h[rule + "_gewinn"][ref][gegner][key]


def block_gewinn():
    h = load("head.json")
    lines = ["| Gegner | MASE alle | Kosten Testjahr | Kosten Fenster | Kosten Testjahr, konform | Kosten Fenster, konform | Orakel: Kosten Testjahr |", "|---|---|---|---|---|---|---|"]
    for key, name in GEWINN_ROWS:
        gm = key if key in ("hw", "none") else f"{key}[mase]"
        gk = key if key in ("hw", "none") else f"{key}[kosten]"
        cells = [pm(_gk(h, "empirical", "cusum[mase]", gm, "mase_alle")), pm(_gk(h, "empirical", "cusum[kosten]", gk, "kosten_alle")), pm(_gk(h, "empirical", "cusum[kosten]", gk, "kosten_fenster")),
                 pm(_gk(h, "conformal", "cusum[kosten]", gk, "kosten_alle")), pm(_gk(h, "conformal", "cusum[kosten]", gk, "kosten_fenster")), pm(_gk(h, "empirical", "oracle[kosten]", gk, "kosten_alle"))]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------------------------------
def _karte_table(obj, key, gain="gewinn_detektor", d=1, datei="karte.json"):
    z = load(datei)["zellen"]
    lines = ["| Rauschen | Autokorrelation | " + " | ".join(f"Sprung +{s} %" for s in C.KARTE_SHIFT) + " |", "|---|---|" + "---|" * len(C.KARTE_SHIFT)]
    for n in C.KARTE_NOISE:
        for a in C.KARTE_AR:
            cells = []
            for s in C.KARTE_SHIFT:
                g = z[f"{n}|{s}|{a}"]["messung"][obj][gain][key]
                cells.append(pm(g, d))
            lines.append(f"| {n} % | {f(a, 1)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def block_karte_kosten_alle():
    return _karte_table("kosten", "kosten_alle")


def block_karte_kosten_fenster():
    return _karte_table("kosten", "kosten_fenster")


def block_karte_mase_alle():
    return _karte_table("mase", "mase_alle")


def block_karte_mase_fenster():
    return _karte_table("mase", "mase_fenster")


def block_karte_orakel():
    return _karte_table("kosten", "kosten_alle", "gewinn_orakel")


def block_karte_orakel_fenster():
    return _karte_table("kosten", "kosten_fenster", "gewinn_orakel")


def block_karte_meldungen():
    z = load("karte.json")["zellen"]
    lines = ["| Rauschen | Autokorrelation | " + " | ".join(f"Sprung +{s} %" for s in C.KARTE_SHIFT) + " |", "|---|---|" + "---|" * len(C.KARTE_SHIFT)]
    for n in C.KARTE_NOISE:
        for a in C.KARTE_AR:
            lines.append(f"| {n} % | {f(a, 1)} | " + " | ".join(f"{f(z[f'{n}|{s}|{a}']['messung']['kosten']['meldungen_je_reihe'], 2)}" for s in C.KARTE_SHIFT) + " |")
    return "\n".join(lines)


def karte_aus(ar=0.0):
    """Zellen, in denen der Detektor praktisch aus ist (unter 0,05 Meldungen je Reihe, Ziel Kosten)."""
    z = load("karte.json")["zellen"]
    return sorted(k for k, v in z.items() if v["messung"]["kosten"]["meldungen_je_reihe"] < 0.05)


# --------------------------------------------------------------------------------------------------------------------------------------------
def block_confirm():
    c = load("confirm.json")["ergebnis"]
    cs = (0, 3, 7, 10, 14, 21)
    forms = ["Stufe +28 %", "Stufe +14 %", "Aktionswoche +25 %", "Aktionswoche +50 %", "Ausreißer +100 %", "Rampe auf +28 %"]
    lines = ["| Form | " + " | ".join(f"c = {x} Tage" for x in cs) + " |", "|---|" + "---|" * len(cs)]
    for fm in forms:
        cells = []
        for x in cs:
            r = c[f"{fm}|1.0|{x}"]
            cells.append(f"{f(r['anteil'] * 100, 1)} % ({f(r['verzoegerung'], 1)} d)" if r["verzoegerung"] is not None else f"{f(r['anteil'] * 100, 1)} % (– d)")
        lines.append(f"| {fm} | " + " | ".join(cells) + " |")
    cells = [f"{f(c[f'keine Form|1.0|{x}']['meldungen_je_jahr'][0], 3)}" for x in cs]
    lines.append("| *keine Form: Meldungen je Jahr* | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def block_confirm_ruhig():
    c = load("confirm_ruhig.json")["ergebnis"]
    cs = (0, 10, 14, 21)
    lines = ["| Form | " + " | ".join(f"c = {x} Tage" for x in cs) + " | c = 10, θ = 1,5 | c = 14, θ = 1,5 |", "|---|" + "---|" * (len(cs) + 2)]
    for fm in ("Stufe +28 %", "Aktionswoche +50 %"):
        cells = []
        for key in [f"{fm}|1.0|{x}" for x in cs] + [f"{fm}|1.5|10", f"{fm}|1.5|14"]:
            r = c[key]
            cells.append(f"{f(r['anteil'] * 100, 1)} % ({f(r['verzoegerung'], 1)} d)" if r["verzoegerung"] is not None else f"{f(r['anteil'] * 100, 1)} % (– d)")
        lines.append(f"| {fm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def block_confirm_theta():
    c = load("confirm.json")["ergebnis"]
    forms = ["Stufe +28 %", "Stufe +14 %", "Aktionswoche +25 %", "Aktionswoche +50 %", "Ausreißer +100 %", "Rampe auf +28 %"]
    lines = ["| Form | θ = 1,0 | θ = 1,5 |", "|---|---|---|"]
    for fm in forms:
        cells = []
        for th in ("1.0", "1.5"):
            r = c[f"{fm}|{th}|10"]
            cells.append(f"{f(r['anteil'] * 100, 1)} % ({f(r['verzoegerung'], 1)} d)" if r["verzoegerung"] is not None else f"{f(r['anteil'] * 100, 1)} % (– d)")
        lines.append(f"| {fm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------------------------------
def kennlinie_betrieb(model):
    """Kleinste Schwelle der Rasters mit höchstens einer Meldung je Jahr: (h, Zeile)."""
    k = load("kennlinie.json")["ergebnis"]
    hs = sorted(float(key.split("|")[1]) for key in k if key.startswith(model + "|"))
    for h in hs:
        row = k[f"{model}|{h}"]
        if row["fehlmeldungen"][0] <= 1.0:
            return h, row
    return None, None


def block_kennlinie():
    lines = ["| Bereinigung | kleinstes h mit höchstens einer Meldung je Jahr | Meldungen je Jahr | Sprung +7 % | Sprung +14 % | Sprung +28 % | Sprung +42 % |", "|---|---|---|---|---|---|---|"]
    for m in C.MODELS:
        h, row = kennlinie_betrieb(m)
        cells = []
        for sh in (0.07, 0.14, 0.28, 0.42):
            r = row[f"sprung_{sh}"]
            cells.append(f"{f(r['anteil'] * 100, 0)} % ({f(r['verzoegerung'], 1)} d)" if r["verzoegerung"] is not None else f"{f(r['anteil'] * 100, 0)} % (– d)")
        lines.append(f"| {C.MODEL_NAMES[m]} | {f(h, 0)} | {f(row['fehlmeldungen'][0], 2)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def block_alpha():
    a = load("alpha.json")["ergebnis"]
    forms = ("Stufe +28 %", "Stufe +14 %", "Rampe auf +28 %")
    lines = ["| Detektor (ohne Bestätigung) | Meldungen je Jahr ohne Form | " + " | ".join(forms) + " |", "|---|---|---|---|---|"]

    def cell(r):
        return f"{f(r['anteil'] * 100, 1)} % ({f(r['verzoegerung'], 1)} d)"
    for al, name in ((0.0, "α = 0 (feste Referenz), h = 6"), (0.03, "α = 0,03, h = 6"), (0.06, "α = 0,06 (Standard der App), h = 6")):
        lines.append(f"| {name} | {f(a[f'fehlmeldungen|{al}'], 2)} | " + " | ".join(cell(a[f"{fm}|{al}"]) for fm in forms) + " |")
    g = a["gleiche_rate"]
    lines.append(f"| α = 0, h = {f(g['h'], 0)} (höchstens so viele Meldungen je Jahr wie α = 0,06 bei h = 6) | {f(g['fehlmeldungen'], 2)} | " + " | ".join(cell(g[fm]) for fm in forms) + " |")
    return "\n".join(lines)


def block_laut():
    z = load("laute_reihe.json")
    lines = ["| Rauschen 24 %, Sprung | Detektor | h, k, α, w | Sprung gemeldet | Verzögerung | Meldungen je Jahr ohne Sprung |", "|---|---|---|---|---|---|"]
    for key, r in z.items():
        sh, name = key.split("|")
        p = r["parameter"]
        verz = f"{f(r['verzoegerung'], 1)} d" if r["verzoegerung"] is not None else "– d"
        lines.append(f"| +{sh} % | {'Standardwerte der App' if name == 'standard' else 'abgestimmt auf Kosten (Zelle)'} | {f(float(p['h']), 0)}, {f(float(p['k']), 1)}, {f(float(p['alpha']), 2)}, {p['wmin']} | "
                     f"{f(r['anteil'] * 100, 1)} % | {verz} | {f(r['fehlmeldungen_je_jahr'], 2)} |")
    return "\n".join(lines)


def block_wiederholung():
    return _karte_table("kosten", "kosten_alle", datei="karte_wiederholung.json")


def block_kosten():
    k = load("kosten.json")
    lines = ["| Rauschen (Gegner) | τ (p) | L | Gewinn Detektor, Testjahr | Gewinn Detektor, Fenster | Gewinn Orakel, Testjahr | Gewinn Orakel, Fenster |", "|---|---|---|---|---|---|---|"]
    for noise in ("8", "14"):
        d = k[noise]
        for key, v in d.items():
            if key == "gegner_familie":
                continue
            tau, lead = key.split("|")
            p = round(float(tau) / (1 - float(tau)))
            lines.append(f"| {noise} % ({d['gegner_familie']}) | {f(float(tau), 2)} ({p}) | {lead} | {pm(v['kosten_alle']['gewinn_detektor'])} | {pm(v['kosten_fenster']['gewinn_detektor'])} | "
                         f"{f(v['kosten_alle']['gewinn_orakel'][0], 1, True)} | {f(v['kosten_fenster']['gewinn_orakel'][0], 1, True)} |")
    return "\n".join(lines)


def block_multi():
    m = load("karte.json")["multi"]
    names = {"step": "ein Sprung +30 %", "step_back": "+30 %, nach 90 Tagen zurück", "step_two": "+30 %, nach 120 Tagen weitere +20 %"}
    lines = ["| Rauschen | Folge | MASE alle | Kosten Testjahr | Kosten Fenster (180 Tage) | Orakel: Kosten Testjahr | Meldungen je Reihe | letzter Sprung hinter dem Reihenende | Fenster über das Reihenende hinaus |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in ("4", "8", "14", "24"):
        for form in ("step", "step_back", "step_two"):
            r = m[n][form]["messung"]
            e = r["hinter_ende"]
            lines.append(f"| {n} % | {names[form]} | {pm(r['mase']['gewinn_detektor']['mase_alle'])} | {pm(r['kosten']['gewinn_detektor']['kosten_alle'])} | {pm(r['kosten']['gewinn_detektor']['kosten_fenster'])} | "
                         f"{f(r['kosten']['gewinn_orakel']['kosten_alle'][0], 1, True)} | {f(r['kosten']['meldungen_je_reihe'], 2)} | {f(e['letzter_sprung'] * 100, 1)} % | {f(e['fenster_verkuerzt'] * 100, 1)} % |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------------------------------------------
# Vormessung (ERGEBNIS.md Teil A, Seeds 0 bis 599 bzw. 0 bis 299; Detektor dort auf den Residuen mit fester Referenz, Karte mit je Zelle abgestimmten Parametern): die Zahlen der Spalte stehen im Text unten.


def block_vergleich():
    h, z, c = load("head.json"), load("karte.json")["zellen"], load("confirm.json")["ergebnis"]
    z14 = z["14|30|0.0"]["messung"]
    c0, c10 = c["Aktionswoche +50 %|1.0|0"], c["Aktionswoche +50 %|1.0|10"]
    s0, s10 = c["Stufe +28 %|1.0|0"], c["Stufe +28 %|1.0|10"]
    rows = [
        ("Gewinn gegen stärksten Gegner, MASE alle (Rauschen 14 %, +30 %)", "+1,3 %", f(_gk(h, "empirical", "cusum[mase]", "ewma[mase]", "mase_alle")[0], 1, True) + " %"),
        ("Gewinn gegen stärksten Gegner, Kosten Testjahr", "+2,4 %", f(_gk(h, "empirical", "cusum[kosten]", "ewma[kosten]", "kosten_alle")[0], 1, True) + " %"),
        ("Gewinn gegen Trigg-Leach, Kosten im Fenster", "−17,0 %", f(_gk(h, "empirical", "cusum[kosten]", "trigg[kosten]", "kosten_fenster")[0], 1, True) + " %"),
        ("Gewinn gegen die feste Glättung, MASE alle", "+13,5 % (Original-Holt-Winters)", f(_gk(h, "empirical", "cusum[mase]", "hw", "mase_alle")[0], 1, True) + " % (Nachbau, α = 0,10)"),
        ("Gewinn gegen die feste Glättung, Kosten Testjahr", "+21,6 % (Original)", f(_gk(h, "empirical", "cusum[kosten]", "hw", "kosten_alle")[0], 1, True) + " % (Nachbau)"),
        ("Orakel gegen stärksten Gegner, Kosten Testjahr", "+16 %", f(_gk(h, "empirical", "oracle[kosten]", "ewma[kosten]", "kosten_alle")[0], 1, True) + " %"),
        ("Orakel gegen stärksten Gegner, Kosten Fenster", "+37,4 %", f(_gk(h, "empirical", "oracle[kosten]", "ewma[kosten]", "kosten_fenster")[0], 1, True) + " %"),
        ("Karte Rauschen 14 %, +50 %: Kosten Testjahr", "+5,7 %", f(z["14|50|0.0"]["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"][0], 1, True) + " %"),
        ("Karte Rauschen 24 %, +50 %: Kosten Testjahr", "+1,1 %", f(z["24|50|0.0"]["messung"]["kosten"]["gewinn_detektor"]["kosten_alle"][0], 1, True) + " %"),
        ("Karte Rauschen 14 %, +30 %: MASE alle", "+1,5 %", f(z14["mase"]["gewinn_detektor"]["mase_alle"][0], 1, True) + " %"),
        ("Aktionswoche +50 % ohne Bestätigung gemeldet", "100,0 %", f(c0["anteil"] * 100, 1) + " %"),
        ("Aktionswoche +50 % mit Bestätigung nach 10 Tagen", "12,2 %", f(c10["anteil"] * 100, 1) + " %"),
        ("Stufe +28 % mit Bestätigung nach 10 Tagen bestätigt", "99,5 %", f(s10["anteil"] * 100, 1) + " %"),
        ("Zusatzverzögerung der Bestätigung (c = 10, Stufe +28 %)", "+10,7 Tage", f(s10["verzoegerung"] - s0["verzoegerung"], 1, True) + " Tage"),
        ("Stufe +14 % ohne Bestätigung gemeldet", "97,2 %", f(c["Stufe +14 %|1.0|0"]["anteil"] * 100, 1) + " %"),
        ("Stufe +14 % mit Bestätigung nach 10 Tagen", "45,5 %", f(c["Stufe +14 %|1.0|10"]["anteil"] * 100, 1) + " %"),
        ("Rampe auf +28 % ohne Bestätigung gemeldet", "100,0 %", f(c["Rampe auf +28 %|1.0|0"]["anteil"] * 100, 1) + " %"),
        ("Aktionswoche +25 % ohne Bestätigung gemeldet", "74,2 %", f(c["Aktionswoche +25 %|1.0|0"]["anteil"] * 100, 1) + " %"),
        ("Ausreißer +100 % ohne Bestätigung gemeldet", "26,8 %", f(c["Ausreißer +100 %|1.0|0"]["anteil"] * 100, 1) + " %"),
    ]
    lines = ["| Größe | Vormessung (Seeds 0 bis 599 bzw. 0 bis 299) | Bau (diese Demo, andere Seeds) |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c_} |" for a, b, c_ in rows]
    return "\n".join(lines)


BLOCKS = {
    "head": block_head, "gewinn": block_gewinn, "karte_kosten_alle": block_karte_kosten_alle, "karte_kosten_fenster": block_karte_kosten_fenster, "karte_mase_alle": block_karte_mase_alle,
    "karte_mase_fenster": block_karte_mase_fenster, "karte_orakel": block_karte_orakel, "karte_orakel_fenster": block_karte_orakel_fenster, "karte_meldungen": block_karte_meldungen,
    "confirm": block_confirm, "alpha": block_alpha, "laut": block_laut, "wiederholung": block_wiederholung, "confirm_theta": block_confirm_theta, "confirm_ruhig": block_confirm_ruhig, "kennlinie": block_kennlinie, "kosten": block_kosten, "multi": block_multi, "vergleich": block_vergleich,
}


def render(readme_text):
    """Ersetzt in `readme_text` jeden Block zwischen den Marken durch die frisch erzeugte Tabelle."""
    for name, fn in BLOCKS.items():
        pattern = re.compile(rf"<!-- BEGIN:{name} -->.*?<!-- END:{name} -->", re.S)
        if pattern.search(readme_text):
            block = f"<!-- BEGIN:{name} -->\n{fn()}\n<!-- END:{name} -->"
            readme_text = pattern.sub(lambda m: block, readme_text)
    return readme_text


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--write" in sys.argv:
        p = ROOT / "README.md"
        p.write_text(render(p.read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
    else:
        for name, fn in BLOCKS.items():
            print(f"### {name}\n{fn()}\n")
