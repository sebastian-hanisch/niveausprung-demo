"""Niveausprung erkennen - interaktive Konzept-Demo (Stück 12 der Reihe "Zeitreihen-Prognose: von Naiv bis Vortraining")
Sebastian Hanisch - Operations Research und Machine Learning

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import nvs_constants as C
import nvs_evaluation as ev
import nvs_karte as K
from nvs_layout import dunkles_design, kopf_css
from nvs_presets import (
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from nvs_visualization import build_comparison, build_confirm, build_karte, build_kennlinie, build_series

st.set_page_config(page_title="Niveausprung erkennen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _kennlinie(settings):
    return ev.kennlinie(settings)


@st.cache_data(show_spinner=False)
def _confirm(settings):
    return ev.confirm_experiment(settings)


@st.cache_data(show_spinner=False)
def _karte():
    return K.load()


@st.cache_data(show_spinner=False)
def _kosten():
    return K.load(K.DATA / "kosten.json")


def de(x, d=1):
    return ev.fmt(x, d)


def tab_rows(summ, a):
    """Tabelle aller Verfahren: Mittel ± Standardfehler je Maß."""
    rows = []
    for m in C.METHODS:
        if m not in a.metrics:
            continue
        row = {"Verfahren": C.METHOD_NAMES[m]}
        for ms in ev.MEASURES:
            mean, se = summ[ms]["means"][m]
            row[ev.MEASURE_NAMES[ms]] = f"{de(mean, 3)} ± {de(se, 3)}"
        rows.append(row)
    return rows


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in nvs_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Niveausprung erkennen")
st.markdown(
    """
Springt das Niveau einer Zeitreihe dauerhaft, braucht jede Prognose, die das Niveau nur aus den Daten schätzt, eine Weile, bis sie es einholt. Hier wird gemessen, ob es sich lohnt, den **Niveausprung zu erkennen** und das Niveau nach der Meldung
**zurückzusetzen**, und zwar in **Bestandskosten**, nicht nur im Prognosefehler: Ein CUSUM-Detektor überwacht die Abweichungen einer Regression (Wochentag, Jahresmuster, Trend, Kalender) vom
Niveau, meldet bei Überschreiten einer Schwelle **h**, und die Bestandsentscheidung (Order-up-to mit Sicherheitsbestand) nutzt das neue Niveau. Gegner sind **Nachführer** auf derselben
Regression, abgestimmt für die Voreinstellung und für die Karte (die Regler der Seitenleiste werden bei geänderten Szenarien nicht neu abgestimmt); das Orakel kennt Tag und Größe des Sprungs. Stück 12 der Reihe „Zeitreihen-Prognose: von Naiv bis Vortraining“ (Vehikel: tägliche Aufträge je Depot), Bezug zu OR: die Bestandsentscheidung nach der Meldung.

Direkt darunter steht der Expander **💬 In einfachen Worten** (ohne Formeln). Weiter unten: **❓ Kernfragen** mit Kennlinie der Schwelle, **Karte** (wo lohnt Erkennen?), **Bestätigung** gegen die Verwechslung mit einer Aktionswoche und die **Orakel-Reserve**, dann **🚧 Wo die
Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/niveausprung-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Springt die Zahl der Tagesaufträge dauerhaft, merkt eine Prognose das nur langsam: Sie ist absichtlich träge, damit einzelne laute Tage sie nicht umwerfen. Die Demo prüft, ob es sich lohnt, den **Niveausprung zu erkennen** und die Prognose dann auf einmal neu anzusetzen, und zwar in **Bestandskosten**: Lager für zu viel, Rückstand für zu wenig. Ein Alarm zählt dafür Abweichungen zusammen und schlägt bei einer Schwelle an. Eine Karte aus Rauschen und Sprunggröße zeigt, wo das etwas bringt und wo nicht. Eine **Bestätigungsregel** soll eine einwöchige Aktion von einem echten Sprung trennen, kostet aber Tage. Gegen eine gut abgestimmte träge Prognose ist der Gewinn klein.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/niveausprung-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert Niveausprung erkennen", expanded=True):
    st.markdown(
        """
1. **Bereinigen:** Eine Regression im Logarithmus (Wochentag, Jahresmuster, Trend, Feiertage, Aktionsplan; angepasst auf den Tagen 28 bis 729) erklärt, was an der Reihe vorhersehbar ist. Übrig bleiben die **Residuen**.
2. **Nachführen:** Das Niveau der Residuen wird täglich mit einem kleinen α fortgeschrieben (langsam, damit das Rauschen nicht durchschlägt).
3. **Überwachen:** Ein zweiseitiger **CUSUM** summiert die Ein-Schritt-Fehler (in Standardabweichungen, abzüglich der Zulage k). Übersteigt die Summe die Schwelle **h**, gibt es eine **Meldung**.
4. **Zurücksetzen:** Nach der Meldung wird das Niveau auf das Mittel der Residuen seit Beginn der Auslenkung gesetzt (mindestens w Tage; Beginn = erster Tag der laufenden Auslenkung). Optional **bestätigt** der Detektor die Meldung erst c Tage später, damit eine einwöchige Aktion nicht als Sprung gilt.
5. **Entscheiden:** Aus der Prognose und den bisherigen Fehlern folgt der Bestellbestand; gezählt werden Lager und Rückstand.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Szenario")
    st.slider("Rauschen (Streuung des Tagesrauschens)", C.NOISE_MIN, C.NOISE_MAX, step=C.NOISE_STEP, key="noise", format="%d %%",
              help="Streuung des log-normalen Tagesrauschens. Je lauter die Reihe, desto schwerer hebt sich ein Sprung ab.")
    st.slider("Sprunggröße", C.SHIFT_MIN, C.SHIFT_MAX, step=C.SHIFT_STEP, key="shift", format="+%d %%",
              help="Relative Änderung des Niveaus. Bei Aktionswoche und Ausreißer: Höhe der Abweichung, bei der Rampe: Endhöhe nach 60 Tagen.")
    st.selectbox("Form und Folge", C.FORMS, key="form", format_func=C.FORM_NAMES.get,
                 help="Dauerhafter Sprung, Sprung mit Rückkehr, zweiter Sprung, Rampe, einwöchige Aktion oder eintägiger Ausreißer. Bei Folgen von Sprüngen ist das Fenster der Gewinne 180 statt 60 Tage.")
    st.slider("Autokorrelation des Rauschens", C.AR_MIN, C.AR_MAX, step=C.AR_STEP, key="ar", format="%.1f",
              help="AR(1)-Koeffizient des Log-Rauschens bei gleicher Streuung: aufeinanderfolgende Tagesfehler hängen zusammen.")
    st.number_input("Zufalls-Seed der ersten Reihe", 0, C.SEED_MAX, step=1, key="seed",
                    help="Die Auswertung nutzt die Reihen mit den Seeds seed bis seed + Zahl der Reihen − 1; die erste davon zeigt das Diagramm.")
    st.slider("Zahl der Reihen", C.RUNS_MIN, C.RUNS_MAX, step=C.RUNS_STEP, key="runs", help="Mehr Reihen verkleinern den Standardfehler der Gewinne.")
    st.header("🧮 Bereinigung")
    st.selectbox("Modell der Regression", C.MODELS, key="model", format_func=C.MODEL_NAMES.get,
                 help="Was die Regression kennt. Was sie nicht kennt, muss der Detektor als Abweichung verarbeiten.")
    st.header("🚨 Detektor (CUSUM)")
    st.slider("Schwelle h", C.H_MIN, C.H_MAX, step=C.H_STEP, key="h", format="%.1f", help="In Standardabweichungen der Ein-Schritt-Fehler. Größer: weniger Fehlmeldungen, aber später oder gar keine Meldung.")
    st.slider("Zulage k", C.K_MIN, C.K_MAX, step=C.K_STEP, key="k", format="%.2f", help="Abweichungen unter k Standardabweichungen werden nicht aufsummiert.")
    st.slider("Nachführung zwischen Meldungen (α)", C.ALPHA_LOW_MIN, C.ALPHA_LOW_MAX, step=C.ALPHA_LOW_STEP, key="alpha_low", format="%.2f",
              help="Langsame EWMA-Nachführung des Niveaus, solange nichts gemeldet wird.")
    st.slider("Mindestfenster beim Zurücksetzen (Tage)", C.WMIN_MIN, C.WMIN_MAX, key="wmin", help="Das neue Niveau ist das Mittel der Residuen seit dem ersten Tag der laufenden Auslenkung, mindestens aber dieser Tage.")
    st.header("🆚 Gegner (Nachführer)")
    st.slider("EWMA: α", C.EWMA_MIN, C.EWMA_MAX, step=C.EWMA_STEP, key="ewma", format="%.2f", help="Feste exponentielle Nachführung des Niveaus der Residuen.")
    st.slider("Mittel der letzten k Tage", C.MA_MIN, C.MA_MAX, key="ma", help="Niveau = Mittel der Residuen der letzten k Tage.")
    st.slider("Trigg-Leach: λ", C.TRIGG_LAM_MIN, C.TRIGG_LAM_MAX, step=C.TRIGG_LAM_STEP, key="trigg_lam", format="%.2f", help="Glättung der Fehlersignale, aus denen das adaptive α entsteht.")
    st.slider("Trigg-Leach: kleinstes α", C.TRIGG_AMIN_MIN, C.TRIGG_AMIN_MAX, step=C.TRIGG_AMIN_STEP, key="trigg_amin", format="%.2f", help="Untergrenze des adaptiven α (Obergrenze 0,5).")
    st.header("✅ Bestätigung")
    st.selectbox("Bestätigungsregel", tuple(C.CONFIRM_RULES), key="confirm_rule", format_func=C.CONFIRM_NAMES.get,
                 help="Eine Meldung gilt erst c Tage nach dem Alarm und nur, wenn das Mittel der letzten min(c, 7) Residuen mindestens θ Standardabweichungen über dem Referenzmittel liegt; sonst wird der Alarm verworfen.")
    st.header("📦 Bestand")
    st.select_slider("Zielservicegrad τ", options=C.TAU_OPTIONS, key="tau", format_func=lambda v: f"{v:.2f}".replace(".", ","),
                     help="Kritisches Verhältnis p / (p + h) mit Lager h = 1 je Stück und Tag und Rückstand p = τ / (1 − τ).")
    st.slider("Wiederbeschaffungszeit L (Tage)", C.LEAD_MIN, C.LEAD_MAX, key="lead", help="Das Schutzintervall der Bestellung umfasst L + 1 Tage.")
    st.selectbox("Regel des Sicherheitsbestands", C.RULES, key="rule", format_func=C.RULE_NAMES.get, help="Quantil der Fehler aus dem Training oder aus den zuletzt realisierten Fehlern (konform).")
    st.button("🎲 Neue Reihen generieren", width="stretch", on_click=randomize_seed)

values = {name: st.session_state[name] for name in ev.SETTING_NAMES}
sync_query_params(values)
settings = ev.Settings(**values)
with st.spinner("Rechne die Reihen …"):
    a = _analysis(settings)
summ = ev.summary(a)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
row = summ["kosten_alle"]
gd, sd = row["gain_detector"]
go_, so = row["gain_oracle"]
found_share = float(a.found.mean())
opp_name = C.METHOD_NAMES[row["opponent"]].replace("Regression + ", "")
m1, m2, m3, m4 = st.columns(4)
# Gewinne sind „Gegner minus Verfahren“ in Prozent des Gegners (positiv: das Verfahren ist besser); keine Farbe nötig.
m1.metric("Gewinn Zurücksetzen", f"{de(gd)} %", delta=f"± {de(sd)} gegen {opp_name}", delta_color="off", delta_arrow="off", help="Bestandskosten im Testjahr, gegen den stärksten eingestellten Gegner, in Prozent des Gegners.")
m2.metric("Gewinn Orakel", f"{de(go_)} %", delta=f"± {de(so)} gegen {opp_name}", delta_color="off", delta_arrow="off", help="Bestandskosten im Testjahr; das Orakel kennt Tag und Größe des Sprungs.")
m3.metric("Sprung gemeldet", f"{de(found_share * 100, 0)} % der Reihen", delta=f"Verzögerung {de(float(a.delay[a.found].mean()))} Tage" if a.found.any() else "keine Meldung", delta_color="off", delta_arrow="off",
          help="Anteil der Reihen mit einer Meldung im Fenster ab dem Sprung (60 Tage plus Wartezeit der Bestätigung).")
m4.metric("Meldungen je Reihe", de(float(a.meldungen.mean()), 2), delta=f"davon vor dem Sprung: {de(float(a.vor_sprung.mean()), 2)}", delta_color="off", delta_arrow="off", help="Meldungen im Testjahr (Tage 730 bis 1094).")
st.caption(f"Alle Zahlen: Mittel über {settings.runs} Reihen (Seeds {settings.seed} bis {settings.seed + settings.runs - 1}), gepaart; Gewinn = (Gegner − Verfahren) / Gegner in Prozent, positiv heißt besser. "
           f"Stärkster Gegner unter den eingestellten Nachführern: {opp_name}.")

st.markdown("**Eine Reihe im Detail** (die erste, Seed " + str(settings.seed) + ")")
st.plotly_chart(build_series(a, row["opponent"]), width="stretch", key="series_chart")
st.caption("Oben: Residuen der Regression (Log) und das Niveau, das jedes Verfahren daraus schätzt; Raute: Meldung, bei der das Niveau zurückgesetzt wird; gestrichelt orange: Beginn und Ende der Sprungfolge. "
           "Unten: CUSUM-Statistik und Schwelle ±h (Standardabweichungen der Ein-Schritt-Fehler); Punkte: Alarme.")

measure = st.radio("Maß", ev.MEASURES, format_func=ev.MEASURE_NAMES.get, horizontal=True, key="main_measure")
st.plotly_chart(build_comparison(summ[measure], measure, C.METHODS), width="stretch", key="comparison_chart")
st.caption("Mittel und Standardfehler je Verfahren auf denselben Reihen (kleiner ist besser); farbig: stärkster Gegner (blau), Zurücksetzen (grün), Orakel (violett). Fenster = "
           f"{a.window} Tage ab dem Sprung. Kosten in Tagesbedarfen je Tag (Lager 1 je Stück und Tag).")

st.markdown("---")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")

st.subheader("1. Wie viele Tage kostet „höchstens eine Fehlmeldung je Jahr“?")
with st.spinner("Messe die Kennlinie …"):
    rows_k = _kennlinie(settings)
st.plotly_chart(build_kennlinie(rows_k), width="stretch", key="kennlinie_chart")
kind, text = ev.verdict_kennlinie(rows_k)
getattr(st, kind)(text)
st.caption(f"Je Punkt eine Schwelle h; {C.KENNLINIE_RUNS} Reihen mit und ohne Sprung (gleiche Zufallszahlen), Detektor wie in der Seitenleiste (Bereinigung, k, α, Bestätigung); Verzögerung nur über die gemeldeten Sprünge.")

st.subheader("2. Wo auf der Karte Rauschen × Sprung lohnt Erkennen?")
karte = _karte()
zellen = karte["zellen"]
c1, c2 = st.columns(2)
karte_measure = c1.radio("Maß der Karte", ["kosten_alle", "kosten_fenster", "mase_alle", "mase_fenster"], format_func=ev.MEASURE_NAMES.get, key="karte_measure")
karte_ar = c2.radio("Autokorrelation", C.KARTE_AR, format_func=lambda v: f"{v:.1f}".replace(".", ","), key="karte_ar")
objective = "mase" if karte_measure.startswith("mase") else "kosten"
st.plotly_chart(build_karte(zellen, karte_measure, objective, karte_ar, current=(settings.noise, settings.shift)), width="stretch", key="karte_chart")
kind, text = ev.verdict_karte(zellen, settings.noise, settings.shift, settings.ar)
getattr(st, kind)(text)
plus, minus, total = ev.karte_counts(zellen, karte_ar, objective, karte_measure)
st.caption(f"Gewinn des CUSUM-Zurücksetzens gegen den stärksten abgestimmten Gegner in Prozent (positiv: Erkennen lohnt), je Zelle {len(C.MEASURE_SEEDS)} Reihen, gepaart, ± Standardfehler. Gegner, Detektor und Orakel sind je Zelle "
           f"auf getrennten Reihen abgestimmt ({len(C.TUNE_SEEDS)} Reihen, halb mit, halb ohne Sprung). Der Standardfehler gilt für die Messreihen bei gegebener Abstimmung; eine neue Abstimmung verschiebt einzelne Zellen um mehrere Prozentpunkte (README, Befund 3). Gilt für die dauerhafte Stufe, die Regression mit Kalender, τ = 0,95, L = 3 und die empirische Regel, unabhängig von den "
           f"übrigen Reglern. In {plus} von {total} Zellen dieser Karte liegt der Gewinn mehr als zwei Standardfehler über null und mindestens bei 0,5 Prozentpunkten, in {minus} von {total} mehr als zwei Standardfehler darunter.")

st.subheader("3. Was kostet das Trennen von Aktionswoche und Sprung?")
with st.spinner("Messe die Bestätigung …"):
    c_used, rows_c = _confirm(settings)
st.plotly_chart(build_confirm(rows_c, c_used), width="stretch", key="confirm_chart")
kind, text = ev.verdict_confirm(c_used, rows_c)
getattr(st, kind)(text)
keine = next(r for r in rows_c if r["form"] == "none")
st.caption(f"Je Form {C.KENNLINIE_RUNS} Reihen in der eingestellten Größe (+{settings.shift} %); Meldung im Fenster von 60 Tagen plus Wartezeit. Ohne Form: {de(keine['ohne']['fehlmeldungen'], 2)} Meldungen je Jahr ohne, "
           f"{de(keine['mit']['fehlmeldungen'], 2)} mit Bestätigung. Steht die Wartezeit in der Seitenleiste auf 0, vergleicht der Versuch mit 10 Tagen.")

st.subheader("4. Wie viel der Orakel-Reserve holt der Detektor, und wohin geht der Rest?")
kind, text = ev.verdict_main(summ, a.delay_days, found_share)
getattr(st, kind)(text)
kind, text = ev.verdict_regression_share(summ)
getattr(st, kind)(text)
st.caption("Das Orakel kennt Tag und Größe jedes Sprungs und setzt das Niveau genau dort; „um die Verzögerung verspätet“ setzt es um die mittlere Verzögerung des Detektors später. Diese Zeile zeigt, wie viel vom Rückstand auf das Orakel "
           "die Verzögerung erklären kann; die Gleichheit zweier Summen ist keine Zerlegung, Fehlmeldungen und Schätzfehler des neuen Niveaus können sich darin aufheben, und die Demo trennt sie nicht weiter auf.")

with st.expander("🔧 Alle Verfahren im Vergleich"):
    st.markdown("Mittel ± Standardfehler je Maß über die Reihen der Auswertung. MASE: mittlerer absoluter Fehler über die Horizonte 1 bis 14 Tage und die Ursprünge Tag 730 bis 1081, geteilt durch den saisonal naiven Fehler auf den Tagen 7 bis 729 derselben Reihe; Kosten: Tagesbedarfe je Tag (geteilt durch die mittlere Tagesnachfrage des Testjahres).")
    st.dataframe(tab_rows(summ, a), hide_index=True, width="stretch")
    st.markdown(f"Zählungen des Detektors: Meldung im Sprungfenster in {int(a.found.sum())} von {settings.runs} Reihen; Meldungen im Testjahr {de(float(a.meldungen.mean()), 2)} je Reihe.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
kosten = _kosten()["14"]
_plus, _minus, _total = ev.karte_counts(zellen, 0.6)
_still = max(zellen[f"24|{s}|0.0"]["messung"]["kosten"]["meldungen_je_reihe"] for s in C.KARTE_SHIFT if s <= 30)
_ar_zeile = (f"Bei Autokorrelation 0,6 kostet das Zurücksetzen auf der Karte in {_minus} von {_total} Zellen im Testjahr mehr als zwei Standardfehler mehr, in {_plus} von {_total} spart es. "
             f"Bei Rauschen 24 % und Sprüngen bis +30 % meldet der Detektor ohne Autokorrelation höchstens {de(_still, 2)} Mal je Reihe.")
_ende = karte["multi"]["14"]["step_two"]["messung"]["hinter_ende"]
_ende_zeile = (f"Der Sprungtag liegt in den Tagen 760 bis 1034. Auf der Karte der Folgen von Sprüngen (README, Befund 3) liegt der Rücksprung in {de(karte['multi']['14']['step_back']['messung']['hinter_ende']['letzter_sprung'] * 100)} %, "
               f"der zweite Sprung in {de(_ende['letzter_sprung'] * 100)} % der Messreihen hinter dem Reihenende, und das Fenster von 180 Tagen ist in {de(_ende['fenster_verkuerzt'] * 100)} % verkürzt. In der Live-Auswertung gilt dasselbe.")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **Die Regression kennt die Struktur der Reihe** (hier erzeugt wie angepasst: Wochentag, zwei Fourier-Paare, Trend, bekannte Feiertage und Aktionen) | Je weniger das Modell kennt, desto höher muss die Schwelle h für höchstens eine Fehlmeldung je Jahr sein und desto später wird gemeldet (Bereinigungs-Regler und Kennlinie oben). In echten Reihen ist das die schwächste Annahme; Modellfehler wurden nicht gemessen. |
| **Kosten nur für die gemessenen Annahmen** (τ = 0,95, L = 3, empirische Regel auf der Karte) | Bei teurem Rückstand (τ = 0,99) und Wiederbeschaffungszeit ab 3 Tagen wird der Gewinn negativ oder verschwindet (Tabelle unten: bei Rauschen 14 % −1,0 ± 0,6 bei L = 3, −2,4 ± 0,9 bei L = 7); eine konforme Regel fängt den Sprung selbst teilweise ab. |
| **Bestätigung nur für die einwöchige Aktion** | Eine längere Aktion würde nach der Regel bestätigt (nicht gemessen); ein kleiner Sprung wird dabei häufiger verworfen. Das Fenster der letzten 7 Tage beginnt c − 6 Tage nach dem Alarm und muss hinter dem Ende der Aktion liegen: meldet der Detektor schon am ersten Tag (ruhige Reihe), reicht c = 10 nicht (Preset „Ruhige Reihe“ und Aktionswoche ausprobieren). |
| **Abstimmung je Zelle** | Gegner und Detektor sind je Zelle auf den Sprung dieser Zelle abgestimmt, also für beide optimistisch; die Abstimmung auf 100 Reihen ist selbst unsicher (bei neuer Abstimmung verschieben sich einzelne Zellen um mehrere Prozentpunkte); die Regler der Seitenleiste sind nicht abgestimmt. |
| **Gleichmäßiges Rauschen, höchstens AR(1)** | AR_ZEILE |
| **Folgen von Sprüngen liegen im Reihenende** | ENDE_ZEILE |
"""
    .replace("AR_ZEILE", _ar_zeile)
    .replace("ENDE_ZEILE", _ende_zeile)
)
st.markdown("**Kostenannahmen** (Rauschen 14 %, Sprung +30 %, Gegner und Detektor für τ = 0,95 und L = 3 abgestimmt, nicht neu abgestimmt; Messreihen der Karte). Gewinn des Zurücksetzens gegen den stärksten Gegner, Prozent ± Standardfehler:")
tab_k = []
for key, v in kosten.items():
    if key == "gegner_familie":
        continue
    tau_s, lead_s = key.split("|")
    tab_k.append({"τ": tau_s.replace(".", ","), "L": int(lead_s), "Testjahr": f"{de(v['kosten_alle']['gewinn_detektor'][0])} ± {de(v['kosten_alle']['gewinn_detektor'][1])}",
                  "Sprungfenster": f"{de(v['kosten_fenster']['gewinn_detektor'][0])} ± {de(v['kosten_fenster']['gewinn_detektor'][1])}",
                  "Orakel (Testjahr)": de(v["kosten_alle"]["gewinn_orakel"][0])})
st.dataframe(tab_k, hide_index=True, width="stretch")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/niveausprung-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist eine tägliche Reihe $y_t$ (Aufträge) mit einem möglichen Niveausprung zu unbekannter Zeit. Gesucht ist ein Niveau $o_t$ für die Prognose, das nach einem Sprung schnell nachzieht, ohne bei Rauschen auszuschlagen, und der Bestellbestand $S_t$, der daraus folgt.

| Symbol | Bedeutung |
|---|---|
| $y_t$, $r_t$ | Aufträge am Tag $t$; Residuum der Regression im Logarithmus |
| $x_t$, $\beta$ | Regressoren (Wochentag, Fourier-Paare, Trend, Kalender) und ihre Koeffizienten |
| $o_t$ | Niveau (Offset) nach Tag $t$, gilt für Prognosen ab $t+1$ |
| $e_t$, $\sigma$, $z_t$ | Ein-Schritt-Fehler $r_t - o_{t-1}$; deren Standardabweichung im Referenzfenster (Tage 639 bis 729); standardisierter Fehler $e_t/\sigma$ |
| $k$, $h$ | Zulage und Schwelle des CUSUM |
| $u_t$, $\ell_t$ | obere und untere CUSUM-Statistik |
| $\alpha$ | Gewicht der langsamen Nachführung zwischen den Meldungen |
| $t_0$, $w$ | Beginn der Auslenkung (erster Tag nach dem letzten Tag mit Statistik 0); Mindestfenster beim Zurücksetzen |
| $c$, $\theta$ | Wartetage und Schwelle der Bestätigung |
| $s$, $\bar r_0$, $s_0$ | Richtung des Alarms $+1$ oder $-1$; Mittel und Standardabweichung der Residuen im Referenzfenster (in der Mittelwert-Summe ist $s$ der Laufindex) |
| $F_m$, $D_m$ | prognostizierte und tatsächliche Nachfrage im Schutzintervall von $m = L+1$ Tagen |
| $q_\tau$, $S_t$ | $\tau$-Quantil der Fehler $\ln\frac{D_m+1}{F_m+1}$; Bestellbestand |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/niveausprung-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(
        r"""
**Modell.** Die Regression wird auf den Tagen 28 bis 729 angepasst (kleinste Quadrate); danach bleibt $\beta$ fest, und nur $o_t$ bewegt sich:
$$\ln(y_t+1) = x_t^\top \beta + o_t + \varepsilon_t,\qquad r_t = \ln(y_t+1) - x_t^\top\beta$$

**Nachführung zwischen Meldungen** und **CUSUM** auf den standardisierten Ein-Schritt-Fehlern $z_t = e_t/\sigma$ (Start: Tag 730):
$$o_t = o_{t-1} + \alpha\, e_t$$
$$u_t = \max(0,\; u_{t-1} + z_t - k),\qquad \ell_t = \max(0,\; \ell_{t-1} - z_t - k)$$

**Meldung und Zurücksetzen.** Wird $u_t > h$ oder $\ell_t > h$, beginnt die Auslenkung am ersten Tag $t_0$ der laufenden Auslenkung (dem Tag nach dem letzten Tag, an dem die Statistik 0 war), und das Niveau wird auf das Mittel seit $t_0$ gesetzt (mindestens $w$ Tage); beide Statistiken
beginnen wieder bei 0:
$$o_t = \frac{1}{\max(t - t_0 + 1,\; w)} \sum_{s=t-\max(t-t_0+1,\,w)+1}^{t} r_s$$

**Bestätigung** (c > 0): Das Niveau wird erst am Tag $t+c$ gesetzt, und nur, wenn das Mittel der letzten $\min(c,7)$ Residuen in Alarmrichtung $s \in \{+1, -1\}$ mindestens $\theta$ Standardabweichungen über dem
Referenzmittel $\bar r_0$ liegt (Standardabweichung $s_0$ der Residuen im Referenzfenster):
$$s\cdot\frac{\bar r_{t+c-\min(c,7)+1\,:\,t+c} - \bar r_0}{s_0} \ge \theta$$

**Prognose und Bestellbestand.** Mittelwerttreu im Original-Maßstab, Restvarianz $\hat s^2$ der Regression:
$$F_{t+j} = \max\!\left(\exp\!\left(x_{t+j}^\top\beta + o_{t-1} + \tfrac{1}{2}\hat s^2\right) - 1,\; 0\right),\qquad S_t = (F_m + 1)\,e^{q_\tau} - 1$$
Kosten je Tag: $\text{Lagerbestand} + p \cdot \text{Rückstand}$ mit $p = \tau/(1-\tau)$ bei Lager 1 je Stück und Tag; angegeben in Tagesbedarfen (geteilt durch die mittlere Tagesnachfrage des Testjahres).

**Orakel.** Das Orakel nutzt dieselbe Regression und die Nachführung mit $\alpha = 0{,}03$, kennt aber Tag und Größe jedes Sprungs (ein Maßstab, keine strenge Obergrenze: $\alpha = 0{,}03$ ist in allen 64 Abstimmungen der Karte der kleinste Wert des Abstimmungsrasters, und in 2 von 32 Zellen liegt das Orakel in der MASE hinter dem Gegner) und addiert $\ln(1+\text{Änderung})$ genau am Sprungtag.

**Skizze, kein Optimierungsmodell:** Die Demo vergleicht Verfahren; sie löst kein Optimierungsproblem. Die Order-up-to-Regel ist die der `forecast-inventory-demo`; ihre Optimalität (Newsvendor) ist hier nicht Gegenstand.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Zeitreihen-Prognose: von Naiv bis Vortraining](https://sebastianhanisch.net/konzepte-zeitreihen-prognose.html)."
)
