# Niveausprung erkennen – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-niveausprung-demo.streamlit.app/)** -->

Stück 12 der **Zeitreihen-Prognose: von Naiv bis Vortraining** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vorgänger sind die Stücke der Linie, die den Preis eines Niveausprungs zeigen
(`naive-forecast-demo`, `exponential-smoothing-demo`, `arima-demo`, `dynamic-regression-demo`) und die Bestandsentscheidung aus Prognosen (`forecast-inventory-demo`); die Kante ist ein **Kontrast**, kein Fix: Eine gut abgestimmte Nachführung holt den
Sprung selbst auf, das Erkennen muss sich gegen sie behaupten. Vehikel: tägliche Aufträge je Depot (`nf_scenario` der Linie, drei Jahre, Ursprünge im letzten Jahr).

## Warum dieses Problem

**Erkennen eines Niveausprungs lohnt in Bestandskosten, aber nur in einem Teil der Karte Rauschen × Sprunggröße, und der Gewinn ist kleiner, als die Regression allein bringt; die Bestätigungsregel trennt Aktionswoche und Sprung gegen Verzögerung (bei Rauschen 14 % trennt c = 10 die einwöchige +50-%-Aktion, bei Rauschen 4 % nicht).**
Ein Orakel, das Tag und Größe des Sprungs kennt, spart in jeder der 32 Zellen der Karte (Abschnitt 3) im Testjahr mehr Bestandskosten als der Detektor; wie viel davon ein CUSUM-Detektor holt, wann das Zurücksetzen nach einer Meldung sogar schadet und was eine Bestätigung kostet,
messen die Kernfragen der App: (1) Wie viele Tage kostet „höchstens eine Fehlmeldung je Jahr“? (2) Wo auf der Karte Rauschen × Sprung lohnt Erkennen? (3) Was kostet das Trennen von Aktionswoche und Sprung? (4) Wie viel der Orakel-Reserve holt der Detektor, und wohin geht der Rest?

**Bezug zu OR:** Die Prognose ist hier Mittel zum Zweck: Aus ihr und ihren bisherigen Fehlern folgt der Bestellbestand einer Order-up-to-Regel (Lager 1 je Stück und Tag, Rückstand p = τ/(1−τ), Wiederbeschaffungszeit L). Die Frage ist die der Bestandsentscheidung nach einer Meldung:
Lohnt es, auf eine Meldung hin das Niveau zu ändern, wenn jede Fehlmeldung und jede Verzögerung Lager- oder Rückstandskosten verursacht?

## Modell

Die Regression im Logarithmus (Wochentag, zwei Fourier-Paare des Jahresmusters, Trend, bekannte Feiertage und Aktionen; kleinste Quadrate auf den Tagen 28 bis 729) ist für alle Nachführer dieselbe (saisonal naiv, Wochenmittel und die feste Glättung nutzen sie nicht); die Nachführer unterscheiden sich nur im **Niveau** (Offset), das aus den Residuen nachgeführt wird:

- **EWMA:** o_t = o_(t−1) + α·e_t mit dem Ein-Schritt-Fehler e_t = r_t − o_(t−1).
- **Mittel der letzten k Tage** und **Trigg-Leach** (adaptives α = |geglätteter Fehler| / geglättetes |Fehler|, auf [α_min, 0,5] geklemmt).
- **CUSUM-Zurücksetzen:** langsame EWMA-Nachführung (α, Standard 0,06) zwischen den Meldungen; zweiseitiger CUSUM auf z_t = e_t/σ (σ aus dem Referenzfenster Tage 639 bis 729) mit Zulage k und Schwelle h; bei Überschreiten wird das Niveau auf das Mittel der Residuen seit Beginn der
  Auslenkung gesetzt (mindestens w Tage; Beginn = erster Tag der laufenden Auslenkung, also der Tag nach dem letzten Tag, an dem die Statistik 0 war). Mit **Bestätigung** (c Wartetage, θ) wird erst c Tage nach dem Alarm gesetzt, und nur, wenn das Mittel der letzten min(c, 7) Residuen in Alarmrichtung mindestens θ Standardabweichungen über dem Referenzmittel liegt.
- **Orakel:** dieselbe Regression, EWMA mit α = 0,03, dazu das Niveau am Sprungtag um die Größe des Sprungs gesetzt (ein Maßstab, keine strenge Obergrenze: α = 0,03 ist in allen 64 Abstimmungen der Karte, also 32 Zellen mal 2 Ziele (MASE und Kosten), der kleinste Wert des Abstimmungsrasters 0,03 / 0,06 / 0,1 / 0,15, und in 2 von 32 Zellen (Rauschen 24 %, Sprung +10 %, ohne und mit Autokorrelation) liegt das Orakel in der MASE hinter dem Gegner, mit −0,15 ± 0,03 % und −0,17 ± 0,05 %; in den Kosten liegt es in keiner der 32 Zellen hinter dem Gegner. Ein Orakel „um die Verzögerung verspätet“ setzt das Niveau um die mittlere Verzögerung des Detektors später).

Die Bestandsentscheidung: Bestellbestand S = (F_m + 1)·exp(q) − 1 mit F_m = Summe der Prognosen über das Schutzintervall m = L + 1 und q = τ-Quantil der Fehler log((D_m + 1)/(F_m + 1)), empirisch (Trainingsfehler) oder konform (letzte 90 realisierte Fehler);
Kosten je Tag = Lagerbestand + p·Rückstand, in Tagesbedarfen. Formeln vollständig im 📐-Expander der App. **Zähleinheiten:** **MASE** je Reihe = mittlerer absoluter Fehler der Prognose über die Horizonte 1 bis 14 Tage und die Prognoseursprünge Tag 730 bis 1081 (Testjahr), geteilt durch den mittleren absoluten Fehler der saisonal naiven Prognose (Wert von vor 7 Tagen) auf den Trainingstagen 7 bis 729 derselben Reihe, gemittelt über die Reihen; im Fenster nur über die Ursprünge im Fenster; Kosten je Tag in Tagesbedarfen (Kosten je Tag geteilt durch die mittlere Tagesnachfrage des Testjahres der Reihe); Fenster = 60 Tage ab dem Sprung (180 Tage bei
Folgen von Sprüngen); **Meldung** = Zurücksetzen des Niveaus (mit Bestätigung: der bestätigte Alarm); Verzögerung in Tagen ab Sprungtag.

## Methodik

- **Aufbau der App** (`app.py`): Presets und alle Regler in der Seitenleiste; im Hauptbereich das Ergebnis (Gewinne gegen den stärksten eingestellten Gegner, eine Reihe im Detail, Vergleich aller Verfahren) und die vier Kernfragen mit Verdikt. Jeder Regler ändert die Kennzahlen seines Verfahrens, und die übrigen Verfahren bleiben bit-gleich
  (`tests/test_regler_wirken.py`); im Hauptergebnis (Gewinn gegen den stärksten Gegner) wirkt ein Regler der Gegner nur, solange dieser Gegner der stärkste ist (im Standardszenario ändern „Mittel der letzten k Tage“ und „Trigg-Leach: λ“ den Gewinn nicht, sie wirken im Vergleich aller Verfahren). Die Auswertung läuft live über 12 bis 60 Reihen mit den Zufalls-Seeds `seed` bis `seed + Zahl der Reihen − 1`; Gewinne sind gepaart (gleiche Reihen) und haben einen Standardfehler.
- **Karte** (`data/karte.json`, erzeugt von `tools/messlauf.py`): je Zelle (Rauschen 4/8/14/24 % × Sprung +10/20/30/50 % × Autokorrelation 0 und 0,6) werden EWMA, Mittel, Trigg-Leach, feste Glättung, Detektor und Orakel auf **getrennten Abstimm-Reihen** (Zufalls-Seeds 12000 bis 12099, halb mit, halb ohne Sprung) abgestimmt,
  je Ziel MASE und Kosten; der stärkste Gegner ist die Familie mit dem kleinsten abgestimmten Zielwert. Gemessen wird auf 300 Reihen mit Sprung (Zufalls-Seeds 10000 bis 10299). Die Karte gilt für die dauerhafte Stufe, die Regression mit Kalender, τ = 0,95, L = 3 und die empirische Regel.
- **Weitere Messläufe** (`tools/messlauf.py`, Tabellen in `tools/readme_tabellen.py`): Standardszenario mit 600 Reihen (Zufalls-Seeds 16000 bis 16599), Bestätigung und Kennlinie mit 600 Reihen (14000 bis 14599), Kostenannahmen mit den 300 Messreihen. Die Tabellen dieser README werden aus den gespeicherten Messungen erzeugt und von `tests/test_claims.py` gegen sie geprüft.
- **Gegner nicht von Hand abgestimmt:** Die Regler von Detektor und Gegnern sind Voreinstellungen aus der Abstimmung der Zelle Rauschen 14 %, Sprung +30 %, keine Autokorrelation (Ziel Kosten). Die Presets „Ruhige Reihe“, „Laute Reihe“ und „Autokorrelierte Fehler“ übernehmen die Abstimmung ihrer eigenen Zelle (Rauschen 4 % bzw. 24 %, Autokorrelation 0,6); wer sonst Rauschen, Sprung oder Form ändert, sieht die nicht neu abgestimmten Verfahren.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen mit ihrer Voraussetzung im Satz; die Tabellen sind erzeugt (`tools/readme_tabellen.py`) und werden gegen `data/*.json` geprüft, die Sätze gegen dieselben Dateien (`tests/test_claims.py`, Zahl → Test siehe Tabelle am Ende dieses Abschnitts). Gewinn = (Gegner − Verfahren) / Gegner in Prozent des Gegners, gepaart,
± Standardfehler; positiv heißt, das Verfahren ist besser.

### 1. Kennlinie: wie viele Tage kostet „höchstens eine Fehlmeldung je Jahr“?

Rauschen 14 %, ein Sprung je Reihe, 600 Reihen mit und 600 ohne Sprung (gleiche Zufallszahlen), Detektor wie im Standard (α = 0,06, k = 0,5, w = 5, ohne Bestätigung), Schwellenraster 3, 4, 5, 6, 7, 8, 10, 12, 15, 20. Je Bereinigung die kleinste Schwelle mit höchstens einer Meldung je Jahr auf den Reihen ohne Sprung:

<!-- BEGIN:kennlinie -->
| Bereinigung | kleinstes h mit höchstens einer Meldung je Jahr | Meldungen je Jahr | Sprung +7 % | Sprung +14 % | Sprung +28 % | Sprung +42 % |
|---|---|---|---|---|---|---|
| nur Wochentag | 10 | 0,53 | 13 % (23,4 d) | 27 % (17,4 d) | 64 % (10,8 d) | 92 % (7,0 d) |
| Wochentag, Jahresmuster, Trend | 8 | 0,90 | 13 % (21,3 d) | 24 % (16,7 d) | 84 % (10,3 d) | 99 % (5,1 d) |
| zusätzlich Feiertage und Aktionsplan | 6 | 0,30 | 9 % (16,3 d) | 44 % (11,1 d) | 98 % (5,5 d) | 100 % (2,7 d) |
<!-- END:kennlinie -->

Mit Kalender und Aktionsplan genügt **h = 6**: 0,30 Meldungen je Jahr, und ein Sprung von +28 % wird in 98 % der Reihen nach 5,5 Tagen gemeldet. Kennt die Regression die Ereignisse nicht (Wochentag, Jahresmuster, Trend), braucht der Betriebspunkt h = 8, und der +28-%-Sprung wird in 84 % der Reihen nach 10,3 Tagen gemeldet; kennt sie
nur den Wochentag, ist es h = 10 mit 64 % nach 10,8 Tagen. Die Voraussetzung ist ein Modell, das die Struktur der Reihe trifft (hier ist die Reihe erzeugt wie angepasst).

### 2. Standardszenario: Rauschen 14 %, Sprung +30 % an einem zufälligen Tag, keine Autokorrelation (600 Reihen, Zufalls-Seeds 16000 bis 16599)

Mittel über 600 Reihen (Standardfehler der Mittel höchstens 0,12; Kosten in Tagesbedarfen je Tag, τ = 0,95, L = 3). „auf MASE“ bzw. „auf Kosten“: Parameter aus der Abstimmung dieser Zelle auf getrennten Reihen.

<!-- BEGIN:head -->
| Verfahren | MASE alle | MASE Fenster | Kosten Testjahr | Kosten Fenster | Kosten Testjahr, konform | Kosten Fenster, konform |
|---|---|---|---|---|---|---|
| Saisonal naiv (Stück 1) | 1,346 | 1,543 | 1,325 | 1,874 | 1,467 | 2,075 |
| Wochenmittel, 4 Wochen (Stück 1) | 1,125 | 1,359 | 1,307 | 2,310 | 1,386 | 2,132 |
| Saisonale Glättung, feste Parameter (α = 0,10) | 1,025 | 1,197 | 1,099 | 1,663 | 1,186 | 1,710 |
| Regression ohne Nachführung | 1,381 | 1,946 | 6,031 | 10,823 | 0,982 | 2,409 |
| Regression + EWMA, abgestimmt auf MASE | 0,908 | 1,127 | 0,958 | 2,314 | 0,965 | 1,854 |
| Regression + Mittel der letzten k Tage, auf MASE | 0,914 | 1,154 | 1,004 | 2,568 | 1,017 | 2,078 |
| Regression + Trigg-Leach, auf MASE | 0,925 | 1,072 | 0,891 | 1,446 | 0,974 | 1,601 |
| Regression + CUSUM-Zurücksetzen, auf MASE | 0,895 | 1,071 | 0,883 | 1,901 | 0,983 | 2,014 |
| Regression + EWMA, abgestimmt auf Kosten | 0,906 | 1,076 | 0,886 | 1,760 | 0,943 | 1,700 |
| Regression + Mittel der letzten k Tage, auf Kosten | 0,914 | 1,084 | 0,906 | 1,763 | 0,983 | 1,810 |
| Regression + Trigg-Leach, auf Kosten | 0,925 | 1,072 | 0,891 | 1,446 | 0,974 | 1,601 |
| Regression + CUSUM-Zurücksetzen, auf Kosten | 0,897 | 1,056 | 0,864 | 1,709 | 0,957 | 1,849 |
| Orakel: kennt Tag und Größe | 0,880 | 0,987 | 0,743 | 1,089 | 0,796 | 1,165 |
| Orakel, um die mittlere Verzögerung des Detektors verspätet | 0,896 | 1,081 | 0,867 | 1,843 | 1,000 | 2,157 |
| wahrer Erwartungswert (Orakel der Linie, kein Nachführer) | 0,844 | 0,952 | 0,666 | 0,734 | 0,702 | 0,773 |
<!-- END:head -->

Gewinn des CUSUM-Zurücksetzens (MASE: auf MASE abgestimmte Fassung, Kosten: auf Kosten abgestimmte) und Gewinn des Orakels gegen jeden Gegner:

<!-- BEGIN:gewinn -->
| Gegner | MASE alle | Kosten Testjahr | Kosten Fenster | Kosten Testjahr, konform | Kosten Fenster, konform | Orakel: Kosten Testjahr |
|---|---|---|---|---|---|---|
| Saisonale Glättung (α = 0,10) | +12,6 ± 0,2 | +21,4 ± 0,6 | −2,8 ± 1,5 | +19,3 ± 0,6 | −8,1 ± 1,5 | +32,4 ± 0,6 |
| Regression ohne Nachführung | +35,2 ± 0,7 | +85,7 ± 1,5 | +84,2 ± 1,0 | +2,6 ± 0,5 | +23,3 ± 1,0 | +87,7 ± 1,4 |
| Regression + EWMA | +1,4 ± 0,1 | +2,5 ± 0,2 | +2,9 ± 0,7 | −1,5 ± 0,2 | −8,7 ± 0,6 | +16,2 ± 0,4 |
| Regression + Mittel der letzten k Tage | +2,0 ± 0,1 | +4,6 ± 0,3 | +3,0 ± 0,8 | +2,6 ± 0,3 | −2,1 ± 0,7 | +18,0 ± 0,4 |
| Regression + Trigg-Leach | +3,2 ± 0,1 | +3,1 ± 0,3 | −18,2 ± 0,8 | +1,7 ± 0,4 | −15,5 ± 0,9 | +16,6 ± 0,4 |
<!-- END:gewinn -->

- **Gegen die stärksten abgestimmten Gegner ist der Gewinn klein:** im Testjahr +1,4 % MASE und +2,5 % Kosten gegen die EWMA-Nachführung (Kosten gegen das Mittel der letzten k Tage +4,6 %, gegen Trigg-Leach +3,1 %), im Sprungfenster gegen die EWMA +2,9 ± 0,7 %.
- **Im Sprungfenster verliert das Zurücksetzen gegen Trigg-Leach:** −18,2 ± 0,8 % Kosten (konforme Regel −15,5 ± 0,9 %). Der stärkste Gegner hängt also vom Maß ab: im Testjahr liegt die EWMA knapp vor Trigg-Leach (Kosten 0,886 gegen 0,891), im Fenster liegt Trigg-Leach klar vorn (1,446 gegen 1,760).
- **Der Vorsprung gegen die Glättung kommt zum größten Teil aus der Regression, nicht aus dem Detektor:** Die Regression mit EWMA-Nachführung erreicht eine MASE von 0,908 gegen 1,025 der festen Glättung; das Zurücksetzen bringt dazu 0,895.
- **Die Orakel-Reserve ist groß:** +16,2 ± 0,4 % Kosten im Testjahr und +38,1 % im Fenster gegen die EWMA. Der Detektor holt davon im Testjahr 16 % (2,5 von 16,2 Prozentpunkten, Quotient der ungerundeten Werte). Ein Orakel, das den Sprung erst nach der mittleren Verzögerung des Detektors (5 Tage, im Mittel über die 99 % gemeldeten Sprünge)
  kennt, kostet im Testjahr 0,867 gegen 0,864 des Detektors, im Sprungfenster aber 1,843 gegen 1,709 (dort ist das verspätete Orakel schlechter als der Detektor). Im Testjahr ist der Rückstand auf das Orakel damit größenordnungsmäßig die Verzögerung; die Gleichheit der Summen ist keine Zerlegung, Fehlmeldungen, streuende Verzögerung und Schätzfehler des neuen Niveaus können sich darin aufheben.
- **Die Regel des Sicherheitsbestands verschiebt das Bild:** Mit der konformen Regel kostet die Regression ganz ohne Nachführung 0,982 gegen 0,957 des Zurücksetzens (+2,6 ± 0,5 %); gegen die EWMA liegt das Zurücksetzen mit der konformen Regel zurück (−1,5 ± 0,2 %). Deutung, nicht getrennt gemessen: Das Erkennen konkurriert mit der konformen Kalibrierung, die einen Teil der Verzögerung über die Fehlerquantile abfängt; Detektor und Gegner sind für die empirische Regel abgestimmt, nicht für die konforme.

### 3. Karte: wo lohnt Erkennen?

Gewinn des CUSUM-Zurücksetzens gegen den stärksten abgestimmten Gegner je Zelle (300 Reihen, gepaart, ± Standardfehler; Gegner, Detektor und Orakel je Zelle abgestimmt, τ = 0,95, L = 3, empirische Regel). **Bestandskosten, Testjahr** (Ziel Kosten):

<!-- BEGIN:karte_kosten_alle -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | +2,8 ± 0,5 | +5,3 ± 0,6 | +5,4 ± 0,5 | +7,1 ± 0,4 |
| 4 % | 0,6 | −2,3 ± 0,2 | −1,8 ± 0,5 | −0,4 ± 0,5 | +3,9 ± 0,5 |
| 8 % | 0,0 | +0,0 ± 0,0 | +2,1 ± 0,5 | +5,7 ± 0,7 | +5,5 ± 0,7 |
| 8 % | 0,6 | −0,1 ± 0,0 | −2,3 ± 0,2 | −2,3 ± 0,4 | −0,9 ± 0,5 |
| 14 % | 0,0 | −0,1 ± 0,0 | +0,7 ± 0,2 | +2,4 ± 0,3 | +9,4 ± 0,6 |
| 14 % | 0,6 | −0,2 ± 0,1 | −0,1 ± 0,0 | −1,5 ± 0,2 | −2,8 ± 0,4 |
| 24 % | 0,0 | +0,0 ± 0,0 | −0,1 ± 0,1 | +0,0 ± 0,0 | +0,0 ± 0,5 |
| 24 % | 0,6 | −0,1 ± 0,3 | −0,2 ± 0,1 | +0,5 ± 0,2 | −0,1 ± 0,0 |
<!-- END:karte_kosten_alle -->

**Bestandskosten im Sprungfenster:**

<!-- BEGIN:karte_kosten_fenster -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | −5,5 ± 1,5 | −3,3 ± 1,3 | +0,8 ± 0,9 | +5,1 ± 0,6 |
| 4 % | 0,6 | −10,2 ± 0,7 | −12,9 ± 1,1 | −7,5 ± 1,1 | +8,2 ± 0,6 |
| 8 % | 0,0 | +0,0 ± 0,0 | −8,4 ± 1,4 | −3,0 ± 1,5 | −1,9 ± 1,3 |
| 8 % | 0,6 | −0,2 ± 0,1 | −10,2 ± 0,7 | −9,5 ± 1,1 | −9,1 ± 1,1 |
| 14 % | 0,0 | −0,3 ± 0,1 | +2,6 ± 0,6 | +2,7 ± 0,9 | +12,5 ± 1,3 |
| 14 % | 0,6 | −0,4 ± 0,2 | −0,2 ± 0,1 | −9,5 ± 0,7 | −13,2 ± 1,2 |
| 24 % | 0,0 | +0,0 ± 0,0 | −0,5 ± 0,2 | −0,1 ± 0,1 | −6,8 ± 1,1 |
| 24 % | 0,6 | +0,8 ± 0,7 | −0,5 ± 0,2 | +5,6 ± 0,8 | −0,3 ± 0,1 |
<!-- END:karte_kosten_fenster -->

**MASE, alle Ursprünge** (Ziel MASE):

<!-- BEGIN:karte_mase_alle -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | +2,0 ± 0,1 | +4,3 ± 0,1 | +3,1 ± 0,1 | +3,5 ± 0,1 |
| 4 % | 0,6 | +0,9 ± 0,2 | +4,8 ± 0,2 | +6,5 ± 0,2 | +6,0 ± 0,2 |
| 8 % | 0,0 | +0,4 ± 0,1 | +2,5 ± 0,1 | +3,5 ± 0,1 | +3,3 ± 0,1 |
| 8 % | 0,6 | −0,7 ± 0,1 | +0,4 ± 0,2 | +3,7 ± 0,2 | +5,6 ± 0,2 |
| 14 % | 0,0 | +0,0 ± 0,0 | +0,0 ± 0,1 | +1,2 ± 0,1 | +2,7 ± 0,1 |
| 14 % | 0,6 | −0,4 ± 0,1 | −0,7 ± 0,1 | +0,2 ± 0,2 | +2,2 ± 0,2 |
| 24 % | 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,3 ± 0,1 |
| 24 % | 0,6 | −0,4 ± 0,1 | −0,6 ± 0,1 | −1,0 ± 0,2 | −0,9 ± 0,2 |
<!-- END:karte_mase_alle -->

**MASE im Sprungfenster:**

<!-- BEGIN:karte_mase_fenster -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | +4,4 ± 0,5 | +10,8 ± 0,5 | +0,9 ± 0,4 | +4,6 ± 0,4 |
| 4 % | 0,6 | +5,1 ± 0,7 | +12,2 ± 0,7 | +13,4 ± 0,7 | −1,1 ± 0,8 |
| 8 % | 0,0 | +2,7 ± 0,4 | +10,2 ± 0,5 | +11,4 ± 0,5 | +1,5 ± 0,5 |
| 8 % | 0,6 | −1,5 ± 0,5 | +3,2 ± 0,6 | +12,6 ± 0,6 | +14,8 ± 0,7 |
| 14 % | 0,0 | −0,1 ± 0,0 | −2,6 ± 0,3 | +4,4 ± 0,4 | +7,8 ± 0,5 |
| 14 % | 0,6 | −1,0 ± 0,4 | −1,5 ± 0,5 | +2,5 ± 0,7 | +6,3 ± 0,7 |
| 24 % | 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,2 ± 0,4 |
| 24 % | 0,6 | −1,0 ± 0,4 | −1,5 ± 0,4 | −2,8 ± 0,6 | −1,5 ± 0,6 |
<!-- END:karte_mase_fenster -->

**Reserve: Gewinn des Orakels gegen denselben Gegner, Bestandskosten im Testjahr:**

<!-- BEGIN:karte_orakel -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | +16,4 ± 0,5 | +19,3 ± 0,5 | +20,3 ± 0,5 | +20,9 ± 0,4 |
| 4 % | 0,6 | +7,1 ± 0,6 | +8,6 ± 0,6 | +11,4 ± 0,6 | +16,1 ± 0,5 |
| 8 % | 0,0 | +10,0 ± 0,4 | +16,2 ± 0,5 | +18,5 ± 0,5 | +20,4 ± 0,5 |
| 8 % | 0,6 | +2,9 ± 0,4 | +6,8 ± 0,6 | +8,1 ± 0,6 | +9,7 ± 0,6 |
| 14 % | 0,0 | +5,2 ± 0,3 | +11,9 ± 0,5 | +15,5 ± 0,5 | +21,0 ± 0,6 |
| 14 % | 0,6 | +1,3 ± 0,3 | +3,4 ± 0,5 | +5,2 ± 0,6 | +7,4 ± 0,6 |
| 24 % | 0,0 | +1,6 ± 0,2 | +6,7 ± 0,4 | +8,8 ± 0,4 | +13,6 ± 0,5 |
| 24 % | 0,6 | +0,7 ± 0,2 | +1,7 ± 0,3 | +3,4 ± 0,4 | +5,7 ± 0,6 |
<!-- END:karte_orakel -->

**Reserve im Sprungfenster:**

<!-- BEGIN:karte_orakel_fenster -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | +34,9 ± 1,4 | +29,8 ± 1,1 | +28,5 ± 0,9 | +26,7 ± 0,7 |
| 4 % | 0,6 | +23,8 ± 2,0 | +14,7 ± 1,5 | +19,5 ± 1,2 | +25,0 ± 0,9 |
| 8 % | 0,0 | +33,7 ± 1,7 | +33,3 ± 1,5 | +29,4 ± 1,3 | +29,7 ± 1,0 |
| 8 % | 0,6 | +12,4 ± 1,6 | +21,9 ± 1,9 | +19,9 ± 1,8 | +16,0 ± 1,4 |
| 14 % | 0,0 | +23,7 ± 1,6 | +37,3 ± 1,7 | +37,9 ± 1,5 | +39,7 ± 1,3 |
| 14 % | 0,6 | +7,1 ± 1,2 | +13,7 ± 1,7 | +16,5 ± 1,9 | +16,9 ± 1,8 |
| 24 % | 0,0 | +8,3 ± 1,1 | +27,8 ± 1,7 | +28,7 ± 1,6 | +33,2 ± 1,6 |
| 24 % | 0,6 | +4,1 ± 1,0 | +8,2 ± 1,3 | +15,4 ± 1,7 | +19,6 ± 1,8 |
<!-- END:karte_orakel_fenster -->

**Meldungen des Detektors je Reihe im Testjahr** (Ziel Kosten):

<!-- BEGIN:karte_meldungen -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | 1,33 | 1,09 | 2,32 | 2,43 |
| 4 % | 0,6 | 0,76 | 1,95 | 1,42 | 5,44 |
| 8 % | 0,0 | 0,00 | 1,35 | 1,41 | 1,14 |
| 8 % | 0,6 | 0,03 | 0,71 | 1,84 | 2,04 |
| 14 % | 0,0 | 0,04 | 0,54 | 1,31 | 1,31 |
| 14 % | 0,6 | 0,05 | 0,03 | 0,11 | 1,76 |
| 24 % | 0,0 | 0,00 | 0,07 | 0,00 | 1,70 |
| 24 % | 0,6 | 0,04 | 0,05 | 0,02 | 0,09 |
<!-- END:karte_meldungen -->

- **Ohne Autokorrelation lohnt Erkennen in den Kosten des Testjahres in 10 von 16 Zellen** (Gewinn mehr als zwei Standardfehler über null und mindestens 0,5 Prozentpunkte), in keiner Zelle ist es mehr als zwei Standardfehler schlechter. Größter Wert der Zeile Rauschen 4 %: +7,1 % (Sprung +50 %); +2,4 ± 0,3 % bei Rauschen 14 % und +30 %; der größte Wert der ganzen Karte ist +9,4 ± 0,6 % bei Rauschen 14 % und +50 %, und gerade er verschiebt sich bei neuer Abstimmung (nächster Punkt).
- **Der Standardfehler der Karte deckt nur die Messreihen bei gegebener Abstimmung, nicht die Abstimmung selbst.** Die Wiederholung der ganzen Karte mit anderen Abstimm-Reihen (Zufalls-Seeds 22000 bis 22099) und anderen Messreihen (20000 bis 20299) verändert den Gewinn in den Kosten des Testjahres in 8 von 32 Zellen um mehr als zwei kombinierte Standardfehler (Zelle Rauschen 14 %, +50 %: +9,4 statt +6,2 %; Rauschen 4 %, +20 %: +5,3 statt +7,7 %). Die Wiederholung (Kosten, Testjahr):

<!-- BEGIN:wiederholung -->
| Rauschen | Autokorrelation | Sprung +10 % | Sprung +20 % | Sprung +30 % | Sprung +50 % |
|---|---|---|---|---|---|
| 4 % | 0,0 | +3,0 ± 0,5 | +7,7 ± 0,5 | +6,2 ± 0,5 | +8,2 ± 0,4 |
| 4 % | 0,6 | −2,6 ± 0,4 | −0,6 ± 0,5 | +0,8 ± 0,5 | +3,7 ± 0,4 |
| 8 % | 0,0 | +0,5 ± 0,3 | +2,3 ± 0,6 | +7,3 ± 0,6 | +6,9 ± 0,5 |
| 8 % | 0,6 | +0,0 ± 0,0 | −1,8 ± 0,3 | −3,7 ± 0,4 | +0,4 ± 0,5 |
| 14 % | 0,0 | −0,1 ± 0,1 | +1,4 ± 0,3 | +2,3 ± 0,3 | +6,2 ± 0,6 |
| 14 % | 0,6 | −0,3 ± 0,1 | +0,0 ± 0,0 | −1,3 ± 0,2 | −3,6 ± 0,4 |
| 24 % | 0,0 | −0,1 ± 0,1 | +0,0 ± 0,1 | −0,1 ± 0,2 | −0,3 ± 0,5 |
| 24 % | 0,6 | −1,0 ± 0,3 | −0,3 ± 0,1 | −0,3 ± 0,1 | +0,0 ± 0,0 |
<!-- END:wiederholung -->

  Die Zählungen bleiben nahe beieinander (ohne Autokorrelation 10 und 11 von 16 Zellen mit Gewinn, keine mit Verlust); einzelne Zellen aber schwanken mit der Abstimmung um mehrere Prozentpunkte, und die Karte ist als Ganzes zu lesen, nicht Zelle für Zelle.
- **Bei Rauschen 24 % bringt der auf Kosten abgestimmte Detektor in keiner der vier Zellen ohne Autokorrelation einen Gewinn:** |Gewinn| höchstens 0,1 Prozentpunkte in den Kosten des Testjahres in den drei Zellen bis +30 %, bei +50 % +0,0 ± 0,5 (Wiederholung: −0,3 ± 0,5; die MASE gewinnt bei +50 % +0,3 ± 0,1, in der Wiederholung +0,7 ± 0,1). Das liegt zum Teil an der Abstimmung: Sie wählt dort Schwelle und Zulage, die den Detektor praktisch abschalten (in 8 von 32 Zellen unter 0,05 Meldungen je Reihe, Tabelle der Meldungen), und dann misst die Zelle nur den Unterschied zwischen der langsamen EWMA und dem Gegner. Der Sprung ist bei Rauschen 24 % nicht unerkennbar:

<!-- BEGIN:laut -->
| Rauschen 24 %, Sprung | Detektor | h, k, α, w | Sprung gemeldet | Verzögerung | Meldungen je Jahr ohne Sprung |
|---|---|---|---|---|---|
| +30 % | Standardwerte der App | 6, 0,5, 0,06, 5 | 58,8 % | 9,9 d | 0,30 |
| +30 % | abgestimmt auf Kosten (Zelle) | 8, 1,0, 0,06, 2 | 0,8 % | 6,4 d | 0,00 |
| +50 % | Standardwerte der App | 6, 0,5, 0,06, 5 | 97,5 % | 6,0 d | 0,30 |
| +50 % | abgestimmt auf Kosten (Zelle) | 6, 0,5, 0,03, 5 | 99,5 % | 5,0 d | 0,65 |
<!-- END:laut -->

  Mit den Standardwerten der App wird +30 % in 58,8 % und +50 % in 97,5 % der Reihen gemeldet (600 Reihen); „Grenze“ heißt hier: Nach der Abstimmung auf Kosten lohnt das Melden nicht mehr, nicht: Der Sprung ist im Rauschen nicht zu sehen. In der Zelle +50 % meldet der Detektor trotzdem 1,70 Mal je Reihe und gewinnt in den Kosten nichts.
- **Mit Autokorrelation 0,6 kippen die Kosten in 6 von 16 Zellen ins Negative:** In 6 von 16 Zellen kostet das Zurücksetzen im Testjahr mehr als zwei Standardfehler mehr, in 1 von 16 spart es (Wiederholung: 6 und 1); die MASE gewinnt der Detektor in 7 von 16 Zellen, verliert in 5 (Wiederholung: 9 und 5). Die Aussage „mehr Zellen mit Verlust als mit Gewinn“ hält in beiden Läufen für die Kosten, die Zählungen selbst hängen an Zellen nahe der Zwei-Standardfehler-Grenze. Gemessen ist: Mit denselben Detektorwerten (α = 0,06, h = 6, k = 0,5, w = 5) meldet der CUSUM auf Reihen ohne Sprung bei Autokorrelation 0,6 8,72 ± 0,19 statt 0,30 ± 0,03 Mal je Jahr (600 Reihen, Rauschen 14 %). Dass diese Meldungen die negativen Kosten erklären, ist Deutung, nicht getrennt gemessen.
- **Folgen von Sprüngen** (Abstimmung des Einzelsprungs; in allen drei Zeilen dasselbe Fenster von 180 Tagen ab dem ersten Sprung, damit die Zeilen vergleichbar sind; Orakel kennt alle Sprünge). Der Sprungtag liegt gleichverteilt in den Tagen 760 bis 1034 (gemessen: in 7,7 % der 300 Messreihen liegt der Rücksprung, in 15,7 % der zweite Sprung hinter dem Reihenende, in 37,7 % ragt das Fenster über das Ende hinaus); dort fehlen die Folgen oder das Fenster ist verkürzt:

<!-- BEGIN:multi -->
| Rauschen | Folge | MASE alle | Kosten Testjahr | Kosten Fenster (180 Tage) | Orakel: Kosten Testjahr | Meldungen je Reihe | letzter Sprung hinter dem Reihenende | Fenster über das Reihenende hinaus |
|---|---|---|---|---|---|---|---|---|
| 4 % | ein Sprung +30 % | +3,1 ± 0,1 | +5,4 ± 0,5 | +2,9 ± 0,8 | +20,3 | 2,32 | 0,0 % | 37,7 % |
| 4 % | +30 %, nach 90 Tagen zurück | +2,2 ± 0,1 | +4,3 ± 0,5 | +1,7 ± 0,8 | +21,0 | 4,20 | 7,7 % | 37,7 % |
| 4 % | +30 %, nach 120 Tagen weitere +20 % | +2,4 ± 0,1 | +2,9 ± 0,5 | +0,6 ± 0,7 | +23,4 | 3,57 | 15,7 % | 37,7 % |
| 8 % | ein Sprung +30 % | +3,5 ± 0,1 | +5,7 ± 0,7 | +2,3 ± 1,1 | +18,5 | 1,41 | 0,0 % | 37,7 % |
| 8 % | +30 %, nach 90 Tagen zurück | +5,5 ± 0,1 | +4,4 ± 0,7 | +0,4 ± 1,1 | +19,5 | 2,47 | 7,7 % | 37,7 % |
| 8 % | +30 %, nach 120 Tagen weitere +20 % | +3,5 ± 0,1 | −0,3 ± 0,7 | −5,8 ± 1,1 | +20,8 | 2,30 | 15,7 % | 37,7 % |
| 14 % | ein Sprung +30 % | +1,2 ± 0,1 | +2,4 ± 0,3 | +2,5 ± 0,6 | +15,5 | 1,31 | 0,0 % | 37,7 % |
| 14 % | +30 %, nach 90 Tagen zurück | +2,4 ± 0,1 | +2,8 ± 0,3 | +3,2 ± 0,6 | +17,6 | 2,26 | 7,7 % | 37,7 % |
| 14 % | +30 %, nach 120 Tagen weitere +20 % | +0,4 ± 0,1 | −0,1 ± 0,4 | −1,1 ± 0,6 | +18,5 | 1,97 | 15,7 % | 37,7 % |
| 24 % | ein Sprung +30 % | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +8,8 | 0,00 | 0,0 % | 37,7 % |
| 24 % | +30 %, nach 90 Tagen zurück | +0,0 ± 0,0 | +0,0 ± 0,0 | −0,1 ± 0,0 | +10,8 | 0,01 | 7,7 % | 37,7 % |
| 24 % | +30 %, nach 120 Tagen weitere +20 % | +0,0 ± 0,0 | +0,0 ± 0,0 | +0,0 ± 0,0 | +11,2 | 0,00 | 15,7 % | 37,7 % |
<!-- END:multi -->

  Der **Rücksprung** (nach 90 Tagen zurück) verkraftet das Zurücksetzen bei Rauschen 14 %: +2,8 ± 0,3 % Kosten im Testjahr gegen +2,4 ± 0,3 % beim Einzelsprung. Der **zweite Sprung nach oben** (+20 %) bringt bei Rauschen 14 % im Testjahr keinen Gewinn (−0,1 ± 0,4 %); er wird dort in 77,5 % der Reihen innerhalb von 60 Tagen gemeldet (187 Reihen, in denen dieses Fenster vor dem Reihenende liegt), der Einzelsprung +20 % allein in 74,3 % der Reihen; bei Rauschen 8 % in 100,0 % (beide). Der negative Fensterwert bei Rauschen 8 % (−5,8 ± 1,1 %) hat also nicht seinen Grund in einem nicht gemeldeten Sprung; die Ursache ist nicht gemessen.

### 4. Bestätigung: Aktionswoche gegen Sprung

Anteil der Reihen mit einer Meldung im Fenster [Beginn, Beginn + 60 Tage + c) und mittlere Verzögerung ab Beginn der Form (600 Reihen, Rauschen 14 %, θ = 1,0; Detektor α = 0,06, h = 6, k = 0,5, w = 5):

<!-- BEGIN:confirm -->
| Form | c = 0 Tage | c = 3 Tage | c = 7 Tage | c = 10 Tage | c = 14 Tage | c = 21 Tage |
|---|---|---|---|---|---|---|
| Stufe +28 % | 98,5 % (5,5 d) | 91,8 % (9,3 d) | 95,7 % (12,6 d) | 96,5 % (15,7 d) | 95,2 % (19,6 d) | 94,7 % (26,5 d) |
| Stufe +14 % | 43,8 % (11,1 d) | 20,8 % (14,7 d) | 20,7 % (18,2 d) | 21,0 % (21,4 d) | 19,0 % (24,9 d) | 19,0 % (31,2 d) |
| Aktionswoche +25 % | 62,8 % (4,7 d) | 30,8 % (6,6 d) | 10,2 % (9,4 d) | 1,0 % (12,0 d) | 0,8 % (25,0 d) | 1,0 % (20,8 d) |
| Aktionswoche +50 % | 100,0 % (2,1 d) | 97,7 % (5,0 d) | 81,3 % (8,8 d) | 15,0 % (11,7 d) | 1,5 % (24,6 d) | 0,7 % (17,8 d) |
| Ausreißer +100 % | 28,2 % (1,7 d) | 1,8 % (10,1 d) | 0,0 % (– d) | 0,2 % (12,0 d) | 0,5 % (26,7 d) | 0,5 % (40,7 d) |
| Rampe auf +28 % | 44,3 % (38,5 d) | 32,3 % (47,3 d) | 35,7 % (51,4 d) | 36,0 % (53,7 d) | 36,8 % (56,6 d) | 39,2 % (61,9 d) |
| *keine Form: Meldungen je Jahr* | 0,300 | 0,013 | 0,000 | 0,003 | 0,007 | 0,005 |
<!-- END:confirm -->

Mit **c = 10 und θ = 1,0** sinkt die Verwechslung der einwöchigen +50-%-Aktion von 100,0 % auf 15,0 %, die der +25-%-Aktion von 62,8 % auf 1,0 %, die des Ausreißers von 28,2 % auf 0,2 %; der dauerhafte +28-%-Sprung bleibt zu 96,5 % bestätigt, mit 15,7 statt 5,5 Tagen Verzögerung (+10,2 Tage).
Mit c = 7 bleiben 81,3 % der +50-%-Aktion (das Mittel der letzten 7 Tage vor der Bestätigung liegt noch in der Aktion), mit c = 14 sind es 1,5 %. Die Grenze ist die Länge der Aktion: Das Beobachtungsfenster muss hinter ihrem Ende liegen. **Kosten der Trennung:** Der kleine Sprung von +14 % wird nur noch zu 21,0 % statt 43,8 % bestätigt, die Rampe
auf +28 % zu 36,0 % statt 44,3 %, und θ = 1,5 trennt besser, verliert aber Sprünge:

<!-- BEGIN:confirm_theta -->
| Form | θ = 1,0 | θ = 1,5 |
|---|---|---|
| Stufe +28 % | 96,5 % (15,7 d) | 79,3 % (17,0 d) |
| Stufe +14 % | 21,0 % (21,4 d) | 5,7 % (20,5 d) |
| Aktionswoche +25 % | 1,0 % (12,0 d) | 0,2 % (8,0 d) |
| Aktionswoche +50 % | 15,0 % (11,7 d) | 2,5 % (10,3 d) |
| Ausreißer +100 % | 0,2 % (12,0 d) | 0,0 % (– d) |
| Rampe auf +28 % | 36,0 % (53,7 d) | 23,0 % (56,9 d) |
<!-- END:confirm_theta -->

Mit θ = 1,5 und c = 10 bleibt der +28-%-Sprung nur noch zu 79,3 % bestätigt. Eine Aktion von 14 Tagen würde nach der Regel mit c = 10 bestätigt (nicht gemessen, folgt aus der Regel).

**Die nötige Wartezeit hängt an der Verzögerung des Alarms.** Das Beobachtungsfenster der Bestätigung (die letzten 7 Tage) beginnt c − 6 Tage nach dem Alarm; meldet der Detektor die Aktion schon am ersten Tag, muss c − 6 ≥ 7 sein, also c ≥ 13 (folgt aus der Regel). Bei Rauschen 4 % (Detektor der Zelle Rauschen 4 %, +30 %:
h = 4, k = 1,0, α = 0,03, w = 2) ist die Verzögerung 0,0 Tage, und c = 10 trennt die Aktionswoche nicht (600 Reihen, Rauschen 4 %):

<!-- BEGIN:confirm_ruhig -->
| Form | c = 0 Tage | c = 10 Tage | c = 14 Tage | c = 21 Tage | c = 10, θ = 1,5 | c = 14, θ = 1,5 |
|---|---|---|---|---|---|---|
| Stufe +28 % | 100,0 % (0,1 d) | 100,0 % (10,1 d) | 100,0 % (14,1 d) | 100,0 % (21,2 d) | 100,0 % (10,1 d) | 100,0 % (14,1 d) |
| Aktionswoche +50 % | 100,0 % (0,0 d) | 100,0 % (10,0 d) | 4,5 % (35,0 d) | 2,3 % (43,8 d) | 100,0 % (10,0 d) | 0,2 % (29,0 d) |
<!-- END:confirm_ruhig -->

Die Aktionswoche +50 % bleibt dort mit c = 10 zu 100,0 % bestätigt (auch mit θ = 1,5), mit c = 14 nur zu 4,5 %. Die Wartezeit der Bestätigung ist also keine Konstante der Aktionslänge, sondern gilt zusammen mit der Alarmverzögerung.

### 5. Kostenannahmen

Gewinn des Detektors gegen den stärksten Gegner, **nicht neu abgestimmt** (Parameter der Zelle für τ = 0,95 und L = 3), 300 Messreihen, Sprung +30 %, Rauschen 8 % bzw. 14 %; Orakel in Prozent des Gegners:

<!-- BEGIN:kosten -->
| Rauschen (Gegner) | τ (p) | L | Gewinn Detektor, Testjahr | Gewinn Detektor, Fenster | Gewinn Orakel, Testjahr | Gewinn Orakel, Fenster |
|---|---|---|---|---|---|---|
| 8 % (trigg) | 0,80 (4) | 0 | +3,4 ± 0,2 | +1,5 ± 0,6 | +6,4 | +14,2 |
| 8 % (trigg) | 0,80 (4) | 3 | +9,1 ± 0,4 | +2,4 ± 1,1 | +15,6 | +25,4 |
| 8 % (trigg) | 0,80 (4) | 7 | +12,0 ± 0,5 | +1,1 ± 1,2 | +19,8 | +24,8 |
| 8 % (trigg) | 0,95 (19) | 0 | +2,5 ± 0,3 | −1,8 ± 0,9 | +8,2 | +20,1 |
| 8 % (trigg) | 0,95 (19) | 3 | +5,7 ± 0,7 | −3,0 ± 1,5 | +18,5 | +29,4 |
| 8 % (trigg) | 0,95 (19) | 7 | +6,3 ± 0,8 | −6,0 ± 1,5 | +20,6 | +22,4 |
| 8 % (trigg) | 0,99 (99) | 0 | −1,1 ± 0,6 | −10,3 ± 1,7 | +11,4 | +27,5 |
| 8 % (trigg) | 0,99 (99) | 3 | −3,3 ± 1,2 | −13,1 ± 1,9 | +22,6 | +30,1 |
| 8 % (trigg) | 0,99 (99) | 7 | −7,1 ± 1,4 | −16,8 ± 1,8 | +18,1 | +16,9 |
| 14 % (ewma) | 0,80 (4) | 0 | +1,2 ± 0,1 | +2,6 ± 0,3 | +4,0 | +13,2 |
| 14 % (ewma) | 0,80 (4) | 3 | +3,5 ± 0,2 | +6,3 ± 0,6 | +11,4 | +29,7 |
| 14 % (ewma) | 0,80 (4) | 7 | +5,4 ± 0,3 | +8,5 ± 0,7 | +16,2 | +34,9 |
| 14 % (ewma) | 0,95 (19) | 0 | +1,3 ± 0,1 | +2,9 ± 0,5 | +5,5 | +19,0 |
| 14 % (ewma) | 0,95 (19) | 3 | +2,4 ± 0,3 | +2,7 ± 0,9 | +15,5 | +37,9 |
| 14 % (ewma) | 0,95 (19) | 7 | +3,5 ± 0,5 | +3,4 ± 1,0 | +21,6 | +39,7 |
| 14 % (ewma) | 0,99 (99) | 0 | +0,8 ± 0,3 | +1,5 ± 1,0 | +7,7 | +26,9 |
| 14 % (ewma) | 0,99 (99) | 3 | −1,0 ± 0,6 | −4,9 ± 1,3 | +22,7 | +44,8 |
| 14 % (ewma) | 0,99 (99) | 7 | −2,4 ± 0,9 | −5,2 ± 1,3 | +28,5 | +40,8 |
<!-- END:kosten -->

Der Vorsprung wächst bei billigem Rückstand (τ = 0,8) und langer Wiederbeschaffungszeit (Rauschen 14 %, τ = 0,8, L = 7: +5,4 ± 0,3 %) und **wird bei τ = 0,99 negativ oder verschwindet**: Mehr als zwei Standardfehler unter null liegt er bei Rauschen 14 % mit L = 7 (−2,4 ± 0,9 %) und bei Rauschen 8 % mit L = 3 (−3,3 ± 1,2 %) und L = 7 (−7,1 ± 1,4 %); innerhalb von zwei Standardfehlern liegen Rauschen 14 % mit L = 3 (−1,0 ± 0,6 %) und Rauschen 8 % mit L = 0 (−1,1 ± 0,6 %).

### Befunde und Korrekturen gegenüber der Vormessung

Die Vormessung (`aenderungspunkt-planung/messreihe_niveausprung`, Runde 2, Zufalls-Seeds 0 bis 599 bzw. 0 bis 299) war die Erwartung; im Bau wurde mit dem Zufallsgenerator der Demo und anderen Zufalls-Seeds neu gemessen. Zwei Eingriffe im Aufbau, nicht nur im Zufall: (a) der Detektor ist hier der CUSUM auf den Ein-Schritt-Fehlern der **langsamen Nachführung**
(α = 0,06), die Vormessung der Bestätigung nutzte einen CUSUM mit **fester Referenz** (α = 0); (b) die Glättung ist ein Nachbau mit festen Parametern (α = 0,10, γ = 0,02), nicht der Original-Holt-Winters der Exponential-Smoothing-Demo.

<!-- BEGIN:vergleich -->
| Größe | Vormessung (Seeds 0 bis 599 bzw. 0 bis 299) | Bau (diese Demo, andere Seeds) |
|---|---|---|
| Gewinn gegen stärksten Gegner, MASE alle (Rauschen 14 %, +30 %) | +1,3 % | +1,4 % |
| Gewinn gegen stärksten Gegner, Kosten Testjahr | +2,4 % | +2,5 % |
| Gewinn gegen Trigg-Leach, Kosten im Fenster | −17,0 % | −18,2 % |
| Gewinn gegen die feste Glättung, MASE alle | +13,5 % (Original-Holt-Winters) | +12,6 % (Nachbau, α = 0,10) |
| Gewinn gegen die feste Glättung, Kosten Testjahr | +21,6 % (Original) | +21,4 % (Nachbau) |
| Orakel gegen stärksten Gegner, Kosten Testjahr | +16 % | +16,2 % |
| Orakel gegen stärksten Gegner, Kosten Fenster | +37,4 % | +38,1 % |
| Karte Rauschen 14 %, +50 %: Kosten Testjahr | +5,7 % | +9,4 % |
| Karte Rauschen 24 %, +50 %: Kosten Testjahr | +1,1 % | +0,0 % |
| Karte Rauschen 14 %, +30 %: MASE alle | +1,5 % | +1,2 % |
| Aktionswoche +50 % ohne Bestätigung gemeldet | 100,0 % | 100,0 % |
| Aktionswoche +50 % mit Bestätigung nach 10 Tagen | 12,2 % | 15,0 % |
| Stufe +28 % mit Bestätigung nach 10 Tagen bestätigt | 99,5 % | 96,5 % |
| Zusatzverzögerung der Bestätigung (c = 10, Stufe +28 %) | +10,7 Tage | +10,2 Tage |
| Stufe +14 % ohne Bestätigung gemeldet | 97,2 % | 43,8 % |
| Stufe +14 % mit Bestätigung nach 10 Tagen | 45,5 % | 21,0 % |
| Rampe auf +28 % ohne Bestätigung gemeldet | 100,0 % | 44,3 % |
| Aktionswoche +25 % ohne Bestätigung gemeldet | 74,2 % | 62,8 % |
| Ausreißer +100 % ohne Bestätigung gemeldet | 26,8 % | 28,2 % |
<!-- END:vergleich -->

- **Bestätigt:** Das Zurücksetzen gewinnt gegen den stärksten abgestimmten Gegner im Jahr wenig (+1,4 % MASE, +2,5 % Kosten gegen die EWMA), verliert im Fenster gegen Trigg-Leach, und die Aktionswoche wird ohne Bestätigung zu 100,0 % als Sprung gemeldet und mit c = 10 nur zu 15,0 % (Vormessung 12,2 %), bei etwa 10 Tagen Mehrverzögerung.
- **Verschoben, weil der Detektor ein anderer ist:** Kleine und langsame Änderungen verschluckt die langsame Nachführung zwischen den Meldungen (Messung mit α = 0 bis 0,06, gleiche Reihen, ohne Bestätigung):

<!-- BEGIN:alpha -->
| Detektor (ohne Bestätigung) | Meldungen je Jahr ohne Form | Stufe +28 % | Stufe +14 % | Rampe auf +28 % |
|---|---|---|---|---|
| α = 0 (feste Referenz), h = 6 | 2,34 | 100,0 % (4,3 d) | 97,0 % (13,6 d) | 99,3 % (29,2 d) |
| α = 0,03, h = 6 | 0,65 | 99,8 % (4,7 d) | 76,5 % (13,8 d) | 90,5 % (36,9 d) |
| α = 0,06 (Standard der App), h = 6 | 0,30 | 98,5 % (5,5 d) | 43,8 % (11,1 d) | 44,3 % (38,5 d) |
| α = 0, h = 10 (höchstens so viele Meldungen je Jahr wie α = 0,06 bei h = 6) | 0,25 | 99,8 % (7,4 d) | 95,2 % (21,8 d) | 99,5 % (38,4 d) |
<!-- END:alpha -->

  Mit α = 0 (feste Referenz) erreicht dieser Detektor bei h = 6 die Anteile der Vormessung (Stufe +14 %: 97,0 % gegen 97,2 %; Rampe: 99,3 % gegen 100,0 %), aber nicht deren Meldungsrate: Bei gleicher Schwelle sind die Meldungen je Jahr sehr verschieden (2,34 bei α = 0, 0,65 bei α = 0,03, 0,30 bei α = 0,06; Vormessung 0,693 bei h = 7), ein Teil des Erkennungsunterschieds ist also der Preis in Meldungen.
  Bei gleicher Meldungsrate bleibt der Abstand: α = 0 mit h = 10 meldet 0,25 Mal je Jahr (höchstens so oft wie α = 0,06 mit h = 6) und erkennt +14 % in 95,2 % der Reihen gegen 43,8 %. Die Erklärung „die langsame Nachführung absorbiert die Änderung“ gilt damit auch bei gleicher Meldungsrate, für diese beiden Formen; für die übrigen Verschiebungen ist sie nicht getrennt gemessen.
  Für die Aussage zur Aktionswoche ändert das nichts, für „Stufe +14 % wird zu 97 % gemeldet“ schon: Das gilt für die feste Referenz, nicht für die Voreinstellung der App.
- **Verschoben:** Karte Rauschen 14 %, +50 %: Kosten +9,4 ± 0,6 % statt +5,7 % (ein Abstimmungsausreißer: bei der Wiederholung mit anderen Abstimm- und Messreihen +6,2 ± 0,6 %); Karte Rauschen 14 %, +30 %: MASE +1,2 % statt +1,5 %; die Kosten bei τ = 0,99 und L = 0 sind bei Rauschen 8 % negativ (−1,1 ± 0,6 %, innerhalb von zwei Standardfehlern), die Vormessung nannte „+0,8 % bis +1,3 %“ (für Rauschen 14 % bleibt es positiv: +0,8 ± 0,3 %).
- **Nicht bestätigt, nicht widerlegt:** Karte Rauschen 24 %, +50 %: In den Kosten des Testjahres +0,0 ± 0,5 % statt +1,1 % (Wiederholung: −0,3 ± 0,5 %); die MASE +0,3 ± 0,1 % statt +0,7 % (Wiederholung: +0,7 ± 0,1 %). Ein kleiner Gewinn von etwa 1 % in den Kosten ist mit dieser Streuung weder belegt noch ausgeschlossen.
- **Neu im Bau gemessen:** der Anteil der Orakel-Reserve, den der Detektor holt (16 %), das um die Verzögerung verspätete Orakel, die Aufteilung des Vorsprungs gegen die Glättung (Regression gegen Zurücksetzen).
- **Nicht neu gemessen:** rückblickende Segmentierung (Änderungspunkt-Verfahren) und Kalenderfehler im Modell (falsche Feiertage); der Original-Holt-Winters der Exponential-Smoothing-Demo (die Vormessung nannte +13,5 % MASE und +21,6 % Kosten gegen ihn, der Nachbau gibt +12,6 % und +21,4 %).

### Zahl → Test

| Aussage | Test |
|---|---|
| alle Tabellen dieses Abschnitts (Kennlinie, Standardszenario, Karte, Folgen von Sprüngen, Bestätigung bei Rauschen 14 % und 4 %, Kostenannahmen, Vergleich mit der Vormessung, Nachführung) stimmen mit `data/*.json` überein | `tests/test_claims.py::test_tabellen_der_readme_sind_erzeugt` (exakt) |
| jede Zahl der Sätze oben steht in `data/*.json` (exakt, als formatierte Zahl) | `tests/test_claims.py::test_saetze_der_readme` (exakt gegen die gespeicherte Messung) |
| die gespeicherten Messungen sind mit dem Code reproduzierbar (Zelle Rauschen 14 %, +30 %: 300 Messreihen; Aktionswoche und Stufe: 600 Reihen) | `tests/test_claims.py::test_messungen_sind_reproduzierbar` (Band 0,05 Prozentpunkte) |
| Zählungen über die Karte und die Wiederholung (10 und 11 von 16, 6 von 16, 8 von 32 Zellen ohne Meldungen, 8 von 32 Zellen mit Änderung über zwei kombinierte Standardfehler) | `tests/test_claims.py::test_zaehlungen_ueber_die_karte`, `test_wiederholung_der_karte` |
| Folgen von Sprüngen: gleiches Fenster, Anteile hinter dem Reihenende, zweiter Sprung gemeldet | `tests/test_claims.py::test_folgen_von_spruengen` |
| Rauschen 24 %: Sprung mit Standardwerten gemeldet | `tests/test_claims.py::test_saetze_der_readme` (Tabelle `data/laute_reihe.json`) |
| Voreinstellungen des Detektors und der Gegner sind die Abstimmung der Zelle Rauschen 14 %, +30 % | `tests/test_claims.py::test_voreinstellungen_sind_die_abstimmung_der_zelle` |
| Presets: jede Verdikt-Zeile | `tests/test_claims.py::test_verdikte_der_presets` |

## Ehrliche Grenzen

- **Das Regressionsmodell ist richtig spezifiziert:** Die Reihen sind erzeugt, wie die Regression sie anpasst (Wochentag, zwei Fourier-Paare, Trend, bekannte Feiertage und Aktionen). In echten Reihen ist das die schwächste Annahme; Fourier-Zahl, Trendmodell und Kalenderfehler wurden nicht variiert (nur die Bereinigungsstufen aus Befund 1).
- **Kosten nur für die gemessenen Annahmen:** Die Karte gilt für τ = 0,95, L = 3 und die empirische Regel; die Kostenannahmen sind nur an der Zelle Rauschen 8 % und 14 %, +30 % gemessen (Befund 5). Das Lager kostet 1 je Stück und Tag, der Rückstand p; es gibt keine Preise, Bestellkosten oder Mindestmengen.
- **Bestätigung nur für die einwöchige Aktion:** θ und c wurden nicht gemeinsam mit der Schwelle h abgestimmt; eine längere Aktion würde bestätigt (nicht gemessen); die Wartezeit, die trennt, hängt an der Verzögerung des Alarms: c = 10 verwirft bei Rauschen 14 % 85,0 % der Aktionswochen +50 %, bei Rauschen 4 % (Alarm am ersten Tag) keine; gemessen ist bei Rauschen 4 % nur die Aktion +50 %, nicht +25 % oder der Ausreißer; die Bestätigung gilt für den Detektor der Messung, nicht für jede Variante (Tabelle der Nachführung).
- **Abstimmung je Zelle:** Gegner und Detektor sind je Zelle auf den Sprung dieser Zelle (halb mit, halb ohne Sprung) abgestimmt, also für beide optimistisch; die Karte stimmt Folgen von Sprüngen nicht neu ab. Die Abstimmung (Auswahl aus einem Raster auf 100 Reihen) ist selbst unsicher: Der Standardfehler der Karte gilt für die Messreihen bei gegebener Abstimmung, eine neue Abstimmung verschiebt einzelne Zellen um mehrere Prozentpunkte (Befund 3). Das Orakel nimmt in allen 64 Abstimmungen den kleinsten Rasterwert α = 0,03; ein noch kleineres α spart etwas mehr (Prüfung im Standardszenario: α = 0,01 gäbe +16,9 statt +16,2 %, nicht in den Tests). Die Regler der App sind **nicht** abgestimmt.
- **Der stärkste Gegner gilt für die gemessene Regel:** „stärkster Gegner“ heißt die Familie mit dem besten abgestimmten Zielwert unter EWMA, Mittel der letzten k Tage, Trigg-Leach und fester Glättung; ein anderer Nachführer, eine andere Abstimmung oder ein anderes Maß kann ihn schlagen (im Sprungfenster tut es Trigg-Leach).
- **Der Vorsprung gegen die Glättung** gilt für das Testjahr und die MASE; im Sprungfenster verliert das Zurücksetzen gegen sie (Kosten −2,8 ± 1,5 %), und er gilt für den Nachbau (feste Parameter), nicht für den Original-Holt-Winters der Linie.
- **Kleine Stichproben der Regler:** Die App wertet 12 bis 60 Reihen aus; Gewinne unter etwa zwei Standardfehlern sind dort Rauschen. Die Karte und die Tabellen oben beruhen auf 300 bis 600 Reihen.
- **Detektor praktisch aus:** In Zellen mit unter 0,05 Meldungen je Reihe ist der Gewinn kein Beitrag des Erkennens (siehe Tabelle der Meldungen).
- **Ein Sprung je Reihe, höchstens zwei Folgen:** mehr als zwei Sprünge, bayessche Änderungspunkte, mehrere Reihen gleichzeitig und echte Daten sind nicht gemessen.
- **Reihenende bei Folgen von Sprüngen:** Der Sprungtag liegt in den Tagen 760 bis 1034; in 7,7 % der Messreihen liegt der Rücksprung, in 15,7 % der zweite Sprung hinter dem Reihenende, und das Fenster von 180 Tagen ist in 37,7 % der Reihen verkürzt (Tabelle der Folgen von Sprüngen). Die Zeilen „zurück“ und „zweiter Sprung“ mitteln über Reihen, in denen die Folge teilweise fehlt.

## Verifikation

- **Kern gegen Handrechnung** (`tests/test_algorithm.py`): EWMA, Mittel, Trigg-Leach, CUSUM mit und ohne Bestätigung (Alarm, Zurücksetzen, Richtung, Verwerfen einer kurzen Aktion), Orakel, Regression ohne Rauschen, Prognose aus Offsets, MASE, Glättung, Sprungformen.
- **Orakel** (unabhängiger Rechenweg): `tests/test_oracle_vehikel.py` (Erzeuger bit-gleich zu einer Kopie von `nf_scenario`; MASE der Stücke 1 bis 3 am Preset der Linie: saisonal naiv 1,36, Wochenmittel 1,08, Erwartungswert 0,90 laut README der Linie, J = 809), `tests/test_oracle_niveau.py` (alle Nachführer
  gegen eine skalare Schleife je Reihe mit reinen Pythonzahlen, inklusive Bestätigung; Regression gegen die Normalgleichungen; Zweig-Test: jede Variante weicht von der Nachbarvariante ab), `tests/test_oracle_bestand.py` (Fortschreibung gegen die geschlossene Formel bei konstantem Bestand und gegen eine Tagesschleife; Regeln gegen Quantil und Rang von Hand; Kosten gegen die Definition).
- **Verdikte** als reine Funktionen (`tests/test_verdicts.py`: Schwellen und jede Zahl im Text an handgesetzten Werten), **Randfälle** (`tests/test_edge_cases.py`: eine und zwei Reihen, kein Alarm, jede Form, alle Regler an den Rändern), **keine toten Regler** (`tests/test_regler_wirken.py`: jeder Regler ändert das Ergebnis dort, wo er hingehört, und sonst nichts).
- **Messungen** (`tests/test_claims.py`): Tabellen und Sätze dieser README gegen `data/*.json`, Stichproben der Messläufe neu gerechnet, Zählungen über die Karte.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt es auf), Footer, Laufzeit-Importe.
- **Browser:** Vorschau mit Streamlit im Browser-Fenster geprüft (Schnellstart-Knöpfe, aktives Preset, Karte mit Markierung der Einstellung, Formeln ohne KaTeX-Fehler, keine Fehlermeldungen in der Konsole, helles und dunkles Design); nicht geprüft: schmales Telefonfenster und die Kennlinien- und Bestätigungsdiagramme im Telefonformat.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `data/*.json` | gespeicherte Messungen (Karte, Standardszenario, Bestätigung, Kennlinie, Nachführung, Kostenannahmen); Tabellen und Sätze dieser README werden dagegen geprüft |
| `nvs_algorithm.py` | Kern: Regression, Nachführer (EWMA, Mittel, Trigg-Leach, CUSUM mit Bestätigung, Orakel), Prognosen, MASE, feste Glättung |
| `nvs_constants.py` | Vehikel, Regler-Grenzen, Sprungformen, Kostenannahmen, Farben, Presets |
| `nvs_evaluation.py` | Auswertung aller Verfahren auf denselben Reihen, gepaarte Gewinne, Kennlinie, Bestätigungsversuch, Verdikte |
| `nvs_inventory.py` | Bestandsfortschreibung und Regeln des Sicherheitsbestands |
| `nvs_karte.py` | Abstimmung und Messung je Zelle der Karte, Laden der Karte |
| `nvs_layout.py` | Fester Kopf (Stylesheet) |
| `nvs_messung.py` | Messläufe für die README (Standardszenario, Bestätigung, Kennlinie, Nachführung, Kostenannahmen) |
| `nvs_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `nvs_scenario.py` | Vehikel „Tagesaufträge“ mit Sprungformen und Autokorrelation |
| `nvs_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tools/messlauf.py`, `tools/readme_tabellen.py` | Messläufe neu erzeugen; README-Tabellen aus den Messungen erzeugen |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handrechnung, Orakel, Verdikte, Randfälle, Regler, Messungen, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht enthalten

- **Sequentielle Tests** (`sequential-test-demo`): Dort geht es darum, ob man in einem laufenden Experiment (Regel A gegen Regel B) zwischendurch hinsehen und bei „signifikant“ stoppen darf. Hier sind Existenz und Zeitpunkt des Sprungs in einer einzelnen Reihe unbekannt, das Maß ist Verzögerung gegen Meldungen je Jahr und der Bestandsnutzen.
- **Anomalie-Linie** (`isolation-forest`, `ocsvm`, `lof` und andere): Sie sucht einzelne ungewöhnliche Punkte; hier ist die gesuchte Struktur eine dauerhafte Verschiebung. Die Berührung zeigt der Bestätigungsversuch: Ein Ausreißer von +100 % wird ohne Bestätigung in 28,2 % der Reihen als Sprung gemeldet; der Unterschied (Dauer, Gegenmeldung) ist, was die Anomalie-Linie nicht prüft.
- **Rückblickende Änderungspunkt-Verfahren** (binäre Segmentierung, PELT), **Kalenderfehler im Modell**, **bayessche Änderungspunkte**, **mehr als zwei Sprünge**, **mehrere Reihen gleichzeitig**, **echte Daten**: nicht gebaut und nicht gemessen.
- **Preise, Bestellkosten, Mindestmengen** im Bestandsmodell.
- **PDF-Export:** nicht vorhanden.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die gespeicherten Messungen lassen sich neu erzeugen (`python tools/messlauf.py alles`, einige Minuten) und die Tabellen dieser README mit `python tools/readme_tabellen.py --write` aktualisieren.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Zeitreihen-Prognose: von Naiv bis Vortraining](https://sebastianhanisch.net/konzepte-zeitreihen-prognose.html).
