"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nvs_constants as C
from nvs_evaluation import fmt
from nvs_scenario import T, T0


def _sign(x, digits=1):
    """Zahl mit Vorzeichen und Dezimalkomma; auf null gerundet: "+0,0"."""
    return ("+" if float(f"{x:.{digits}f}") >= 0 else "-") + fmt(abs(x), digits)


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_series(a, opponent: str, index=0):
    """Eine Reihe (die erste der Auswertung): oben die Residuen der Regression (Log) mit den Niveaus von Gegner, Detektor und Orakel, unten die CUSUM-Statistik mit Schwelle."""
    s = a.settings
    day0 = T0 - 60
    d = np.arange(day0, T)
    R = a.prepared.R[index, day0:]
    J = int(a.prepared.batch.J[index])
    jumps = a.prepared.batch.jumps if a.prepared.batch.has_jump[index] else ()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=d, y=R, mode="markers", marker=dict(size=4, color=C.COLOR_RAW), name="Residuum der Regression", hovertemplate="Tag %{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    for name, label, color, dash in ((opponent, f"Niveau: {C.METHOD_NAMES[opponent].replace('Regression + ', '')}", C.COLOR_GEGNER, "solid"), ("detector", "Niveau: CUSUM-Zurücksetzen", C.COLOR_DETEKTOR, "solid"),
                                     ("oracle", "Niveau: Orakel", C.COLOR_ORAKEL, "dot")):
        fig.add_trace(go.Scatter(x=d, y=a.tracks[name]["O"][index, day0:], mode="lines", line=dict(color=color, width=2.5, dash=dash), name=label,
                                 hovertemplate="Tag %{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    for off, _ in jumps[:1] + jumps[-1:] if len(jumps) > 1 else jumps:
        fig.add_vline(x=J + off, line=dict(color=C.COLOR_JUMP, dash="dash", width=1.5), row="all", col=1)
    meld = np.where(a.tracks["detector"]["meldung"][index, day0:])[0]
    if len(meld):
        fig.add_trace(go.Scatter(x=d[meld], y=a.tracks["detector"]["O"][index, day0:][meld], mode="markers", marker=dict(size=11, color=C.COLOR_ALARM, symbol="diamond"), name="Meldung (Niveau zurückgesetzt)"), row=1, col=1)
    st_ = a.tracks["detector"]["stat"][index, day0:]
    fig.add_trace(go.Scatter(x=d, y=st_, mode="lines", line=dict(color=C.COLOR_DETEKTOR, width=1.5), name="CUSUM-Statistik", showlegend=False, hovertemplate="Tag %{x}: %{y:.1f}<extra></extra>"), row=2, col=1)
    for sign in (1, -1):
        fig.add_hline(y=sign * s.h, line=dict(color=C.COLOR_ALARM, dash="dot", width=1), row=2, col=1)
    alarm = np.where(a.tracks["detector"]["alarm"][index, day0:])[0]
    if len(alarm):
        fig.add_trace(go.Scatter(x=d[alarm], y=st_[alarm], mode="markers", marker=dict(size=8, color=C.COLOR_ALARM), name="Alarm", showlegend=False), row=2, col=1)
    fig.update_yaxes(title_text="Log-Abweichung", row=1, col=1)
    fig.update_yaxes(title_text="CUSUM (σ)", row=2, col=1)
    fig.update_xaxes(title_text="Tag (Testjahr ab 730; gestrichelt: Sprung)", row=2, col=1)
    return _base(fig, 520)


def build_comparison(summ_row: dict, measure: str, names):
    """Mittel und Standardfehler je Verfahren für ein Maß; Detektor und Orakel farbig."""
    means = summ_row["means"]
    names = [m for m in names if m in means]
    colors = [C.COLOR_DETEKTOR if m == "detector" else C.COLOR_ORAKEL if m.startswith("oracle") else C.COLOR_GEGNER if m == summ_row["opponent"] else C.COLOR_RAW for m in names]
    fig = go.Figure(go.Bar(x=[C.METHOD_NAMES[m] for m in names], y=[means[m][0] for m in names], error_y=dict(type="data", array=[means[m][1] for m in names], visible=True),
                           marker_color=colors, hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="MASE" if measure.startswith("mase") else "Kosten (Tagesbedarfe je Tag)")
    fig.update_xaxes(tickangle=-30)
    return _base(fig, 420)


def build_kennlinie(rows, s_h=None):
    """Kennlinie: Fehlmeldungen je Jahr (x) gegen mittlere Verzögerung in Tagen (y); Beschriftung = Schwelle h."""
    pts = [r for r in rows if r["verzoegerung"] is not None]
    fig = go.Figure(go.Scatter(x=[r["fehlmeldungen"] for r in pts], y=[r["verzoegerung"] for r in pts], mode="lines+markers+text", text=[f"h = {r['h']:g}" for r in pts], textposition="top right",
                               line=dict(color=C.COLOR_DETEKTOR, width=2), marker=dict(size=8), hovertemplate="Fehlmeldungen/Jahr %{x:.2f}, Verzögerung %{y:.1f} Tage<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color=C.COLOR_ALARM, dash="dot"), annotation_text="eine Fehlmeldung je Jahr", annotation_position="top")
    fig.update_xaxes(title_text="Fehlmeldungen je Jahr (Reihen ohne Sprung)", rangemode="tozero")
    fig.update_yaxes(title_text="Verzögerung bis zur Meldung (Tage)", rangemode="tozero")
    return _base(fig, 380)


def build_karte(zellen: dict, measure: str, objective: str, ar: float, gain_key="gewinn_detektor", current=None):
    """Heatmap Rauschen (Zeilen) x Sprung (Spalten): Gewinn in Prozent (positiv: Erkennen lohnt). current = (Rauschen, Sprung) der Einstellungen, wird markiert, wenn es ein Gitterpunkt ist."""
    noises, shifts = C.KARTE_NOISE, C.KARTE_SHIFT
    z = [[zellen[f"{n}|{s}|{ar}"]["messung"][objective][gain_key][measure][0] for s in shifts] for n in noises]
    se = [[zellen[f"{n}|{s}|{ar}"]["messung"][objective][gain_key][measure][1] for s in shifts] for n in noises]
    text = [[f"{_sign(z[i][j])} ± {fmt(se[i][j])}" for j in range(len(shifts))] for i in range(len(noises))]
    lim = max(2.0, max(abs(v) for row in z for v in row))
    fig = go.Figure(go.Heatmap(z=z, x=[f"+{s} %" for s in shifts], y=[f"{n} %" for n in noises], text=text, texttemplate="%{text}", colorscale="RdBu", zmid=0, zmin=-lim, zmax=lim,
                               colorbar=dict(title="Gewinn %"), hovertemplate="Rauschen %{y}, Sprung %{x}: %{text} %<extra></extra>"))
    if current and current[0] in noises and current[1] in shifts:
        fig.add_trace(go.Scatter(x=[f"+{current[1]} %"], y=[f"{current[0]} %"], mode="markers", marker=dict(size=26, symbol="square-open", color="black", line=dict(width=3)),
                                 name="aktuelle Einstellung", hoverinfo="skip"))
    fig.update_xaxes(title_text="Sprunggröße", type="category", categoryorder="array", categoryarray=[f"+{s} %" for s in shifts])       # "14 %" liest Plotly sonst als Zahl
    fig.update_yaxes(title_text="Rauschen", type="category", categoryorder="array", categoryarray=[f"{n} %" for n in noises])
    return _base(fig, 330)


def build_confirm(rows, c):
    """Anteil der Reihen mit Meldung je Form, ohne und mit Bestätigung (Wartezeit c)."""
    forms = [r for r in rows if r["form"] != "none"]
    names = [C.FORM_NAMES.get(r["form"], r["form"]).split(":")[0].split(",")[0] for r in forms]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["ohne"]["anteil"] * 100 for r in forms], name="ohne Bestätigung", marker_color=C.COLOR_GEGNER))
    fig.add_trace(go.Bar(x=names, y=[r["mit"]["anteil"] * 100 for r in forms], name=f"mit Bestätigung nach {c} Tagen", marker_color=C.COLOR_DETEKTOR))
    fig.update_yaxes(title_text="Reihen mit Meldung (%)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _base(fig, 340)
