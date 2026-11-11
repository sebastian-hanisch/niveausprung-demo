"""Messläufe der README: schreibt `data/*.json` (Karte, Standardszenario, Bestätigung, Kennlinie, Kostenannahmen). Aufruf im Demo-Ordner:

    python tools/messlauf.py karte | head | confirm | kennlinie | ruhig | alpha | zweiter | laut | ar | wiederholung | kosten | alles

`karte` braucht einige Minuten (acht Prozesse); die übrigen Läufe lesen die Abstimmung aus `data/karte.json`. Zufalls-Seeds: siehe nvs_constants.py (Messreihen 10000 bis 10299,
Abstimmreihen 12000 bis 12099, Bestätigung 14000 bis 14599, Standardszenario 16000 bis 16599)."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import nvs_constants as C  # noqa: E402
import nvs_karte as K  # noqa: E402
import nvs_messung as M  # noqa: E402

DATA = ROOT / "data"


def _write(name, obj):
    DATA.mkdir(exist_ok=True)
    (DATA / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


def main(what):
    t0 = time.time()
    if what in ("karte", "alles"):
        K.build()
        print("karte", round(time.time() - t0), "s", flush=True)
    karte = K.load()
    best = karte["zellen"]["14|30|0.0"]["best"]
    if what in ("head", "alles"):
        _write("head.json", M.head(best))
        print("head", round(time.time() - t0), "s", flush=True)
    det = best["kosten"]["detektor_params"]
    if what in ("confirm", "alles"):
        _write("confirm.json", {"detektor": det, "ergebnis": M.confirm(det)})
        print("confirm", round(time.time() - t0), "s", flush=True)
    if what in ("kennlinie", "alles"):
        _write("kennlinie.json", {"detektor": det, "ergebnis": M.kennlinie_modelle(det)})
        print("kennlinie", round(time.time() - t0), "s", flush=True)
    if what in ("ruhig", "alles"):
        det4 = karte["zellen"]["4|30|0.0"]["best"]["kosten"]["detektor_params"]
        _write("confirm_ruhig.json", {"detektor": det4, "ergebnis": M.confirm_ruhig(det4)})
        print("ruhig", round(time.time() - t0), "s", flush=True)
    if what in ("alpha", "alles"):
        _write("alpha.json", {"detektor": det, "ergebnis": M.nachfuehrung(det)})
        print("alpha", round(time.time() - t0), "s", flush=True)
    if what in ("zweiter", "alles"):
        _write("zweiter_sprung.json", {"8": M.zweiter_sprung(karte["zellen"]["8|30|0.0"]["best"]["kosten"]["detektor_params"], 8), "14": M.zweiter_sprung(det, 14)})
        print("zweiter", round(time.time() - t0), "s", flush=True)
    if what in ("laut", "alles"):
        z = karte["zellen"]
        _write("laute_reihe.json", M.laute_reihe({"30": z["24|30|0.0"]["best"]["kosten"]["detektor_params"], "50": z["24|50|0.0"]["best"]["kosten"]["detektor_params"]}))
        print("laut", round(time.time() - t0), "s", flush=True)
    if what in ("wiederholung", "alles"):
        K.build(DATA / "karte_wiederholung.json", tune_seeds=C.TUNE_SEEDS_B, measure_seeds=C.MEASURE_SEEDS_B, multi=False)
        print("wiederholung", round(time.time() - t0), "s", flush=True)
    if what in ("ar", "alles"):
        _write("autokorrelation.json", {"detektor": det, "ergebnis": M.autokorrelation_fehlmeldungen(det)})
        print("ar", round(time.time() - t0), "s", flush=True)
    if what in ("kosten", "alles"):
        _write("kosten.json", {"8": M.kostenannahmen(karte["zellen"]["8|30|0.0"]["best"], 8), "14": M.kostenannahmen(best, 14)})
        print("kosten", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "alles")
