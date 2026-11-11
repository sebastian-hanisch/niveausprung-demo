"""Gemeinsame Fixtures: kleine Reihen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def ref_ten():
    """Zehn Referenztage mit abwechselnd +1 und -1: Mittel 0, Standardabweichung (ddof=1) = sqrt(10 / 9). Danach (Tage 10 bis 29) ist das Residuum 3 (Aufwärts-) bzw. -3 (Abwärtssprung)."""
    def make(level, n=30):
        r = np.zeros((1, n))
        r[0, :10] = [1, -1] * 5
        r[0, 10:] = level
        return r
    return make
