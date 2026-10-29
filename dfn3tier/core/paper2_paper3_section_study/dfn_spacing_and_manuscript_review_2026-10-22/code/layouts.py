"""Scanline layouts for the spacing study (predefined before any run). Scanlines are vertical 0.1 m strips placed on a GLOBAL
grid in face coordinates (x_face = window x0 + local x), positions offset + k*spacing (k >= 0), then clipped into each 20 m window.
S05ref is the Plan D L0 layout (local x = 5, 10, 15 m in every window; lines on window seams omitted), kept as the reproducible
5 m reference. Offsets are chosen so that no line falls on a window seam (multiple of 20 m). DISPLAY = offset used in figures."""
import numpy as np
W = 20.0; AP = 0.1
LAYOUTS = {'S05ref': None, 'S05o2.5': (5, 2.5), 'S10o2.5': (10, 2.5), 'S10o5': (10, 5), 'S10o7.5': (10, 7.5),
           'S15o2.5': (15, 2.5), 'S15o7.5': (15, 7.5), 'S15o12.5': (15, 12.5), 'S20o5': (20, 5), 'S20o10': (20, 10), 'S20o15': (20, 15)}
SPACING = {k: (5 if v is None else v[0]) for k, v in LAYOUTS.items()}
DISPLAY = {5: 'S05ref', 10: 'S10o5', 15: 'S15o7.5', 20: 'S20o10'}


def local_lines(spec, x0):
    """Local x (m) of the strip centres inside the window starting at face position x0."""
    if LAYOUTS[spec] is None: return [5.0, 10.0, 15.0]
    sp, off = LAYOUTS[spec]; k0 = int(np.ceil((x0 - off) / sp)); out = []
    for k in range(max(k0, 0), k0 + int(W / sp) + 2):
        p = off + k * sp - x0
        if AP / 2 < p < W - AP / 2: out.append(round(float(p), 6))
        assert not (abs(p) < AP / 2 or abs(p - W) < AP / 2), 'line on a window seam'
    return out


def strips(spec, x0):
    """Strips in window units (as run_revision.strips: (lo, hi) / 20)."""
    return [((p - AP / 2) / W, (p + AP / 2) / W) for p in local_lines(spec, x0)]
