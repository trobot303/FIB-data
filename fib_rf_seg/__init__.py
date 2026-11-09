"""Random-forest pixel classification for FIB-SEM cathode slices.

Phases (label codes, matching the cathode_analysis convention):
    0 = unlabelled (only used in training scribbles)
    1 = pore   (black)
    2 = LCO    (grey)
    3 = LLZTO  (almost white)
    4 = crack  (thin dark lines; optional, cycled samples)
"""

PHASES = {1: "pore", 2: "LCO", 3: "LLZTO", 4: "crack"}
PHASE_COLORS = {1: (0.10, 0.10, 0.85), 2: (0.10, 0.75, 0.10), 3: (0.95, 0.20, 0.20), 4: (1.00, 0.85, 0.10)}
