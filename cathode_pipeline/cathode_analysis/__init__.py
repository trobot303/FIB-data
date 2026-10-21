"""3D microstructure analysis of LCO/LLZTO composite cathodes.

Reproduces the geometric analysis of Clausnitzer et al., Batteries & Supercaps
2023, e202300167: volume fractions, connectivity, active interface area,
tortuosity, grain segmentation and through-thickness profiles.

Phase label codes used throughout the package (uint8 volumes):

    0 = pore
    1 = LCO   (active material)
    2 = LLZTO (solid electrolyte)

Codes 3-5 are used by the connectivity stage for the *isolated* part of each
phase (3 = isolated pore, 4 = isolated LCO, 5 = isolated LLZTO).
"""

__version__ = "0.1.0"

PHASES = ("pore", "lco", "llzto")
PHASE_CODE = {"pore": 0, "lco": 1, "llzto": 2}
ISOLATED_CODE = {"pore": 3, "lco": 4, "llzto": 5}
