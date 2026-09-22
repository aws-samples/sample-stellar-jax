"""Loader for the RGB MESA reference tracks (F5 / issue #159).

Identical-physics MESA tracks evolved from pre-main-sequence to the tip of the
red-giant branch (pre-helium-flash) for 1.0, 1.2, 1.5, 2.0 Msun. Stored under
``data/mesa_comparison/rgb/<M>Msun/`` as:

  - ``history.data.gz``  : full MESA history (gzipped, all models, all columns)
  - ``tip.FGONG.gz``     : FGONG structure at the final (RGB-tip) model

Physics (MODE A, identical-physics — see project steering): MESA f12c70cf,
alpha_MLT=2.0 (Cox MLT), Z=0.014, Y=0.2695, gs98 opacities, Krishna-Swamy
T(tau) atmosphere, diffusion/overshoot/rotation OFF. Provenance and the pinned
inlist live next to the data in ``data/mesa_comparison/rgb/``.

These are REAL MESA outputs, not synthetic. The sanity test
(``test_rgb_reference_sanity`` in validate.py) and its anti-synthetic guards
exist to keep them that way.
"""
import gzip
import os

import numpy as np

RGB_MASSES = (1.0, 1.2, 1.5, 2.0)

_RGB_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "stellar_jax", "data", "mesa_comparison", "rgb")


def rgb_dir(mass):
    """Path to the reference directory for a given mass (may not exist yet)."""
    return os.path.join(_RGB_DIR, f"{mass}Msun")


def has_rgb_track(mass):
    """True iff both the history and tip-FGONG files are present for ``mass``."""
    d = rgb_dir(mass)
    return os.path.exists(os.path.join(d, "history.data.gz")) and os.path.exists(
        os.path.join(d, "tip.FGONG.gz")
    )


def _open_maybe_gz(path):
    """Open a possibly-gzipped text file, returning a list of lines."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return f.readlines()
    with open(path) as f:
        return f.readlines()


def load_rgb_history(mass):
    """Load a MESA ``history.data`` (gzipped) into a dict ``{column: ndarray}``.

    Mirrors the parsing convention of ``test_mesa_comparison``: skip the MESA
    header block, locate the column-name row (the line containing
    ``model_number``), and ``loadtxt`` the numeric rows below it.
    """
    path = os.path.join(rgb_dir(mass), "history.data.gz")
    lines = _open_maybe_gz(path)
    header = None
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("model_number"):
            header = line.split()
            data_start = i + 1
            break
    if header is None:
        raise ValueError(f"no 'model_number' header row in {path}")
    data = np.loadtxt(lines[data_start:])
    return {name: data[:, idx] for idx, name in enumerate(header)}


def load_rgb_tip_fgong(mass):
    """Parse the tip FGONG into a dict.

    Returns ``{nn, iconst, ivar, iversion, glob, data}`` where ``glob`` is the
    length-``iconst`` global-constants vector and ``data`` is the best-effort
    ``(nn, ivar)`` variable array (parsed by whitespace; may be ``None`` if the
    FGONG uses glued fixed-width fields, in which case structural metadata is
    still valid).
    """
    path = os.path.join(rgb_dir(mass), "tip.FGONG.gz")
    lines = _open_maybe_gz(path)
    # FGONG: 4 free-form header lines, then the "nn iconst ivar iversion" line,
    # then iconst global constants, then nn*ivar variable values.
    hdr_idx = None
    nn = iconst = ivar = iversion = None
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 4 and all(p.lstrip("-").isdigit() for p in parts):
            nn, iconst, ivar, iversion = (int(p) for p in parts)
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError(f"no FGONG 'nn iconst ivar iversion' line in {path}")

    def _floats(line):
        out = []
        for tok in line.split():
            try:
                out.append(float(tok.replace("D", "E").replace("d", "e")))
            except ValueError:
                return None  # glued fixed-width field; bail to structural-only
        return out

    nums = []
    glued = False
    for line in lines[hdr_idx + 1:]:
        vals = _floats(line)
        if vals is None:
            glued = True
            break
        nums.extend(vals)

    glob = np.array(nums[:iconst]) if len(nums) >= iconst else None
    data = None
    if not glued and len(nums) >= iconst + nn * ivar:
        data = np.array(nums[iconst:iconst + nn * ivar]).reshape(nn, ivar)
    return {
        "nn": nn,
        "iconst": iconst,
        "ivar": ivar,
        "iversion": iversion,
        "glob": glob,
        "data": data,
    }
