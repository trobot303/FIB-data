"""Memory-aware operations on (z, y, x) uint8 label volumes.

All heavy operations work on z-chunks with an overlap margin so that the
int32 arrays produced by ``scipy.ndimage.label`` only ever exist for one
chunk. Objects that cross chunk borders are handled conservatively (kept /
not filled) and the margin makes that rare for the small structures these
functions target.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
from scipy import ndimage as ndi

LOG = logging.getLogger(__name__)

_STRUCTURES = {6: 1, 18: 2, 26: 3}


def structure(connectivity: int) -> np.ndarray:
    """3D binary structure for 6 / 18 / 26 neighbourhoods."""
    return ndi.generate_binary_structure(3, _STRUCTURES[int(connectivity)])


def iter_z_chunks(nz: int, chunk: int | None, margin: int = 0) -> Iterator[tuple[int, int, int, int]]:
    """Yield (core_start, core_end, ext_start, ext_end) along z."""
    chunk = nz if not chunk or chunk <= 0 else int(chunk)
    for z0 in range(0, nz, chunk):
        z1 = min(nz, z0 + chunk)
        yield z0, z1, max(0, z0 - margin), min(nz, z1 + margin)


def histogram_uint8(vol: np.ndarray, chunk_z: int = 64) -> np.ndarray:
    """Exact 256-bin histogram computed chunk-wise (no temporaries of full size)."""
    hist = np.zeros(256, dtype=np.int64)
    for z0, z1, _, _ in iter_z_chunks(vol.shape[0], chunk_z):
        hist += np.bincount(vol[z0:z1].ravel(), minlength=256)
    return hist


def apply_lut_inplace(vol: np.ndarray, lut: np.ndarray, chunk_z: int = 64) -> np.ndarray:
    """vol[...] = lut[vol] without allocating a full-size copy."""
    for z0, z1, _, _ in iter_z_chunks(vol.shape[0], chunk_z):
        vol[z0:z1] = lut[vol[z0:z1]]
    return vol


def count_codes(labels: np.ndarray, n_codes: int = 6, chunk_z: int = 64) -> np.ndarray:
    """Voxel count per label code."""
    counts = np.zeros(n_codes, dtype=np.int64)
    for z0, z1, _, _ in iter_z_chunks(labels.shape[0], chunk_z):
        counts += np.bincount(labels[z0:z1].ravel(), minlength=n_codes)[:n_codes]
    return counts


# --------------------------------------------------------------------------- #
# small-object removal with neighbour-majority refill
# --------------------------------------------------------------------------- #
def _fill_by_neighbour_majority(labels: np.ndarray, unassigned: np.ndarray, codes: tuple[int, ...],
                                max_iter: int = 16) -> int:
    """Assign ``unassigned`` voxels to the majority code among their assigned
    26-neighbours, growing inwards iteratively. Returns the number left over."""
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    for _ in range(max_iter):
        n_open = int(unassigned.sum())
        if n_open == 0:
            return 0
        best_count = np.zeros(labels.shape, dtype=np.uint8)
        best_code = np.zeros(labels.shape, dtype=np.uint8)
        for c in codes:
            src = ((labels == c) & ~unassigned).astype(np.uint8)
            cnt = ndi.convolve(src, kernel, mode="constant", cval=0, output=np.uint8)
            better = cnt > best_count
            best_count[better] = cnt[better]
            best_code[better] = c
        newly = unassigned & (best_count > 0)
        if not newly.any():
            break
        labels[newly] = best_code[newly]
        unassigned &= ~newly
    return int(unassigned.sum())


def fill_unassigned(labels: np.ndarray, sentinel: int, codes: tuple[int, ...], fallback: int,
                    chunk_z: int | None = 128, margin: int = 16) -> int:
    """Replace ``sentinel`` voxels by the majority code of their assigned
    26-neighbours (grown iteratively, z-chunked). Voxels that are still open
    afterwards (large sentinel blobs) get ``fallback``. Returns the number of
    voxels that needed the fallback.
    """
    nz = labels.shape[0]
    n_fallback = 0
    for z0, z1, e0, e1 in iter_z_chunks(nz, chunk_z, margin):
        block = labels[e0:e1].copy()
        unassigned = block == sentinel
        if not unassigned.any():
            continue
        left = _fill_by_neighbour_majority(block, unassigned, codes)
        core = slice(z0 - e0, z1 - e0)
        if left:
            still = unassigned[core]
            n_fallback += int(still.sum())
            block[core][still] = fallback
        labels[z0:z1] = block[core]
    return n_fallback


def remove_small_objects(labels: np.ndarray, code: int, min_size: int, other_codes: tuple[int, ...],
                         conn: int = 6, chunk_z: int | None = 128, margin: int = 16) -> int:
    """Remove islands of ``code`` smaller than ``min_size`` voxels (in place).

    Removed voxels are reassigned to the majority of their neighbours among
    ``other_codes``. Islands touching a chunk's extended z-faces are kept
    (their true size is unknown), so ``margin`` should exceed the expected
    island extent. Returns the number of voxels reassigned.
    """
    if min_size <= 1:
        return 0
    st = structure(conn)
    nz = labels.shape[0]
    n_removed = 0
    for z0, z1, e0, e1 in iter_z_chunks(nz, chunk_z, margin):
        block = labels[e0:e1]
        mask = block == code
        if not mask.any():
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        small = sizes < min_size
        small[0] = False
        # objects reaching the extended block's z-faces (inside the volume) may continue: keep them
        if e0 > 0:
            small[np.unique(lab[0])] = False
        if e1 < nz:
            small[np.unique(lab[-1])] = False
        core = slice(z0 - e0, z1 - e0)
        removed = small[lab[core]]
        del lab
        n_rm = int(removed.sum())
        if n_rm == 0:
            continue
        # fill using an extended window so neighbours outside the core are seen
        sub = block
        unassigned = np.zeros(sub.shape, dtype=bool)
        unassigned[core] = removed
        left = _fill_by_neighbour_majority(sub, unassigned, other_codes)
        if left:
            sub[unassigned] = other_codes[0]
        labels[z0:z1] = sub[core]
        n_removed += n_rm
    return n_removed


def fill_holes(labels: np.ndarray, code: int, max_voxels: int, cavity_code: int = 0,
               chunk_z: int | None = 128, margin: int = 16) -> int:
    """Fill small cavities of ``cavity_code`` fully enclosed by ``code`` (in place).

    A cavity is a connected component of ``labels != code`` that touches no face
    of the extended chunk, consists only of ``cavity_code`` voxels (so enclosed
    particles of a third phase are never touched) and has at most ``max_voxels``
    voxels. Returns the number of voxels filled.
    """
    if max_voxels <= 0:
        return 0
    st = structure(6)
    nz = labels.shape[0]
    n_filled = 0
    for z0, z1, e0, e1 in iter_z_chunks(nz, chunk_z, margin):
        block = labels[e0:e1]
        lab, n = ndi.label(block != code, structure=st)
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        cavity_only = np.bincount(lab[block == cavity_code].ravel(), minlength=n + 1) == sizes
        touching = np.zeros(n + 1, dtype=bool)
        for face in (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]):
            touching[np.unique(face)] = True
        touching[0] = True
        fill = ~touching & cavity_only & (sizes <= max_voxels)
        core = slice(z0 - e0, z1 - e0)
        holes = fill[lab[core]]
        del lab
        cnt = int(holes.sum())
        if cnt:
            sub = labels[z0:z1]
            sub[holes] = code
            n_filled += cnt
    return n_filled


def majority_filter(labels: np.ndarray, radius: int, codes: tuple[int, ...], chunk_z: int | None = 128) -> None:
    """3D majority vote in a ball of the given radius (in place)."""
    if radius <= 0:
        return
    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    kernel = ((zz ** 2 + yy ** 2 + xx ** 2) <= r * r).astype(np.uint8)
    nz = labels.shape[0]
    for z0, z1, e0, e1 in iter_z_chunks(nz, chunk_z, r):
        block = labels[e0:e1]
        best_count = np.zeros(block.shape, dtype=np.uint16)
        best_code = np.zeros(block.shape, dtype=np.uint8)
        for c in codes:
            cnt = ndi.convolve((block == c).astype(np.uint16), kernel.astype(np.uint16), mode="nearest")
            better = cnt > best_count
            best_count[better] = cnt[better]
            best_code[better] = c
        labels[z0:z1] = best_code[z0 - e0:z1 - e0]


# --------------------------------------------------------------------------- #
# adjacency (voxel faces)
# --------------------------------------------------------------------------- #
def face_counts(labels: np.ndarray, n_codes: int = 6, chunk_z: int | None = 128,
                axes: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    """Symmetric matrix M[a, b] = number of voxel faces between codes a and b.

    Faces between two voxels of the same code are counted on the diagonal.
    """
    M = np.zeros((n_codes, n_codes), dtype=np.int64)
    nz = labels.shape[0]
    for z0, z1, _, _ in iter_z_chunks(nz, chunk_z):
        z_end = min(nz, z1 + 1) if 0 in axes else z1
        block = labels[z0:z_end]
        for ax in axes:
            if ax == 0:
                a, b = block[:-1], block[1:]
            elif ax == 1:
                a, b = block[:z1 - z0, :-1], block[:z1 - z0, 1:]
            else:
                a, b = block[:z1 - z0, :, :-1], block[:z1 - z0, :, 1:]
            if a.size == 0:
                continue
            pair = a.astype(np.int16) * n_codes + b.astype(np.int16)
            M += np.bincount(pair.ravel(), minlength=n_codes * n_codes).reshape(n_codes, n_codes)
    return M + M.T - np.diag(np.diag(M))


def face_counts_along(labels: np.ndarray, axis: int, n_codes: int = 6) -> np.ndarray:
    """Per-position face counts along ``axis`` (for profiles): shape (n-1, n_codes, n_codes)
    of faces between position i and i+1 along that axis plus in-plane faces at i."""
    lab = np.moveaxis(labels, axis, 0)
    n = lab.shape[0]
    out = np.zeros((n, n_codes, n_codes), dtype=np.int64)
    for i in range(n):
        sl = lab[i]
        planes = [(sl[:-1], sl[1:]), (sl[:, :-1], sl[:, 1:])]
        if i + 1 < n:
            planes.append((sl, lab[i + 1]))
        for a, b in planes:
            pair = a.astype(np.int16) * n_codes + b.astype(np.int16)
            m = np.bincount(pair.ravel(), minlength=n_codes * n_codes).reshape(n_codes, n_codes)
            out[i] += m + m.T - np.diag(np.diag(m))
    return out
