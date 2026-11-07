"""Volume-fraction preserving downsampling for the tortuosity solver."""
import numpy as np
from scipy import ndimage as ndi

from cathode_analysis.tortuosity import block_reduce_mask, spans_axis


def _blobby_minority(shape=(64, 64, 64), vf=0.19, sigma=2.0, seed=0):
    """Smooth random field thresholded to a minority phase with curved boundaries."""
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.random(shape), sigma)
    return f > np.quantile(f, 1 - vf)


def test_majority_vote_is_biased_against_minority_phase_and_preserving_is_not():
    m = _blobby_minority()
    majority = block_reduce_mask(m, 2, preserve_vf=False)
    kept = block_reduce_mask(m, 2, preserve_vf=True)
    assert majority.mean() < m.mean() - 0.01          # boundary blocks below half occupancy are dropped
    assert abs(kept.mean() - m.mean()) < 0.005


def test_volume_fraction_is_matched_for_random_and_blobby_masks():
    rng = np.random.default_rng(0)
    for k in (2, 3, 4):
        noise = rng.random((48, 48, 48)) < 0.19
        assert abs(block_reduce_mask(noise, k).mean() - noise.mean()) < 0.01
        blobs = _blobby_minority((48, 48, 48), 0.3, 1.5, seed=k)
        assert abs(block_reduce_mask(blobs, k).mean() - blobs.mean()) < 0.01


def test_percolating_slab_survives_downsampling():
    m = np.zeros((32, 32, 32), dtype=bool)
    m[:, 5:8, :] = True                                # 3-voxel slab spanning axis 0, straddling 2x2x2 blocks
    assert spans_axis(m)
    assert not spans_axis(block_reduce_mask(m, 2, preserve_vf=False)) or block_reduce_mask(m, 2, preserve_vf=False).mean() < m.mean() - 0.02
    kept = block_reduce_mask(m, 2, preserve_vf=True)
    assert spans_axis(kept) and abs(kept.mean() - m.mean()) < 0.01


def test_downsample_one_is_identity_and_result_is_reproducible():
    m = _blobby_minority((32, 32, 32))
    assert block_reduce_mask(m, 1) is m
    assert np.array_equal(block_reduce_mask(m, 2), block_reduce_mask(m, 2))
