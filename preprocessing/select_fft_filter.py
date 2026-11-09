r"""
Interactively choose which parts of the FFT spectrum to delete from a
representative slice of the merged BSE/SE stack, with a live filtered
preview, then save that exact selection as a mask for remove_curtaining.py
to reuse on the whole stack.

Why: remove_curtaining.py's automatic Gaussian notch assumes the artifact is
a plain axis-aligned band through the origin. Some datasets (or curtaining
severities) don't fit that well - this tool lets you look at the actual FFT
magnitude of your own data and paint out exactly the regions responsible.

Self-contained (doesn't depend on any sibling script).

Usage
-----
    python select_fft_filter.py                              # previews the first merged slice
    python select_fft_filter.py --image-path "/path/to/slice-0001.tif"
    python select_fft_filter.py --mask-out "my_mask.npy" --feather 3

Controls (in the popup window)
-------------------------------
    left-drag on the FFT panel : mark a region to delete (its point-mirrored
                                  counterpart is added automatically, since a
                                  real image's spectrum is conjugate-symmetric
                                  - deleting only one side would bias the result)
    u                          : undo the last region
    r                          : reset all regions
    s / enter                  : save the mask and exit
    q / escape                 : quit without saving

Output
------
    A boolean .npy file (True = frequency removed) matching the image shape.
    Use it with: python remove_curtaining.py --mask-path fft_mask.npy
"""

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

import platform

import matplotlib
if platform.system() == "Darwin":
    # macOS ships an ancient system Tcl/Tk (8.5.9) that renders Tkinter
    # windows blank on modern macOS - use the native Cocoa backend instead,
    # which doesn't go through Tk at all.
    matplotlib.use("macosx")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402

SLICE_RE = re.compile(r"(\d{3,6})\.tiff?$", re.IGNORECASE)

DEFAULT_IN_DIR = '/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_5050_cycled/A970_5to5_cycled/try3/Dataset 1/Merged'
DEFAULT_MASK_OUT = str(Path(__file__).resolve().parent / "fft_mask.npy")


def index_files(folder: Path) -> dict[int, Path]:
    """Map slice index (parsed from filename) -> file path."""
    files = {}
    for f in sorted(folder.glob("*.tif*")):
        m = SLICE_RE.search(f.name)
        if m:
            files[int(m.group(1))] = f
    return files


def img_as_float01(img: np.ndarray) -> np.ndarray:
    """Scale an integer image to float in [0, 1] based on its dtype's max value."""
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img, 0.0, 1.0).astype(np.float64)
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def apply_fft_weight(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Multiply an image's (fftshifted) spectrum by a per-frequency weight in
    [0, 1] (0 = fully removed, 1 = fully kept) and invert back to image space."""
    img_f = img_as_float01(img)
    spectrum = np.fft.fftshift(np.fft.fft2(img_f)) * weight
    result = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def mirror_rect(y0: int, y1: int, x0: int, x1: int, h: int, w: int) -> tuple[int, int, int, int]:
    cy, cx = h // 2, w // 2
    my0, my1 = np.clip([2 * cy - y1, 2 * cy - y0], 0, h)
    mx0, mx1 = np.clip([2 * cx - x1, 2 * cx - x0], 0, w)
    return int(my0), int(my1), int(mx0), int(mx1)


def build_delete_mask(rects: list, shape: tuple[int, int], dc_radius: float = 0.0) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for y0, y1, x0, x1 in rects:
        mask[y0:y1, x0:x1] = True
    if dc_radius > 0:
        # Always protect a small disk around the true DC term: it holds the image's
        # overall brightness, and a careless drag that sweeps across the center would
        # otherwise zero it out and crush the whole preview toward black.
        h, w = shape
        cy, cx = h // 2, w // 2
        yy, xx = np.mgrid[0:h, 0:w]
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= dc_radius ** 2] = False
    return mask


def weight_from_delete_mask(delete_mask: np.ndarray, feather: float) -> np.ndarray:
    weight = (~delete_mask).astype(np.float64)
    if feather > 0:
        weight = gaussian_filter(weight, sigma=feather)
    return weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-path", default=None, help="Specific slice to preview (default: first slice in --in-dir)")
    parser.add_argument("--in-dir", default=DEFAULT_IN_DIR, help="Merged BSE/SE stack (output of merge_bse_se.py)")
    parser.add_argument("--mask-out", default=DEFAULT_MASK_OUT)
    parser.add_argument("--feather", type=float, default=2.0,
                         help="Gaussian blur (in FFT pixels) applied to the mask edges to reduce ringing.")
    parser.add_argument("--dc-radius", type=float, default=5.0,
                         help="Radius around the true center (DC / overall brightness) that's always kept, "
                              "even if a dragged region covers it - shown as a dashed circle.")
    args = parser.parse_args()

    if args.image_path:
        img = tifffile.imread(args.image_path)
    else:
        files = index_files(Path(args.in_dir))
        if not files:
            raise SystemExit(f"No slices found in {args.in_dir}")
        first_idx = sorted(files)[0]
        img = tifffile.imread(files[first_idx])
        print(f"Previewing: {files[first_idx]}")

    h, w = img.shape[:2]
    img_f = img_as_float01(img)
    log_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img_f))))

    delete_rects: list[tuple[int, int, int, int]] = []
    patches: list[Rectangle] = []

    fig, (ax_fft, ax_preview) = plt.subplots(1, 2, figsize=(14, 7))
    ax_fft.imshow(log_mag, cmap="gray")
    ax_fft.set_title("FFT log-magnitude - drag to mark a region to delete")
    ax_fft.add_patch(plt.Circle((w // 2, h // 2), args.dc_radius, edgecolor="cyan", facecolor="none",
                                 linestyle="--", linewidth=1.2))
    preview_im = ax_preview.imshow(img_f, cmap="gray", vmin=0, vmax=1)
    ax_preview.set_title("Live filtered preview")
    fig.suptitle("drag=add region | u=undo | r=reset | s/enter=save | q/esc=quit  "
                 "(cyan circle = protected DC region, always kept)")

    def redraw_preview() -> None:
        delete_mask = build_delete_mask(delete_rects, (h, w), args.dc_radius)
        weight = weight_from_delete_mask(delete_mask, args.feather)
        preview_im.set_data(apply_fft_weight(img, weight))
        fig.canvas.draw_idle()

    def redraw_patches() -> None:
        for p in patches:
            p.remove()
        patches.clear()
        for y0, y1, x0, x1 in delete_rects:
            rect = Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", facecolor="none", linewidth=1.2)
            ax_fft.add_patch(rect)
            patches.append(rect)
        fig.canvas.draw_idle()

    def onselect(eclick, erelease) -> None:
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        x0, x1 = int(np.clip(x0, 0, w)), int(np.clip(np.ceil(x1), 0, w))
        y0, y1 = int(np.clip(y0, 0, h)), int(np.clip(np.ceil(y1), 0, h))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return
        delete_rects.append((y0, y1, x0, x1))
        delete_rects.append(mirror_rect(y0, y1, x0, x1, h, w))
        redraw_patches()
        redraw_preview()

    selector = RectangleSelector(
        ax_fft, onselect, useblit=True, button=[1], minspanx=1, minspany=1, spancoords="data",
    )

    def on_key(event) -> None:
        nonlocal delete_rects
        if event.key == "u" and delete_rects:
            delete_rects = delete_rects[:-2]
            redraw_patches()
            redraw_preview()
        elif event.key == "r":
            delete_rects = []
            redraw_patches()
            redraw_preview()
        elif event.key in ("s", "enter"):
            delete_mask = build_delete_mask(delete_rects, (h, w), args.dc_radius)
            np.save(args.mask_out, delete_mask)
            print(f"Saved mask with {len(delete_rects) // 2} region(s) (each mirrored) to {args.mask_out}")
            plt.close(fig)
        elif event.key in ("q", "escape"):
            print("Quit without saving.")
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print(__doc__)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
