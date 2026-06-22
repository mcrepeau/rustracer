#!/usr/bin/env python3
"""
render_qa.py  —  Artifact detector and quality grader for path-traced renders.

Detects three artifact classes in a tone-mapped PNG:

  black_dots    Isolated dark pixels surrounded by bright neighbours.
                Indicates convergence bugs, missed NEE emission, or adaptive-
                sampling marking pixels converged too early.

  fireflies     Isolated pixels that are far brighter than their local
                neighbourhood.  Caused by high-variance light paths (caustics,
                small light sources, insufficient path clamping).

  hue_outliers  Pixels whose hue departs sharply from the local circular median.
                Catches chromatic energy spikes from dispersive materials
                (diamond fire, spectral glass) that landed on a single pixel.

Usage:
  python render_qa.py render.png
  python render_qa.py a.png b.png              # compare multiple renders
  python render_qa.py render.png --annotate    # save *_qa.png with circles
  python render_qa.py render.png --json        # machine-readable report

Dependencies:
  pip install numpy Pillow scipy
"""

import sys
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy.ndimage import maximum_filter, uniform_filter, median_filter
except ImportError:
    sys.exit("scipy is required:  pip install scipy")


# ── Detection thresholds ──────────────────────────────────────────────────────
#
# Calibrated against 2000-spp renders from rustracer (1200×800 and similar).
# Increase the *_FLOOR / *_MIN values to reduce false-positive sensitivity.

# Black dots ─────────────────────────────────────────────────────────────────
# A pixel qualifies when it is very dark, its 3×3 neighbourhood max is bright,
# AND it is genuinely isolated (very few dark pixels in its 7×7 neighbourhood).
# The isolation check prevents flagging legitimate edge pixels at the boundary
# between a large dark border/background and the lit scene content.
BD_PIXEL_MAX     = 0.04   # pixel luminance ceiling  (gamma [0, 1])
BD_NEIGHBOR_MIN  = 0.10   # 3×3 neighbourhood max must exceed this
BD_ISOLATION_MAX = 0.06   # fraction of dark pixels in the 7×7 window (<~3/49)

# Fireflies ───────────────────────────────────────────────────────────────────
# Pixel must be k·σ above its 11×11 local mean, above an absolute floor,
# and at least N× the local mean.
FF_SIGMA        = 6.0     # local σ multiplier
FF_HALF_WIN     = 5       # half-window; full window = (2·w+1)² = 11×11
FF_ABS_FLOOR    = 0.80    # absolute brightness floor in gamma [0, 1]
FF_RATIO_FLOOR  = 4.0     # pixel / local mean must exceed this

# Hue outliers ────────────────────────────────────────────────────────────────
# Pixel hue departs sharply from the circular median of its 7×7 neighbourhood.
# Using a tight 7×7 window avoids flagging large coloured regions (e.g. the
# random-spheres scene) while catching single-pixel chromatic spikes.
HO_HALF_WIN     = 3       # half-window; full window = 7×7
HO_DIFF_DEG     = 55.0    # minimum hue angular difference (degrees)
HO_SAT_MIN      = 0.35    # pixel saturation floor (hue is meaningless near grey)
HO_VAL_MIN      = 0.15    # pixel brightness floor

# Annotation
ANNOTATE_CAP    = 300     # max circles drawn per class (keeps image readable)
CIRCLE_R        = 7       # circle radius in pixels


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_float(path: str) -> np.ndarray:
    """Load any PIL-readable image as float32 [H, W, 3] ∈ [0, 1]."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def gamma_lum(rgb: np.ndarray) -> np.ndarray:
    """Perceptual luminance in gamma space (sufficient for artifact detection)."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def rgb_to_hsv(rgb: np.ndarray) -> tuple:
    """Vectorised RGB [H,W,3] → (hue°, saturation, value) each [H,W]."""
    r, g, b  = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax     = np.maximum.reduce([r, g, b])
    cmin     = np.minimum.reduce([r, g, b])
    delta    = cmax - cmin
    safe_d   = np.where(delta > 1e-9, delta, 1.0)

    sat = np.where(cmax > 1e-6, delta / np.maximum(cmax, 1e-6), 0.0)

    hue = np.zeros_like(r)
    m   = delta > 1e-6
    mr  =  m & (cmax == r)
    mg  =  m & (cmax == g) & ~mr
    mb  =  m & (cmax == b) & ~mr & ~mg
    hue[mr] = (60.0 * (g[mr] - b[mr]) / safe_d[mr]) % 360.0
    hue[mg] = (60.0 * (b[mg] - r[mg]) / safe_d[mg] + 120.0) % 360.0
    hue[mb] = (60.0 * (r[mb] - g[mb]) / safe_d[mb] + 240.0) % 360.0

    return hue, sat, cmax   # (value = cmax)


# ── Detectors ─────────────────────────────────────────────────────────────────

def detect_black_dots(lum: np.ndarray) -> np.ndarray:
    """
    Returns a bool mask: True where a pixel is anomalously dark.

    Logic:
      is_dark     = pixel ≤ BD_PIXEL_MAX
      max_nbr     = maximum of the 3×3 window — bright neighbours present
      dark_frac   = fraction of dark pixels in the 7×7 window
      → flag when is_dark AND max_nbr ≥ BD_NEIGHBOR_MIN AND dark_frac < BD_ISOLATION_MAX

    The isolation check prevents flagging edge pixels at the legitimate boundary
    of a large dark region (e.g. the black frame around a Cornell-box scene).
    A single isolated black dot has dark_frac ≈ 1/49 ≈ 0.02; an edge pixel at
    the border of a dark background has dark_frac ≈ 0.3–0.5.
    """
    is_dark   = lum <= BD_PIXEL_MAX
    max_nbr   = maximum_filter(lum, size=3)
    dark_frac = uniform_filter(is_dark.astype(np.float32), size=7)
    return is_dark & (max_nbr >= BD_NEIGHBOR_MIN) & (dark_frac < BD_ISOLATION_MAX)


def detect_fireflies(lum: np.ndarray) -> np.ndarray:
    """
    Returns a bool mask: True where a pixel is an isolated bright spike.

    A pixel is flagged when:
      • lum > local_mean + FF_SIGMA * local_std     (statistical outlier)
      • lum > FF_RATIO_FLOOR * local_mean           (ratio guard)
      • lum ≥ FF_ABS_FLOOR                          (absolute brightness floor)

    The 11×11 local stats are computed via uniform_filter (fast O(1) per pixel).
    """
    sz          = 2 * FF_HALF_WIN + 1
    local_mean  = uniform_filter(lum,      size=sz)
    local_sq    = uniform_filter(lum ** 2, size=sz)
    local_var   = np.maximum(local_sq - local_mean ** 2, 0.0)
    local_std   = np.sqrt(local_var)

    stat_thr    = local_mean + FF_SIGMA * local_std
    ratio_thr   = local_mean * FF_RATIO_FLOOR

    return (lum > stat_thr) & (lum > ratio_thr) & (lum >= FF_ABS_FLOOR)


def detect_hue_outliers(rgb: np.ndarray) -> np.ndarray:
    """
    Returns a bool mask: True where a pixel's hue is an isolated chromatic spike.

    Uses a circular-median hue via sin/cos median-filter to handle the 0°/360°
    wraparound, then checks the angular distance.  Only considers pixels with
    sufficient saturation and brightness.

    Pixels near the edge of a large dark region are excluded: the circular
    median is computed across a 7×7 window that can straddle the scene/background
    boundary, producing a spurious hue estimate for the last row of scene content.
    """
    hue, sat, val = rgb_to_hsv(rgb)
    lum       = gamma_lum(rgb)
    candidate = (sat >= HO_SAT_MIN) & (val >= HO_VAL_MIN)

    # Exclude pixels whose 7×7 neighbourhood contains significant dark content —
    # these sit on the edge of a background region and the circular median is
    # unreliable there.
    dark_frac = uniform_filter((lum <= BD_PIXEL_MAX).astype(np.float32), size=7)
    candidate = candidate & (dark_frac < BD_ISOLATION_MAX)

    rad     = np.radians(hue)
    sz      = 2 * HO_HALF_WIN + 1
    med_hue = (np.degrees(
        np.arctan2(
            median_filter(np.sin(rad), size=sz),
            median_filter(np.cos(rad), size=sz),
        )
    ) % 360.0)

    diff = np.abs(hue - med_hue)
    diff = np.minimum(diff, 360.0 - diff)   # wrap to [0, 180]

    return candidate & (diff >= HO_DIFF_DEG)


# ── Grading ───────────────────────────────────────────────────────────────────

# (minor_threshold, notable_threshold) — per-megapixel counts
_SEVERITY_THRESHOLDS = {
    "black_dots":   (1,   5),    # even 1/MP is notable; > 5/MP is severe
    "fireflies":    (10,  100),
    "hue_outliers": (10,  80),
}

def _severity(name: str, per_mp: float) -> str:
    minor_t, notable_t = _SEVERITY_THRESHOLDS[name]
    if per_mp == 0:              return "none"
    if per_mp <= minor_t:        return "minor"
    if per_mp <= notable_t:      return "notable"
    return "severe"


def _top_instances(mask: np.ndarray, score: np.ndarray, n: int = 12) -> list:
    """Return the N highest-score artifact locations as [(x, y, score), ...]."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []
    s   = score[ys, xs]
    idx = np.argsort(s)[-n:][::-1]
    return [(int(xs[i]), int(ys[i]), round(float(s[i]), 4)) for i in idx]


def _cluster_info(mask: np.ndarray) -> dict:
    """
    Quick spatial distribution summary: whether artifacts are clustered
    (suggesting a region-specific bug) or scattered (statistical noise).
    """
    ys, xs = np.where(mask)
    if len(ys) < 2:
        return {"clustered": False, "regions": 0}
    from scipy.ndimage import label as nd_label
    # Dilate slightly so nearby pixels merge into one cluster.
    from scipy.ndimage import binary_dilation
    dilated       = binary_dilation(mask, iterations=8)
    labeled, n_cc = nd_label(dilated)
    return {"clustered": n_cc < max(1, len(ys) // 5), "regions": int(n_cc)}


def compute_score(bd_pmp: float, ff_pmp: float, ho_pmp: float) -> float:
    """
    Map per-megapixel artifact counts to a 0–100 quality score.

    Black dots carry the heaviest penalty because they signal outright bugs,
    not just statistical noise.
    """
    bd_penalty = min(bd_pmp * 5.0,  45.0)   # up to −45
    ff_penalty = min(ff_pmp * 0.4,  30.0)   # up to −30
    ho_penalty = min(ho_pmp * 0.3,  20.0)   # up to −20
    return max(0.0, 100.0 - bd_penalty - ff_penalty - ho_penalty)


def score_to_grade(score: float) -> str:
    if score >= 97: return "A+"
    if score >= 93: return "A"
    if score >= 90: return "A−"
    if score >= 87: return "B+"
    if score >= 83: return "B"
    if score >= 80: return "B−"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


# ── Full analysis pipeline ────────────────────────────────────────────────────

def analyze(path: str) -> dict:
    """
    Load the image at `path`, run all detectors, and return a structured report
    dict suitable for both human printing and JSON serialisation.
    """
    rgb  = load_float(path)
    H, W = rgb.shape[:2]
    mpx  = H * W / 1_000_000.0
    lum  = gamma_lum(rgb)

    # ── Run detectors ─────────────────────────────────────────────────────────
    bd_mask = detect_black_dots(lum)
    ff_mask = detect_fireflies(lum)
    ho_mask = detect_hue_outliers(rgb)

    bd_n = int(bd_mask.sum())
    ff_n = int(ff_mask.sum())
    ho_n = int(ho_mask.sum())

    bd_pmp = bd_n / mpx
    ff_pmp = ff_n / mpx
    ho_pmp = ho_n / mpx

    # ── Score maps for worst-instance ranking ──────────────────────────────────
    max_nbr   = maximum_filter(lum, size=3)
    bd_score  = np.where(bd_mask, max_nbr - lum, 0.0)   # higher = more isolated

    sz          = 2 * FF_HALF_WIN + 1
    local_mean  = uniform_filter(lum, size=sz)
    ff_score    = np.where(ff_mask, lum / np.maximum(local_mean, 1e-6), 0.0)

    hue, sat, val = rgb_to_hsv(rgb)
    rad     = np.radians(hue)
    sz_ho   = 2 * HO_HALF_WIN + 1
    med_hue = (np.degrees(
        np.arctan2(
            median_filter(np.sin(rad), size=sz_ho),
            median_filter(np.cos(rad), size=sz_ho),
        )
    ) % 360.0)
    hue_diff = np.abs(hue - med_hue)
    hue_diff = np.minimum(hue_diff, 360.0 - hue_diff)
    ho_score = np.where(ho_mask, hue_diff, 0.0)

    # ── Grade ─────────────────────────────────────────────────────────────────
    score = compute_score(bd_pmp, ff_pmp, ho_pmp)
    grade = score_to_grade(score)

    # ── Build structured report ────────────────────────────────────────────────
    def _artifact(name, mask, n, pmp, score_map, desc):
        return {
            "name":        name,
            "count":       n,
            "per_mp":      round(pmp, 2),
            "severity":    _severity(name, pmp),
            "description": desc,
            "top_instances": _top_instances(mask, score_map),
            "distribution":  _cluster_info(mask),
        }

    artifacts = [
        _artifact("black_dots", bd_mask, bd_n, bd_pmp, bd_score,
                  "Isolated dark pixel in a bright region — convergence / sampling bug"),
        _artifact("fireflies",  ff_mask, ff_n, ff_pmp, ff_score,
                  "Isolated bright spike — high-variance path sample"),
        _artifact("hue_outliers", ho_mask, ho_n, ho_pmp, ho_score,
                  "Isolated chromatic spike vs local hue median"),
    ]

    parts = []
    for a in artifacts:
        if a["count"]:
            parts.append(f"{a['count']} {a['name'].replace('_', ' ')}")
    summary = ("No artifacts detected." if not parts
               else "Found: " + ", ".join(parts) + ".")

    return {
        "path":       path,
        "width":      W,
        "height":     H,
        "megapixels": round(mpx, 3),
        "score":      round(score, 1),
        "grade":      grade,
        "summary":    summary,
        "artifacts":  artifacts,
        # Keep masks for annotation (stripped before JSON output).
        "_masks": {"black_dots": bd_mask, "fireflies": ff_mask, "hue_outliers": ho_mask},
        "_rgb":   rgb,
    }


# ── Annotation ────────────────────────────────────────────────────────────────

# (circle fill RGBA, circle outline RGBA)
_ARTIFACT_STYLE = {
    "black_dots":    ((255,  60,  60, 200), (160,   0,   0, 255)),   # red
    "fireflies":     ((255, 220,  30, 200), (180, 130,   0, 255)),   # amber
    "hue_outliers":  ((200,  50, 240, 200), (130,   0, 180, 255)),   # magenta
}

_LEGEND_LABELS = [
    ("black_dots",    "Black dot  (dark in bright region)"),
    ("fireflies",     "Firefly    (isolated bright spike)"),
    ("hue_outliers",  "Hue outlier (chromatic spike)"),
]


def _circle(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int,
            fill: tuple, outline: tuple) -> None:
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=fill, outline=outline, width=2)


def build_annotated(report: dict) -> Image.Image:
    """
    Overlay coloured circles on detected artifact locations and add a legend.
    Returns an RGB PIL Image.
    """
    rgb  = report["_rgb"]
    base = Image.fromarray((rgb * 255).astype(np.uint8), "RGB").convert("RGBA")
    over = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)

    for name, (fill, outline) in _ARTIFACT_STYLE.items():
        mask = report["_masks"][name]
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        # Sort: worst instances first (from top_instances list), then the rest.
        top_xy = {(x, y) for x, y, _ in
                  next(a["top_instances"] for a in report["artifacts"]
                       if a["name"] == name)}
        coords = list(zip(xs.tolist(), ys.tolist()))
        ordered = [c for c in coords if c in top_xy]
        ordered += [c for c in coords if c not in top_xy]

        for cx, cy in ordered[:ANNOTATE_CAP]:
            _circle(draw, cx, cy, CIRCLE_R, fill, outline)

    # ── Legend ────────────────────────────────────────────────────────────────
    W, H   = base.size
    pad    = 12
    lh     = 22          # line height
    box_w  = 280
    box_h  = pad * 2 + 26 + len(_LEGEND_LABELS) * lh
    bx     = pad
    by     = H - box_h - pad

    draw.rectangle([bx, by, bx + box_w, by + box_h], fill=(0, 0, 0, 170))

    grade_line = f"Grade: {report['grade']}  ({report['score']:.0f} / 100)"
    draw.text((bx + pad, by + pad), grade_line, fill=(255, 255, 255, 255))

    for i, (name, label_text) in enumerate(_LEGEND_LABELS):
        fy    = by + pad + 26 + i * lh
        fc, _ = _ARTIFACT_STYLE[name]
        draw.ellipse([bx + pad, fy + 4, bx + pad + 13, fy + 17], fill=fc)
        draw.text((bx + pad + 18, fy), label_text, fill=(230, 230, 230, 255))

    return Image.alpha_composite(base, over).convert("RGB")


# ── Printing ──────────────────────────────────────────────────────────────────

_SEV_COLOUR = {
    "none":    "\033[32m",   # green
    "minor":   "\033[33m",   # yellow
    "notable": "\033[33m",   # yellow
    "severe":  "\033[31m",   # red
}
_GRADE_COLOUR = {
    "A+": "\033[32;1m", "A": "\033[32;1m", "A−": "\033[32m",
    "B+": "\033[36m",   "B": "\033[36m",   "B−": "\033[36m",
    "C":  "\033[33m",
    "D":  "\033[31m",
    "F":  "\033[31;1m",
}
_RST = "\033[0m"


def _sev(label: str, use_color: bool) -> str:
    if not use_color:
        return label
    return f"{_SEV_COLOUR.get(label, '')}{label}{_RST}"


def _grade_str(grade: str, score: float, use_color: bool) -> str:
    text = f"{grade}  ({score:.0f}/100)"
    if not use_color:
        return text
    return f"{_GRADE_COLOUR.get(grade, '')}{text}{_RST}"


def print_report(report: dict, use_color: bool = True) -> None:
    W, H  = report["width"], report["height"]
    mpx   = report["megapixels"]
    print(f"\n{'─' * 60}")
    print(f"  {report['path']}  ({W}×{H}, {mpx:.2f} MP)")
    print(f"{'─' * 60}")

    for a in report["artifacts"]:
        sev  = a["severity"]
        line = (f"  {a['name']:<16}"
                f"  {a['count']:>6} px"
                f"  ({a['per_mp']:>7.1f}/MP)"
                f"  [{_sev(sev, use_color)}]")
        print(line)
        if a["top_instances"]:
            coords_str = "  ".join(f"({x},{y})" for x, y, _ in a["top_instances"][:5])
            print(f"  {'':16}  worst: {coords_str}")
        if a["distribution"]["regions"] > 1:
            print(f"  {'':16}  {a['distribution']['regions']} spatial clusters"
                  f"{'  ← concentrated region' if a['distribution']['clustered'] else ''}")

    print()
    print(f"  Grade: {_grade_str(report['grade'], report['score'], use_color)}")
    print(f"  {report['summary']}")
    print(f"{'─' * 60}\n")


def print_comparison(reports: list[dict], use_color: bool = True) -> None:
    """Side-by-side comparison table for multiple renders."""
    names = [Path(r["path"]).name for r in reports]
    col_w = max(len(n) for n in names) + 2

    header = f"  {'Artifact':<16}" + "".join(f"  {n:>{col_w}}" for n in names)
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    classes = ["black_dots", "fireflies", "hue_outliers"]
    for cls in classes:
        row = f"  {cls:<16}"
        for r in reports:
            a    = next(x for x in r["artifacts"] if x["name"] == cls)
            cell = f"{a['count']} ({a['per_mp']:.0f}/MP)"
            row += f"  {cell:>{col_w}}"
        print(row)

    row = f"  {'grade':<16}"
    for r in reports:
        g = f"{r['grade']} ({r['score']:.0f})"
        row += f"  {g:>{col_w}}"
    print(f"{'─' * len(header)}")
    print(row)
    print(f"{'─' * len(header)}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Artifact detector and quality grader for path-traced renders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    ap.add_argument("images", nargs="+", metavar="image.png",
                    help="One or more PNG renders to analyse.")
    ap.add_argument("--annotate", action="store_true",
                    help="Save annotated image alongside each input (*_qa.png).")
    ap.add_argument("--json", action="store_true",
                    help="Print machine-readable JSON report to stdout.")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable ANSI colour in terminal output.")
    args = ap.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    reports = []
    for path in args.images:
        if not Path(path).exists():
            print(f"File not found: {path}", file=sys.stderr)
            continue

        print(f"Analysing {path}…", end=" ", flush=True)
        report = analyze(path)
        reports.append(report)
        print("done.")

        if args.annotate:
            ann_path = str(Path(path).with_stem(Path(path).stem + "_qa"))
            build_annotated(report).save(ann_path)
            print(f"  → annotated image saved to {ann_path}")

    if not reports:
        sys.exit(1)

    # ── Output ────────────────────────────────────────────────────────────────
    if args.json:
        # Strip numpy masks before serialisation.
        clean = []
        for r in reports:
            c = {k: v for k, v in r.items() if not k.startswith("_")}
            clean.append(c)
        print(json.dumps(clean if len(clean) > 1 else clean[0], indent=2))
    else:
        if len(reports) == 1:
            print_report(reports[0], use_color)
        else:
            for r in reports:
                print_report(r, use_color)
            print_comparison(reports, use_color)


if __name__ == "__main__":
    main()
