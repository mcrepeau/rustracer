#!/usr/bin/env python3
"""report_to_toml.py — turn an IGI/GIA grading-report PDF into a renderable
rustracer scene TOML with a [[gems]] block.

Usage:
    py report_to_toml.py REPORT.pdf [-o out.toml] [--material PRESET] [--cut CUT]
                                [--gap-mm 0.1] [--no-chart] [--lights] [--no-caustics] [--no-animation] [--dark]

    py report_to_toml.py LG625469749.pdf            # writes scenes/LG625469749.toml

Standard library only: PDF text is pulled from FlateDecode content streams
with zlib + regex (works for IGI electronic-copy reports; GIA layouts are
matched best-effort). Fields that fail to parse are reported as warnings and
omitted from the TOML, so the output always loads.

Field mapping:
    Report number/date, Description  -> [[gems]] report + material preset
    Shape and Cutting Style          -> cut preset (see CUT_MAP)
    Measurements L x W x D mm        -> measurements_mm (ranges are averaged)
    Carat Weight                     -> carat
    Color Grade                      -> color_grade
    Proportions diagram (crown% table% pavilion% culet depth%)
                                     -> crown_pct / table_pct / pavilion_pct / depth_pct
"""

import argparse
import os
import re
import sys
import zlib

# ── PDF text extraction (stdlib only) ────────────────────────────────────────

def extract_pdf_text(path):
    data = open(path, "rb").read()
    chunks = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", data, re.S):
        try:
            d = zlib.decompress(m.group(1))
        except Exception:
            continue  # not a Flate stream (images, fonts, ...)
        # Text-show operators, in stream order; TJ arrays may interleave kerning.
        for op in re.finditer(rb"\[(.*?)\]\s*TJ|\(((?:[^()\\]|\\.)*)\)\s*Tj", d, re.S):
            if op.group(1) is not None:
                chunks.extend(re.findall(rb"\(((?:[^()\\]|\\.)*)\)", op.group(1)))
            else:
                chunks.append(op.group(2))
    text = b" ".join(chunks).decode("latin-1")
    return text.replace("\\(", "(").replace("\\)", ")")

# ── Report parsing ───────────────────────────────────────────────────────────

CUT_MAP = [
    ("OVAL", "brilliant"), ("ROUND", "brilliant"), ("BRILLIANT", "brilliant"),
    ("MODIFIED", "brilliant"),
    ("PRINCESS", "princess"),
    ("EMERALD", "step"),
    ("CUSHION", "cushion"),
    ("MARQUISE", "marquise"),
    ("PEAR", "pear"),
    ("RADIANT", "radiant"),
    ("ASSCHER", "asscher"),
    ("BAGUETTE", "baguette"),
    ("TRILLION", "trillion"), ("TRIANGLE", "trillion"),
    ("HEART", None),  # non-convex: not supported by the analytic shapes
]

MATERIAL_MAP = [
    ("SAPPHIRE", "sapphire"), ("RUBY", "ruby"), ("EMERALD", "emerald"),
    ("DIAMOND", "diamond"),
]

def cut_from_style(style, warnings):
    s = style.upper()
    for key, cut in CUT_MAP:
        if key in s:
            if cut is None:
                warnings.append("cut '%s' (heart) is non-convex and not supported "
                                "by the analytic shapes; falling back to 'brilliant'" % style)
                return "brilliant"
            return cut
    warnings.append("unknown cutting style '%s'; falling back to 'brilliant'" % style)
    return "brilliant"

def material_from_description(desc, warnings):
    d = desc.upper()
    for key, mat in MATERIAL_MAP:
        if key in d:
            return mat
    warnings.append("unknown description '%s'; defaulting material to 'diamond'" % desc)
    return "diamond"

# GIA-style girdle descriptor -> band thickness as % of the girdle diameter
GIRDLE_TO_PCT = {
    "Extremely Thin": 0.5, "Very Thin": 1.0, "Thin": 1.75, "Medium": 3.0,
    "Slightly Thick": 4.0, "Thick": 5.0, "Very Thick": 6.5, "Extremely Thick": 8.0,
}

def parse_measurements(text, warnings):

    m = re.search(r"Measurements\s+([\d.]+(?:\s*-\s*[\d.]+)?)\s*X\s*([\d.]+)\s*X\s*([\d.]+)\s*MM", text)
    if m:
        return [float(m.group(1).replace(" ", "")), float(m.group(2)), float(m.group(3))]
    m = re.search(r"Measurements\s+([\d.]+(?:\s*-\s*[\d.]+)?)\s*X\s*([\d.]+)\s*MM", text)
    if m:  # round stone: "5.26 - 5.31 X 3.24 MM" = diameter range x depth
        first = m.group(1)
        if "-" in first:
            lo, hi = first.split("-")
            l = (float(lo) + float(hi)) / 2.0
        else:
            l = float(first)
        return [l, l, float(m.group(2))]
    warnings.append("no Measurements line found; size will come from carat only")
    return None

def parse_report(text):
    warnings = []
    f = {}
    m = re.search(r"(?:IGI Report (?:Number|No)|GIA Report (?:Number|No\.?))\s+([A-Z0-9]+)", text)
    f["report"] = m.group(1) if m else "unknown"
    if not m:
        warnings.append("no report number found")
    m = re.search(r"Description\s+(.+?)\s+Shape and Cutting Style", text, re.S)
    f["description"] = re.sub(r"\s+", " ", m.group(1)).strip() if m else "unknown"
    m = re.search(r"Shape and Cutting Style\s+(.+?)\s+Measurements", text, re.S)
    f["style"] = re.sub(r"\s+", " ", m.group(1)).strip() if m else "unknown"
    if not m:
        warnings.append("no Shape and Cutting Style found")
    m = re.search(r"Carat Weight\s+([\d.]+)\s*CARAT", text, re.I)
    f["carat"] = float(m.group(1)) if m else None
    if not m:
        warnings.append("no Carat Weight found")
    m = re.search(r"Color Grade\s+([D-Z])\b", text)
    f["color_grade"] = m.group(1) if m else "D"
    if not m:
        warnings.append("no Color Grade found; defaulting to D")
    m = re.search(r"Clarity Grade\s+([A-Z]{1,3}\s?\d?)", text)
    f["clarity"] = re.sub(r"\s+", "", m.group(1)) if m else "unknown"
    m = re.search(r"(\d{1,3}(?:\.\d+)?)%\s*(\d{1,3}(?:\.\d+)?)%\s*(\d{1,3}(?:\.\d+)?)%\s*"
                  r"([A-Za-z]+)\s*(\d{1,3}(?:\.\d+)?)%\s*PROPORTIONS", text)
    # proportions diagram order: crown%  table%  pavilion%  culet  depth%
    if m:
        f["crown_pct"], f["table_pct"], f["pavilion_pct"] = float(m.group(1)), float(m.group(2)), float(m.group(3))
        f["culet"], f["depth_pct"] = m.group(4), float(m.group(5))
    else:
        warnings.append("no proportions diagram found; cut angles will use defaults")
        f["crown_pct"] = f["table_pct"] = f["pavilion_pct"] = f["depth_pct"] = None
    gm = re.search(r"(Extremely Thin|Very Thin|Thin|Medium|Slightly Thick|Thick|Very Thick|Extremely Thick)", text)
    f["girdle"] = gm.group(1) if gm else None
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}", text)
    f["date"] = m.group(0) if m else "unknown date"
    f["measurements_mm"] = parse_measurements(text, warnings)
    return f, warnings

# ── TOML emission ────────────────────────────────────────────────────────────

def render_toml(f, args):
    warnings = []
    cut = args.cut or cut_from_style(f["style"], warnings)
    material = args.material or material_from_description(f["description"], warnings)

    meas = f["measurements_mm"]
    if meas:
        l_mm, w_mm, d_mm = meas
    elif f["carat"]:
        l_mm = w_mm = (f["carat"] / 0.00355) ** (1.0 / 3.0)
        d_mm = None
    else:
        raise SystemExit("error: neither measurements nor carat weight found in the report")

    # Normalize so the stone's length is exactly 1 scene unit: all distances derive
    # from it, and the renderer's absolute t_min = 0.001 becomes 0.1% of the stone
    # (at mm->m scale a 9 mm stone is 0.009 units, so the epsilon is ~11% of it).
    scale = 1.0 / l_mm

    # "oval" only matters for brilliant cuts (smooth-ellipse girdle); step cuts
    # are rectangular by design, so L != W there is not an oval.
    oval = cut == "brilliant" and l_mm / w_mm > 1.05
    # pavilion height in mm = W * pavilion_pct/100 (see build_gem angle conversion)
    pav_mm = w_mm * (f["pavilion_pct"] or 41.0) / 100.0
    size = l_mm * scale                            # stone length in scene units
    # Optional grey-chart card under the stone: the OBJ is in centimetres
    # (18 cm wide, 0.4 cm tall); scale it so the card is --chart-width stone
    # lengths wide. Its top becomes the new "floor" the culet tip rests on.
    card_top = 0.0
    chart_scale = 0.0
    if args.chart:
        chart_scale = (args.chart_width * size) / 18.0   # OBJ units are cm
        card_top = 0.4 * chart_scale                     # OBJ y max = 0.4 cm
    cy = (pav_mm + args.gap_mm) * scale + card_top       # girdle centre height above the floor
    dist = max(3.2 * size, 0.012)                  # camera distance
    lf = (0.43 * dist, cy + 0.26 * dist, -0.86 * dist)  # same view direction as gem_LG625469749

    # Staging by default: the stone rests centered on the grey chart card on the
    # Perlin-mottled floor, lit by the env map alone (product_studio.hdr, or
    # studio_dark.hdr with --dark). --lights adds geometric softbox quads,
    # --no-chart drops the card.
    if args.dark:
        bg_lines = ['type = "env_map"', 'path = "assets/studio_dark.hdr"']
        floor_material = '{ type = "pbr", albedo = [0.03, 0.03, 0.035], roughness = 0.95, metallic = 0.0 }'
        staging = "dark-room studio: the stone pops against a near-black surround"
        lights_block = None
    else:
        # Bright product staging (Blue Nile style): gray cyclorama env map for ambient +
        # backdrop + caustic seed (product_studio.hdr); with --lights, crisp angled softboxes as
        # geometric [[lights]] quads — quad reflections in facets are perfect rectangles,
        # unlike HDR strips which blur at ~0.18 deg/texel. All panel geometry scales with
        # the stone (size = 1 scene unit), and sits outside the camera frame, so only its
        # reflection is visible — like real product photography.
        bg_lines = ['type = "env_map"', 'path = "assets/product_studio.hdr"']
        # subtle Perlin-mottled floor (seamless-paper look, like Blue Nile product
        # shots): separates the floor from the smooth gradient backdrop at the horizon
        floor_material = '{ type = "noise", lo = [0.375, 0.375, 0.395], hi = [0.49, 0.49, 0.51], scale = 2.0 }'

        staging = "bright product staging: gradient cyclorama env map lights the scene + perlin-mottled floor (Blue Nile style)"
        s, d = size, dist

        def norm3(v):
            l = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
            return (v[0] / l, v[1] / l, v[2] / l)

        def quad(comment, center, normal, w, h, color):
            # light quad centred at `center`, facing `normal` (quads emit along u x v);
            # corner/u/v derive from an orthonormal basis in the panel plane
            nx, ny, nz = normal
            if abs(nz) > 0.999:
                ex = (1.0, 0.0, 0.0)
            else:
                ex = norm3((-ny, nx, 0.0))
            ey = (ny * ex[2] - nz * ex[1], nz * ex[0] - nx * ex[2], nx * ex[1] - ny * ex[0])
            cx, cyy, cz = center
            ox = cx - 0.5 * w * ex[0] - 0.5 * h * ey[0]
            oy = cyy - 0.5 * w * ex[1] - 0.5 * h * ey[1]
            oz = cz - 0.5 * w * ex[2] - 0.5 * h * ey[2]
            return [
                "[[lights]]   # " + comment,
                "corner = [%.5g, %.5g, %.5g]" % (ox, oy, oz),
                "u      = [%.5g, %.5g, %.5g]" % (w * ex[0], w * ex[1], w * ex[2]),
                "v      = [%.5g, %.5g, %.5g]" % (h * ey[0], h * ey[1], h * ey[2]),
                "color  = [%.5g, %.5g, %.5g]" % color,
                "",
            ]

        stone = (0.0, cy, 0.0)
        _lb = []  # quad list is cheap to build; emitted only with --lights
        # key overhead softbox, high enough to stay out of frame; lights the crown
        _lb += quad("key overhead softbox, high enough to stay out of frame; lights the crown",
                             (0.0, cy + 4.0 * d, 0.0), (0.0, -1.0, 0.0), 3.0 * s, 3.0 * s, (5.0, 5.0, 5.0))
        # left/right softboxes: OVERHEAD and angled down at the stone, so they stay above
        # the camera top ray at every turntable angle (never in frame) while still raking
        # crisp streaks across the crown facets
        lc = (-1.8 * s, cy + 1.8 * d, -0.6 * s)
        _lb += quad("left softbox: overhead, angled at the stone (cool streaks)",
                             lc, norm3((stone[0] - lc[0], stone[1] - lc[1], stone[2] - lc[2])),
                             1.7 * s, 1.7 * s, (4.0, 4.2, 4.6))
        rc = (1.8 * s, cy + 1.8 * d, -0.6 * s)
        _lb += quad("right softbox: overhead, angled at the stone (warm streaks)",
                             rc, norm3((stone[0] - rc[0], stone[1] - rc[1], stone[2] - rc[2])),
                             1.7 * s, 1.7 * s, (4.6, 4.2, 4.0))
        # front fill: horizontal strip high above the camera side, out of frame at any
        # orbit angle; lifts the crown facing the default camera
        _lb += quad("front fill: high strip above the camera side, out of frame at any orbit angle",
                             (0.0, cy + 3.2 * d, -3.2 * s), (0.0, -1.0, 0.0), 4.0 * s, 1.2 * s, (2.0, 2.0, 2.0))
        lights_block = _lb[:-1] if args.lights else None  # env-map-only lighting unless --lights

    def esc(t):
        return t.replace("\\", "\\\\").replace('"', '\\"')

    lines = []
    lines.append("# Scene generated by report_to_toml.py from %s" % os.path.basename(args.pdf))
    lines.append("# IGI report %s (%s): %s, %s," % (f["report"], f["date"], f["description"], f["style"]))
    if f["carat"]:
        desc = "%.6g ct, color %s, clarity %s" % (f["carat"], f["color_grade"], f["clarity"])
        if d_mm:
            desc += ", %.6g x %.6g x %.6g mm" % (l_mm, w_mm, d_mm)
        else:
            desc += ", %.6g mm diameter" % l_mm
        lines.append("# " + desc)
    if f["table_pct"] is not None:
        lines.append("# table %.6g%%, crown %.6g%% / pavilion %.6g%%, depth %.6g%%, culet %s, girdle %s"
                     % (f["table_pct"], f["crown_pct"], f["pavilion_pct"], f["depth_pct"], f["culet"], f.get("girdle") or "n/a"))
    lines.append("#")
    if args.chart:
        lines.append("# Staging: stone rests centered on a grey chart card (%.6g stone lengths wide); " % args.chart_width + staging + ";")
        lines.append("#          (needs the local greyChart OBJ+TIFF in scenes/assets; regenerate with --no-chart to skip)")
    else:
        lines.append("# Staging: stone rests on the floor (culet tip + %.6g mm gap); " % args.gap_mm + staging + ";")
    lines.append("# Lighting: environment map only" + (" plus [[lights]] softbox quads (--lights)" if args.lights else " (--lights adds geometric softboxes)") + ";")
    lines.append("# caustics on. Interactive view:  cargo run --release -- --scene scenes/" + os.path.basename(args.output))
    lines.append("# Render still:     --render --preset final --width 480 --height 300 --output out.png")
    lines.append("# Render turntable: --render --preset final --width 1280 --height 720 --output frames")
    lines.append("#                   (the [animation.orbit] block at the bottom frames the stone automatically)")
    lines.append("")
    lines.append('name = "%s %s"' % (f["report"], f["style"]))
    lines.append("")
    lines.append("[camera]")
    lines.append("look_from  = [%.5g, %.5g, %.5g]" % lf)
    lines.append("look_at    = [0.0, %.5g, 0.0]" % cy)
    lines.append("vfov       = 30.0")
    lines.append("aperture   = 0.0")
    lines.append("focus_dist = %.5g" % dist)
    lines.append("move_speed = %.5g" % (dist / 28.0))
    lines.append("")
    lines.append("[background]")
    lines.extend(bg_lines)
    lines.append("")
    lines.append("[[objects]]")
    lines.append('shape    = { type = "infinite_plane", point = [0, 0, 0], normal = [0, 1, 0] }')
    lines.append('material = %s' % floor_material)
    lines.append("")
    if args.chart:
        lines.append("[[objects]]")
        lines.append('# grey chart card the stone rests on (OBJ in cm: %s)' % args.chart_obj)
        lines.append('shape    = { type = "mesh", path = "%s" }' % args.chart_obj)
        lines.append('material = { type = "textured_pbr", albedo_path = "%s", roughness = 0.9, metallic = 0.0 }' % args.chart_tex)
        lines.append("scale    = %.6g              # card 18 cm wide -> %.6g stone lengths" % (chart_scale, args.chart_width))
        lines.append("translate = [0.0, 0.0, 0.0]  # card bottom on the floor; top at y = %.6g" % card_top)
        lines.append("")
    if lights_block:
        lines.extend(lights_block)
        lines.append("")
    lines.append("")
    lines.append("[[gems]]")
    lines.append('report          = "%s"' % esc('%s (IGI %s, %s)' % (f["report"], f["date"], f["description"])))
    if f["carat"]:
        lines.append("carat           = %.6g" % f["carat"])
    if meas:
        note = "   # L x W x depth, mm" + (" -> oval" if oval else "")
        lines.append("measurements_mm = [%.6g, %.6g, %.6g]%s" % (l_mm, w_mm, d_mm, note))
    if f["table_pct"] is not None:
        lines.append("table_pct       = %.6g" % f["table_pct"])
        lines.append("crown_pct       = %.6g                 # %% of minor axis" % f["crown_pct"])
        lines.append("pavilion_pct    = %.6g" % f["pavilion_pct"])
        lines.append("depth_pct       = %.6g" % f["depth_pct"])
        gp = GIRDLE_TO_PCT.get(f.get("girdle"), 2.0)
        lines.append("girdle_pct       = %.6g                 # girdle band, %% of diameter (report: %s)" % (gp, f.get("girdle") or "n/a"))
    if oval and cut == "brilliant":
        lines.append("girdle_facets   = 64      # smooth oval outline (16 would show polygon edges)")
    lines.append('color_grade     = "%s"' % f["color_grade"])
    lines.append('cut             = "%s"' % cut)
    lines.append('material        = "%s"' % material)
    if args.chart:
        lines.append("center          = [0.0, %.5g, 0.0]   # pavilion (%.2f mm) + %.6g mm gap on chart top y=%.6g" % (cy, pav_mm, args.gap_mm, card_top))
    else:
        lines.append("center          = [0.0, %.5g, 0.0]   # pavilion (%.2f mm) + %.6g mm gap above the floor" % (cy, pav_mm, args.gap_mm))
    lines.append("scale_mm        = %.6g               # mm -> scene units (stone length = 1)" % scale)
    lines.append("")
    if not args.no_caustics:
        lines.append("[caustics]")
        lines.append("enabled       = true")
        lines.append("gather_radius = %.5g" % (1.2 * w_mm * scale))
        lines.append("num_photons   = 300000")
        lines.append("")

    if not args.no_animation:
        # Auto-framed 360-degree orbit: radius/height/focus_dist all derive from the
        # camera distance `dist` = 3.2 x stone length, so with vfov = 30 deg the gem
        # fills the same fraction of the frame whatever its measured size.
        lines.append("# --- Animation: auto-framed 360-degree orbit (scales with the stone) ---")
        lines.append("# radius = camera distance, height = still-camera elevation, look_at = girdle centre:")
        lines.append("# the framing is identical for every stone. Turntable render command is in the header.")
        lines.append("[animation]")
        lines.append("fps      = 24")
        lines.append("duration = 4.0   # 96 frames")
        lines.append("")
        lines.append("[animation.orbit]")
        lines.append("center      = [0.0, 0.0, 0.0]    # orbit axis (XZ only; Y comes from `height`)")
        lines.append("radius      = %.5g              # = camera distance: constant framing" % dist)
        lines.append("height      = %.5g              # same elevation as the still camera" % (cy + 0.26 * dist))
        lines.append("look_at     = [0.0, %.5g, 0.0]  # the stone (girdle centre)" % cy)
        lines.append("start_angle = 180.0              # start on the default camera side (-Z)")
        lines.append("end_angle   = 540.0              # full 360-degree orbit")
        lines.append("vfov        = 30.0")
        lines.append("focus_dist  = %.5g" % dist)
        lines.append("")
    return "\n".join(lines), warnings, dict(cut=cut, material=material, oval=oval)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="IGI/GIA report PDF -> rustracer [[gems]] scene TOML")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--output", default=None, help="output .toml (default: scenes/<report>.toml next to this script)")
    ap.add_argument("--material", default=None, help="override material preset (diamond|sapphire|ruby|emerald)")
    ap.add_argument("--cut", default=None, help="override cut preset (brilliant|step|princess|...)")
    ap.add_argument("--gap-mm", type=float, default=0.1, help="air gap under the stone, mm (default 0.1: culet tip resting)")
    ap.add_argument("--no-chart", dest="chart", action="store_false",
                    help="no grey chart card under the stone (default: card on, stone resting centered on it)")
    ap.add_argument("--lights", action="store_true",
                    help="add geometric [[lights]] softbox quads on top of env-map lighting (default: env map only)")
    ap.add_argument("--chart-width", type=float, default=4.0,
                    help="chart card width in stone lengths (default 4.0)")
    ap.add_argument("--chart-obj", default="assets/greyChart_OBJ.obj", help="grey chart OBJ path (cm units)")
    ap.add_argument("--chart-tex", default="assets/greyChart_default_default_COL_sRGB.tiff",
                    help="grey chart albedo TIFF (sRGB, decoded to linear at sample time)")
    ap.add_argument("--no-caustics", action="store_true")
    ap.add_argument("--no-animation", action="store_true",
                    help="do not emit the [animation] / [animation.orbit] turntable block")
    ap.add_argument("--dark", action="store_true",
                    help="dark-room studio staging (default: bright product staging, Blue Nile style)")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit("error: %s not found" % args.pdf)
    text = extract_pdf_text(args.pdf)
    if len(text) < 200:
        raise SystemExit("error: could not extract text from %s "
                         "(got %d chars; scanned/image-only PDF?)" % (args.pdf, len(text)))
    f, warnings = parse_report(text)
    if args.output is None:
        scenes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes")
        args.output = os.path.join(scenes_dir, os.path.splitext(os.path.basename(args.pdf))[0] + ".toml")
    toml, warnings2, info = render_toml(f, args)
    warnings += warnings2

    with open(args.output, "w", encoding="utf-8") as fh:  # platform line endings (CRLF on Windows)
        fh.write(toml)

    print("report   : %s (%s)" % (f["report"], f["date"]))
    print("stone    : %s / %s  ->  cut = %s, material = %s" % (f["description"], f["style"], info["cut"], info["material"]))
    if f["carat"]:
        print("weight   : %.6g ct" % f["carat"])
    if f["measurements_mm"]:
        print("measured : %s mm%s" % (f["measurements_mm"], " (oval)" if info["oval"] else ""))
    if f["table_pct"] is not None:
        print("props    : table %.6g%%  crown %.6g%%  pavilion %.6g%%  depth %.6g%%  culet %s"
              % (f["table_pct"], f["crown_pct"], f["pavilion_pct"], f["depth_pct"], f["culet"]))
    print("grades   : color %s, clarity %s" % (f["color_grade"], f["clarity"]))
    for w in warnings:
        print("warning  : %s" % w, file=sys.stderr)
    print("wrote    : %s" % args.output)

if __name__ == "__main__":
    main()
