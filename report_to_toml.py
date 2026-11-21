#!/usr/bin/env python3
"""report_to_toml.py — turn an IGI/GIA grading-report PDF into a renderable
rustracer scene TOML with a [[gems]] block.

Usage:
    py report_to_toml.py REPORT.pdf [-o out.toml] [--material PRESET] [--cut CUT]
                                [--gap-mm 15.0] [--no-caustics]

    py report_to_toml.py LG625469749.pdf            # writes LG625469749.toml

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
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}", text)
    f["date"] = m.group(0) if m else "unknown date"
    f["measurements_mm"] = parse_measurements(text, warnings)
    return f, warnings

# ── TOML emission ────────────────────────────────────────────────────────────

def render_toml(f, args):
    warnings = []
    cut = args.cut or cut_from_style(f["style"], warnings)
    material = args.material or material_from_description(f["description"], warnings)

    scale = 0.001  # mm -> metres
    meas = f["measurements_mm"]
    if meas:
        l_mm, w_mm, d_mm = meas
    elif f["carat"]:
        l_mm = w_mm = (f["carat"] / 0.00355) ** (1.0 / 3.0)
        d_mm = None
    else:
        raise SystemExit("error: neither measurements nor carat weight found in the report")

    # "oval" only matters for brilliant cuts (smooth-ellipse girdle); step cuts
    # are rectangular by design, so L != W there is not an oval.
    oval = cut == "brilliant" and l_mm / w_mm > 1.05
    # pavilion height in mm = W * pavilion_pct/100 (see build_gem angle conversion)
    pav_mm = w_mm * (f["pavilion_pct"] or 41.0) / 100.0
    cy = (pav_mm + args.gap_mm) * scale          # girdle centre height above the floor
    size = l_mm * scale                            # stone length in scene units
    dist = max(3.2 * size, 0.012)                  # camera distance
    lf = (0.43 * dist, cy + 0.26 * dist, -0.86 * dist)  # same view direction as gem_LG625469749

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
        lines.append("# table %.6g%%, crown %.6g%% / pavilion %.6g%%, depth %.6g%%, culet %s"
                     % (f["table_pct"], f["crown_pct"], f["pavilion_pct"], f["depth_pct"], f["culet"]))
    lines.append("#")
    lines.append("# Staging: stone floats above a dark floor in the studio light-tent env map;")
    lines.append("# caustics on. Interactive view:  cargo run --release -- --scene " + os.path.basename(args.output))
    lines.append("# Render still: --render --preset final --width 480 --height 300 --output out.png")
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
    lines.append('type = "env_map"')
    lines.append('path = "assets/studio_softbox.hdr"')
    lines.append("")
    lines.append("[[objects]]")
    lines.append('shape    = { type = "infinite_plane", point = [0, 0, 0], normal = [0, 1, 0] }')
    lines.append('material = { type = "pbr", albedo = [0.04, 0.04, 0.05], roughness = 0.25, metallic = 0.0 }')
    lines.append("")
    lines.append("[[gems]]")
    lines.append('report          = "%s (IGI %s, %s)"' % (f["report"], f["date"], f["description"]))
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
    if oval and cut == "brilliant":
        lines.append("girdle_facets   = 64      # smooth oval outline (16 would show polygon edges)")
    lines.append('color_grade     = "%s"' % f["color_grade"])
    lines.append('cut             = "%s"' % cut)
    lines.append('material        = "%s"' % material)
    lines.append("center          = [0.0, %.5g, 0.0]   # pavilion (%.2f mm) + %.6g mm air gap" % (cy, pav_mm, args.gap_mm))
    lines.append("scale_mm        = 0.001                # mm -> metres")
    lines.append("")
    if not args.no_caustics:
        lines.append("[caustics]")
        lines.append("enabled       = true")
        lines.append("gather_radius = %.5g" % (1.2 * w_mm * scale))
        lines.append("num_photons   = 300000")
        lines.append("")
    return "\n".join(lines), warnings, dict(cut=cut, material=material, oval=oval)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="IGI/GIA report PDF -> rustracer [[gems]] scene TOML")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--output", default=None, help="output .toml (default: <report>.toml next to the PDF)")
    ap.add_argument("--material", default=None, help="override material preset (diamond|sapphire|ruby|emerald)")
    ap.add_argument("--cut", default=None, help="override cut preset (brilliant|step|princess|...)")
    ap.add_argument("--gap-mm", type=float, default=15.0, help="air gap under the stone, mm (default 15)")
    ap.add_argument("--no-caustics", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        raise SystemExit("error: %s not found" % args.pdf)
    text = extract_pdf_text(args.pdf)
    if len(text) < 200:
        raise SystemExit("error: could not extract text from %s "
                         "(got %d chars; scanned/image-only PDF?)" % (args.pdf, len(text)))
    f, warnings = parse_report(text)
    args.output = args.output or os.path.splitext(args.pdf)[0] + ".toml"
    toml, warnings2, info = render_toml(f, args)
    warnings += warnings2

    with open(args.output, "w") as fh:  # platform line endings (CRLF on Windows)
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
