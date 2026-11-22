"""Generate scenes/assets/studio_softbox.hdr: a procedural "light tent" environment map.

Equirectangular Radiance RGBE (RLE), 2048x1024, v=0 = zenith (+Y), u wraps azimuth.
Near-uniform bright base (zenith ~2.4 -> nadir ~0.8) so the visible background is
an even light from all sides; the four softbox positions only add very soft,
low-contrast glows (no hard rectangle edges, nothing pixellated). Directional
lighting and caustics come from this map. A warm global tint (TINT) sets the
color temperature.
Run from the repo root:  py make_studio_softbox.py           # bright white-tent map
                         py make_studio_softbox.py --dark    # same softboxes, near-black room
                                                            # (studio_dark.hdr) for gem scenes
"""
import math
import os
import sys

W, H = 2048, 1024
HERE = os.path.dirname(os.path.abspath(__file__))
DARK = "--dark" in sys.argv
OUT = os.path.join(HERE, "scenes", "assets", "studio_dark.hdr" if DARK else "studio_softbox.hdr")
COMPARE = os.path.join(HERE, "scenes", "assets", "sunset_puresky_4k.hdr")


def rgbe(r, g, b):
    v = max(r, g, b)
    if v < 1e-32:
        return bytes((0, 0, 0, 0))
    m, e = math.frexp(v)
    s = m * 256.0 / v
    return bytes((min(255, int(r * s)), min(255, int(g * s)), min(255, int(b * s)), e + 128))


def smoothstep(d, r):
    # d = normalized distance from rect centre (half-size maps to 0.5), r = softness
    t = max(0.0, min(1.0, (0.5 + r - d) / (2.0 * r)))
    return t * t * (3.0 - 2.0 * t)


def rect(u, v, cu, cv, hu, hv, soft):
    du = abs(((u - cu + 0.5) % 1.0) - 0.5) / hu  # wrap-aware in u
    dv = abs(v - cv) / hv
    d = max(du, dv) / 0.5
    return smoothstep(d, soft)


# soft glows at the softbox positions: (cu, cv, hu, hv, radiance, tint)
# soft=0.9 makes the edge wider than the panel itself -> smooth blob, no square.
PANELS = [
    (0.50, 0.10, 0.22, 0.13, 12.0, (1.00, 1.00, 1.00)),  # key overhead glow
    (0.10, 0.50, 0.13, 0.26,  6.0, (0.90, 0.95, 1.10)),  # left, cool
    (0.90, 0.48, 0.13, 0.26,  6.0, (1.10, 1.00, 0.90)),  # right, warm
    (0.50, 0.42, 0.30, 0.15,  4.0, (0.85, 0.92, 1.10)),  # back rim, blue-ish
]
SOFT = 0.9

# Global color-temperature tint applied to everything: warm (~3500 K look),
# R up / B down relative to neutral daylight.
TINT = (1.12, 1.00, 0.84)


def build_rows(dark=False):
    rows = []
    for j in range(H):
        v = (j + 0.5) / H
        # base gradient: bright "white tent" by default (zenith ~2.4 -> nadir ~0.8);
        # --dark keeps only a faint room (zenith ~0.065 -> nadir ~0.015) so the visible
        # background goes dark and the stone pops — the softbox panels still light it.
        base = (0.015 + 0.05 * (1.0 - v) ** 1.5) if dark else (0.8 + 1.6 * (1.0 - v) ** 1.5)
        row = bytearray()
        for i in range(W):
            u = (i + 0.5) / W
            r = g = b = base
            for (cu, cv, hu, hv, rad, tint) in PANELS:
                w = rect(u, v, cu, cv, hu, hv, SOFT)
                if w > 0.0:
                    r += rad * tint[0] * w
                    g += rad * tint[1] * w
                    b += rad * tint[2] * w
            row += rgbe(r * TINT[0], g * TINT[1], b * TINT[2])
        rows.append(bytes(row))
    return rows


def rle_channel(data):
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        run = 1
        while i + run < n and run < 127 and data[i + run] == data[i]:
            run += 1
        if run >= 2:
            out += bytes((128 + run, data[i]))
            i += run
        else:
            start = i
            i += 1
            while i < n and i - start < 127:
                run = 1
                while i + run < n and run < 127 and data[i + run] == data[i]:
                    run += 1
                if run >= 2:
                    break
                i += 1
            out += bytes((i - start,)) + data[start:i]
    return bytes(out)


def write_hdr(path, rows):
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(("-Y %d +X %d\n" % (H, W)).encode())
        for row in rows:
            assert len(row) == W * 4
            f.write(bytes((2, 2, (W >> 8) & 0xFF, W & 0xFF)))
            for ch in range(4):
                f.write(rle_channel(row[ch::4]))


def read_hdr(path):
    with open(path, "rb") as f:
        data = f.read()
    end = data.index(b"\n\n")
    pos = end + 2
    line = data[pos:data.index(b"\n", pos)].decode()
    pos = data.index(b"\n", pos) + 1
    parts = line.split()
    h = int(parts[1])
    w = int(parts[3])
    img = []
    for _ in range(h):
        assert data[pos] == 2 and data[pos + 1] == 2
        sw = (data[pos + 2] << 8) | data[pos + 3]
        pos += 4
        chans = []
        for _c in range(4):
            chan = bytearray()
            while len(chan) < sw:
                cnt = data[pos]
                pos += 1
                if cnt > 128:
                    chan += bytes((data[pos],)) * (cnt - 128)
                    pos += 1
                else:
                    chan += data[pos:pos + cnt]
                    pos += cnt
            chans.append(bytes(chan))
        row = []
        for i in range(sw):
            r, g, b, e = chans[0][i], chans[1][i], chans[2][i], chans[3][i]
            if e == 0:
                row.append((0.0, 0.0, 0.0))
            else:
                s = math.ldexp(1.0, e - 136)
                row.append((r * s, g * s, b * s))
        img.append(row)
    return w, h, img


def stats(path):
    w, h, img = read_hdr(path)
    tot = 0.0
    mx = 0.0
    n = w * h
    for row in img:
        for (r, g, b) in row:
            l = 0.2126 * r + 0.7152 * g + 0.0722 * b
            tot += l
            if l > mx:
                mx = l
    return w, h, tot / n, mx


if __name__ == "__main__":
    write_hdr(OUT, build_rows(DARK))
    w, h, mean, mx = stats(OUT)
    print("%s: %dx%d  mean lum %.3f  max lum %.1f" % (os.path.basename(OUT), w, h, mean, mx))
    if os.path.exists(COMPARE):
        w, h, mean, mx = stats(COMPARE)
        print("sunset_puresky_4k.hdr: %dx%d  mean lum %.3f  max lum %.1f" % (w, h, mean, mx))
