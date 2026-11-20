"""Generate assets/studio_softbox.hdr: a procedural "jewelry store" environment map.

Equirectangular Radiance RGBE (RLE), 1024x512, v=0 = zenith (+Y), u wraps azimuth.
Vertical gradient base plus four softbox panels with soft edges (key overhead,
cool left, warm right, blue back rim) so a gem reflects bright rectangles.
Run from the repo root:  py make_studio_softbox.py
"""
import math
import os

W, H = 1024, 512
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "studio_softbox.hdr")
COMPARE = os.path.join(HERE, "assets", "sunset_puresky_4k.hdr")


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


# softbox panels: (cu, cv, hu, hv, radiance, tint)
PANELS = [
    (0.50, 0.10, 0.20, 0.11, 40.0, (1.00, 1.00, 1.00)),  # key overhead
    (0.10, 0.50, 0.11, 0.24, 14.0, (0.90, 0.95, 1.10)),  # left, cool
    (0.90, 0.48, 0.11, 0.24, 14.0, (1.10, 1.00, 0.90)),  # right, warm
    (0.50, 0.40, 0.26, 0.13, 7.0, (0.85, 0.92, 1.10)),   # back rim, blue-ish
]


def build_rows():
    rows = []
    for j in range(H):
        v = (j + 0.5) / H
        # base gradient: brighter upper hemisphere, dim floor
        if v < 0.5:
            base = 0.04 + 0.45 * (1.0 - 2.0 * abs(v - 0.25)) ** 1.5
        else:
            base = 0.03 + 0.02 * (2.0 - 2.0 * v)
        base = max(0.02, base)
        row = bytearray()
        for i in range(W):
            u = (i + 0.5) / W
            r = g = b = base * (1.0 - 0.03 * abs(v - 0.5) * 2.0)
            for (cu, cv, hu, hv, rad, tint) in PANELS:
                w = rect(u, v, cu, cv, hu, hv, 0.35)
                if w > 0.0:
                    r += rad * tint[0] * w
                    g += rad * tint[1] * w
                    b += rad * tint[2] * w
            row += rgbe(r, g, b)
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
    write_hdr(OUT, build_rows())
    w, h, mean, mx = stats(OUT)
    print("studio_softbox.hdr: %dx%d  mean lum %.3f  max lum %.1f" % (w, h, mean, mx))
    if os.path.exists(COMPARE):
        w, h, mean, mx = stats(COMPARE)
        print("sunset_puresky_4k.hdr: %dx%d  mean lum %.3f  max lum %.1f" % (w, h, mean, mx))
