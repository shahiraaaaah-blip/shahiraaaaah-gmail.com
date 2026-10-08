#!/usr/bin/env python3
"""Podcast Tentera Darat - cinematic microphone -> logo reveal (16:9, 7s).

Procedural render: numpy + PIL frames piped to ffmpeg, with synthesized audio.
The logo (logo.png) is never redrawn or altered; only scale, glow and a brief
pixel/glitch build-in are applied until it locks at full fidelity.

Usage:
  python3 video/render_reveal.py out.mp4
  python3 video/render_reveal.py --stills OUTDIR 0.8 2.5 3.8 4.6 4.85 6.5
"""
import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(HERE, "..", "logo.png")

W, H = 1920, 1080
CX, CY = 960, 540
FPS = 30
DUR = 7.0
NF = int(round(FPS * DUR))
T_LOGO = 3.0      # logo starts materializing
T_IMP = 4.8       # bass impact / lock
LOGO_FINAL = 860  # px
PULSES = [1.65, 2.05, 2.40, 2.70, 2.95, 3.17, 3.37, 3.55, 3.72, 3.88,
          4.02, 4.15, 4.27, 4.38, 4.48, 4.57, 4.65, 4.72]
GOLD = np.array([1.0, 0.80, 0.42], np.float32)
WGOLD = np.array([1.0, 0.93, 0.75], np.float32)


def clip01(x):
    return min(max(x, 0.0), 1.0)


def smooth(x):
    x = clip01(x)
    return x * x * (3 - 2 * x)


def p_flick(t):
    if t < 0.25 or t > 4.8:
        return 0.0
    return 0.25 + 0.5 * smooth((t - 1.0) / 1.6)


def is_flick(fi):
    return np.random.default_rng(1000 + fi).random() < p_flick(fi / FPS)


def cam_s(t):
    if t < 3.0:
        return 1.15 + 0.30 * (t / 3.0)
    if t < T_IMP:
        u = (t - 3.0) / 1.8
        return 1.45 + 0.50 * u * u
    if t < 5.3:
        u = (t - T_IMP) / 0.5
        return 1.95 + (1.25 - 1.95) * (1 - (1 - u) ** 3)
    return 1.25 + 0.05 * (t - 5.3) / 1.7


def pulse_env(t):
    e = 0.0
    for p in PULSES:
        if p <= t:
            e = max(e, math.exp(-(t - p) * 9.0))
    if t >= T_IMP:
        e = max(e, 1.4 * math.exp(-(t - T_IMP) * 5.0))
    return e


# --------------------------------------------------------------------------
# Procedural microphone sprite
# --------------------------------------------------------------------------
K = 2.0
SX0, SX1, SY0, SY1 = 90, 810, 100, 1300  # logical crop of the sprite
ANCH = (450.0, 500.0)                    # capsule centre (logical)


def build_mic():
    sw, sh = int((SX1 - SX0) * K), int((SY1 - SY0) * K)
    xs = SX0 + (np.arange(sw, dtype=np.float32) + 0.5) / K
    ys = SY0 + (np.arange(sh, dtype=np.float32) + 0.5) / K
    X, Y = np.meshgrid(xs, ys)
    rgb = np.zeros((sh, sw, 3), np.float32)
    a = np.zeros((sh, sw), np.float32)
    rg = np.random.default_rng(3)
    rown = rg.random(sh).astype(np.float32)[:, None]
    coln = rg.random(sw).astype(np.float32)[None, :]
    Ld = np.array([-0.45, -0.55, 0.70], np.float32)
    Ld /= np.linalg.norm(Ld)
    Hv = Ld + np.array([0, 0, 1], np.float32)
    Hv /= np.linalg.norm(Hv)
    steel = np.array([0.70, 0.72, 0.74], np.float32)

    def shade(nx, ny, nz, brushed=0.0):
        d = np.clip(nx * Ld[0] + ny * Ld[1] + nz * Ld[2], 0, 1)
        s = np.clip(nx * Hv[0] + ny * Hv[1] + nz * Hv[2], 0, 1) ** 48
        rimr = np.clip(nx, 0, 1) ** 3
        riml = np.clip(-nx, 0, 1) ** 3
        c = (steel * (0.10 + 0.65 * d)[..., None]
             + (s * 0.95)[..., None]
             + rimr[..., None] * np.array([1.0, 0.75, 0.35], np.float32) * 0.55
             + riml[..., None] * np.array([0.35, 0.55, 0.35], np.float32) * 0.25)
        if brushed:
            c = c * (1 + brushed * (rown - 0.5) * 0.25)[..., None]
        return c

    def over(c, al):
        nonlocal rgb, a
        al = np.clip(al, 0, 1)
        rgb = c * al[..., None] + rgb * (1 - al[..., None])
        a = al + a * (1 - al)

    # stand pole
    nx = np.clip((X - 450) / 26, -1, 1)
    nz = np.sqrt(1 - nx ** 2)
    al = np.clip(0.5 - (np.abs(X - 450) - 26) * K, 0, 1) * (Y > 930)
    al *= np.clip((1300 - Y) / 200, 0, 1)
    over(shade(nx, nx * 0, nz, 1.0) * 0.85, al)
    # collar
    nx = np.clip((X - 450) / 120, -1, 1)
    nz = np.sqrt(1 - nx ** 2)
    al = np.clip(0.5 - (np.abs(X - 450) - 120) * K, 0, 1) * (Y > 860) * (Y < 960)
    c = shade(nx, nx * 0, nz, 1.0)
    c *= (1 - 0.7 * np.exp(-((Y - 925) / 3) ** 2))[..., None]
    over(c, al)
    # yoke (U cradle)
    dist = np.hypot(X - 450, Y - 580)
    nx = np.clip((dist - 352) / 10, -1, 1)
    nz = np.sqrt(1 - nx ** 2)
    al = np.clip(0.5 - (np.abs(dist - 352) - 10) * K, 0, 1) * (Y >= 580)
    over(shade(nx, nx * 0, nz, 0.5) * 0.8, al)
    for sx in (450 - 352, 450 + 352):
        dk = np.hypot(X - sx, Y - 580)
        nx = np.clip((X - sx) / 22, -1, 1)
        ny = np.clip((Y - 580) / 22, -1, 1)
        nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
        over(shade(nx, ny, nz), np.clip(0.5 - (dk - 22) * K, 0, 1))
    # capsule (stadium)
    dyc = Y - np.clip(Y, 420, 580)
    d = np.hypot(X - 450, dyc) - 300
    al = np.clip(0.5 - d * K, 0, 1)
    nx = np.clip((X - 450) / 300, -1, 1)
    ny = np.clip(dyc / 300, -1, 1)
    nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
    c = shade(nx, ny, nz, 1.0)
    # perforated grille
    sq = np.sqrt(np.clip(1 - ny ** 2, 0.02, 1))
    lon = np.arcsin(np.clip(nx / sq, -0.999, 0.999))
    lat = np.arcsin(ny)
    u = lon * 300 * sq
    v = np.where(Y < 420, 420 + lat * 300, np.where(Y > 580, 580 + lat * 300, Y))
    pitch = 21.0
    rh = pitch * 0.866
    row = np.floor(v / rh)
    off = (row % 2) * pitch / 2
    px = np.mod(u + off, pitch) - pitch / 2
    py = np.mod(v, rh) - rh / 2
    dd = np.hypot(px, py)
    hole = np.clip((6.4 + 0.5 / K - dd) * K, 0, 1) * (Y < 690)
    mesh = np.where(Y < 690, 0.88, 1.0)
    c = c * mesh[..., None]
    inner = np.array([0.015, 0.02, 0.015], np.float32) + 0.05 * np.clip(
        1 - dd / 6.4, 0, 1)[..., None] * np.array([0.9, 0.8, 0.5], np.float32)
    c = c * (1 - 0.94 * hole)[..., None] + inner * (0.94 * hole)[..., None]
    c *= (1 - 0.30 * np.clip((Y - 790) / 90, 0, 1))[..., None]
    # engraved seam
    c *= (1 - 0.6 * np.exp(-((Y - 800) / 1.2) ** 2))[..., None]
    over(c, al)
    # polished steel band
    nxb = np.clip((X - 450) / 308, -1, 1)
    nzb = np.sqrt(1 - nxb ** 2)
    alb = np.clip(0.5 - (np.abs(X - 450) - 308) * K, 0, 1) * np.clip(
        0.5 - (np.abs(Y - 710) - 20) * K, 0, 1)
    cb = shade(nxb, nxb * 0, nzb, 0.3) * 1.15
    cb *= (1 - 0.7 * np.exp(-((np.abs(Y - 710) - 19) / 1.2) ** 2))[..., None]
    cb += 0.12 * np.exp(-((Y - 700) / 4) ** 2)[..., None]
    over(cb, alb)
    # top key light from the beam
    rgb += (0.10 * np.clip(1 - (Y - 120) / 520, 0, 1))[..., None] * \
        np.array([1.0, 0.85, 0.55], np.float32) * (a > 0.5)[..., None]
    img = np.dstack([np.clip(rgb, 0, 1), a])
    return Image.fromarray((img * 255).astype(np.uint8), "RGBA")


# --------------------------------------------------------------------------
# Static assets (built once, shared with workers through fork)
# --------------------------------------------------------------------------
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)
D = np.hypot(XX - CX, YY - CY).astype(np.float32)
TH = np.arctan2(YY - CY, XX - CX).astype(np.float32)
NB = 120
BIN = (((TH + math.pi) / (2 * math.pi)) * NB).astype(np.float32)
BIDX = np.minimum(BIN.astype(np.int32), NB - 1)
BFRAC = BIN - np.floor(BIN)


def make_bg():
    r = np.hypot((XX - CX) / 1150, (YY - CY) / 720)
    base = np.array([0.040, 0.062, 0.034], np.float32)
    g = np.clip(1 - r, 0, 1) ** 1.6
    bg = base * g[..., None]
    # floor sheen
    fl = np.clip((YY - 700) / 380, 0, 1) * np.exp(-((XX - CX) / 800) ** 2)
    bg += fl[..., None] * np.array([0.02, 0.03, 0.015], np.float32)
    # faint grid
    gx = (np.mod(XX, 120) < 1.2) | (np.mod(YY, 120) < 1.2)
    bg += gx[..., None] * g[..., None] * np.array([0.010, 0.016, 0.009], np.float32)
    return bg.astype(np.float32)


def make_haze(seed):
    rg = np.random.default_rng(seed)
    sm = rg.random((9, 16)).astype(np.float32)
    im = Image.fromarray((sm * 255).astype(np.uint8)).resize((W * 2, H), Image.BICUBIC)
    return np.asarray(im, np.float32) / 255


def make_beam():
    wd = 170 + YY * 0.22
    b = np.exp(-((XX - CX) / wd) ** 2) * np.clip(1 - YY / (H * 1.05), 0, 1) ** 1.3
    return b.astype(np.float32)


def make_hud():
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    m = 54
    ln, wd = 110, 4
    for (x, y, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1),
                           (W - m, H - m, -1, -1)):
        dr.line([(x, y), (x + sx * ln, y)], fill=255, width=wd)
        dr.line([(x, y), (x, y + sy * ln)], fill=255, width=wd)
    dr.rectangle([m + 20, m + 20, W - m - 20, H - m - 20], outline=40, width=1)
    # compass ring with NATO-style graduations
    R = 505
    dr.ellipse([CX - R, CY - R, CX + R, CY + R], outline=70, width=1)
    for deg in range(0, 360, 3):
        a = math.radians(deg)
        ln2 = 30 if deg % 90 == 0 else (18 if deg % 15 == 0 else 8)
        x0, y0 = CX + (R - 1) * math.cos(a), CY + (R - 1) * math.sin(a)
        x1, y1 = CX + (R - 1 - ln2) * math.cos(a), CY + (R - 1 - ln2) * math.sin(a)
        dr.line([(x0, y0), (x1, y1)], fill=150 if deg % 15 == 0 else 90, width=1)
    # edge rulers
    for side in (m, W - m):
        for y in range(200, 881, 24):
            major = ((y - 200) // 24) % 5 == 0
            l2 = 22 if major else 10
            dr.line([(side, y), (side + (l2 if side == m else -l2), y)], fill=170 if major else 100, width=1)
    for (x, y) in ((220, 190), (W - 220, 190), (220, H - 190), (W - 220, H - 190)):
        dr.line([(x - 10, y), (x + 10, y)], fill=140, width=1)
        dr.line([(x, y - 10), (x, y + 10)], fill=140, width=1)
    dr.polygon([(CX, 22), (CX - 9, 8), (CX + 9, 8)], fill=200)
    dr.polygon([(CX, H - 22), (CX - 9, H - 8), (CX + 9, H - 8)], fill=200)
    return np.asarray(im, np.float32) / 255


def make_vignette():
    r = np.hypot((XX - CX) / 1000, (YY - CY) / 640)
    return (1 - 0.62 * np.clip(r - 0.35, 0, 1) ** 1.5).astype(np.float32)


LOGO = Image.open(LOGO_PATH).convert("RGBA")
MIC = build_mic()
BG = make_bg()
HAZE1, HAZE2 = make_haze(11), make_haze(12)
BEAM = make_beam()
HUD = make_hud()
VIG = make_vignette()
FIXR = np.random.default_rng(99).random((1024, 1024)).astype(np.float32)

# particles ------------------------------------------------------------------
_pr = np.random.default_rng(21)


def _mk_particles():
    out = []
    # ambient sparks
    n = 700
    u = _pr.random(n)
    birth = np.where(u < 0.15, _pr.uniform(0.5, 1.5, n),
                     np.where(u < 0.82, _pr.uniform(1.5, 4.8, n), _pr.uniform(5.0, 6.8, n)))
    ang = _pr.uniform(0, 2 * np.pi, n)
    r0 = _pr.uniform(300, 420, n)
    sp = _pr.uniform(40, 260, n)
    out.append(dict(kind=0, birth=birth, life=_pr.uniform(1.2, 3.0, n),
                    x=CX + r0 * np.cos(ang), y=CY + r0 * np.sin(ang) * 1.05,
                    vx=sp * np.cos(ang), vy=sp * np.sin(ang) - 25, k=np.full(n, 0.8),
                    b=_pr.uniform(0.4, 1.0, n), w=_pr.random(n)))
    # converging "digital" particles that build the logo
    n = 600
    ang = _pr.uniform(0, 2 * np.pi, n)
    r0 = _pr.uniform(650, 1100, n)
    r1 = np.sqrt(_pr.random(n)) * 430
    out.append(dict(kind=1, birth=_pr.uniform(T_LOGO - 0.1, T_IMP - 0.5, n),
                    life=_pr.uniform(0.6, 1.1, n), ang=ang, r0=r0, r1=r1,
                    b=_pr.uniform(0.5, 1.0, n), w=_pr.random(n)))
    # impact burst
    n = 1100
    ang = _pr.uniform(0, 2 * np.pi, n)
    r0 = _pr.uniform(380, 450, n)
    sp = _pr.uniform(200, 1500, n)
    out.append(dict(kind=0, birth=np.full(n, T_IMP) + _pr.uniform(0, 0.04, n),
                    life=_pr.uniform(0.7, 2.2, n),
                    x=CX + r0 * np.cos(ang), y=CY + r0 * np.sin(ang),
                    vx=sp * np.cos(ang), vy=sp * np.sin(ang), k=np.full(n, 2.2),
                    b=_pr.uniform(0.5, 1.0, n), w=_pr.random(n)))
    return out


PARTS = _mk_particles()


# --------------------------------------------------------------------------
# Frame renderer
# --------------------------------------------------------------------------
def splat(layer, xs, ys, vals):
    """Bilinear additive splat of (N,3) vals at float positions."""
    ok = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
    xs, ys, vals = xs[ok], ys[ok], vals[ok]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    fx = (xs - x0).astype(np.float32)[:, None]
    fy = (ys - y0).astype(np.float32)[:, None]
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(layer, (y0 + dy, x0 + dx), vals * wgt)


def particle_layer(t):
    layer = np.zeros((H, W, 3), np.float32)
    for P in PARTS:
        tau = t - P["birth"]
        alive = (tau >= 0) & (tau <= P["life"])
        if not alive.any():
            continue
        idx = np.nonzero(alive)[0]
        tt = tau[idx]
        u = tt / P["life"][idx]
        if P["kind"] == 0:
            k = P["k"][idx]
            f = (1 - np.exp(-k * tt)) / k
            xs = P["x"][idx] + P["vx"][idx] * f
            ys = P["y"][idx] + P["vy"][idx] * f
            fade = (1 - u) ** 1.5
        else:
            e = u * u * (3 - 2 * u)
            r = P["r0"][idx] + (P["r1"][idx] - P["r0"][idx]) * e
            xs = CX + r * np.cos(P["ang"][idx])
            ys = CY + r * np.sin(P["ang"][idx])
            fade = np.sin(np.pi * u) ** 0.8
        tw = 0.6 + 0.4 * np.sin(t * 40 + P["w"][idx] * 50)
        br = (P["b"][idx] * fade * tw * 1.6).astype(np.float32)
        col = GOLD[None, :] * (1 - P["w"][idx][:, None] * 0.5) + \
            WGOLD[None, :] * (P["w"][idx][:, None] * 0.5)
        splat(layer, xs, ys, col * br[:, None])
    return layer


def lightning_layer(t, rg):
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    s = cam_s(t)
    n = int(rg.integers(2, 5))
    for _ in range(n):
        a = rg.uniform(0, 2 * math.pi)
        r = 300 * s + rg.uniform(-10, 10)
        x, y = CX + r * math.cos(a) * 1.0, CY + r * math.sin(a) * 1.0
        pts = [(x, y)]
        for _ in range(int(rg.integers(5, 11))):
            a += rg.normal(0, 0.55)
            x += math.cos(a) * rg.uniform(8, 24) * 1.2
            y += math.sin(a) * rg.uniform(8, 24) * 1.2
            pts.append((x, y))
        dr.line(pts, fill=int(rg.uniform(170, 255)), width=int(rg.integers(1, 3)))
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    return np.asarray(im, np.float32) / 255


def logo_layer(t, rg):
    """Returns (rgb HxWx3, alpha HxW, Lc, p) or None."""
    if t < T_LOGO:
        return None
    p = clip01((t - T_LOGO) / (T_IMP - T_LOGO)) if t < T_IMP else 1.0
    if t < T_IMP:
        Lc = 90 + (LOGO_FINAL * 1.06 - 90) * (p ** 2.2)
    else:
        tau = t - T_IMP
        Lc = LOGO_FINAL * (1 + 0.06 * math.exp(-8 * tau) * math.cos(10 * tau))
    Lc = max(int(round(Lc)), 8)
    bob = 0 if t < T_IMP + 0.4 else 3 * math.sin((t - T_IMP) * 2.2)
    n = int(8 * 2 ** (p * 6.5))
    if t < T_IMP and p < 0.97 and n < Lc * 0.85:
        small = np.asarray(LOGO.resize((n, n), Image.BOX), np.float32) / 255
        ii = (np.arange(n) * 1024 // n)
        Rc = FIXR[ii[:, None], ii[None, :]]
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        rad = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) / (n * 0.7071)
        v = 0.6 * Rc + 0.4 * np.clip(rad, 0, 1)
        f = smooth(p / 0.8)
        shown = v <= f
        edge = (v > f) & (v <= f + 0.10) & (small[..., 3] > 0.3)
        out = small.copy()
        out[~shown, 3] = 0
        out[edge, :3] = WGOLD
        out[edge, 3] = 0.9
        big = Image.fromarray((out * 255).astype(np.uint8), "RGBA").resize((Lc, Lc), Image.NEAREST)
    else:
        big = LOGO.resize((Lc, Lc), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(big, (CX - Lc // 2, int(CY - Lc // 2 + bob)))
    arr = np.asarray(canvas, np.float32) / 255
    rgb, a = arr[..., :3].copy(), arr[..., 3].copy()
    if t < T_IMP and p < 0.97:
        g = (1 - p) ** 1.2
        for _ in range(6):
            y0 = int(rg.integers(max(0, CY - Lc // 2), min(H - 8, CY + Lc // 2)))
            hh = int(rg.integers(6, 40))
            dx = int(g * rg.normal(0, 50))
            rgb[y0:y0 + hh] = np.roll(rgb[y0:y0 + hh], dx, axis=1)
            a[y0:y0 + hh] = np.roll(a[y0:y0 + hh], dx, axis=1)
        d = int(g * 16)
        if d:
            rgb[..., 0] = np.roll(rgb[..., 0], d, axis=1)
            rgb[..., 2] = np.roll(rgb[..., 2], -d, axis=1)
        a *= smooth(p * 8)
        rgb *= 1 + 0.8 * (1 - p) ** 2
    return rgb, a, Lc, p


def render_frame(fi, stills_t=None):
    t = fi / FPS if stills_t is None else stills_t
    rg = np.random.default_rng(1000 + fi)
    flick = (rg.random() < p_flick(t))
    pe = pulse_env(t)
    s = cam_s(t)
    tau_i = t - T_IMP

    # ---- background + haze + beam
    sh = int(t * 9)
    haze = 0.5 * (np.roll(HAZE1, sh, axis=1)[:, :W] + np.roll(HAZE2, -sh, axis=1)[:, :W])
    breath = 0.85 + 0.15 * math.sin(t * 1.7)
    frame = BG.copy()
    frame += (haze * 0.05)[..., None] * np.array([0.5, 0.75, 0.4], np.float32) * (BG[..., 1:2] > 0.002)
    frame += (BEAM * (0.10 + 0.05 * pe) * breath)[..., None] * np.array([0.95, 0.9, 0.7], np.float32)

    # ---- microphone
    fx = CX + 1.5 * math.sin(t * 1.3)
    fy = CY + 1.5 * math.cos(t * 1.1)
    a_ = K / s
    c_ = K * (ANCH[0] - SX0) - K * fx / s
    f_ = K * (ANCH[1] - SY0) - K * fy / s
    mic = MIC.transform((W, H), Image.AFFINE, (a_, 0, c_, 0, a_, f_), resample=Image.BICUBIC)
    blur = 0.0
    if t > 3.4:
        blur = 7.0 * smooth((t - 3.4) / 1.8)
    if blur > 0.3:
        mic = mic.filter(ImageFilter.GaussianBlur(blur))
    marr = np.asarray(mic, np.float32) / 255
    dim = 1.0 - 0.65 * smooth((t - 3.2) / 2.0)
    base_lit = 0.50 + 0.12 * smooth(t / 1.5)
    lf = base_lit * dim * (1 + 0.9 * pe) * (1.35 if flick else 1.0)
    ma = marr[..., 3:4]
    frame = frame * (1 - ma) + marr[..., :3] * ma * lf

    # ---- circular energy field, rings and spectrum ring
    add = np.zeros((H, W, 3), np.float32)
    Rf = 330 * smooth((t - 1.6) / 1.4)
    Lc_est = 0
    lg = logo_layer(t, rg)
    if lg is not None:
        Lc_est = lg[2]
    Rv = max(Rf, Lc_est / 2 + 24) if t >= 1.6 else 0
    if Rv > 0:
        Iv = smooth((t - 1.6) / 1.4)
        if t > T_IMP + 0.3:
            Iv = 0.55 + 0.1 * math.sin(t * 2.0)
        if t < T_LOGO + 0.3:
            fieldI = 0.35 * Iv * (0.8 + 0.2 * math.sin(t * 38))
            fld = np.exp(-(D / max(Rv, 1)) ** 4) * fieldI
            edge = np.exp(-((D - Rv) / 7) ** 2) * 0.5 * Iv
            add += (fld + edge)[..., None] * GOLD * 0.6
        amp = 0.35 + 0.65 * np.abs(np.sin(np.arange(NB) * 0.37 + t * 9) *
                                   np.sin(np.arange(NB) * 0.11 - t * 5))
        ln = (amp * (0.4 + 1.0 * pe) * (14 + 40 * Iv)).astype(np.float32)
        msk = (D > Rv + 4) & (D < Rv + 4 + ln[BIDX]) & (np.abs(BFRAC - 0.5) < 0.28)
        add += msk[..., None] * WGOLD * (0.55 * Iv)
        add += (np.exp(-((D - (Rv + 3)) / 2.5) ** 2) * 0.5 * Iv)[..., None] * GOLD

    for k, tp in enumerate(PULSES):
        age = t - tp
        if 0 <= age < 1.4 and t < T_IMP + 0.05:
            r = 250 * s / 1.2 + age * 1000
            w = 9 + age * 22
            wob = 10 * np.sin(6 * TH + age * 10) * np.minimum(age * 3, 1)
            ring = np.exp(-((D - r - wob) / w) ** 2) * ((1 - age / 1.4) ** 2) * (0.35 + 0.35 * k / len(PULSES))
            add += ring[..., None] * GOLD
            add += (np.exp(-((D - r * 0.93) / (w * 0.5)) ** 2) * ((1 - age / 1.4) ** 2) * 0.18)[..., None] * WGOLD

    # shockwave on impact (behind the logo)
    if tau_i >= 0 and tau_i < 1.4:
        r = 440 + tau_i * 2600
        w = 18 + tau_i * 160
        sw_ = np.exp(-((D - r) / w) ** 2) * math.exp(-tau_i * 3.2) * 0.5
        r2 = 440 + tau_i * 1500
        sw2 = np.exp(-((D - r2) / (6 + tau_i * 30)) ** 2) * math.exp(-tau_i * 2.8) * 0.5
        add += sw_[..., None] * GOLD + sw2[..., None] * WGOLD

    # electrical arcs around the grille
    if 0.2 <= t <= 4.6 and (flick or rg.random() < 0.15):
        add += lightning_layer(t, rg)[..., None] * WGOLD * 0.9

    frame += add

    # ---- logo
    if lg is not None:
        rgb, a, Lc, p = lg
        glow = np.exp(-((D - Lc / 2) / 45) ** 2) * (0.20 + 0.45 * pe) * smooth(p * 2.5)
        frame += glow[..., None] * GOLD
        frame = frame * (1 - a[..., None]) + rgb * a[..., None]

    # ---- particles
    if t > 0.4:
        frame += particle_layer(t)

    # ---- final waveform lines
    if t > 4.9:
        ia = smooth((t - 4.9) / 0.7) * (0.6 + 0.4 * (0.5 + 0.5 * math.sin(t * 6.28 * 0.9)))
        pitch, bw = 12, 6
        x_in = CX + LOGO_FINAL // 2 + 40
        xs_ = np.arange(W)
        col_idx = (xs_ - x_in) // pitch
        in_bar = ((xs_ - x_in) % pitch < bw) & (xs_ >= x_in)
        rcol = np.where(in_bar, col_idx, 0)
        hs = (np.abs(np.sin(rcol * 0.9 - t * 7.0)) * 0.6 + np.abs(np.sin(rcol * 0.37 + t * 4.3)) * 0.4)
        env = np.exp(-rcol * 0.045)
        hh = (6 + 70 * hs * env * (0.7 + 0.8 * pe)).astype(np.float32)
        right = np.where(in_bar, hh, 0)
        left = right[::-1]
        hcol = np.maximum(right, left)
        wmask = (np.abs(YY[:, :1] - CY) <= hcol[None, :]) & (hcol[None, :] > 0)
        fade = np.clip(1 - np.abs(XX - CX) / 960, 0, 1) ** 0.6
        frame += (wmask * fade)[..., None] * GOLD * (0.9 * ia)
        base = (np.abs(YY - CY) < 1.0) & (np.abs(XX - CX) > LOGO_FINAL / 2 + 20)
        frame += (base * fade * 0.5 * ia)[..., None] * GOLD

    # ---- HUD
    hud_a = (0.22 + 0.38 * smooth((t - 1.0) / 3.0)) * (0.8 + 0.2 * math.sin(t * 3))
    frame += (HUD * hud_a)[..., None] * np.array([0.85, 0.82, 0.5], np.float32)

    # ---- scanlines / rolling bar
    sl = 0.05 + 0.13 * smooth((t - 2.5) / 1.5) * (1 - 0.6 * smooth((t - 5.2) / 1.0))
    rows = ((np.arange(H) + int(t * 30)) % 4 == 0)
    frame *= (1 - sl * rows)[:, None, None]
    if 1.6 < t < 5.0:
        by = (t * 400) % (H + 200) - 100
        frame += (np.exp(-((YY - by) / 40) ** 2) * 0.05)[..., None] * WGOLD

    # ---- static
    if t < T_IMP:
        loc = np.exp(-(D / 420) ** 2)
        st_loc = (0.55 if flick else 0.06) * (1.0 if t > 0.25 else 0.0)
        st_glob = 0.5 * smooth((t - 1.6) / 3.2) * 0.18
        st = st_loc * loc + st_glob
    else:
        st = 0.30 * math.exp(-tau_i * 12) + 0.02
    if np.max(st) > 0.004:
        N = rg.random((H, W)).astype(np.float32)
        rowj = ((rg.random(H) > 0.96) * rg.random(H)).astype(np.float32)[:, None]
        stat = (N ** 4 * 1.4 + rowj * 0.6)
        frame += (stat * st)[..., None] * WGOLD

    # ---- impact: chroma burst + flash
    if 0 <= tau_i < 0.25:
        d = int(5 * math.exp(-tau_i * 18)) + 1
        frame[..., 0] = np.roll(frame[..., 0], d, axis=1)
        frame[..., 2] = np.roll(frame[..., 2], -d, axis=1)
    if tau_i >= 0:
        frame += (0.28 * math.exp(-tau_i / 0.06)) * WGOLD

    # ---- bloom
    br = np.clip(frame - 0.55, 0, 1)
    bim = Image.fromarray((br * 255).astype(np.uint8))
    b1 = np.asarray(bim.resize((W // 4, H // 4), Image.BOX).filter(ImageFilter.GaussianBlur(4))
                    .resize((W, H), Image.BILINEAR), np.float32) / 255
    b2 = np.asarray(bim.resize((W // 8, H // 8), Image.BOX).filter(ImageFilter.GaussianBlur(5))
                    .resize((W, H), Image.BILINEAR), np.float32) / 255
    frame += b1 * 0.9 + b2 * 0.7

    frame *= VIG[..., None]
    frame += (rg.random((H, W, 1)).astype(np.float32) - 0.5) * 0.03
    frame = np.clip(frame, 0, 1) ** 0.95
    out = (frame * 255 + 0.5).astype(np.uint8)

    # ---- camera shake
    amp = 0.0
    if tau_i >= 0:
        amp = 24 * math.exp(-tau_i * 9)
    elif t > 4.3:
        amp = 4 * smooth((t - 4.3) / 0.5)
    if amp > 0.3:
        dx = int(round(amp * rg.uniform(-1, 1)))
        dy = int(round(amp * rg.uniform(-1, 1)))
        out = np.roll(out, (dy, dx), axis=(0, 1))
    return out


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
def build_audio(path):
    sr = 44100
    n = int(sr * DUR)
    t = np.arange(n) / sr
    rg = np.random.default_rng(5)
    m = np.zeros(n)

    def sm(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)

    def hp(x):
        return x - np.convolve(x, np.ones(24) / 24, mode="same")

    def lp(x, k):
        return np.convolve(x, np.ones(k) / k, mode="same")

    # room tone + hum
    m += 0.03 * np.sin(2 * np.pi * 55 * t) * sm(t / 1.6)
    m += 0.012 * np.sin(2 * np.pi * 110 * t) * sm(t / 2.0)
    m += 0.05 * lp(rg.normal(0, 1, n), 40) * (0.4 + 0.6 * sm(t / 3))
    # static crackle synced with flicker frames
    for fi in range(NF):
        if is_flick(fi):
            s0 = int(fi / FPS * sr)
            ln = int(0.035 * sr)
            seg = hp(rg.normal(0, 1, ln)) * np.exp(-np.arange(ln) / (0.012 * sr))
            g = 0.10 + 0.14 * sm(fi / FPS / 3)
            m[s0:s0 + ln] += seg[:max(0, min(ln, n - s0))] * g
    # pulses
    for k, tp in enumerate(PULSES):
        tau = t - tp
        on = tau >= 0
        amp = 0.30 + 0.45 * k / len(PULSES)
        ph = 2 * np.pi * (48 * tau + (1 - np.exp(-30 * np.clip(tau, 0, None))))
        m += np.where(on, np.sin(ph) * np.exp(-np.clip(tau, 0, None) * 8), 0) * amp
        m += np.where(on, hp(rg.normal(0, 1, n)) * np.exp(-np.clip(tau, 0, None) * 120), 0) * 0.10 * amp
        m += np.where(on, np.sin(2 * np.pi * (1800 + k * 60) * tau) *
                      np.exp(-np.clip(tau, 0, None) * 14), 0) * 0.03 * amp
    # riser
    u = np.clip((t - 1.6) / 3.16, 0, 1)
    live = (t >= 1.6) & (t < 4.76)
    ph = 2 * np.pi * 180 * 3.16 / (4.2 * math.log(2)) * (2 ** (4.2 * u) - 1)
    tone = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    trem = 0.7 + 0.3 * np.sin(2 * np.pi * (6 + 10 * u) * t)
    m += np.where(live, tone * 0.16 * u ** 1.5 * trem, 0)
    m += np.where(live, hp(rg.normal(0, 1, n)) * 0.14 * u ** 2, 0)
    # impact
    tau = np.clip(t - T_IMP, 0, None)
    on = t >= T_IMP
    ph = 2 * np.pi * (36 * tau + 110 * (1 - np.exp(-9 * tau)) / 9)
    m += np.where(on, np.sin(ph) * np.exp(-tau * 1.8), 0) * 1.0
    m += np.where(on, np.sin(2 * np.pi * 85 * tau) * np.exp(-tau * 6), 0) * 0.5
    m += np.where(on, lp(rg.normal(0, 1, n), 4) * np.exp(-tau * 9), 0) * 0.45
    m += np.where(on, rg.normal(0, 1, n) * np.exp(-tau * 80), 0) * 0.5
    for f in (880, 1320, 1760, 2640):
        m += np.where(on, np.sin(2 * np.pi * f * tau) * np.exp(-tau * 3), 0) * 0.03
    m += np.where(on, hp(rg.normal(0, 1, n)) * np.exp(-tau * 12), 0) * 0.20
    # final drone with slow tremolo
    m += 0.05 * np.sin(2 * np.pi * 55 * t) * sm((t - T_IMP) / 0.3) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.9 * t))
    m *= sm((DUR - t) / 0.5)
    m = np.tanh(1.4 * m) / np.tanh(1.4)
    m *= 0.9 / np.max(np.abs(m))
    L = m + 0.01 * rg.normal(0, 1, n)
    R = m + 0.01 * rg.normal(0, 1, n)
    st = np.stack([L, R], 1)
    pcm = (np.clip(st, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--stills":
        od = sys.argv[2]
        os.makedirs(od, exist_ok=True)
        for ts in sys.argv[3:]:
            tt = float(ts)
            fr = render_frame(int(round(tt * FPS)), stills_t=tt)
            Image.fromarray(fr).save(os.path.join(od, f"still_{tt:.2f}.png"))
        return
    out = sys.argv[1] if len(sys.argv) > 1 else "podcast-tentera-darat-reveal.mp4"
    wav = out.rsplit(".", 1)[0] + ".wav"
    build_audio(wav)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for i, fr in enumerate(pool.imap(render_frame, range(NF), chunksize=2)):
            pr.stdin.write(fr.tobytes())
            if i % 15 == 0:
                print(f"frame {i}/{NF}", flush=True)
    pr.stdin.close()
    pr.wait()
    os.remove(wav)
    print("done", out)


if __name__ == "__main__":
    main()
