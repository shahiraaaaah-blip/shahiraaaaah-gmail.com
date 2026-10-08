#!/usr/bin/env python3
"""Podcast Tentera Darat - cinematic intro v2 (16:9, 8.6s, ~2.2s hero hold).

Microphone -> vertical soundwave -> logo materializes and RISES out of the
microphone, camera pushes in then tracks upward, final hero shot with the
logo floating above the microphone, joined by a vertical golden soundwave.

Procedural render (numpy + PIL -> ffmpeg) with synthesized audio. logo.png is
never redrawn: it is only scaled, masked (block-wise reveal) and glowed, and
the final frames show the untouched logo.

Usage:
  python3 video/render_intro_v2.py out.mp4
  python3 video/render_intro_v2.py --stills OUTDIR 0.5 2.8 3.8 4.6 5.6 6.5 8.0
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
sys.path.insert(0, HERE)
import render_reveal as R0  # noqa: E402  (reuses mic sprite, background, helpers)

W, H = 1920, 1080
CX = 960
FPS = 30
DUR = 8.6
NF = int(round(FPS * DUR))
T_C0 = 2.3        # camera starts pulling up / out
T_W0 = 2.4        # vertical soundwave starts travelling up
T_L0 = 3.3        # logo starts materializing
T_IMP = 6.4       # logo locks, bass impact (hold = DUR - T_IMP)
ZH = 0.30         # hero-shot zoom (logical -> screen px)
LUP = 1713.0      # logo centre height above the mic anchor (logical) at the end
DF = 2133.0       # logo diameter (logical) at the end
GOLD = R0.GOLD
WGOLD = R0.WGOLD
clip01, smooth, splat = R0.clip01, R0.smooth, R0.splat
YY, XX = R0.YY, R0.XX

# radio pulses: slow at first, accelerating into the lock
PULSES = []
_t, _iv = 1.3, 0.50
while _t < T_IMP - 0.1:
    PULSES.append(round(_t, 3))
    _iv = max(0.11, _iv * 0.88)
    _t += _iv


def p_flick(t):
    if t < 0.9 or t > T_IMP - 0.1:
        return 0.0
    return 0.20 + 0.5 * smooth((t - 1.2) / 2.5)


def is_flick(fi):
    return np.random.default_rng(1000 + fi).random() < p_flick(fi / FPS)


def pulse_env(t):
    e = 0.0
    for p in PULSES:
        if p <= t:
            e = max(e, math.exp(-(t - p) * 9.0))
    if t >= T_IMP:
        e = max(e, 1.4 * math.exp(-(t - T_IMP) * 5.0))
    return e


# --------------------------------------------------------------------------
# Camera / choreography. World units are "logical" mic units relative to the
# capsule centre; screen = (CX + xw*z, my + yw*z).
# --------------------------------------------------------------------------
def state(t):
    if t < T_C0:
        x = t / T_C0
        z = 1.05 + 0.20 * x * (2 - x)          # slow push-in, settles at 1.25
    else:
        z = 1.25 * (ZH / 1.25) ** smooth((t - T_C0) / (T_IMP - T_C0))
    e = smooth((t - T_C0) / (T_IMP - T_C0))
    my = 590 + 314 * e                          # mic drops as camera tracks up
    v = clip01((t - T_L0) / (T_IMP - T_L0))
    yl = -LUP * smooth(v)
    dl = 300 + (DF - 300) * smooth(v)
    sy = my + yl * z
    Lc = dl * z
    return dict(z=z, my=my, sy=sy, Lc=Lc, yl=yl, dl=dl)


# --------------------------------------------------------------------------
# Static assets
# --------------------------------------------------------------------------
LOGO = R0.LOGO
MIC_K = {2.0: R0.MIC}
MIC_K[1.0] = R0.MIC.resize((R0.MIC.width // 2, R0.MIC.height // 2), Image.LANCZOS)
MIC_K[0.5] = R0.MIC.resize((R0.MIC.width // 4, R0.MIC.height // 4), Image.LANCZOS)
BG, HAZE1, HAZE2, BEAM, VIG, FIXR = R0.BG, R0.HAZE1, R0.HAZE2, R0.BEAM, R0.VIG, R0.FIXR


def build_reveal_times():
    """Per-12px-block reveal time (0..1): frame ring sweeps first, then the
    inner plate, crest, PODCAST left->right, TENTERA DARAT."""
    n = 90
    c = (np.arange(n) + 0.5) * 12
    bx, by = np.meshgrid(c, c)
    rg = np.random.default_rng(7)
    nz = rg.random((n, n))
    r = np.hypot(bx - 540, by - 540)
    ring = r > 452
    crest = (~ring) & (bx > 395) & (bx < 685) & (by > 70) & (by < 520)
    tent = (~ring) & (by >= 785)
    pod = (~ring) & (~crest) & (by >= 500) & (by < 785)
    rest = ~(ring | crest | tent | pod)
    angf = (np.arctan2(bx - 540, -(by - 540)) % (2 * np.pi)) / (2 * np.pi)
    xf = np.clip((bx - 60) / 960, 0, 1)
    cd = np.clip(np.hypot(bx - 540, by - 290) / 260, 0, 1)
    rt = np.zeros((n, n))
    rt[ring] = (0.02 + 0.26 * angf + 0.03 * nz)[ring]
    rt[rest] = (0.10 + 0.42 * nz)[rest]
    rt[crest] = (0.42 + 0.15 * cd + 0.03 * nz)[crest]
    rt[pod] = (0.60 + 0.18 * xf + 0.04 * nz)[pod]
    rt[tent] = (0.78 + 0.13 * xf + 0.03 * nz)[tent]
    return rt.astype(np.float32)


RT = build_reveal_times()


def make_hud():
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    m, ln, wd = 54, 110, 4
    for (x, y, sx, sy) in ((m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1),
                           (W - m, H - m, -1, -1)):
        dr.line([(x, y), (x + sx * ln, y)], fill=255, width=wd)
        dr.line([(x, y), (x, y + sy * ln)], fill=255, width=wd)
    dr.rectangle([m + 20, m + 20, W - m - 20, H - m - 20], outline=40, width=1)
    for side in (m, W - m):
        for y in range(200, 881, 24):
            major = ((y - 200) // 24) % 5 == 0
            l2 = 22 if major else 10
            dr.line([(side, y), (side + (l2 if side == m else -l2), y)],
                    fill=170 if major else 100, width=1)
    for (x, y) in ((220, 190), (W - 220, 190), (220, H - 190), (W - 220, H - 190)):
        dr.line([(x - 10, y), (x + 10, y)], fill=140, width=1)
        dr.line([(x, y - 10), (x, y + 10)], fill=140, width=1)
    dr.polygon([(CX, 22), (CX - 9, 8), (CX + 9, 8)], fill=200)
    dr.polygon([(CX, H - 22), (CX - 9, H - 8), (CX + 9, H - 8)], fill=200)
    return np.asarray(im, np.float32) / 255


def make_ring_sprite():
    """Compass / HUD ring that follows the logo (logo radius = 540 in sprite)."""
    S = 1500
    im = Image.new("L", (S, S), 0)
    dr = ImageDraw.Draw(im)
    c = S // 2
    for R, w, f in ((640, 2, 120), (668, 1, 70), (700, 1, 45)):
        dr.ellipse([c - R, c - R, c + R, c + R], outline=f, width=w)
    for deg in range(0, 360, 3):
        a = math.radians(deg)
        ln = 34 if deg % 90 == 0 else (22 if deg % 15 == 0 else 9)
        x0, y0 = c + 640 * math.cos(a), c + 640 * math.sin(a)
        x1, y1 = c + (640 - ln) * math.cos(a), c + (640 - ln) * math.sin(a)
        dr.line([(x0, y0), (x1, y1)], fill=210 if deg % 15 == 0 else 120, width=2)
    for deg in range(0, 360, 90):
        a = math.radians(deg)
        dr.line([(c + 700 * math.cos(a), c + 700 * math.sin(a)),
                 (c + 740 * math.cos(a), c + 740 * math.sin(a))], fill=190, width=3)
    return im


def make_map():
    """Very subtle tactical dot-matrix map with grid and markers."""
    rg = np.random.default_rng(31)
    sm = rg.random((7, 12)).astype(np.float32)
    land = np.asarray(Image.fromarray((sm * 255).astype(np.uint8))
                      .resize((W, H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(30)),
                      np.float32) / 255
    land = (land - land.min()) / (land.max() - land.min())
    pitch = 20
    px = (np.mod(XX, pitch) < 2) & (np.mod(YY, pitch) < 2)
    dots = px * (0.25 + 0.75 * (land > 0.55))
    grid = ((np.mod(XX, 180) < 1) | (np.mod(YY, 180) < 1)) * 0.35
    out = np.maximum(dots, grid).astype(np.float32)
    im = Image.fromarray((out * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for (x, y) in ((420, 300), (1510, 260), (1380, 800), (500, 790), (1700, 560), (220, 520)):
        dr.ellipse([x - 14, y - 14, x + 14, y + 14], outline=230, width=1)
        dr.line([(x - 26, y), (x - 8, y)], fill=230, width=1)
        dr.line([(x + 8, y), (x + 26, y)], fill=230, width=1)
    return np.asarray(im, np.float32) / 255


HUD = make_hud()
RING_SPR = make_ring_sprite()
MAP = make_map()
MAPW = np.exp(-((XX - CX) / 900) ** 2 - ((YY - 540) / 560) ** 2).astype(np.float32)

# particles ------------------------------------------------------------------
_pr = np.random.default_rng(21)


def _mk_particles():
    out = []
    # 0: ambient sparks around the mic (world coords)
    n = 650
    u = _pr.random(n)
    birth = np.where(u < 0.15, _pr.uniform(0.6, 1.6, n),
                     np.where(u < 0.8, _pr.uniform(1.6, T_IMP, n), _pr.uniform(T_IMP, DUR - 1, n)))
    ang = _pr.uniform(0, 2 * np.pi, n)
    r0 = _pr.uniform(300, 430, n)
    sp = _pr.uniform(40, 260, n)
    out.append(dict(kind=0, birth=birth, life=_pr.uniform(1.2, 3.0, n),
                    x=r0 * np.cos(ang), y=r0 * np.sin(ang) * 1.05,
                    vx=sp * np.cos(ang), vy=sp * np.sin(ang) - 25, k=np.full(n, 0.8),
                    b=_pr.uniform(0.4, 1.0, n), w=_pr.random(n)))
    # 1: digital fragments converging on the logo (screen coords at frame time)
    n = 700
    out.append(dict(kind=1, birth=_pr.uniform(T_L0 - 0.1, T_IMP - 0.4, n),
                    life=_pr.uniform(0.6, 1.1, n), ang=_pr.uniform(0, 2 * np.pi, n),
                    r0=_pr.uniform(1.5, 3.2, n), r1=np.sqrt(_pr.random(n)) * 0.95,
                    b=_pr.uniform(0.5, 1.0, n), w=_pr.random(n)))
    # 2: impact burst (screen coords, centred on the locked logo)
    n = 1000
    ang = _pr.uniform(0, 2 * np.pi, n)
    r0 = _pr.uniform(300, 340, n)
    sp = _pr.uniform(200, 1400, n)
    out.append(dict(kind=2, birth=np.full(n, T_IMP) + _pr.uniform(0, 0.04, n),
                    life=_pr.uniform(0.7, 2.0, n),
                    x=CX + r0 * np.cos(ang), y=390 + r0 * np.sin(ang),
                    vx=sp * np.cos(ang), vy=sp * np.sin(ang), k=np.full(n, 2.2),
                    b=_pr.uniform(0.5, 1.0, n), w=_pr.random(n)))
    # 3: particles streaming up the soundwave from the mic top (world coords)
    n = 500
    out.append(dict(kind=3, birth=_pr.uniform(T_W0, DUR - 0.6, n),
                    life=_pr.uniform(0.9, 1.8, n),
                    x=_pr.normal(0, 55, n), y=np.full(n, -380.0) + _pr.normal(0, 20, n),
                    vy=-_pr.uniform(900, 2300, n), k=np.full(n, 1.1),
                    sw=_pr.uniform(0, 6.28, n),
                    b=_pr.uniform(0.5, 1.0, n), w=_pr.random(n)))
    return out


PARTS = _mk_particles()


def particle_layer(t, S):
    layer = np.zeros((H, W, 3), np.float32)
    z, my = S["z"], S["my"]
    for P in PARTS:
        tau = t - P["birth"]
        alive = (tau >= 0) & (tau <= P["life"])
        if not alive.any():
            continue
        idx = np.nonzero(alive)[0]
        tt = tau[idx]
        u = tt / P["life"][idx]
        kind = P["kind"]
        if kind in (0, 2):
            k = P["k"][idx]
            f = (1 - np.exp(-k * tt)) / k
            xw = P["x"][idx] + P["vx"][idx] * f
            yw = P["y"][idx] + P["vy"][idx] * f
            if kind == 0:
                xs, ys = CX + xw * z, my + yw * z
            else:
                xs, ys = xw, yw
            fade = (1 - u) ** 1.5
        elif kind == 1:
            e = u * u * (3 - 2 * u)
            rr = S["Lc"] / 2 * (P["r0"][idx] + (P["r1"][idx] - P["r0"][idx]) * e)
            xs = CX + rr * np.cos(P["ang"][idx])
            ys = S["sy"] + rr * np.sin(P["ang"][idx])
            fade = np.sin(np.pi * u) ** 0.8
        else:
            k = P["k"][idx]
            f = (1 - np.exp(-k * tt)) / k
            xw = P["x"][idx] + 40 * np.sin(P["sw"][idx] + tt * 6)
            yw = P["y"][idx] + P["vy"][idx] * f
            xs, ys = CX + xw * z, my + yw * z
            fade = np.sin(np.pi * np.clip(u, 0, 1)) ** 0.7
        tw = 0.6 + 0.4 * np.sin(t * 40 + P["w"][idx] * 50)
        br = (P["b"][idx] * fade * tw * 1.6).astype(np.float32)
        col = GOLD[None, :] * (1 - P["w"][idx][:, None] * 0.5) + \
            WGOLD[None, :] * (P["w"][idx][:, None] * 0.5)
        splat(layer, xs, ys, col * br[:, None])
    return layer


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
def mic_layer(S):
    z, my = S["z"], S["my"]
    kk = 2.0 if z >= 0.9 else (1.0 if z >= 0.45 else 0.5)
    spr = MIC_K[kk]
    a = kk / z
    c = kk * 360 - a * CX
    # sprite row v = kk*(yw + 400), yw = (Y-my)/z  ->  v = a*Y + kk*400 - a*my
    f = kk * 400 - a * my
    mic = spr.transform((W, H), Image.AFFINE, (a, 0, c, 0, a, f), resample=Image.BICUBIC)
    return np.asarray(mic, np.float32) / 255


def lightning_layer(t, rg, S):
    im = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(im)
    n = int(rg.integers(2, 5))
    for _ in range(n):
        a = rg.uniform(0, 2 * math.pi)
        r = 300 * S["z"] + rg.uniform(-10, 10)
        x, y = CX + r * math.cos(a), S["my"] + r * math.sin(a)
        pts = [(x, y)]
        for _ in range(int(rg.integers(5, 11))):
            a += rg.normal(0, 0.55)
            x += math.cos(a) * rg.uniform(8, 24) * 1.2
            y += math.sin(a) * rg.uniform(8, 24) * 1.2
            pts.append((x, y))
        dr.line(pts, fill=int(rg.uniform(170, 255)), width=int(rg.integers(1, 3)))
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    return np.asarray(im, np.float32) / 255


def blit(frame, rgb, a, x0, y0):
    h, w = a.shape
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, W), min(y0 + h, H)
    if xb <= xa or yb <= ya:
        return
    sa = a[ya - y0:yb - y0, xa - x0:xb - x0, None]
    sr = rgb[ya - y0:yb - y0, xa - x0:xb - x0]
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - sa) + sr * sa


def logo_layer(t, rg, S):
    """Returns (rgb, a, Lc, p, x0, y0) or None."""
    if t < T_L0:
        return None
    p = clip01((t - T_L0) / (T_IMP - 0.1 - T_L0))
    Lc = S["Lc"]
    if t >= T_IMP:
        tau = t - T_IMP
        Lc *= 1 + 0.05 * math.exp(-8 * tau) * math.cos(10 * tau)
    Lc = max(int(round(Lc)), 16)
    base = LOGO.resize((Lc, Lc), Image.LANCZOS)
    arr = np.asarray(base, np.float32) / 255
    rgb, a = arr[..., :3].copy(), arr[..., 3].copy()
    if p < 1.0:
        shown = RT <= p
        glow = np.clip(1 - (p - RT) / 0.05, 0, 1) * shown
        m = np.asarray(Image.fromarray((shown * 255).astype(np.uint8))
                       .resize((Lc, Lc), Image.NEAREST), np.float32) / 255
        g = np.asarray(Image.fromarray((glow * 255).astype(np.uint8))
                       .resize((Lc, Lc), Image.NEAREST), np.float32) / 255
        rgb = rgb * (1 - 0.8 * g[..., None]) + WGOLD * (0.8 * g[..., None]) * 1.2
        a = a * m
        if p < 0.97:
            gl = (1 - p) ** 1.2
            for _ in range(5):
                y0 = int(rg.integers(0, max(1, Lc - 8)))
                hh = int(rg.integers(4, max(5, Lc // 14)))
                dx = int(gl * rg.normal(0, 40))
                rgb[y0:y0 + hh] = np.roll(rgb[y0:y0 + hh], dx, axis=1)
                a[y0:y0 + hh] = np.roll(a[y0:y0 + hh], dx, axis=1)
            d = int(gl * 10)
            if d:
                rgb[..., 0] = np.roll(rgb[..., 0], d, axis=1)
                rgb[..., 2] = np.roll(rgb[..., 2], -d, axis=1)
    return rgb, a, Lc, p, CX - Lc // 2, int(round(S["sy"] - Lc / 2))


def vwave(frame, t, ytop, ybot, pe, ia):
    """Vertical golden waveform: stacked horizontal bars travelling UP."""
    if ybot - ytop < 6 or ia <= 0:
        return
    y0, y1 = int(max(ytop, 0)), int(min(ybot, H))
    if y1 - y0 < 3:
        return
    xa, xb = CX - 150, CX + 150
    ys = np.arange(y0, y1, dtype=np.float32)
    pitch, bw = 11, 5
    dist = ybot - ys
    idx = np.floor(dist / pitch)
    in_bar = (np.mod(dist, pitch) < bw)
    hs = np.abs(np.sin(idx * 0.9 - t * 9.0)) * 0.6 + np.abs(np.sin(idx * 0.37 + t * 4.3)) * 0.4
    fr = np.clip(dist / max(ybot - ytop, 1), 0, 1)
    def sa(x):
        x = np.clip(x, 0, 1)
        return x * x * (3 - 2 * x)
    env = (0.25 + 0.75 * sa(fr * 6)) * (0.35 + 0.65 * sa((1 - fr) * 6))
    hw = (4 + 46 * hs * env * (0.55 + 0.9 * pe)) * ia
    xs = np.arange(xa, xb, dtype=np.float32)[None, :]
    bars = (np.abs(xs - CX) <= hw[:, None]) & in_bar[:, None]
    core = np.abs(xs - CX) <= 1.3
    strip = (bars * 0.9 + core * 0.55) * ia
    sub = frame[y0:y1, xa:xb]
    sub += strip[..., None] * GOLD
    sub += (bars * 0.25 * ia)[..., None] * WGOLD
    glow = np.exp(-((xs - CX) / 26) ** 2) * (0.10 + 0.14 * pe) * ia
    sub += glow[..., None] * GOLD


def render_frame(fi, stills_t=None):
    t = fi / FPS if stills_t is None else stills_t
    rg = np.random.default_rng(1000 + fi)
    flick = rg.random() < p_flick(t)
    pe = pulse_env(t)
    S = state(t)
    z, my, sy, Lc = S["z"], S["my"], S["sy"], S["Lc"]
    tau_i = t - T_IMP
    cyl = 390.0 if t >= T_IMP else sy   # logo centre (screen y)

    # ---- background, haze, key beam, smoke around the mic
    sh = int(t * 9)
    haze = 0.5 * (np.roll(HAZE1, sh, axis=1)[:, :W] + np.roll(HAZE2, -sh, axis=1)[:, :W])
    frame = BG.copy()
    frame += (haze * 0.05)[..., None] * np.array([0.5, 0.75, 0.4], np.float32) * (BG[..., 1:2] > 0.002)
    breath = 0.85 + 0.15 * math.sin(t * 1.7)
    frame += (BEAM * (0.10 + 0.05 * pe) * breath)[..., None] * np.array([0.95, 0.9, 0.7], np.float32)
    smk = np.exp(-((XX - CX) / 520) ** 2 - ((YY - (my + 60)) / 380) ** 2)
    smk2 = np.roll(HAZE2, -int(t * 14), axis=1)[:, :W]
    frame += (smk * (0.05 + 0.20 * haze * smk2) * (1.0 - 0.45 * smooth((t - 3) / 3)))[..., None] \
        * np.array([0.42, 0.62, 0.40], np.float32)
    # subtle tactical map
    map_a = 0.10 * smooth((t - 0.5) / 2.5) * (0.85 + 0.15 * math.sin(t * 2.3))
    frame += (MAP * MAPW * map_a)[..., None] * np.array([0.55, 0.85, 0.45], np.float32)

    # ---- microphone
    marr = mic_layer(S)
    dim = 1.0 - 0.18 * smooth((t - T_C0) / (T_IMP - T_C0))
    base_lit = 0.50 + 0.12 * smooth(t / 1.5)
    lf = base_lit * dim * (1 + 0.9 * pe) * (1.35 if flick else 1.0)
    ma = marr[..., 3:4]
    frame = frame * (1 - ma) + marr[..., :3] * ma * lf

    # ---- circular energy field / spectrum ring + expanding RF rings
    add = np.zeros((H, W, 3), np.float32)
    Dm = np.hypot(XX - CX, YY - my).astype(np.float32)
    ring_on = t >= 1.4
    if ring_on:
        Iv = smooth((t - 1.4) / 1.4)
        if t > T_IMP + 0.3:
            Iv = 0.55 + 0.1 * math.sin(t * 2.0)
        rb = clip01((t - T_L0) / 0.6)
        rb = smooth(rb)
        rcy = my * (1 - rb) + cyl * rb
        Rg = 230 * z
        Rl = Lc / 2 + 24
        Rv = Rg * (1 - rb) + Rl * rb
        Dr = np.hypot(XX - CX, YY - rcy).astype(np.float32)
        TH = np.arctan2(YY - rcy, XX - CX)
        NB = 120
        BIN = (((TH + math.pi) / (2 * math.pi)) * NB).astype(np.float32)
        BIDX = np.minimum(BIN.astype(np.int32), NB - 1)
        BFRAC = BIN - np.floor(BIN)
        if t < T_L0 + 0.3:
            fieldI = 0.35 * Iv * (0.8 + 0.2 * math.sin(t * 38))
            fld = np.exp(-(Dr / max(Rv, 1)) ** 4) * fieldI
            edge = np.exp(-((Dr - Rv) / 7) ** 2) * 0.5 * Iv
            add += (fld + edge)[..., None] * GOLD * 0.6
        amp = 0.35 + 0.65 * np.abs(np.sin(np.arange(NB) * 0.37 + t * 9) *
                                   np.sin(np.arange(NB) * 0.11 - t * 5))
        ln = (amp * (0.4 + 1.0 * pe) * (10 + 40 * Iv) * max(min(Lc / 640, 1.4), 0.6)).astype(np.float32)
        msk = (Dr > Rv + 4) & (Dr < Rv + 4 + ln[BIDX]) & (np.abs(BFRAC - 0.5) < 0.28)
        add += msk[..., None] * WGOLD * (0.55 * Iv)
        add += (np.exp(-((Dr - (Rv + 3)) / 2.5) ** 2) * 0.5 * Iv)[..., None] * GOLD

    for k, tp in enumerate(PULSES):
        age = t - tp
        if 0 <= age < 1.4 and t < T_IMP + 0.05:
            r = 330 * z + age * 900
            w = 7 + age * 20
            wob = 8 * np.sin(6 * np.arctan2(YY - my, XX - CX) + age * 10) * min(age * 3, 1)
            ring = np.exp(-((Dm - r - wob) / w) ** 2) * ((1 - age / 1.4) ** 2) * (0.30 + 0.35 * k / len(PULSES))
            add += ring[..., None] * GOLD
            add += (np.exp(-((Dm - r * 0.93) / (w * 0.5)) ** 2) * ((1 - age / 1.4) ** 2) * 0.16)[..., None] * WGOLD

    # shockwave on impact, centred on the logo
    if 0 <= tau_i < 1.4:
        Dl = np.hypot(XX - CX, YY - 390).astype(np.float32)
        r = 330 + tau_i * 2400
        w = 16 + tau_i * 150
        add += (np.exp(-((Dl - r) / w) ** 2) * math.exp(-tau_i * 3.2) * 0.5)[..., None] * GOLD
        r2 = 330 + tau_i * 1400
        add += (np.exp(-((Dl - r2) / (6 + tau_i * 28)) ** 2) * math.exp(-tau_i * 2.8) * 0.5)[..., None] * WGOLD

    # electrical arcs around the grille
    if 0.9 <= t <= 4.4 and (flick or rg.random() < 0.12):
        add += lightning_layer(t, rg, S)[..., None] * WGOLD * 0.9

    # ---- vertical soundwave (behind the logo)
    ia = smooth((t - T_W0) / 0.6)
    mic_top = my - 380 * z
    top_w = mic_top - 900 * smooth((t - T_W0) / 1.0)
    bl = smooth((t - T_L0) / 0.5)
    top = min(top_w * (1 - bl) + cyl * bl, mic_top)
    vwave(add, t, top, mic_top + 6, pe, ia)

    frame += add

    # ---- logo (with HUD ring that follows it)
    lg = logo_layer(t, rg, S)
    if lg is not None:
        rgb, a, Lcc, p, x0, y0 = lg
        pa = smooth((t - T_L0) / 1.2) * (0.5 if t > T_IMP else 0.35)
        gsz = int(Lcc * 1500 / 1080)
        spr = np.asarray(RING_SPR.resize((gsz, gsz), Image.BILINEAR), np.float32) / 255
        hud_rgb = np.broadcast_to(np.array([0.85, 0.82, 0.5], np.float32), spr.shape + (3,))
        ring_rgb = np.zeros_like(hud_rgb)
        gx0, gy0 = CX - gsz // 2, int(round(cyl - gsz / 2))
        sub = np.zeros((H, W, 3), np.float32)
        blit(sub, hud_rgb * spr[..., None], np.ones_like(spr), gx0, gy0)
        frame += sub * pa
        Dl2 = np.hypot(XX - CX, YY - cyl)
        gl = np.exp(-((Dl2 - Lcc / 2) / 45) ** 2) * (0.20 + 0.45 * pe) * smooth(p * 2.5)
        frame += gl[..., None] * GOLD
        blit(frame, rgb, a, x0, y0)

    # ---- particles
    if t > 0.4:
        frame += particle_layer(t, S)

    # ---- HUD
    hud_a = (0.22 + 0.38 * smooth((t - 1.0) / 3.0)) * (0.8 + 0.2 * math.sin(t * 3))
    frame += (HUD * hud_a)[..., None] * np.array([0.85, 0.82, 0.5], np.float32)

    # ---- scanlines / rolling bar
    sl = 0.05 + 0.13 * smooth((t - 2.5) / 1.5) * (1 - 0.6 * smooth((t - 6.6) / 1.0))
    rows = ((np.arange(H) + int(t * 30)) % 4 == 0)
    frame *= (1 - sl * rows)[:, None, None]
    if 1.6 < t < T_IMP:
        by = (t * 400) % (H + 200) - 100
        frame += (np.exp(-((YY - by) / 40) ** 2) * 0.05)[..., None] * WGOLD

    # ---- radio static / interference
    if t < T_IMP:
        loc = np.exp(-((XX - CX) ** 2 + (YY - my) ** 2) / (2 * (330 * z) ** 2))
        st_loc = (0.55 if flick else 0.05) * (1.0 if t > 0.25 else 0.0)
        st_glob = 0.5 * smooth((t - 1.6) / 3.2) * 0.18
        st = st_loc * loc + st_glob
    else:
        st = 0.30 * math.exp(-tau_i * 12) + 0.02
    if np.max(st) > 0.004:
        N = rg.random((H, W)).astype(np.float32)
        rowj = ((rg.random(H) > 0.96) * rg.random(H)).astype(np.float32)[:, None]
        frame += ((N ** 4 * 1.4 + rowj * 0.6) * st)[..., None] * WGOLD

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

    # ---- camera shake (impact only) -> static hero shot
    amp = 0.0
    if tau_i >= 0:
        amp = 24 * math.exp(-tau_i * 9)
    elif t > T_IMP - 0.8:
        amp = 3 * smooth((t - (T_IMP - 0.8)) / 0.7)
    if amp > 0.3:
        dx = int(round(amp * rg.uniform(-1, 1)))
        dy = int(round(amp * rg.uniform(-1, 1)))
        out = np.roll(out, (dy, dx), axis=(0, 1))
    return out


# --------------------------------------------------------------------------
# Audio: radio static -> low electrical hum -> rising waveform -> bass impact
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

    # room tone, then electrical hum swelling in (50 Hz + harmonics)
    m += 0.05 * lp(rg.normal(0, 1, n), 40) * (0.3 + 0.7 * sm((t - 0.2) / 3))
    hum = (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
           + 0.25 * np.sin(2 * np.pi * 150 * t))
    m += 0.035 * hum * sm((t - 0.6) / 1.6)
    # radio static crackle synced with the flicker frames
    for fi in range(NF):
        if is_flick(fi):
            s0 = int(fi / FPS * sr)
            ln = int(0.035 * sr)
            seg = hp(rg.normal(0, 1, ln)) * np.exp(-np.arange(ln) / (0.012 * sr))
            m[s0:s0 + ln] += seg[:max(0, min(ln, n - s0))] * (0.10 + 0.14 * sm(fi / FPS / 3.5))
    # waveform pulses (accelerating)
    for k, tp in enumerate(PULSES):
        tau = t - tp
        on = tau >= 0
        amp = 0.28 + 0.45 * k / len(PULSES)
        ph = 2 * np.pi * (48 * tau + (1 - np.exp(-30 * np.clip(tau, 0, None))))
        m += np.where(on, np.sin(ph) * np.exp(-np.clip(tau, 0, None) * 8), 0) * amp
        m += np.where(on, hp(rg.normal(0, 1, n)) * np.exp(-np.clip(tau, 0, None) * 120), 0) * 0.10 * amp
        m += np.where(on, np.sin(2 * np.pi * (1800 + k * 60) * tau) *
                      np.exp(-np.clip(tau, 0, None) * 14), 0) * 0.03 * amp
    # rising waveform riser
    t0, t1 = T_W0, T_IMP - 0.04
    u = np.clip((t - t0) / (t1 - t0), 0, 1)
    live = (t >= t0) & (t < t1)
    ph = 2 * np.pi * 180 * (t1 - t0) / (4.2 * math.log(2)) * (2 ** (4.2 * u) - 1)
    tone = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    trem = 0.7 + 0.3 * np.sin(2 * np.pi * (6 + 10 * u) * t)
    m += np.where(live, tone * 0.16 * u ** 1.5 * trem, 0)
    m += np.where(live, hp(rg.normal(0, 1, n)) * 0.14 * u ** 2, 0)
    # impact: cinematic bass hit when the logo locks
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
    # hero-shot drone
    m += 0.05 * np.sin(2 * np.pi * 55 * t) * sm((t - T_IMP) / 0.3) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.9 * t))
    m *= sm((DUR - t) / 0.5)
    m = np.tanh(1.4 * m) / np.tanh(1.4)
    m *= 0.9 / np.max(np.abs(m))
    L = m + 0.01 * rg.normal(0, 1, n)
    R = m + 0.01 * rg.normal(0, 1, n)
    pcm = (np.clip(np.stack([L, R], 1), -1, 1) * 32767).astype("<i2")
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
    out = sys.argv[1] if len(sys.argv) > 1 else "podcast-tentera-darat-intro-v2.mp4"
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
