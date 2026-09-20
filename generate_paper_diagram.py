"""
Figura estilo paper: predicción de jugadas de ajedrez con el conectoma de Drosophila.
  a) Codificación sensorial  ->  b) Simulación del conectoma (pesos congelados)  ->  c) Readout entrenado

Uso:  python generate_paper_diagram.py
Salida: fig_fly_chess.pdf (vectorial, para el paper) y fig_fly_chess.png (300 dpi)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

# ----------------------------------------------------------------------------
# PARÁMETROS QUE PUEDES EDITAR
# ----------------------------------------------------------------------------
N_VISUAL = "1,546"          # neuronas visuales de entrada
N_NEURONS = "30,000"        # neuronas simuladas
FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"
MOVE_EXAMPLE = ("d2", "d3")  # jugada de ejemplo que se dibuja en el panel c
RESULT_TEXT = None           # p. ej. "Top-1 = 12.3 %" (None = no mostrar resultado en la figura)
OUT_BASENAME = "fig_fly_chess"
SEED = 7

plt.rcParams["font.family"] = ["Liberation Sans", "DejaVu Sans"]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["pdf.fonttype"] = 42

# Paleta
SLATE, GREY = "#1e293b", "#64748b"
C_A, C_B, C_C = "#2563eb", "#dc2626", "#059669"       # paneles a, b, c
C_OPTIC, C_CENTRAL, C_DESC, C_VNC = "#8b5cf6", "#10b981", "#f59e0b", "#3b82f6"
C_SPIKE = "#facc15"

rng = np.random.default_rng(SEED)

fig = plt.figure(figsize=(11, 5.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 110)
ax.set_ylim(0, 56)
ax.axis("off")


# ----------------------------------------------------------------------------
# Utilidades de dibujo
# ----------------------------------------------------------------------------
def panel(x0, x1, y0, y1, color, letter, title, tag, tag_color):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc=color, ec="none", alpha=0.045, zorder=0))
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc="none", ec=color, lw=1.1, alpha=0.9, zorder=0))
    ax.text(x0 + 0.3, y1 + 2.2, letter, fontsize=15, fontweight="bold", color=SLATE, va="center")
    ax.text(x0 + 2.6, y1 + 2.2, title, fontsize=10.5, fontweight="bold", color=color, va="center")
    ax.text(x1 - 0.3, y1 + 2.2, tag, fontsize=7.5, fontweight="bold", color="white", ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=tag_color, ec="none"))


def block(x0, y0, w, h, fc="white", ec="#334155", lw=1.0, z=2):
    p = FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(p0, p1, color="#334155", rad=0.0, lw=1.4, ms=9, z=5, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


def badge(x, y, n, color=SLATE):
    ax.add_patch(Circle((x, y), 1.15, fc=color, ec="white", lw=1.0, zorder=9))
    ax.text(x, y - 0.05, str(n), fontsize=8, fontweight="bold", color="white", ha="center", va="center", zorder=10)


GLYPH = {"k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f"}


def parse_fen(fen):
    rows = fen.split()[0].split("/")
    board = []
    for r in rows:
        line = []
        for ch in r:
            if ch.isdigit():
                line += [None] * int(ch)
            else:
                line.append(ch)
        board.append(line)
    return board  # board[0] = rank 8


def sq_to_rc(sq):
    return 8 - int(sq[1]), "abcdefgh".index(sq[0])


def draw_board(x0, y0, size, fen, fs, move=None, coords=False):
    board = parse_fen(fen)
    s = size / 8
    for r in range(8):
        for c in range(8):
            light = (r + c) % 2 == 0
            ax.add_patch(Rectangle((x0 + c * s, y0 + (7 - r) * s), s, s,
                                   fc="#f1e4c8" if light else "#7fa066", ec="none", zorder=3))
    if move:
        for sq in move:
            r, c = sq_to_rc(sq)
            ax.add_patch(Rectangle((x0 + c * s, y0 + (7 - r) * s), s, s, fc="#facc15", alpha=0.55, ec="none", zorder=3.5))
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p is None:
                continue
            white = p.isupper()
            t = ax.text(x0 + (c + 0.5) * s, y0 + (7 - r + 0.47) * s, GLYPH[p.lower()], fontsize=fs,
                        ha="center", va="center", family="DejaVu Sans",
                        color="white" if white else "#111827", zorder=4)
            if white:
                t.set_path_effects([pe.withStroke(linewidth=1.3, foreground="#111827")])
    ax.add_patch(Rectangle((x0, y0), size, size, fc="none", ec=SLATE, lw=1.2, zorder=5))
    if coords:
        for i in range(8):
            ax.text(x0 + (i + 0.5) * s, y0 - 0.7, "abcdefgh"[i], fontsize=5.5, color=GREY, ha="center", va="center")
            ax.text(x0 - 0.7, y0 + (i + 0.5) * s, str(i + 1), fontsize=5.5, color=GREY, ha="center", va="center")
    if move:
        (r0, c0), (r1, c1) = sq_to_rc(move[0]), sq_to_rc(move[1])
        arrow((x0 + (c0 + 0.5) * s, y0 + (7 - r0 + 0.5) * s), (x0 + (c1 + 0.5) * s, y0 + (7 - r1 + 0.5) * s),
              color="#b91c1c", lw=2.0, ms=8, z=6)


# ----------------------------------------------------------------------------
# PANELES
# ----------------------------------------------------------------------------
YB, YT = 6.0, 51.0
PA = (1.0, 29.0)
PB = (32.0, 82.0)
PC = (85.0, 109.0)

panel(*PA, YB, YT, C_A, "a", "Sensory encoding", "FIXED", "#64748b")
panel(*PB, YB, YT, C_B, "b", "Connectome simulation", "FROZEN WEIGHTS", "#64748b")
panel(*PC, YB, YT, C_C, "c", "Readout", "TRAINED", C_C)

# ---------------------------------------------------------------- Panel a
cxa = (PA[0] + PA[1]) / 2
ax.text(cxa, 49.4, "Board state $s_t$ (FEN)", fontsize=8.5, fontweight="bold", color=SLATE, ha="center", va="center")
draw_board(cxa - 7.2, 34.0, 14.4, FEN, fs=10.5, coords=True)
badge(PA[0] + 2.0, 47.6, 1)
arrow((cxa, 32.6), (cxa, 28.6))

# Bloque: planos de características piezas x casillas
block(PA[0] + 1.4, 17.0, PA[1] - PA[0] - 2.8, 11.2)
badge(PA[0] + 2.0, 27.3, 2)
board = parse_fen(FEN)
px0, py0, ps = PA[0] + 4.6, 18.6, 6.6
for k in range(3, -1, -1):
    off = 1.0 * k
    ax.add_patch(Rectangle((px0 + off, py0 + off), ps, ps, fc="white" if k else "#eef2ff", ec=C_A, lw=0.8,
                           zorder=3 + (3 - k) * 0.1))
cell = ps / 8
for r in range(8):
    for c in range(8):
        if board[r][c] == "P":
            ax.add_patch(Rectangle((px0 + c * cell, py0 + (7 - r) * cell), cell, cell, fc=C_A, ec="white", lw=0.3,
                                   zorder=3.6))
for i in range(9):
    ax.plot([px0, px0 + ps], [py0 + i * cell] * 2, color="#c7d2fe", lw=0.35, zorder=3.55)
    ax.plot([px0 + i * cell] * 2, [py0, py0 + ps], color="#c7d2fe", lw=0.35, zorder=3.55)
ax.text(px0 + ps + 5.2, 24.2, "Piece\u2013square", fontsize=7.8, fontweight="bold", color=SLATE, va="center")
ax.text(px0 + ps + 5.2, 22.4, "feature planes", fontsize=7.8, fontweight="bold", color=SLATE, va="center")
ax.text(px0 + ps + 5.2, 20.4, "64 \u00d7 12", fontsize=7.5, color=GREY, va="center", style="italic")
arrow((cxa, 16.6), (cxa, 14.9))

# Bloque: estimulación retiniana (red hexagonal de fotorreceptores)
block(PA[0] + 1.4, 7.4, PA[1] - PA[0] - 2.8, 7.4)
badge(PA[0] + 2.0, 13.9, 3)
hx0, hy0 = PA[0] + 4.2, 8.7
for i in range(8):
    for j in range(4):
        x = hx0 + i * 1.25 + (0.62 if j % 2 else 0)
        y = hy0 + j * 1.05
        v = rng.random() ** 1.6
        ax.add_patch(Circle((x, y), 0.46, fc=plt.cm.YlOrRd(0.15 + 0.8 * v), ec="#94a3b8", lw=0.3, zorder=4))
ax.text(hx0 + 11.6, 11.9, "Input currents", fontsize=7.6, fontweight="bold", color=SLATE, va="center")
ax.text(hx0 + 11.6, 10.3, f"{N_VISUAL} visual", fontsize=7.4, color=GREY, va="center", style="italic")
ax.text(hx0 + 11.6, 9.0, "neurons", fontsize=7.4, color=GREY, va="center", style="italic")

# ---------------------------------------------------------------- Panel b: mosca
S = 0.35  # Mosca mucho más pequeña
CX, CY = 58.6, 18.2


def W(pts):
    p = np.atleast_2d(np.asarray(pts, dtype=float))
    return np.column_stack([CX - S * p[:, 1], CY + S * p[:, 0]])


def ell(xl, yl, w, h, ang=0.0, **kw):
    X, Y = W([xl, yl])[0]
    e = Ellipse((X, Y), w * S, h * S, angle=ang + 90, **kw)
    ax.add_patch(e)
    return e


def poly(points, **kw):
    p = W(points)
    return ax.plot(p[:, 0], p[:, 1], **kw)

# ESTILO SILUETA: Solo contornos grises, sin relleno de color.
F_COL = "none"
E_COL = "#94a3b8"

# Patas (coxa - fémur - tibia - tarso), simétricas
legs = {
    "front": [(3.4, 9.6), (7.6, 12.6), (11.3, 9.6), (13.4, 6.6)],
    "mid": [(4.2, 6.0), (9.4, 7.2), (12.8, 4.4), (14.3, 0.6)],
    "hind": [(3.8, 2.6), (8.2, 0.2), (11.4, -3.8), (12.0, -8.6)],
}
for side in (-1, 1):
    for pts in legs.values():
        pts = [(side * x, y) for x, y in pts]
        for (a, b), lw in zip(zip(pts[:-1], pts[1:]), (3.0, 2.2, 1.3)):
            poly([a, b], color=E_COL, lw=lw, solid_capstyle="round", zorder=1)

# Abdomen con bandas de tergitos
abd = ell(0, -8.6, 9.2, 18.4, fc=F_COL, ec=E_COL, lw=1.0, zorder=2)
for k in range(6):
    y_hi = -1.6 - 2.9 * k
    y_lo = y_hi - 1.35
    (xa, ya), (xb, yb) = W([-6, y_hi])[0], W([6, y_lo])[0]
    r = Rectangle((min(xa, xb), min(ya, yb)), abs(xb - xa), abs(yb - ya), fc=E_COL, ec="none", alpha=0.2, zorder=2.1)
    ax.add_patch(r)
    r.set_clip_path(abd)

# Halterios
for side in (-1, 1):
    poly([(side * 4.4, 0.9), (side * 6.3, -0.3)], color=E_COL, lw=0.9, zorder=2.2)
    ell(side * 6.6, -0.5, 1.5, 1.5, fc=F_COL, ec=E_COL, lw=0.6, zorder=2.3)

# Tórax
thx = ell(0, 6.3, 9.8, 10.8, fc=F_COL, ec=E_COL, lw=1.0, zorder=3)
for sx in (-1.6, 0, 1.6):
    p = W([(sx, 10.6), (sx, 2.6)])
    ln, = ax.plot(p[:, 0], p[:, 1], color=E_COL, lw=1.5 if sx == 0 else 1.1, alpha=0.7, zorder=3.1)
    ln.set_clip_path(thx)
ell(0, 1.3, 4.2, 2.6, fc=F_COL, ec=E_COL, lw=0.7, zorder=3.2)

# Cabeza, ojos compuestos, ocelos, antenas, probóscide
ell(0, 14.9, 8.4, 5.8, fc=F_COL, ec=E_COL, lw=1.0, zorder=3)
for side in (-1, 1):
    poly([(side * 0.8, 17.6), (side * 1.7, 19.5)], color=E_COL, lw=1.1, zorder=2.5)
    ell(side * 2.0, 19.9, 1.5, 1.1, ang=side * 25, fc=F_COL, ec=E_COL, lw=0.6, zorder=2.6)
poly([(0, 17.7), (0, 19.9)], color=E_COL, lw=1.8, solid_capstyle="round", zorder=2.5)
for side in (-1, 1):
    eye = ell(side * 3.9, 15.1, 3.7, 5.3, ang=side * 12, fc=F_COL, ec=E_COL, lw=0.9, zorder=3.4)
    # facetas
    gx, gy = np.meshgrid(np.arange(-2.4, 2.5, 0.62), np.arange(-3.2, 3.3, 0.62))
    gx = gx + (np.arange(gx.shape[0])[:, None] % 2) * 0.31
    pts = W(np.column_stack([side * 3.9 + gx.ravel(), 15.1 + gy.ravel()]))
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=E_COL, lw=0, zorder=3.45)
    sc.set_clip_path(eye)
for x in (-0.9, 0, 0.9):
    ell(x, 16.7 if x == 0 else 16.3, 0.55, 0.55, fc=E_COL, ec="none", zorder=3.5)

# Alas
wing_centre, wing_w, wing_h = (6.7, -5.4), 7.2, 22.5
for side in (-1, 1):
    phi = side * 9.4
    ell(side * wing_centre[0], wing_centre[1], wing_w, wing_h, ang=phi, fc=F_COL, ec=E_COL, lw=0.9, zorder=4)
    A = np.array([side * 4.6, 5.0])
    a_, b_ = wing_w / 2, wing_h / 2
    th = np.deg2rad(phi)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for alpha in (-90, -62, -118, -40, -140, -18):
        loc = np.array([a_ * np.cos(np.deg2rad(alpha)), b_ * np.sin(np.deg2rad(alpha))])
        tip = R @ loc + np.array([side * wing_centre[0], wing_centre[1]])
        poly([A, tip * 0.94 + A * 0.06], color=E_COL, lw=0.5, zorder=4.1)

# ---------------------------------------------------------------- Neuronas dentro del cuerpo
def sample(cx, cy, w, h, n):
    pts = []
    while len(pts) < n:
        u, v = rng.uniform(-1, 1, 2)
        if u * u + v * v <= 1:
            pts.append((cx + u * w / 2, cy + v * h / 2))
    return np.array(pts)


def knn_edges(pts, k=2):
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    e = set()
    for i in range(len(pts)):
        for j in np.argsort(d[i])[1:k + 1]:
            e.add(tuple(sorted((i, j))))
    return e


regions = {
    "optic_L": (sample(-3.7, 14.9, 3.0, 4.2, 34), C_OPTIC),
    "optic_R": (sample(3.7, 14.9, 3.0, 4.2, 34), C_OPTIC),
    "central": (sample(0, 14.7, 3.4, 3.1, 40), C_CENTRAL),
    "thoracic": (sample(0, 6.4, 3.8, 7.6, 55), C_VNC),
    "abdominal": (sample(0, -0.6, 2.6, 3.4, 18), C_VNC),
}
cord = np.column_stack([rng.uniform(-0.25, 0.25, 9), np.linspace(-2.6, -13.5, 9)])
regions["cord"] = (cord, C_VNC)
neck = np.column_stack([np.linspace(-0.9, 0.9, 6), np.array([12.3, 12.0, 11.6, 11.8, 12.1, 12.4]) - 1.3])
regions["neck"] = (neck, C_DESC)

# Aristas dentro de cada región
seg_by_color = {}
active_pts, all_pts = [], []
for name, (pts, col) in regions.items():
    edges = knn_edges(pts, 2 if name != "cord" else 1)
    segs = [W([pts[i], pts[j]]) for i, j in edges]
    ax.add_collection(LineCollection(segs, colors=col, linewidths=0.55, alpha=0.75, zorder=6))
    wp = W(pts)
    ax.scatter(wp[:, 0], wp[:, 1], s=6.5, c=col, ec="white", lw=0.25, zorder=7)

# Conexiones entre regiones (vía visual -> cerebro central -> descendentes -> cordón)
def link(pa, pb, col, lw=0.6, alpha=0.65):
    ax.add_collection(LineCollection([W([pa, pb])], colors=col, linewidths=lw, alpha=alpha, zorder=6))

cen = regions["central"][0]
for key in ("optic_L", "optic_R"):
    pts = regions[key][0]
    for i in rng.choice(len(pts), 9, replace=False):
        j = np.argmin(np.linalg.norm(cen - pts[i], axis=1))
        link(pts[i], cen[j], "#7c3aed")
for i in range(len(cen)):
    if rng.random() < 0.5:
        j = np.argmin(np.linalg.norm(neck - cen[i] + np.array([0, 0]), axis=1))
        link(cen[i], neck[j], C_DESC, 0.6)
th_pts = regions["thoracic"][0]
for i in range(len(neck)):
    j = np.argmin(np.linalg.norm(th_pts - neck[i], axis=1))
    link(neck[i], th_pts[j], C_DESC, 0.7, 0.7)
ab = regions["abdominal"][0]
for i in rng.choice(len(th_pts), 7, replace=False):
    if th_pts[i][1] < 5:
        j = np.argmin(np.linalg.norm(ab - th_pts[i], axis=1))
        link(th_pts[i], ab[j], C_VNC)
for i in range(4):
    link(ab[i], cord[0], C_VNC)
# salidas motoras a las patas
for side in (-1, 1):
    for leg in legs.values():
        j = np.argmin(np.linalg.norm(th_pts - np.array([0, leg[0][1]]), axis=1))
        link(th_pts[j], (side * leg[0][0] * 0.98, leg[0][1]), C_VNC, 0.6, 0.6)

# Ruta de propagación destacada: ojo -> lóbulo óptico -> cerebro central -> descendentes -> VNC
waypoints = [(-4.6, 15.4), (-2.6, 14.9), (-0.9, 14.6), (0, 12.6), (0.1, 10.6), (0.0, 8.0), (-0.3, 5.6)]
pool = np.vstack([regions[k][0] for k in ("optic_L", "central", "neck", "thoracic")])
path = []
for wp_ in waypoints:
    path.append(pool[np.argmin(np.linalg.norm(pool - np.array(wp_), axis=1))])
path = np.array(path)
pw = W(path)
ax.plot(pw[:, 0], pw[:, 1], color="#f59e0b", lw=2.6, alpha=0.25, zorder=8, solid_capstyle="round")
ax.plot(pw[:, 0], pw[:, 1], color="#ea580c", lw=1.1, zorder=8.1, solid_capstyle="round")

# Neuronas disparando (halo + núcleo)
act = [path]
for name, prob in (("optic_L", 0.45), ("optic_R", 0.12), ("central", 0.35), ("thoracic", 0.25), ("abdominal", 0.1)):
    pts = regions[name][0]
    act.append(pts[rng.random(len(pts)) < prob])
act = np.vstack(act)
aw = W(act)
ax.scatter(aw[:, 0], aw[:, 1], s=55, c=C_SPIKE, alpha=0.28, lw=0, zorder=8.5)
ax.scatter(aw[:, 0], aw[:, 1], s=11, c="#fde047", ec="#b45309", lw=0.5, zorder=8.6)

# Abanico de entradas visuales hacia el ojo
badge(PB[0] + 2.0, 49.0, 4)
src = (PA[1] + 0.6, 11.1)
for k, yy in enumerate(np.linspace(-6.3, -1.8, 5)):
    tgt = W([yy * 0.9 - 0.8, 15.4 + 0.5 * k / 4])[0] if False else W([-3.9 + 0.25 * (k - 2), 15.0 + 0.5 * (k - 2)])[0]
    arrow(src, tgt, color=C_A, rad=0.16, lw=0.8, ms=6, z=9)
ax.text(PB[0] + 1.0, 8.4, "visual input", fontsize=6.6, color=C_A, ha="left", va="center", style="italic", zorder=9)

# ---------------------------------------------------------------- Panel b: raster + ecuación
mid_b = (PB[0] + PB[1]) / 2
ax.text(mid_b, 48.3, r"$\tau\,\dot{v}_i = -(v_i - v_{\mathrm{rest}}) + \sum_j w_{ij}\,s_j(t)$",
        fontsize=11, ha="center", va="center", color=SLATE)
ax.text(mid_b, 45.8, r"leaky integrate-and-fire; $w_{ij}$ = FlyWire synapse counts (signed), no plasticity",
        fontsize=6.8, ha="center", va="center", color=GREY, style="italic")

rx0, rx1, ry0 = 44.0, 78.0, 33.4
rows = [("Visual", C_OPTIC, 3, 5, 32), ("Central", C_CENTRAL, 3, 20, 50), ("Descending", C_DESC, 3, 38, 72)]
rh = 0.72
y = ry0 + 0.6
for label, col, nrows, t0, t1 in rows:
    ax.text(rx0 - 0.9, y + nrows * rh / 2, label, fontsize=7, ha="right", va="center", color=col, fontweight="bold")
    for r_ in range(nrows):
        n = rng.integers(6, 12)
        ts = np.sort(rng.uniform(t0, t1, n))
        xs = rx0 + (ts / 100) * (rx1 - rx0)
        yy = y + r_ * rh
        ax.vlines(xs, yy + 0.12, yy + rh - 0.12, color=col, lw=0.8, zorder=6)
    y += nrows * rh + 0.5
ry1 = y - 0.3
ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fc="white", ec="#cbd5e1", lw=0.8, zorder=2))
for t in (0, 50, 100):
    xx = rx0 + t / 100 * (rx1 - rx0)
    ax.plot([xx, xx], [ry0 - 0.3, ry0], color=GREY, lw=0.6)
    ax.text(xx, ry0 - 1.1, str(t), fontsize=6.2, ha="center", va="center", color=GREY)
ax.text(rx1, ry0 - 2.2, "time (ms)", fontsize=6.4, ha="right", va="center", color=GREY)
ax.text(rx0 + 0.2, ry1 + 1.0, "Spike raster (one position)", fontsize=7.2, fontweight="bold", color=SLATE, va="center")
ax.annotate("", xy=(rx1 - 0.2, ry1 + 1.0), xytext=(rx1 - 7.0, ry1 + 1.0),
            arrowprops=dict(arrowstyle="-|>", color=C_DESC, lw=1.0, mutation_scale=6))
ax.text(rx1 - 7.4, ry1 + 1.0, "propagation", fontsize=6.2, color=C_DESC, ha="right", va="center", style="italic")

# Flecha de la mosca al raster
arrow((CX - S * 4.2, CY + S * 5.3), (rx0 + 6.5, ry0 - 0.5), color=C_DESC, rad=-0.22, lw=1.1, ms=7, z=9)
ax.text(rx0 + 7.8, ry0 - 3.4, "output spikes", fontsize=6.4, color=C_DESC, ha="left", va="center", style="italic")

# ---------------------------------------------------------------- Panel c
cxc = (PC[0] + PC[1]) / 2
arrow((PB[1] - 3.6, 38.2), (PC[0] + 1.2, 38.2), color="#334155", lw=1.4)
ax.text((PB[1] + PC[0]) / 2, 39.6, "rates", fontsize=6.6, color=GREY, ha="center", style="italic")

block(PC[0] + 1.2, 35.4, PC[1] - PC[0] - 2.4, 12.0)
badge(PC[0] + 2.0, 46.3, 5)
ax.text(cxc + 0.8, 44.7, "Firing-rate vector $\\mathbf{r}$", fontsize=7.6, fontweight="bold", color=SLATE, ha="center", va="center")
hts = rng.random(16) ** 1.5
bx0, bw = PC[0] + 3.2, (PC[1] - PC[0] - 6.4) / 16
cols = [C_OPTIC] * 3 + [C_CENTRAL] * 7 + [C_DESC] * 6
for i, h in enumerate(hts):
    ax.add_patch(Rectangle((bx0 + i * bw, 36.7), bw * 0.78, 1.0 + 5.4 * h, fc=cols[i], ec="none", zorder=4))
ax.plot([bx0 - 0.2, bx0 + 16 * bw], [36.7, 36.7], color=SLATE, lw=0.8, zorder=4)
arrow((cxc, 35.0), (cxc, 32.8))

block(PC[0] + 1.2, 22.0, PC[1] - PC[0] - 2.4, 10.4, ec=C_C, lw=1.5)
ax.text(cxc, 30.8, "Linear readout", fontsize=8.0, fontweight="bold", color=C_C, ha="center", va="center")
in_y = np.linspace(24.1, 28.6, 6)
out_y = np.linspace(25.0, 27.7, 4)
xi, xo = PC[0] + 5.0, PC[1] - 5.0
for a_ in in_y:
    for b_ in out_y:
        ax.plot([xi, xo], [a_, b_], color="#a7f3d0", lw=0.5, zorder=3)
ax.scatter([xi] * 6, in_y, s=12, c=[C_OPTIC, C_CENTRAL, C_CENTRAL, C_CENTRAL, C_DESC, C_DESC], ec="white", lw=0.4, zorder=4)
ax.scatter([xo] * 4, out_y, s=14, c=C_C, ec="white", lw=0.4, zorder=4)
ax.text(cxc, 22.9, r"$\hat{y} = W\mathbf{r} + b$", fontsize=7.4, color=SLATE, ha="center", va="center")
arrow((cxc, 21.6), (cxc, 19.6))

block(PC[0] + 1.2, 15.6, PC[1] - PC[0] - 2.4, 3.6)
ax.text(cxc, 17.4, "Legal-move mask \u2192 argmax", fontsize=7.0, fontweight="bold", color=SLATE, ha="center", va="center")
arrow((cxc, 15.2), (cxc, 13.6))

bsz = 7.4
draw_board(PC[0] + 1.4, 6.6, bsz, FEN, fs=5.4, move=MOVE_EXAMPLE)
ax.text(PC[0] + 1.4 + bsz + 1.0, 11.6, "Predicted", fontsize=7.2, fontweight="bold", color=SLATE, va="center")
ax.text(PC[0] + 1.4 + bsz + 1.0, 10.0, "move", fontsize=7.2, fontweight="bold", color=SLATE, va="center")
if RESULT_TEXT:
    ax.text(PC[0] + 1.4 + bsz + 1.0, 8.2, RESULT_TEXT, fontsize=6.8, color="#b91c1c", va="center", fontweight="bold")

# ----------------------------------------------------------------------------
# Leyenda inferior (posiciones medidas con el renderer)
# ----------------------------------------------------------------------------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def text_width(t):
    bb = t.get_window_extent(rend)
    (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return x1 - x0


items = [("Optic lobes / visual input", C_OPTIC), ("Central brain", C_CENTRAL), ("Descending neurons", C_DESC),
         ("Ventral nerve cord / motor", C_VNC), ("Spiking neuron", None)]
lx, ly = 3.0, 2.6
for label, col in items:
    if col is None:
        ax.scatter([lx], [ly], s=90, c=C_SPIKE, alpha=0.3, lw=0)
        ax.scatter([lx], [ly], s=22, c="#fde047", ec="#b45309", lw=0.6)
    else:
        ax.scatter([lx], [ly], s=22, c=col, lw=0)
    t = ax.text(lx + 1.2, ly, label, fontsize=7.4, va="center", color=SLATE)
    lx += 1.2 + text_width(t) + 3.4
ax.text(107.5, ly, "Neurons drawn schematically", fontsize=7.0, va="center", ha="right", color=GREY, style="italic")

fig.savefig(f"{OUT_BASENAME}.png", dpi=300)
fig.savefig(f"{OUT_BASENAME}.pdf")
print("OK ->", OUT_BASENAME)
