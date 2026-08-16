"""
HD 93205: 3D Density Isosurface Figure (Publication Figure + Rotation GIF)

Author: Asrate Gaulle Wedegebral
Affiliation: Department of Physics, Ariel University, Ariel, Israel

Figure Features:
    - Marching-cubes isosurfaces at fixed density thresholds
    - Gaussian-smoothed density field (removes marching-cubes noise)
    - Zoomable sub-domain in AU
    - Clean journal-style axes, bounding box, and legend
    - Static PNG/PDF render plus a rotating-view GIF of the same isosurfaces

Project: Hydrodynamic simulations of the colliding-wind binary HD 93205
"""

import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from skimage import measure

import yt

# ---------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------
BASE_PATH = "/home/asrat/mass_hd93w"
SNAPSHOT = os.path.join(BASE_PATH, "massive_star_hdf5_plt_cnt_")  # TODO: append the snapshot index, e.g. "0030"
OUT_STEM = "HD93205_density_isosurface"

AU = 1.496e13

ISO_LEVELS = [1e-17, 1e-16, 1e-15, 1e-14, 1e-13]

COLORS = [
    "#2c7bb6",  # low density
    "#1f968b",
    "#5ec962",
    "#fde725",
    "#fdae61",  # high density
]

ALPHAS = [0.1, 0.2, 0.35, 0.5, 0.7]

ZOOM = 1.0

# GIF rotation settings
GIF_NAME = f"{OUT_STEM}.gif"
GIF_N_FRAMES = 60
GIF_FPS = 15
GIF_ELEV = 25

# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------
print("Loading dataset...")
ds = yt.load(SNAPSHOT)
ds.force_periodicity()

t_day = float(ds.current_time.to("day"))

cg = ds.arbitrary_grid(
    ds.domain_left_edge, ds.domain_right_edge, dims=[256, 256, 128]
)

rho = cg[("gas", "density")].to_value("g/cm**3")
rho[rho <= 0] = np.nan
rho = gaussian_filter(rho, sigma=0.6)

log_rho = np.log10(rho)
nx, ny, nz = rho.shape

# ---------------------------------------------------------
# 3. DOMAIN (AU)
# ---------------------------------------------------------
XMIN_FULL, XMAX_FULL = -1.0, 1.0
YMIN_FULL, YMAX_FULL = -1.0, 1.0
ZMIN_FULL, ZMAX_FULL = -0.5, 0.5

xc, yc, zc = 0.0, 0.0, 0.0

dx = (XMAX_FULL - XMIN_FULL) * ZOOM / 2
dy = (YMAX_FULL - YMIN_FULL) * ZOOM / 2
dz = (ZMAX_FULL - ZMIN_FULL) * ZOOM / 2

XMIN, XMAX = xc - dx, xc + dx
YMIN, YMAX = yc - dy, yc + dy
ZMIN, ZMAX = zc - dz, zc + dz


def phys_to_idx(val, vmin_full, vmax_full, n):
    return int(np.clip((val - vmin_full) / (vmax_full - vmin_full) * n, 0, n - 1))


ix0 = phys_to_idx(XMIN, XMIN_FULL, XMAX_FULL, nx)
ix1 = phys_to_idx(XMAX, XMIN_FULL, XMAX_FULL, nx)

iy0 = phys_to_idx(YMIN, YMIN_FULL, YMAX_FULL, ny)
iy1 = phys_to_idx(YMAX, YMIN_FULL, YMAX_FULL, ny)

iz0 = phys_to_idx(ZMIN, ZMIN_FULL, ZMAX_FULL, nz)
iz1 = phys_to_idx(ZMAX, ZMIN_FULL, ZMAX_FULL, nz)

log_rho_zoom = log_rho[ix0:ix1, iy0:iy1, iz0:iz1]

if log_rho_zoom.size == 0:
    raise ValueError("Zoom too small -- empty region selected.")

vmin = np.nanmin(log_rho_zoom)
vmax = np.nanmax(log_rho_zoom)

nxz, nyz, nzz = log_rho_zoom.shape
spacing = (
    (XMAX - XMIN) / max(nxz, 1),
    (YMAX - YMIN) / max(nyz, 1),
    (ZMAX - ZMIN) / max(nzz, 1),
)

# safer data (avoid fake surfaces from NaNs)
data_clean = np.nan_to_num(log_rho_zoom, nan=vmin - 1)

# ---------------------------------------------------------
# 4. FIGURE
# ---------------------------------------------------------
fig = plt.figure(figsize=(8.2, 6.8))
ax = fig.add_subplot(111, projection="3d")

ax.view_init(elev=GIF_ELEV, azim=35)
ax.set_box_aspect([XMAX - XMIN, YMAX - YMIN, ZMAX - ZMIN])

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("#cccccc")

ax.grid(False)

# ---------------------------------------------------------
# 5. ISOSURFACES
# ---------------------------------------------------------
print("Rendering isosurfaces...")

for i, level in enumerate(ISO_LEVELS):
    log_level = np.log10(level)

    if not (vmin < log_level < vmax):
        continue

    try:
        verts, faces, _, _ = measure.marching_cubes(
            data_clean, level=log_level, spacing=spacing
        )

        verts[:, 0] += XMIN
        verts[:, 1] += YMIN
        verts[:, 2] += ZMIN

        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            color=COLORS[i],
            alpha=ALPHAS[i],
            linewidth=0,
            shade=True,
        )

    except RuntimeError:
        continue

# ---------------------------------------------------------
# 6. AXES
# ---------------------------------------------------------
ax.set_xlim(XMIN, XMAX)
ax.set_ylim(YMIN, YMAX)
ax.set_zlim(ZMIN, ZMAX)

ax.set_xticks(np.linspace(XMIN, XMAX, 5))
ax.set_yticks(np.linspace(YMIN, YMAX, 5))
ax.set_zticks(np.linspace(ZMIN, ZMAX, 5))

ax.set_xlabel(r"$x\ (\mathrm{AU})$")
ax.set_ylabel(r"$y\ (\mathrm{AU})$")
ax.set_zlabel("")

fig.text(0.06, 0.5, r"$z\ (\mathrm{AU})$", rotation=90, va="center", ha="center")

# ---------------------------------------------------------
# 7. BOUNDING BOX
# ---------------------------------------------------------
def draw_box(ax, xmin, xmax, ymin, ymax, zmin, zmax):
    corners = [
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot(
            [corners[i][0], corners[j][0]],
            [corners[i][1], corners[j][1]],
            [corners[i][2], corners[j][2]],
            color="k",
            alpha=0.2,
            lw=0.6,
        )


draw_box(ax, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

# ---------------------------------------------------------
# 8. LEGEND
# ---------------------------------------------------------
handles = [
    Patch(
        color=COLORS[i], alpha=ALPHAS[i],
        label=rf"$10^{{{int(np.log10(lv))}}}$ g cm$^{{-3}}$",
    )
    for i, lv in enumerate(ISO_LEVELS)
]

ax.legend(handles=handles, title="Isodensity surface", loc="upper left")

# ---------------------------------------------------------
# 9. SAVE STATIC FIGURE
# ---------------------------------------------------------
plt.tight_layout()

plt.savefig(f"{OUT_STEM}.png", dpi=300)
plt.savefig(f"{OUT_STEM}.pdf", dpi=300)

print("Saved:", f"{OUT_STEM}.png", "/", f"{OUT_STEM}.pdf")

# ---------------------------------------------------------
# 10. SAVE ROTATING ISOSURFACE GIF
# ---------------------------------------------------------
print("Rendering rotation GIF...")

azimuths = np.linspace(0, 360, GIF_N_FRAMES, endpoint=False)

with imageio.get_writer(GIF_NAME, mode="I", fps=GIF_FPS) as writer:
    for i, azim in enumerate(azimuths):
        ax.view_init(elev=GIF_ELEV, azim=azim)
        fig.canvas.draw()

        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)

        print(f"  frame {i + 1}/{GIF_N_FRAMES} done")

plt.close(fig)

print(f"\nSaved: {GIF_NAME}")
print(f"t = {t_day:.1f} d")
