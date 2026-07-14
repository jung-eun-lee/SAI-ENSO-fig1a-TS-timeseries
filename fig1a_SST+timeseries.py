#!/usr/bin/env python3
"""
DJF SST time series
  HIST (1980-2014) / SSP245 (2016-2069) / SAI×7 (2035-2069)
"""

import os
import glob

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from scipy.ndimage import uniform_filter1d


MM2IN = 1.0 / 25.4

# ──────────────────────────────────────────────────────────────────
# 0.  rcParams
# ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"        : "sans-serif",
    "font.sans-serif"    : ["Inter", "Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size"          : 7,
    "axes.linewidth"     : 0.7,
    "xtick.major.width"  : 0.7,
    "ytick.major.width"  : 0.7,
    "xtick.major.size"   : 3.5,
    "ytick.major.size"   : 3.5,
    "xtick.minor.size"   : 0.0,
    "ytick.minor.size"   : 0.0,
    "xtick.direction"    : "out",
    "ytick.direction"    : "out",
    "lines.linewidth"    : 0.8,
    "figure.dpi"         : 1000,
    "savefig.dpi"        : 1000,
    "savefig.bbox"       : "tight",
    "savefig.pad_inches" : 0.05,
    "pdf.fonttype"       : 42,
    "ps.fonttype"        : 42,
})

# ──────────────────────────────────────────────────────────────────
# 1.  Constants and paths
# ──────────────────────────────────────────────────────────────────
HIST_FILE = (
    "/home/hkim/model_data/sai_new/monthly/hist/atm/2d/"
    "CESM2LE.ensm.TS.p1980_2014.nc"
)
SSP_FILE  = (
    "/home/hkim/model_data/sai_new/monthly/ssp245/atm/2d/"
    "CESM2.SSP245.ensm.TS.20160101_20691231.nc"
)
SAI_DIR = "/home/hkim/model_data/sai_new/monthly/sai/atm/2d/"
FIGDIR  = "/home/jelee/research/ENSO_SAI/ai_polished/fig/"
os.makedirs(FIGDIR, exist_ok=True)

LAT_LIST = ["45S", "30S", "15S", "0N", "15N", "30N", "45N"]

COLORS = [
    "#2C2C2C",  # HIST    [0]
    "#C462C7",  # SSP245  [1]
    "#d07c58",  # SAI@45N [2]
    "#d9a189",  # SAI@30N [3]
    "#e5bead",  # SAI@15N [4]
    "#F1EAC8",  # SAI@0N  [5]
    "#B1C7B3",  # SAI@15S [6]
    "#72AAA1",  # SAI@30S [7]
    "#009392",  # SAI@45S [8]
]

LABELS = [
    "HIST",
    "SSP245",
    "SAI@45N", "SAI@30N", "SAI@15N",
    "SAI@0N",
    "SAI@15S", "SAI@30S", "SAI@45S",
]

# Data loading order: HIST(0), SSP(1), 45S(2)…45N(8)
PLOT_IDX = [0, 1, 8, 7, 6, 5, 4, 3, 2]

# ──────────────────────────────────────────────────────────────────
# 2.  Helpers
# ──────────────────────────────────────────────────────────────────
def area_avg_djf(ds, sm, em):
    da   = ds["x"].isel(time=slice(sm, em))
    lat  = da["latitude"].values.astype(np.float64)      # (lat,) — verify coord name
    raw  = da.values.astype(np.float64)             # (time, lat, lon)
    wgt  = np.cos(np.deg2rad(lat))                   # cos(lat) area weight
    num  = np.nansum(raw * wgt[None, :, None], axis=(1, 2))
    den  = np.nansum(np.broadcast_to(
               wgt[None, :, None], raw.shape) * ~np.isnan(raw), axis=(1, 2))
    ts   = num / den

    # ── DJF extraction ──────────
    n_months    = len(ts)
    start_month = sm % 12
    djf = []
    for i in range(n_months - 2):
        m = (start_month + i) % 12
        if m == 11:
            djf.append(np.mean(ts[i:i + 3]))
    return np.array(djf, dtype=np.float64)


def compute_ci(ts):
    valid   = ~np.isnan(ts)
    std_arr = np.full_like(ts, np.nan)
    idx     = np.where(valid)[0]
    for j in idx:
        w          = idx[(idx >= j - 2) & (idx <= j + 2)]
        std_arr[j] = ts[w].std(ddof=1) if len(w) > 1 else 0.0
    return ts + 1.96 * std_arr, ts - 1.96 * std_arr


# ──────────────────────────────────────────────────────────────────
# 3.  Load and process data
# ──────────────────────────────────────────────────────────────────
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

print("[1/3]  Loading HIST ...")
ds_hist  = xr.open_dataset(HIST_FILE, decode_times=time_coder)
sm_h     = (1980 - 1980) * 12
em_h     = (2014 - 1980 + 1) * 12
djf_hist = area_avg_djf(ds_hist, sm_h, em_h)
ds_hist.close()

print("[2/3]  Loading SSP245 ...")
ds_ssp   = xr.open_dataset(SSP_FILE, decode_times=time_coder)
sm_s     = (2016 - 2016) * 12
em_s     = (2069 - 2016 + 1) * 12
djf_ssp  = area_avg_djf(ds_ssp, sm_s, em_s)
ds_ssp.close()

print("[3/3]  Loading SAI ...")
djf_sai = []
for lat in LAT_LIST:
    pattern = os.path.join(SAI_DIR, "CESM2.INJANN*ensm*TS*")
    files   = sorted(glob.glob(pattern))
    matched = [
        f for f in files
        if f"INJANN{lat.upper()}" in os.path.basename(f).upper()
    ]
    if not matched:
        raise FileNotFoundError(f"No SAI file for {lat}")
    ds_sai = xr.open_dataset(matched[0], decode_times=time_coder)
    sm_sai = (2035 - 2035) * 12
    em_sai = (2069 - 2035 + 1) * 12
    djf    = area_avg_djf(ds_sai, sm_sai, em_sai)
    djf_sai.append(djf)
    ds_sai.close()
    print(f"  {lat:>4s}: {len(djf)} DJF years")

# ──────────────────────────────────────────────────────────────────
# 4.  Build common time axis and xres
# ──────────────────────────────────────────────────────────────────
TIME   = np.arange(1980, 2070)
N_TIME = len(TIME)
N_SCEN = 9
xres   = np.full((N_SCEN, N_TIME), np.nan, dtype=np.float64)

xres[0, 0:len(djf_hist)] = djf_hist

idx_s = 2016 - 1980
xres[1, idx_s : idx_s + len(djf_ssp)] = djf_ssp

idx_sai = 2035 - 1980
for ll, djf in enumerate(djf_sai):
    xres[ll + 2, idx_sai : idx_sai + len(djf)] = djf

xres_plot = xres[PLOT_IDX, :]

# ──────────────────────────────────────────────────────────────────
# 5.  Figure
# ──────────────────────────────────────────────────────────────────
print("[Plotting] ...")

fig_w = 100 * MM2IN
fig_h =  70 * MM2IN

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# 95% CI shading — HIST, SSP245
for ci_i, ci_color in zip([0, 1], [COLORS[0], COLORS[1]]):
    ts           = xres_plot[ci_i, :]
    upper, lower = compute_ci(ts)
    mask         = ~np.isnan(ts)
    ax.fill_between(
        TIME[mask], lower[mask], upper[mask],
        color     = ci_color,
        alpha     = 0.12,
        linewidth = 0,
        zorder    = -1,
    )

# Time series lines
for i in range(N_SCEN):
    mask = ~np.isnan(xres_plot[i, :])
    ax.plot(
        TIME[mask], xres_plot[i, mask],
        color          = COLORS[i],
        linewidth      = 0.9,
        zorder         = 3,
        alpha          = 1.0,
        solid_capstyle = "butt",
    )

# Vertical reference lines
vline_kw = dict(linewidth=0.5, linestyle="--", zorder=100, alpha=1.0)
ax.axvline(2035, color=COLORS[7], **vline_kw)
ax.axvline(2040, color="#A9A9A9", **vline_kw)

# Injection start arrow + label
y_arrow_inj = 7.10 + 0.1 + 9.05 + 273.15
x_start_inj = 2028
x_end_inj   = 2036
color_inj   = COLORS[7]

ax.annotate(
    "",
    xy         = (x_end_inj,   y_arrow_inj),
    xytext     = (x_start_inj, y_arrow_inj),
    arrowprops = dict(
        arrowstyle     = "->",
        color          = color_inj,
        lw             = 0.9,
        mutation_scale = 6,
    ),
    annotation_clip = False,
    zorder          = 5,
)
ax.text(
    x_start_inj - 3,
    y_arrow_inj,
    "Injection start",
    color        = color_inj,
    fontsize     = 5,
    fontfamily   = "Inter",
    fontweight   = "bold",
    va           = "center",
    ha           = "center",
    path_effects = [pe.withStroke(linewidth=2.0, foreground="white")],
    zorder       = 5,
)

# Analysis period arrow + label
y_arrow  = 7.10 + 0.1 + 9.05 + 273.15
x_start  = 2039
x_end    = 2069
color_ap = "#A9A9A9"

ax.annotate(
    "",
    xy         = (x_end,   y_arrow),
    xytext     = (x_start, y_arrow),
    arrowprops = dict(
        arrowstyle     = "<->",
        color          = color_ap,
        lw             = 0.9,
        mutation_scale = 6,
    ),
    annotation_clip = False,
    zorder          = 5,
)
ax.text(
    (x_start + x_end) / 2.0,
    y_arrow,
    "Analysis period",
    color        = color_ap,
    fontsize     = 5,
    fontfamily   = "Inter",
    fontweight   = "bold",
    va           = "center",
    ha           = "center",
    path_effects = [pe.withStroke(linewidth=2.0, foreground="white")],
    zorder       = 5,
)

# y-axis ticks
y_ticks = np.arange(287, 290, 1)        
ax.set_yticks(y_ticks)
ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.45, alpha=0.60, zorder=-2)
ax.set_yticklabels(
    [f"{int(t)}" for t in y_ticks],
    fontfamily = "Inter",
    fontsize   = 6.0,
)
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.25))   
ax.yaxis.grid(True, which="minor", color="#CCCCCC", linewidth=0.25,
              alpha=0.55 * 0.65, zorder=-3)
ax.tick_params(axis="y", which="minor", length=0, labelsize = 0.0, labelcolor = "white")          
ax.tick_params(axis="x", which="minor", length=0) 

# x-axis ticks
x_ticks = [1980, 2000, 2020, 2040, 2060]
ax.set_xticks(x_ticks)
ax.set_xticklabels(
    [str(t) for t in x_ticks],
    fontfamily = "Inter",
    fontsize   = 6.0,
    zorder = 200,
)

ax.set_xlim(x_ticks[0] - 1, 2068)
ax.set_ylim(286.65, 289.45)

# Spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_linewidth(0.7)
ax.spines["bottom"].set_color("#333333")
ax.spines["bottom"].set_zorder(300)

# Tick style
ax.tick_params(axis="x", which="major", length=3, pad=5)
ax.tick_params(axis="y", which="major", length=0, pad=4)

# Axis labels
ax.set_xlabel("Year",  fontsize=7, labelpad=6, fontfamily="Inter")
ax.set_ylabel("GMST [K]", fontsize=7, labelpad=6, fontfamily="Inter")

# Colored-text legend with background box
n_labels  = len(LABELS)
box_x     = 0.02 + 0.01
box_y_top = 1.00
row_h     = 0.042
box_pad   = 0.015
box_w     = 0.13
box_h     = n_labels * row_h + box_pad * 1.7

ax.add_patch(mpatches.FancyBboxPatch(
    (box_x - box_pad, box_y_top - box_h),
    box_w, box_h,
    boxstyle    = "square, pad=0",
    transform   = ax.transAxes,
    linewidth   = 0.5,
    edgecolor   = "#CCCCCC",
    facecolor   = "white",
    zorder      = 4,
    clip_on     = False,
))

for k, (label, color) in enumerate(zip(LABELS, COLORS)):
    ax.text(
        box_x,
        0.98 - k * row_h,
        label,
        transform    = ax.transAxes,
        fontsize     = 4.5,
        fontweight   = "bold",
        fontfamily   = "Inter",
        color        = color,
        va           = "top",
        ha           = "left",
        zorder       = 5,
        path_effects = [pe.withStroke(linewidth=1.5, foreground="white")],
    )

# ──────────────────────────────────────────────────────────────────
# 6.  Save
# ──────────────────────────────────────────────────────────────────
stem_pdf = os.path.join(FIGDIR, "pdf/FIG1-2_timeseries_SST")
stem_png = os.path.join(FIGDIR, "png/FIG1-2_timeseries_SST")
fig.savefig(stem_pdf + ".pdf", format="pdf")
fig.savefig(stem_png + ".png", dpi=1000)
plt.close(fig)
print(f"\nSaved → {stem_pdf}.pdf / {stem_png}.png")