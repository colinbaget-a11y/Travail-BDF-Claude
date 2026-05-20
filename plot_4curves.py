"""
Pour chaque pays, 4 courbes :
  (1) YoY officielle de CP00
  (2) YoY du CP00 reconstruit par Laspeyres (à partir de CP01..CP12)
  (3) Σ contributions Ribe à 12 divisions
  (4) Σ contributions Ribe à 4 agrégats (SERV + IGD_NNRG + NRG + FOOD)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
idx = pd.read_csv(ROOT / "data" / "hicp_midx.csv")
wts = pd.read_csv(ROOT / "data" / "hicp_weights.csv")
idx["date"] = pd.to_datetime(idx["period"], format="%Y-%m")

GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]
DIV12 = [f"CP{i:02d}" for i in range(1, 13)]
AGG4 = ["SERV", "IGD_NNRG", "NRG", "FOOD"]


def laspeyres_yoy(geo, components):
    """YoY implicite de la reconstruction Laspeyres."""
    sub = idx[idx["geo"] == geo]
    w = wts[wts["geo"] == geo]
    wide = sub.pivot(index="date", columns="coicop", values="value").sort_index()
    w_wide = w.pivot(index="year", columns="coicop", values="weight").sort_index()

    recon = {}
    for y in sorted({d.year for d in wide.index}):
        dec_prev = pd.Timestamp(year=y - 1, month=12, day=1)
        if dec_prev not in wide.index or y not in w_wide.index:
            continue
        if any(c not in w_wide.columns or pd.isna(w_wide.loc[y, c]) for c in components):
            continue
        I_dec = wide.loc[dec_prev, components]
        if I_dec.isna().any():
            continue
        I_tot_dec = wide.loc[dec_prev, "CP00"]
        w_y = w_wide.loc[y, components] / 1000.0
        months = wide.loc[wide.index.year == y]
        ratio = months[components] / I_dec
        recon_y = I_tot_dec * (ratio.mul(w_y, axis=1)).sum(axis=1)
        recon.update(recon_y.to_dict())
    recon_s = pd.Series(recon).sort_index()
    yoy = (recon_s / recon_s.shift(12) - 1) * 100
    return yoy


def ribe_sum(geo, components):
    """Somme des contributions Ribe pour les composantes données."""
    sub = idx[idx["geo"] == geo]
    w = wts[wts["geo"] == geo]
    wide = sub.pivot(index="date", columns="coicop", values="value").sort_index()
    w_wide = w.pivot(index="year", columns="coicop", values="weight").sort_index()

    out = {}
    for t in wide.index:
        t_prev = t - pd.DateOffset(years=1)
        y, y_prev = t.year, t.year - 1
        dec_y1 = pd.Timestamp(year=y - 1, month=12, day=1)
        dec_y2 = pd.Timestamp(year=y - 2, month=12, day=1)
        if any(d not in wide.index for d in [t_prev, dec_y1, dec_y2]):
            continue
        if y not in w_wide.index or y_prev not in w_wide.index:
            continue
        if any(c not in w_wide.columns for c in components):
            continue
        w_y = w_wide.loc[y, components] / 1000.0
        w_y1 = w_wide.loc[y_prev, components] / 1000.0
        if w_y.isna().any() or w_y1.isna().any():
            continue
        I_t = wide.loc[t, components]
        I_tm12 = wide.loc[t_prev, components]
        I_dy1 = wide.loc[dec_y1, components]
        I_dy2 = wide.loc[dec_y2, components]
        if any(x.isna().any() for x in [I_t, I_tm12, I_dy1, I_dy2]):
            continue
        I_tot_dy1 = wide.loc[dec_y1, "CP00"]
        I_tot_dy2 = wide.loc[dec_y2, "CP00"]
        I_tot_tm12 = wide.loc[t_prev, "CP00"]
        contrib = (
            I_tot_dy1 * w_y * I_t / I_dy1
            - I_tot_dy2 * w_y1 * I_tm12 / I_dy2
        ) / I_tot_tm12 * 100.0
        out[t] = contrib.sum()
    return pd.Series(out).sort_index()


fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
axes = axes.flatten()

for ax, g in zip(axes, GEOS):
    sub = idx[(idx["geo"] == g) & (idx["coicop"] == "CP00")].sort_values("date")
    off_yoy = sub.set_index("date")["value"].pct_change(12) * 100
    las_yoy = laspeyres_yoy(g, DIV12)
    ribe12 = ribe_sum(g, DIV12)
    ribe4 = ribe_sum(g, AGG4)

    # Restreint à 2002+ (4-agg dispo à partir de 2001, YoY à partir de 2002)
    mask = lambda s: s[s.index >= "2002-01-01"]

    ax.plot(mask(off_yoy).index, mask(off_yoy).values,
            label="Officiel CP00", color="black", linewidth=1.6)
    ax.plot(mask(las_yoy).index, mask(las_yoy).values,
            label="Laspeyres 12 div.", color="#1f77b4", linewidth=1.0, linestyle="--")
    ax.plot(mask(ribe12).index, mask(ribe12).values,
            label="Σ Ribe 12 div.", color="#d62728", linewidth=1.0, linestyle=":")
    ax.plot(mask(ribe4).index, mask(ribe4).values,
            label="Σ Ribe 4 agrégats", color="#2ca02c", linewidth=1.0, linestyle="-.")
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axhline(2, color="red", linewidth=0.4, alpha=0.5)
    ax.set_title(f"{g}")
    ax.set_ylabel("% YoY")
    ax.grid(alpha=0.3)
    if g == "EA":
        ax.legend(fontsize=8, loc="upper left")

fig.suptitle("HICP YoY — officiel vs reconstructions Laspeyres et sommes de contributions Ribe",
             fontsize=13, y=1.00)
fig.tight_layout()
out = ROOT / "data" / "comparison_4curves.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"OK → {out}")
