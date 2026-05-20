"""
Graphiques de contributions à l'inflation HICP — style BCE.
Barres empilées : 4 agrégats (Services, Biens hors énergie, Énergie, Alimentation)
Ligne : HICP total YoY officiel
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parent
idx = pd.read_csv(ROOT / "data" / "hicp_midx.csv")
wts = pd.read_csv(ROOT / "data" / "hicp_weights.csv")
idx["date"] = pd.to_datetime(idx["period"], format="%Y-%m")

GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]
NAMES = {"EA": "Zone Euro", "DE": "Allemagne", "FR": "France",
         "IT": "Italie", "ES": "Espagne", "NL": "Pays-Bas"}

# Style BCE — ordre des composantes et couleurs
COMPONENTS = [
    ("SERV",     "Services",                         "#7E8FC6"),
    ("IGD_NNRG", "Produits manufacturés hors énergie", "#E03A88"),
    ("FOOD",     "Alimentation",                     "#3EB371"),
    ("NRG",      "Énergie",                          "#F58220"),
]


def ribe_contributions(geo: str) -> pd.DataFrame:
    """
    Contributions à la YoY HICP — formule de Ribe (1999) exacte.

    La fenêtre 12 mois est scindée en deux sous-périodes :
      • de t-12 à Dec(y-1)  → poids de l'année y-1
      • de Dec(y-1) à t     → poids de l'année y

    C^i_t = [I_tot_Dec(y-1) / I_tot_t-12] · (w^i_y / 1000)
            · (I^i_t - I^i_Dec(y-1)) / I^i_Dec(y-1) · 100
          + [I_tot_Dec(y-2) / I_tot_t-12] · (w^i_(y-1) / 1000)
            · (I^i_Dec(y-1) - I^i_t-12) / I^i_Dec(y-2) · 100

    Propriété : Σ_i C^i_t = π_t exactement (identité Laspeyres).
    """
    sub = idx[idx["geo"] == geo]
    w = wts[wts["geo"] == geo]
    wide = sub.pivot(index="date", columns="coicop", values="value").sort_index()
    w_wide = w.pivot(index="year", columns="coicop", values="weight").sort_index()
    cols = [c for c, _, _ in COMPONENTS]

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
        if any(c not in w_wide.columns for c in cols):
            continue
        w_y = w_wide.loc[y, cols] / 1000.0
        w_y1 = w_wide.loc[y_prev, cols] / 1000.0
        if w_y.isna().any() or w_y1.isna().any():
            continue
        I_t = wide.loc[t, cols]
        I_tm12 = wide.loc[t_prev, cols]
        I_dy1 = wide.loc[dec_y1, cols]
        I_dy2 = wide.loc[dec_y2, cols]
        if any(x.isna().any() for x in [I_t, I_tm12, I_dy1, I_dy2]):
            continue
        I_tot_tm12 = wide.loc[t_prev, "CP00"]
        I_tot_dy1 = wide.loc[dec_y1, "CP00"]
        I_tot_dy2 = wide.loc[dec_y2, "CP00"]

        term1 = (I_tot_dy1 / I_tot_tm12) * w_y * (I_t - I_dy1) / I_dy1
        term2 = (I_tot_dy2 / I_tot_tm12) * w_y1 * (I_dy1 - I_tm12) / I_dy2
        out[t] = (term1 + term2) * 100.0
    df = pd.DataFrame(out).T.sort_index()
    df.index.name = "date"
    return df


def official_yoy(geo: str) -> pd.Series:
    sub = idx[(idx["geo"] == geo) & (idx["coicop"] == "CP00")].sort_values("date")
    return (sub.set_index("date")["value"].pct_change(12) * 100)


def plot_panel(ax, geo: str, start="2021-01-01", end=None, freq="M"):
    contribs = ribe_contributions(geo)
    yoy = official_yoy(geo)
    if freq == "Q":
        contribs = contribs.resample("QE").mean()
        yoy = yoy.resample("QE").mean()
    contribs = contribs.loc[start:end]
    yoy = yoy.loc[start:end]

    bar_w = 25 if freq == "M" else 70
    bottom_pos = np.zeros(len(contribs))
    bottom_neg = np.zeros(len(contribs))
    x = contribs.index
    for code, label, color in COMPONENTS:
        vals = contribs[code].values
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)
        ax.bar(x, pos, bottom=bottom_pos, color=color, width=bar_w, label=label, linewidth=0)
        ax.bar(x, neg, bottom=bottom_neg, color=color, width=bar_w, linewidth=0)
        bottom_pos += pos
        bottom_neg += neg

    ax.plot(yoy.index, yoy.values, color="#0A2A5E", linewidth=2.0, label="HICP (YoY)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(2, color="red", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_title(NAMES[geo], fontsize=11)
    ax.set_ylabel("% / pp")
    ax.grid(alpha=0.25, axis="y")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


START = "2020-01-01"

# --- Grille 2x3, mensuel, 2023 → dernier point ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
for ax, g in zip(axes.flatten(), GEOS):
    plot_panel(ax, g, start=START, freq="M")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, 1.00), frameon=False)
fig.suptitle("Contributions à l'inflation HICP (YoY mensuel, décomposition Ribe)",
             fontsize=13, y=1.04)
fig.tight_layout()
out = ROOT / "data" / "contributions_2023_now.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"OK → {out}")

# --- Un PNG par pays, mensuel, 2023 → dernier point ---
for g in GEOS:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_panel(ax, g, start=START, freq="M")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.set_title(f"{NAMES[g]} — Contributions à l'inflation HICP (YoY mensuel, Ribe)")
    fig.tight_layout()
    out_g = ROOT / "data" / f"contributions_{g}_2023_now.png"
    fig.savefig(out_g, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"OK → {out_g}")
