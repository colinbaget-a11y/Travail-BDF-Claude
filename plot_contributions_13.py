"""
Contributions à l'inflation HICP — décomposition par les 13 divisions COICOP
(ECOICOP v.2). Formule de Ribe (1999) exacte.
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

DIV13 = [f"CP{i:02d}" for i in range(1, 14)]
LABELS = {
    "CP01": "Alimentation & boissons NA",
    "CP02": "Alcool & tabac",
    "CP03": "Habillement",
    "CP04": "Logement, eau, énergie",
    "CP05": "Ameublement, équip. ménager",
    "CP06": "Santé",
    "CP07": "Transport",
    "CP08": "Information & communication",
    "CP09": "Loisirs, sport, culture",
    "CP10": "Éducation",
    "CP11": "Restauration & hébergement",
    "CP12": "Assurance & services financiers",
    "CP13": "Soins, protection sociale, divers",
}
# Palette qualitative 13 couleurs
COLORS = dict(zip(DIV13, [
    "#3EB371", "#A04E9E", "#E8B0CE", "#F58220", "#9C6B4F", "#D62728",
    "#7E8FC6", "#17BECF", "#FFD92F", "#8C564B", "#E03A88", "#1F77B4", "#BCBD22",
]))


def ribe_contributions_13(geo: str) -> pd.DataFrame:
    sub = idx[idx["geo"] == geo]
    w = wts[wts["geo"] == geo]
    wide = sub.pivot(index="date", columns="coicop", values="value").sort_index()
    w_wide = w.pivot(index="year", columns="coicop", values="weight").sort_index()
    cols = DIV13

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


def plot_panel_13(ax, geo: str, start="2023-01-01"):
    contribs = ribe_contributions_13(geo).loc[start:]
    yoy = official_yoy(geo).loc[start:]
    x = contribs.index
    bottom_pos = np.zeros(len(contribs))
    bottom_neg = np.zeros(len(contribs))
    for code in DIV13:
        vals = contribs[code].values
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)
        ax.bar(x, pos, bottom=bottom_pos, color=COLORS[code], width=25,
               label=LABELS[code], linewidth=0)
        ax.bar(x, neg, bottom=bottom_neg, color=COLORS[code], width=25, linewidth=0)
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


# --- Grille 2x3 ---
fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharey=True)
for ax, g in zip(axes.flatten(), GEOS):
    plot_panel_13(ax, g, start="2023-01-01")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.04), frameon=False)
fig.suptitle("Contributions à l'inflation HICP — décomposition par les 13 divisions COICOP (Ribe exacte)",
             fontsize=13, y=1.00)
fig.tight_layout()
out = ROOT / "data" / "contributions_13div_2023_now.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"OK → {out}")

# --- Un graphique par pays ---
for g in GEOS:
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_panel_13(ax, g, start="2023-01-01")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title(f"{NAMES[g]} — Contributions HICP par division COICOP (mensuel, Ribe)")
    fig.tight_layout()
    out_g = ROOT / "data" / f"contributions_13div_{g}_2023_now.png"
    fig.savefig(out_g, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"OK → {out_g}")
