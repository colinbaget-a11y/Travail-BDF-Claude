"""
Pays vs Zone Euro — deux vues :
  1) Heatmap : écart de contribution (pays − ZE) au dernier point, par composante
     (4 agrégats ET 13 divisions COICOP)
  2) Stacked bar : écart de contribution par composante dans le temps,
     ligne = écart de YoY total (= somme des barres)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

ROOT = Path(__file__).parent
idx = pd.read_csv(ROOT / "data" / "hicp_midx.csv")
wts = pd.read_csv(ROOT / "data" / "hicp_weights.csv")
idx["date"] = pd.to_datetime(idx["period"], format="%Y-%m")

GEOS_NON_EA = ["DE", "FR", "IT", "ES", "NL"]
NAMES = {"EA": "ZE", "DE": "Allemagne", "FR": "France",
         "IT": "Italie", "ES": "Espagne", "NL": "Pays-Bas"}

AGG4 = ["SERV", "IGD_NNRG", "FOOD", "NRG"]
AGG4_LBL = {"SERV": "Services", "IGD_NNRG": "Biens hors énergie",
            "FOOD": "Alimentation", "NRG": "Énergie"}
DIV13 = [f"CP{i:02d}" for i in range(1, 14)]
DIV13_LBL = {
    "CP01": "Alimentation", "CP02": "Alcool & tabac", "CP03": "Habillement",
    "CP04": "Logement, énergie", "CP05": "Ameublement", "CP06": "Santé",
    "CP07": "Transport", "CP08": "Info & communication",
    "CP09": "Loisirs, culture", "CP10": "Éducation",
    "CP11": "Resto & héberg.", "CP12": "Assurance & finance",
    "CP13": "Soins & divers",
}
AGG4_COLORS = {"SERV": "#7E8FC6", "IGD_NNRG": "#E03A88",
               "FOOD": "#3EB371", "NRG": "#F58220"}


def ribe(geo: str, cols: list[str]) -> pd.DataFrame:
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
        if any(c not in w_wide.columns for c in cols):
            continue
        w_y = w_wide.loc[y, cols] / 1000.0
        w_y1 = w_wide.loc[y_prev, cols] / 1000.0
        if w_y.isna().any() or w_y1.isna().any():
            continue
        I_t, I_tm12 = wide.loc[t, cols], wide.loc[t_prev, cols]
        I_dy1, I_dy2 = wide.loc[dec_y1, cols], wide.loc[dec_y2, cols]
        if any(x.isna().any() for x in [I_t, I_tm12, I_dy1, I_dy2]):
            continue
        I_tot_tm12 = wide.loc[t_prev, "CP00"]
        I_tot_dy1 = wide.loc[dec_y1, "CP00"]
        I_tot_dy2 = wide.loc[dec_y2, "CP00"]
        term1 = (I_tot_dy1 / I_tot_tm12) * w_y * (I_t - I_dy1) / I_dy1
        term2 = (I_tot_dy2 / I_tot_tm12) * w_y1 * (I_dy1 - I_tm12) / I_dy2
        out[t] = (term1 + term2) * 100.0
    return pd.DataFrame(out).T.sort_index()


def gap(geo: str, cols: list[str]) -> pd.DataFrame:
    """Écart de contributions pays - ZE."""
    return ribe(geo, cols) - ribe("EA", cols)


# ============================================================
# 1) HEATMAP au dernier point — 4 agrégats puis 13 divisions
# ============================================================
REF_START, REF_END = "2015-01-01", "2024-12-31"  # 10 ans, inclut volatilité COVID


def make_heatmap(cols, labels_dict, title, outpath, figwidth, ma_months=3):
    """
    Heatmap colorée par z-score (vs 2015-2019), annotée valeur brute en pp.
    Cellules à |z|<1 : grisées (banal). |z|>2 : couleur saturée (inhabituel).
    """
    raw_rows, z_rows = [], []
    for g in GEOS_NON_EA:
        gap_ts = gap(g, cols).rolling(ma_months).mean()
        # Référence : écart-type historique du même écart pays-ZE
        ref = gap_ts.loc[REF_START:REF_END]
        sigma = ref.std()
        current = gap_ts.iloc[-1]
        z = current / sigma
        raw_rows.append(current.rename(g))
        z_rows.append(z.rename(g))
    H = pd.DataFrame(raw_rows)[cols]
    Z = pd.DataFrame(z_rows)[cols]

    fig, ax = plt.subplots(figsize=(figwidth, 3.6))
    vmax = max(3.0, min(6.0, np.nanpercentile(np.abs(Z.values), 95)))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([labels_dict[c] for c in cols], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(GEOS_NON_EA)))
    ax.set_yticklabels([NAMES[g] for g in GEOS_NON_EA], fontsize=10)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v_raw = H.values[i, j]
            v_z = Z.values[i, j]
            if abs(v_z) < 1:
                color, weight = "#666", "normal"
            elif abs(v_z) > vmax * 0.55:
                color, weight = "white", "bold"
            else:
                color, weight = "black", "bold" if abs(v_z) > 2 else "normal"
            ax.text(j, i, f"{v_raw:+.2f}", ha="center", va="center",
                    color=color, fontsize=8.5, weight=weight)

    end = ribe("EA", cols).index[-1]
    start = end - pd.DateOffset(months=ma_months - 1)
    period = f"{start.strftime('%b')}–{end.strftime('%b %Y')}"
    ax.set_title(
        f"{title}\nÉcart de contribution (pays − ZE), moy. {ma_months} mois ({period}). "
        f"Couleur = z-score vs {REF_START[:4]}–{REF_END[:4]}, valeur = pp.",
        fontsize=10.5)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("z-score (σ historique)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"OK → {outpath}  (vmax z = {vmax:.1f}σ)")
    return H, Z


H4, Z4 = make_heatmap(AGG4, AGG4_LBL, "Décomposition à 4 agrégats",
                      ROOT / "data" / "heatmap_gap_4agg.png", 8.5)
H13, Z13 = make_heatmap(DIV13, DIV13_LBL, "Décomposition à 13 divisions COICOP",
                        ROOT / "data" / "heatmap_gap_13div.png", 13)

# ============================================================
# 2) STACKED BAR de l'écart, mensuel 2023+
# ============================================================
START = "2023-01-01"


def plot_gap_bars(ax, geo: str):
    g = gap(geo, AGG4).loc[START:]
    yoy_country = (idx[(idx.geo==geo) & (idx.coicop=="CP00")]
                   .set_index("date")["value"].pct_change(12) * 100)
    yoy_ea = (idx[(idx.geo=="EA") & (idx.coicop=="CP00")]
              .set_index("date")["value"].pct_change(12) * 100)
    yoy_gap = (yoy_country - yoy_ea).loc[START:]

    x = g.index
    bottom_pos = np.zeros(len(g))
    bottom_neg = np.zeros(len(g))
    for code in AGG4:
        vals = g[code].values
        pos = np.where(vals > 0, vals, 0)
        neg = np.where(vals < 0, vals, 0)
        ax.bar(x, pos, bottom=bottom_pos, color=AGG4_COLORS[code], width=25,
               label=AGG4_LBL[code], linewidth=0)
        ax.bar(x, neg, bottom=bottom_neg, color=AGG4_COLORS[code], width=25, linewidth=0)
        bottom_pos += pos
        bottom_neg += neg
    ax.plot(yoy_gap.index, yoy_gap.values, color="#0A2A5E", linewidth=2.0,
            label="Écart YoY total")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"{NAMES[geo]} − ZE", fontsize=11)
    ax.set_ylabel("pp")
    ax.grid(alpha=0.25, axis="y")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey=True)
axes = axes.flatten()
for ax, g in zip(axes, GEOS_NON_EA):
    plot_gap_bars(ax, g)
axes[-1].axis("off")  # dernière case vide (on a 5 pays)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=10,
           bbox_to_anchor=(0.5, 1.00), frameon=False)
fig.suptitle("Écart de contributions vs ZE — décomposition à 4 agrégats (Ribe exacte)",
             fontsize=13, y=1.04)
fig.tight_layout()
out = ROOT / "data" / "gap_bars_4agg.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"OK → {out}")

# Top 3 du dernier point par pays — résumé console
print(f"\n=== Top 3 écarts les + INHABITUELS (z-score vs {REF_START[:4]}-{REF_END[:4]}) — 13 div ===")
for g in GEOS_NON_EA:
    z_row = Z13.loc[g].sort_values(key=abs, ascending=False)
    parts = [f"{DIV13_LBL[c]} (z={z_row[c]:+.1f}σ, {H13.loc[g,c]:+.2f} pp)"
             for c in z_row.index[:3]]
    print(f"  {NAMES[g]:10s}: " + " | ".join(parts))
