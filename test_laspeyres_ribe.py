"""
Vérification empirique de la méthodologie HICP :
  1. Reconstruction Laspeyres du total CP00 à partir des 12 divisions CP01..CP12
     et des poids annuels publiés (prc_hicp_inw, déjà price-updated à Dec y-1).
  2. Décomposition Ribe : les contributions doivent sommer à la YoY officielle.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
idx = pd.read_csv(ROOT / "data" / "hicp_midx.csv")
wts = pd.read_csv(ROOT / "data" / "hicp_weights.csv")

idx["date"] = pd.to_datetime(idx["period"], format="%Y-%m")
idx["year"] = idx["date"].dt.year
idx["month"] = idx["date"].dt.month

DIV = [f"CP{i:02d}" for i in range(1, 13)]
GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]


def reconstruct_one_geo(geo: str):
    sub = idx[idx["geo"] == geo].copy()
    w = wts[wts["geo"] == geo].copy()

    # wide: index par date, colonnes par COICOP
    wide = sub.pivot(index="date", columns="coicop", values="value").sort_index()
    # poids wide : par année & COICOP
    w_wide = w.pivot(index="year", columns="coicop", values="weight").sort_index()

    years = sorted(set(wide.index.year))
    out = []
    for y in years:
        # Dec y-1 doit exister dans l'index officiel
        dec_prev = pd.Timestamp(year=y - 1, month=12, day=1)
        if dec_prev not in wide.index or y not in w_wide.index:
            continue
        I_div_dec = wide.loc[dec_prev, DIV]  # I^i_Dec,y-1
        I_tot_dec = wide.loc[dec_prev, "CP00"]
        w_y = w_wide.loc[y, DIV] / 1000.0
        if w_y.isna().any() or I_div_dec.isna().any():
            continue
        # Mois de l'année y
        months = wide.loc[wide.index.year == y]
        ratio = months[DIV] / I_div_dec  # I^i_t / I^i_Dec,y-1, broadcast par colonne
        recon = I_tot_dec * (ratio.mul(w_y, axis=1)).sum(axis=1)
        for d, val in recon.items():
            out.append({
                "geo": geo, "date": d,
                "I_official": wide.loc[d, "CP00"],
                "I_recon": val,
            })
    return pd.DataFrame(out)


def ribe_one_geo(geo: str):
    sub = idx[idx["geo"] == geo]
    w = wts[wts["geo"] == geo]
    wide = sub.pivot(index="date", columns="coicop", values="value").sort_index()
    w_wide = w.pivot(index="year", columns="coicop", values="weight").sort_index()

    out = []
    for t in wide.index:
        t_prev = t - pd.DateOffset(years=1)
        if t_prev not in wide.index:
            continue
        y = t.year
        y_prev = y - 1
        dec_y1 = pd.Timestamp(year=y - 1, month=12, day=1)
        dec_y2 = pd.Timestamp(year=y - 2, month=12, day=1)
        if dec_y1 not in wide.index or dec_y2 not in wide.index:
            continue
        if y not in w_wide.index or y_prev not in w_wide.index:
            continue
        I_t = wide.loc[t, DIV]
        I_tm12 = wide.loc[t_prev, DIV]
        I_dec_y1 = wide.loc[dec_y1, DIV]
        I_dec_y2 = wide.loc[dec_y2, DIV]
        I_tot_dec_y1 = wide.loc[dec_y1, "CP00"]
        I_tot_dec_y2 = wide.loc[dec_y2, "CP00"]
        I_tot_tm12 = wide.loc[t_prev, "CP00"]
        I_tot_t = wide.loc[t, "CP00"]

        w_y = w_wide.loc[y, DIV] / 1000.0
        w_y1 = w_wide.loc[y_prev, DIV] / 1000.0
        if any(x.isna().any() for x in [I_t, I_tm12, I_dec_y1, I_dec_y2, w_y, w_y1]):
            continue

        contrib = (
            I_tot_dec_y1 * w_y * I_t / I_dec_y1
            - I_tot_dec_y2 * w_y1 * I_tm12 / I_dec_y2
        ) / I_tot_tm12 * 100.0
        yoy_official = (I_tot_t / I_tot_tm12 - 1) * 100.0
        out.append({
            "geo": geo, "date": t,
            "yoy_official": yoy_official,
            "sum_contrib": contrib.sum(),
            **{f"C_{c}": contrib[c] for c in DIV},
        })
    return pd.DataFrame(out)


print("=" * 70)
print("TEST 1 — Reconstruction Laspeyres (CP00 vs Σ CP01..CP12 pondéré)")
print("=" * 70)
all_recon = []
for g in GEOS:
    r = reconstruct_one_geo(g)
    r["diff"] = r["I_recon"] - r["I_official"]
    r["abs_diff"] = r["diff"].abs()
    all_recon.append(r)
    print(f"{g}: n={len(r):5d}  RMSE={np.sqrt((r['diff']**2).mean()):.4f}  "
          f"max|err|={r['abs_diff'].max():.4f}  "
          f"max|err|/I={r['abs_diff'].max()/r['I_official'].mean()*100:.3f}%")

print()
print("=" * 70)
print("TEST 2 — Décomposition Ribe (Σ contributions vs YoY officielle)")
print("=" * 70)
all_ribe = []
for g in GEOS:
    r = ribe_one_geo(g)
    r["diff"] = r["sum_contrib"] - r["yoy_official"]
    all_ribe.append(r)
    print(f"{g}: n={len(r):5d}  "
          f"RMSE={np.sqrt((r['diff']**2).mean()):.4f} pp  "
          f"max|err|={r['diff'].abs().max():.4f} pp")

# Sauvegarde
out_recon = pd.concat(all_recon, ignore_index=True)
out_ribe = pd.concat(all_ribe, ignore_index=True)
out_recon.to_csv(ROOT / "data" / "laspeyres_check.csv", index=False)
out_ribe.to_csv(ROOT / "data" / "ribe_contributions.csv", index=False)

# Aperçu Ribe ZE pour les pires écarts
print("\n--- 5 plus gros écarts Ribe (toutes zones) ---")
worst = pd.concat(all_ribe, ignore_index=True).reindex(
    pd.concat(all_ribe).reset_index(drop=True)["diff"].abs().sort_values(ascending=False).index
).head(5)
print(worst[["geo", "date", "yoy_official", "sum_contrib", "diff"]])
