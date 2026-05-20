"""
Récupère les poids HICP.
- prc_hicp_iw (ECOICOP v.2) pour 2024+ (nouveau dataset, dimension coicop18, dispo 2026)
- prc_hicp_inw (ECOICOP v.1) pour 1996-2023 (historique complet, dimension coicop)
On les concatène en un seul CSV avec coicop renommé en CP00 (au lieu de TOTAL) pour rétro-compat.
"""
from pathlib import Path
import time
import eurostat
import pandas as pd

GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]
# v1 (prc_hicp_inw) : ECOICOP v.1, pas de CP13
COICOPS_V1 = [
    "CP00", "TOT_X_NRG_FOOD", "IGD_NNRG", "SERV", "NRG", "FOOD",
    "CP01", "CP02", "CP03", "CP04", "CP05", "CP06",
    "CP07", "CP08", "CP09", "CP10", "CP11", "CP12",
]
# v2 (prc_hicp_iw) : ECOICOP v.2, TOTAL au lieu de CP00, ajoute CP13
COICOPS_V2 = [
    "TOTAL", "TOT_X_NRG_FOOD", "IGD_NNRG", "SERV", "NRG", "FOOD",
    "CP01", "CP02", "CP03", "CP04", "CP05", "CP06",
    "CP07", "CP08", "CP09", "CP10", "CP11", "CP12", "CP13",
]
OUT = Path(__file__).parent / "data" / "hicp_weights.csv"


def fetch(ds: str, filter_pars: dict) -> pd.DataFrame:
    delay = 2
    for attempt in range(1, 5):
        try:
            df = eurostat.get_data_df(ds, filter_pars=filter_pars)
            if df is None or df.empty:
                return pd.DataFrame()
            return df
        except Exception as e:  # noqa: BLE001
            print(f"  {ds} tentative {attempt} échouée : {e!r}")
            if attempt == 4:
                raise
            time.sleep(delay)
            delay *= 2


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    id_cols = [c for c in df.columns if not c[:4].isdigit()]
    long = df.melt(id_vars=id_cols, var_name="year", value_name="weight")
    long = long.dropna(subset=["weight"])
    long = long.rename(columns={c: c.split("\\")[0] for c in long.columns})
    if "coicop18" in long.columns:
        long = long.rename(columns={"coicop18": "coicop"})
    long["coicop"] = long["coicop"].replace({"TOTAL": "CP00"})
    long["year"] = long["year"].astype(int)
    return long[["coicop", "geo", "year", "weight"]]


def main():
    # 1) Ancien dataset (v1) pour l'historique
    print("→ prc_hicp_inw (v1, 1996-2023+)")
    df_v1 = to_long(fetch("prc_hicp_inw", {"geo": GEOS, "coicop": COICOPS_V1}))
    print(f"  {len(df_v1)} lignes, années {df_v1['year'].min()}-{df_v1['year'].max()}")

    # 2) Nouveau dataset (v2) pour les années récentes
    print("→ prc_hicp_iw (v2, 2024-2026)")
    df_v2 = to_long(fetch("prc_hicp_iw", {"geo": GEOS, "coicop18": COICOPS_V2, "statinfo": ["IW"]}))
    print(f"  {len(df_v2)} lignes, années {df_v2['year'].min()}-{df_v2['year'].max()}")

    # Concat : v2 prioritaire sur les années qu'il couvre
    v2_years = set(df_v2["year"].unique())
    keep_v1 = df_v1[~df_v1["year"].isin(v2_years)]
    combined = pd.concat([keep_v1, df_v2], ignore_index=True)
    combined = combined.sort_values(["geo", "coicop", "year"]).reset_index(drop=True)

    OUT.parent.mkdir(exist_ok=True)
    combined.to_csv(OUT, index=False)
    print(f"\nOK — {len(combined)} lignes → {OUT}")
    print(f"Couverture : {combined['year'].min()} → {combined['year'].max()}")

    # Sanity check : somme CP01..CP12 ≈ 1000 par (geo, année)
    div = combined[combined["coicop"].str.match(r"CP0[1-9]|CP1[0-2]")]
    tot = div.groupby(["geo", "year"])["weight"].sum().reset_index()
    print("\nSomme CP01..CP12 — échantillon récent :")
    print(tot[tot["year"] >= 2023].pivot(index="year", columns="geo", values="weight"))


if __name__ == "__main__":
    main()
