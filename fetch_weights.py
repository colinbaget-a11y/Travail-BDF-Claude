"""Récupère les poids HICP (prc_hicp_inw) pour les mêmes pays et codes COICOP."""
from pathlib import Path
import time
import eurostat
import pandas as pd

GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]
COICOPS = [
    "CP00", "TOT_X_NRG_FOOD", "IGD_NNRG", "SERV", "NRG", "FOOD",
    "CP01", "CP02", "CP03", "CP04", "CP05", "CP06",
    "CP07", "CP08", "CP09", "CP10", "CP11", "CP12",
]
OUT = Path(__file__).parent / "data" / "hicp_weights.csv"


def main():
    delay = 2
    for attempt in range(1, 5):
        try:
            df = eurostat.get_data_df(
                "prc_hicp_inw",
                filter_pars={"geo": GEOS, "coicop": COICOPS},
            )
            break
        except Exception as e:  # noqa: BLE001
            print(f"tentative {attempt} échouée : {e!r}")
            if attempt == 4:
                raise
            time.sleep(delay)
            delay *= 2

    id_cols = [c for c in df.columns if not c[:4].isdigit()]
    long = df.melt(id_vars=id_cols, var_name="year", value_name="weight")
    long = long.dropna(subset=["weight"])
    long = long.rename(columns={c: c.split("\\")[0] for c in long.columns})
    long["year"] = long["year"].astype(int)
    long = long.sort_values(["geo", "coicop", "year"]).reset_index(drop=True)
    OUT.parent.mkdir(exist_ok=True)
    long.to_csv(OUT, index=False)
    print(f"OK — {len(long)} lignes → {OUT}")
    print(long.head())
    print("\nCouverture annuelle :", long["year"].min(), "→", long["year"].max())
    # Sanity: CP01..CP12 doivent sommer ~1000 par (geo, année)
    div = long[long["coicop"].str.match(r"CP0[1-9]|CP1[0-2]")]
    tot = div.groupby(["geo", "year"])["weight"].sum().reset_index()
    print("\nSomme CP01..CP12 (devrait ≈ 1000) — échantillon :")
    print(tot.sample(10, random_state=0).sort_values(["geo", "year"]))


if __name__ == "__main__":
    main()
