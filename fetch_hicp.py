"""
Récupération des données HICP (Eurostat) — indice mensuel base 2015=100.

Stratégie :
- Une seule requête API avec filtres (geo, coicop, unit) — évite de tirer
  tout le dataset (Eurostat plafonne à 50 catégories par requête).
- Retry avec backoff exponentiel sur erreurs réseau.
- Cache CSV local : on ne re-télécharge que si --refresh est passé.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import eurostat
import pandas as pd

DATASET = "prc_hicp_midx"  # HICP - indice mensuel (2015 = 100)

GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]  # Zone Euro + 5 pays

# Agrégats spéciaux + 12 divisions COICOP
COICOPS = [
    "CP00",            # Total (all-items HICP)
    "TOT_X_NRG_FOOD",  # Core (hors énergie, alimentation, alcool, tabac)
    "IGD_NNRG",        # Biens industriels hors énergie
    "SERV",            # Services
    "NRG",             # Énergie
    "FOOD",            # Alimentation (y c. alcool & tabac)
    "CP01", "CP02", "CP03", "CP04", "CP05", "CP06",
    "CP07", "CP08", "CP09", "CP10", "CP11", "CP12",
]

UNIT = "I15"  # Indice 2015 = 100

OUT_DIR = Path(__file__).parent / "data"
OUT_FILE = OUT_DIR / "hicp_midx.csv"


def fetch_with_retry(filter_pars: dict, max_attempts: int = 4) -> pd.DataFrame:
    """Appel Eurostat avec backoff exponentiel (2s, 4s, 8s)."""
    delay = 2
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[{attempt}/{max_attempts}] Requête Eurostat {DATASET}...", flush=True)
            df = eurostat.get_data_df(DATASET, filter_pars=filter_pars)
            if df is None or df.empty:
                raise RuntimeError("Réponse vide d'Eurostat")
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  échec : {e!r}", flush=True)
            if attempt < max_attempts:
                print(f"  retry dans {delay}s...", flush=True)
                time.sleep(delay)
                delay *= 2
    raise RuntimeError(f"Échec après {max_attempts} tentatives : {last_err!r}")


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Eurostat renvoie un format large (colonnes = périodes). On passe en long."""
    id_cols = [c for c in df.columns if not c[:4].isdigit()]
    long = df.melt(id_vars=id_cols, var_name="period", value_name="value")
    long = long.dropna(subset=["value"])
    # 'geo\\TIME_PERIOD' devient 'geo' pour propreté
    long = long.rename(columns={c: c.split("\\")[0] for c in long.columns})
    return long.sort_values(["geo", "coicop", "period"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Télécharge HICP Eurostat (indice mensuel).")
    parser.add_argument("--refresh", action="store_true", help="Force le re-téléchargement.")
    args = parser.parse_args()

    if OUT_FILE.exists() and not args.refresh:
        print(f"Cache présent : {OUT_FILE} (utilise --refresh pour rafraîchir).")
        df = pd.read_csv(OUT_FILE)
        print(df.head())
        print(f"\n{len(df)} lignes en cache.")
        return 0

    filter_pars = {"geo": GEOS, "coicop": COICOPS, "unit": [UNIT]}
    raw = fetch_with_retry(filter_pars)
    long = to_long(raw)

    OUT_DIR.mkdir(exist_ok=True)
    long.to_csv(OUT_FILE, index=False)

    print(f"\nOK — {len(long)} observations écrites dans {OUT_FILE}")
    print("\nAperçu :")
    print(long.head())
    print("\nCouverture :")
    print(long.groupby(["geo", "coicop"]).size().head(20))
    return 0


if __name__ == "__main__":
    sys.exit(main())
