"""
Récupération HICP — dataset prc_hicp_minr (ECOICOP v.2).
Données les plus récentes (mises à jour mensuelles, ~M-1).
On renomme TOTAL → CP00 pour rester cohérent avec le reste du code.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import eurostat
import pandas as pd

DATASET = "prc_hicp_minr"  # ECOICOP v.2, indices et taux mensuels

GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]

# Agrégats spéciaux + 12 divisions. NB: TOTAL au lieu de CP00 dans ce dataset.
COICOPS = [
    "TOTAL", "TOT_X_NRG_FOOD", "IGD_NNRG", "SERV", "NRG", "FOOD",
    "CP01", "CP02", "CP03", "CP04", "CP05", "CP06",
    "CP07", "CP08", "CP09", "CP10", "CP11", "CP12", "CP13",
]

UNIT = "I15"  # Indice 2015 = 100

OUT_DIR = Path(__file__).parent / "data"
OUT_FILE = OUT_DIR / "hicp_midx.csv"


def fetch_with_retry(filter_pars: dict, max_attempts: int = 4) -> pd.DataFrame:
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
    id_cols = [c for c in df.columns if not c[:4].isdigit()]
    long = df.melt(id_vars=id_cols, var_name="period", value_name="value")
    long = long.dropna(subset=["value"])
    long = long.rename(columns={c: c.split("\\")[0] for c in long.columns})
    # Le dataset utilise coicop18 — on l'aligne sur 'coicop' pour le reste du code.
    if "coicop18" in long.columns:
        long = long.rename(columns={"coicop18": "coicop"})
    # TOTAL → CP00 pour rester rétro-compatible
    long["coicop"] = long["coicop"].replace({"TOTAL": "CP00"})
    return long.sort_values(["geo", "coicop", "period"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Télécharge HICP (prc_hicp_minr, ECOICOP v.2).")
    parser.add_argument("--refresh", action="store_true", help="Force le re-téléchargement.")
    args = parser.parse_args()

    if OUT_FILE.exists() and not args.refresh:
        print(f"Cache présent : {OUT_FILE} (utilise --refresh pour rafraîchir).")
        df = pd.read_csv(OUT_FILE)
        print(df.head())
        print(f"\n{len(df)} lignes en cache.")
        return 0

    filter_pars = {"geo": GEOS, "coicop18": COICOPS, "unit": [UNIT]}
    raw = fetch_with_retry(filter_pars)
    long = to_long(raw)

    OUT_DIR.mkdir(exist_ok=True)
    long.to_csv(OUT_FILE, index=False)

    print(f"\nOK — {len(long)} observations écrites dans {OUT_FILE}")
    print(f"Plage : {long['period'].min()} → {long['period'].max()}")
    print("\nDernières obs CP00 par zone :")
    last = long[long["coicop"] == "CP00"].groupby("geo")["period"].max()
    print(last)
    return 0


if __name__ == "__main__":
    sys.exit(main())
