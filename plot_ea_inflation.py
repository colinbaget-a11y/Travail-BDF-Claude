"""Trace l'inflation annuelle Zone Euro (CP00) à partir de l'indice HICP."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(Path(__file__).parent / "data" / "hicp_midx.csv")
ea = df[(df["geo"] == "EA") & (df["coicop"] == "CP00")].copy()
ea["date"] = pd.to_datetime(ea["period"], format="%Y-%m")
ea = ea.sort_values("date").set_index("date")
ea["yoy"] = ea["value"].pct_change(12) * 100

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(ea.index, ea["yoy"], color="#1f4e79", linewidth=1.5)
ax.axhline(2.0, color="red", linestyle="--", linewidth=0.8, label="Cible BCE (2 %)")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Zone Euro — Inflation annuelle HICP (CP00, glissement 12 mois)")
ax.set_ylabel("% YoY")
ax.set_xlabel("")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

out = Path(__file__).parent / "data" / "ea_inflation.png"
fig.savefig(out, dpi=120)
print(f"Sauvegardé : {out}")
print(f"Dernière obs : {ea.index[-1].date()} → {ea['yoy'].iloc[-1]:.2f} %")
print(f"Pic : {ea['yoy'].idxmax().date()} → {ea['yoy'].max():.2f} %")
