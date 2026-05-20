"""Graphiques d'erreurs : reconstruction Laspeyres et décomposition Ribe."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).parent
recon = pd.read_csv(ROOT / "data" / "laspeyres_check.csv", parse_dates=["date"])
ribe = pd.read_csv(ROOT / "data" / "ribe_contributions.csv", parse_dates=["date"])
recon["err_pct"] = (recon["I_recon"] - recon["I_official"]) / recon["I_official"] * 100

GEOS = ["EA", "DE", "FR", "IT", "ES", "NL"]
COLORS = dict(zip(GEOS, ["#1f4e79", "#000000", "#0055A4", "#009246", "#AA151B", "#FF6600"]))

# ---------- FIG 1 : erreurs de reconstruction Laspeyres ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
for g in GEOS:
    sub = recon[recon["geo"] == g].sort_values("date")
    ax.plot(sub["date"], sub["err_pct"], label=g, color=COLORS[g], linewidth=0.9, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Erreur relative (%)")
ax.set_title("Reconstruction Laspeyres — écart (I_recon − I_officiel) / I_officiel")
ax.grid(alpha=0.3)
ax.legend(ncol=6, loc="upper center", fontsize=9)

ax = axes[1]
data_box = [recon[recon["geo"] == g]["err_pct"].values for g in GEOS]
bp = ax.boxplot(data_box, labels=GEOS, showfliers=True, patch_artist=True)
for patch, g in zip(bp["boxes"], GEOS):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Erreur relative (%)")
ax.set_title("Distribution des erreurs de reconstruction par zone")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out1 = ROOT / "data" / "err_laspeyres.png"
fig.savefig(out1, dpi=120)
plt.close(fig)

# ---------- FIG 2 : erreurs Ribe ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax = axes[0]
for g in GEOS:
    sub = ribe[ribe["geo"] == g].sort_values("date")
    ax.plot(sub["date"], sub["diff"], label=g, color=COLORS[g], linewidth=0.9, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Écart (points de %)")
ax.set_title("Décomposition Ribe — Σ contributions − YoY officielle")
ax.grid(alpha=0.3)
ax.legend(ncol=6, loc="upper center", fontsize=9)

ax = axes[1]
data_box = [ribe[ribe["geo"] == g]["diff"].values for g in GEOS]
bp = ax.boxplot(data_box, labels=GEOS, showfliers=True, patch_artist=True)
for patch, g in zip(bp["boxes"], GEOS):
    patch.set_facecolor(COLORS[g])
    patch.set_alpha(0.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Écart (points de %)")
ax.set_title("Distribution des écarts Ribe par zone")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
out2 = ROOT / "data" / "err_ribe.png"
fig.savefig(out2, dpi=120)
plt.close(fig)

# ---------- FIG 3 : ZE focus — overlay des deux séries ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ea_recon = recon[recon["geo"] == "EA"].sort_values("date")
ea_ribe = ribe[ribe["geo"] == "EA"].sort_values("date")

ax = axes[0]
ax.plot(ea_recon["date"], ea_recon["I_official"], label="Officiel CP00", color="black", linewidth=1.2)
ax.plot(ea_recon["date"], ea_recon["I_recon"], label="Reconstruit Σ CP01..CP12", color="#d62728", linewidth=1.0, linestyle="--")
ax.set_ylabel("Indice (2015=100)")
ax.set_title("Zone Euro — Indice HICP officiel vs reconstruit Laspeyres")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(ea_ribe["date"], ea_ribe["yoy_official"], label="YoY officielle", color="black", linewidth=1.2)
ax.plot(ea_ribe["date"], ea_ribe["sum_contrib"], label="Σ contributions Ribe", color="#d62728", linewidth=1.0, linestyle="--")
ax.set_ylabel("% YoY")
ax.set_title("Zone Euro — YoY officielle vs somme des contributions Ribe")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
out3 = ROOT / "data" / "err_ea_overlay.png"
fig.savefig(out3, dpi=120)
plt.close(fig)

print(f"Sauvegardés :\n  {out1}\n  {out2}\n  {out3}")
