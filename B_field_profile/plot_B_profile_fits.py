import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['mathtext.fontset'] = 'cm'

DATA_STYLE = dict(marker="o", linestyle="none", markersize=3.5,
                   markeredgecolor="black", markeredgewidth=0.6, alpha=0.9)

# RHIC: B/mpi^2 = C0 (tau/tauB)^k / [1 + (tau/tauB)^(k+n)]
RHIC_FILES = {
    "0-20":  "Bfield_Au_200_cent_0-20.txt",
    "20-40": "Bfield_Au_200_cent_20-40.txt",
    "60-80": "Bfield_Au_200_cent_60-80.txt",
}
RHIC_INIT = {
    "0-20":  (3.6, 0.035, 0.55, 3.4),
    "20-40": (5.2, 0.055, 0.70, 3.4),
    "60-80": (7.2, 0.055, 1.00, 3.4),
}

def B_rhic(tau, C0, tauB, k, n):
    r = tau / tauB
    return C0 * r**k / (1.0 + r**(k + n))

RHIC = {}
for cent, fname in RHIC_FILES.items():
    tau_d, B_d = np.loadtxt(fname)[:, :2].T
    popt, _ = curve_fit(lambda t, *p: np.log(B_rhic(t, *p)), tau_d, np.log(B_d),
                         p0=RHIC_INIT[cent], bounds=(0, np.inf), maxfev=20000)
    RHIC[cent] = dict(zip(("C0", "tauB", "k", "n"), popt), datafile=fname)

# LHC: B/mpi^2 = C0 r^-k0 + C1 r^-k1 + C2 r^-k2
LHC_FILES = {
    "0-20":  "Bfield_Pb_5020_cent_0-20.txt",
    "20-40": "Bfield_Pb_5020_cent_20-40.txt",
    "60-80": "Bfield_Pb_5020_cent_60-80.txt",
}
LHC_INIT = (5.8e-4, 7.05, 1.03e-2, 4.31, 1.81e-2, 3.01, 0.05)

def B_lhc(tau, C0, k0, C1, k1, C2, k2, tauB):
    r = tau / tauB
    return C0 * r**-k0 + C1 * r**-k1 + C2 * r**-k2

LHC = {}
for cent, fname in LHC_FILES.items():
    tau_d, B_d = np.loadtxt(fname)[:, :2].T
    popt, _ = curve_fit(lambda t, *p: np.log(B_lhc(t, *p)), tau_d, np.log(B_d),
                         p0=LHC_INIT, bounds=(0, np.inf), maxfev=50000)
    LHC[cent] = dict(zip(("C0", "k0", "C1", "k1", "C2", "k2", "tauB"), popt), datafile=fname)

# plot fits vs. tabulated UrQMD data
tau = np.linspace(0.015, 1.0, 500)
fig, (ax_rhic, ax_lhc) = plt.subplots(1, 2, figsize=(13, 5.5))

for color, (cent, p) in zip(["C0", "C1", "C2"], RHIC.items()):
    ax_rhic.plot(tau, B_rhic(tau, p["C0"], p["tauB"], p["k"], p["n"]), color=color, lw=2.5, label=f"{cent}% (fit)")
    data = np.loadtxt(p["datafile"])
    ax_rhic.plot(data[::3, 0], data[::3, 1], color=color, label=f"{cent}% (UrQMD)", **DATA_STYLE)
ax_rhic.set_yscale("log")
ax_rhic.set_xlabel(r"$\tau$ [fm]")
ax_rhic.set_ylabel(r"$B(\tau)/m_\pi^2$")
ax_rhic.grid(alpha=0.3, which="both")
ax_rhic.legend(title="Centrality")

for color, (cent, p) in zip(["C0", "C1", "C2"], LHC.items()):
    ax_lhc.plot(tau, B_lhc(tau, p["C0"], p["k0"], p["C1"], p["k1"], p["C2"], p["k2"], p["tauB"]), color=color, lw=2.5, label=f"{cent}% (fit)")
    data = np.loadtxt(p["datafile"])
    ax_lhc.plot(data[::3, 0], data[::3, 1], color=color, label=f"{cent}% (UrQMD)", **DATA_STYLE)
ax_lhc.set_yscale("log")
ax_lhc.set_xlabel(r"$\tau$ [fm]")
ax_lhc.set_ylabel(r"$B(\tau)/m_\pi^2$")
ax_lhc.grid(alpha=0.3, which="both")
ax_lhc.legend(title="Centrality")

fig.tight_layout()
fig.savefig("B_profile_fits.png", dpi=150, bbox_inches="tight")
