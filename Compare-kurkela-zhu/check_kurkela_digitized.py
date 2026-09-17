import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['mathtext.fontset'] = 'cm'

# Kurkela & Zhu (arXiv:1506.06647) Fig. 3 setup: xi=4, lambda=1, Qs*tau0=1, B=0
Qs, xi, lam = 2.0, 4.0, 1.0
tau0 = 1.0 / Qs
Nc = 3.0
dA = Nc**2 - 1.0
nu_g = 2.0 * dA
c_ell = 1.25
pT_mean = 1.8 * Qs
REG2 = 1.0e-24

N = 70
pxmax = pymax = 16.0
petamax = tau0 * 8.0
pxe = np.linspace(-pxmax, pxmax, N + 1); px = 0.5 * (pxe[:-1] + pxe[1:])
pye = np.linspace(-pymax, pymax, N + 1); py = 0.5 * (pye[:-1] + pye[1:])
pee = np.linspace(-petamax, petamax, N + 1); peta = 0.5 * (pee[:-1] + pee[1:])
dpx, dpy, dpeta = px[1] - px[0], py[1] - py[0], peta[1] - peta[0]
PX, PY, PETA = np.meshgrid(px, py, peta, indexing="ij")

def kurkela_shape(px, py, peta, tau0_):
    pT = np.sqrt(px**2 + py**2)
    pz0 = -peta / tau0_
    rhat2 = (pT / pT_mean)**2 + (xi * pz0 / pT_mean)**2
    rhat = np.sqrt(rhat2 + 1e-30)
    return np.exp(-(2.0 / 3.0) * rhat2) / rhat

def p_tau_of(px, py, peta, tau):
    return np.sqrt(px**2 + py**2 + (peta / tau)**2 + REG2)

shape0 = kurkela_shape(PX, PY, PETA, tau0)

# amplitude fixed by the comoving-energy target, Eq. kurkelaEnergyTargetExplicit
ptau0 = p_tau_of(PX, PY, PETA, tau0)
IE = nu_g * np.sum(ptau0 * shape0) * dpx * dpy * (dpeta / tau0) / (2 * np.pi)**3
target = pT_mean * c_ell * dA * Qs**2 / (np.pi * lam)
f0 = (target / (tau0 * IE)) * shape0

# B=0: f(tau) = f0 exactly, so no need to trace characteristics here.
# P_L, P_T without the nu_g factor -- matches Kurkela & Zhu's own plotted P_T
# (their number density carries nu_g, their P_L/P_T never explicitly does).
def pressures_no_nug(fvals, tau_):
    pz = PETA / tau_
    ptau = p_tau_of(PX, PY, PETA, tau_)
    w = dpx * dpy * (dpeta / tau_)
    PL = np.sum((pz**2 / ptau) * fvals) * w / (2 * np.pi)**3
    PT = 0.5 * np.sum(((PX**2 + PY**2) / ptau) * fvals) * w / (2 * np.pi)**3
    return PL, PT

Qtau_vals = np.logspace(np.log10(1.0), np.log10(15.0), 30)
tau_vals = Qtau_vals / Qs
PL_arr = np.empty_like(tau_vals)
PT_arr = np.empty_like(tau_vals)
for i, tau_f in enumerate(tau_vals):
    PL_arr[i], PT_arr[i] = pressures_no_nug(f0, tau_f)
scale = tau_vals**(4.0 / 3.0) / Qs**(8.0 / 3.0)

pl_points = np.loadtxt("kurkela_zhu_PL_points.txt", delimiter=",", skiprows=1)
pt_points = np.loadtxt("kurkela_zhu_PT_points.txt", delimiter=",", skiprows=1)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.loglog(Qtau_vals, PT_arr * scale, "--", color="blue", lw=1.75, label=r"$P_T$, collisionless")
ax.loglog(Qtau_vals, PL_arr * scale, "--", color="C1", lw=1.75, label=r"$P_L$, collisionless")
ax.loglog(pt_points[:, 0], pt_points[:, 1], "^", color="blue", markersize=8,
          markeredgecolor="white", markeredgewidth=0.6, linestyle="none",
          label=r"$P_T$, EKT (Kurkela & Zhu 2015)")
ax.loglog(pl_points[:, 0], pl_points[:, 1], "s", color="C1", markersize=4.5,
          markeredgecolor="white", markeredgewidth=0.6, linestyle="none",
          label=r"$P_L$, EKT (Kurkela & Zhu 2015)")
ax.set_xlabel(r"$Q_s\tau$")
ax.set_ylabel(r"$\tau^{4/3}P/Q_s^{8/3}$")
ax.grid(alpha=0.25, which="both", linewidth=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig("check_kurkela_digitized.png", dpi=150, bbox_inches="tight")
