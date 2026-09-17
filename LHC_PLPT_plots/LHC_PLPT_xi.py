import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 14
rcParams['mathtext.fontset'] = 'cm'

FM_TO_GEV = 5.067731

Qs = 2.0
tau0_fm = 0.01
tau0 = tau0_fm * FM_TO_GEV
lam = 10.0
Nc = 3.0
dA = Nc**2 - 1.0
nu_g = 2.0 * dA
c_ell = 1.25
pT_mean = 1.8 * Qs
mq = 0.100
n0 = 1.674
alpha_s = lam / (4.0 * np.pi * Nc)
e_abs = 0.30282212
qe_abs = e_abs / 3.0
CF = 4.0 / 3.0
qc_abs = np.sqrt(4.0 * np.pi * alpha_s) * np.sqrt(CF)
m_pi2 = 0.13957**2
REG2 = 1.0e-24

# LHC 60-80%: B(tau)/mpi^2 = C0 r^-k0 + C1 r^-k1 + C2 r^-k2, r = tau/tauB
case = dict(AT_fm2=23.0, dETdy=1.1e2, C0=7.272e-4, k0=6.829, C1=1.107e-2, k1=4.281, C2=1.978e-2, k2=3.011, tauB_fm=0.05)
case["AT"] = case["AT_fm2"] * FM_TO_GEV**2
case["tauB"] = case["tauB_fm"] * FM_TO_GEV

def B_of_tau(tau, case):
    r = max(tau / case["tauB"], 1.0e-10)
    return m_pi2 * (case["C0"] * r**-case["k0"] + case["C1"] * r**-case["k1"] + case["C2"] * r**-case["k2"])

def eps_parallel(p_tau, B):
    qeB = qe_abs * B
    return 1.0 + (3.0 * n0 * alpha_s / (8.0 * np.pi)) * (qeB / mq**2) * np.exp(-p_tau**2 / (2.0 * qeB))

def a_B(tau, p_tau, case):
    if case is None:
        return np.zeros_like(p_tau)
    B = B_of_tau(tau, case)
    if B == 0.0:
        return np.zeros_like(p_tau)
    eps = eps_parallel(p_tau, B)
    pref = (qc_abs * qe_abs / mq**2) * B * np.sqrt((2.0 / tau0) * case["dETdy"]) / (case["AT"] * Qs)
    return pref * eps**0.75

def p_tau_of(px, py, peta, tau):
    return np.sqrt(px**2 + py**2 + (peta / tau)**2 + REG2)

def build_grid(N, pxmax, pymax, petamax):
    pxe = np.linspace(-pxmax, pxmax, N + 1); px = 0.5 * (pxe[:-1] + pxe[1:])
    pye = np.linspace(-pymax, pymax, N + 1); py = 0.5 * (pye[:-1] + pye[1:])
    pee = np.linspace(-petamax, petamax, N + 1); peta = 0.5 * (pee[:-1] + pee[1:])
    return px, py, peta, px[1] - px[0], py[1] - py[0], peta[1] - peta[0]

def _rhs(tau, y, PY_flat, case):
    n = PY_flat.size
    px, peta = y[:n], y[n:]
    ptau = np.sqrt(px**2 + PY_flat**2 + (peta / tau)**2 + REG2)
    pperp = np.sqrt(px**2 + (peta / tau)**2 + REG2)
    aB = a_B(tau, ptau, case)
    dpx = aB * PY_flat * peta / (tau * ptau * pperp)
    dpeta = -tau * aB * PY_flat * px / (ptau * pperp)
    return np.concatenate([dpx, dpeta])

def backward_trace(tau_f, PX, PY, PETA, case):
    if case is None or tau_f <= tau0:
        return PX.copy(), PETA.copy()
    shape = PX.shape
    y0 = np.concatenate([PX.ravel(), PETA.ravel()])
    sol = solve_ivp(_rhs, [tau_f, tau0], y0, args=(PY.ravel(), case),
                     method="RK45", rtol=1.0e-7, atol=1.0e-10)
    yf = sol.y[:, -1]
    n = PX.size
    return yf[:n].reshape(shape), yf[n:].reshape(shape)

def kurkela_shape(px, py, peta, tau0_, xi_val):
    pT = np.sqrt(px**2 + py**2)
    pz0 = -peta / tau0_
    rhat2 = (pT / pT_mean)**2 + (xi_val * pz0 / pT_mean)**2
    rhat = np.sqrt(rhat2 + 1.0e-30)
    return np.exp(-(2.0 / 3.0) * rhat2) / rhat

def energy_integral(shape_vals, PX, PY, PETA, dpx, dpy, dpeta, tau_):
    ptau = p_tau_of(PX, PY, PETA, tau_)
    return nu_g * np.sum(ptau * shape_vals) * dpx * dpy * (dpeta / tau_) / (2.0 * np.pi)**3

def normalize_amplitude(shape_vals, PX, PY, PETA, dpx, dpy, dpeta):
    target = pT_mean * c_ell * dA * Qs**2 / (np.pi * lam)
    IE = energy_integral(shape_vals, PX, PY, PETA, dpx, dpy, dpeta, tau0)
    return target / (tau0 * IE)

def pressures_and_number(fvals, PX, PY, PETA, tau_, dpx, dpy, dpeta):
    pz = PETA / tau_
    ptau = p_tau_of(PX, PY, PETA, tau_)
    w = dpx * dpy * (dpeta / tau_)
    PL = nu_g * np.sum((pz**2 / ptau) * fvals) * w / (2.0 * np.pi)**3
    PT = 0.5 * nu_g * np.sum(((PX**2 + PY**2) / ptau) * fvals) * w / (2.0 * np.pi)**3
    return PL, PT

N = 90
PXMAX = PYMAX = 16.0
PETAMAX = 18.0 * tau0
px, py, peta, dpx, dpy, dpeta = build_grid(N, PXMAX, PYMAX, PETAMAX)
PX, PY, PETA = np.meshgrid(px, py, peta, indexing="ij")

def evolve(case, A, xi_val, tau_vals):
    PL_arr = np.empty_like(tau_vals)
    PT_arr = np.empty_like(tau_vals)
    for i, tau_f in enumerate(tau_vals):
        px0, peta0 = backward_trace(tau_f, PX, PY, PETA, case)
        fvals = A * kurkela_shape(px0, PY, peta0, tau0, xi_val)
        PL_arr[i], PT_arr[i] = pressures_and_number(fvals, PX, PY, PETA, tau_f, dpx, dpy, dpeta)
    return PL_arr, PT_arr

tau_fm_vals = np.linspace(tau0_fm, 0.5, 80)
tau_vals = tau_fm_vals * FM_TO_GEV
xi_values = (10.0, 4.0, 1.0)
linestyles = {10.0: "-", 4.0: "--", 1.0: ":"}

fig, ax = plt.subplots(figsize=(7, 6))
for xi_val in xi_values:
    shape0 = kurkela_shape(PX, PY, PETA, tau0, xi_val)
    A_xi = normalize_amplitude(shape0, PX, PY, PETA, dpx, dpy, dpeta)
    PL_B0, PT_B0 = evolve(None, A_xi, xi_val, tau_vals)
    PL_on, PT_on = evolve(case, A_xi, xi_val, tau_vals)

    ls = linestyles[xi_val]
    ax.semilogy(tau_fm_vals, PL_B0 / PT_B0, ls, color="C0", linewidth=2, label=fr"$B=0$, $\xi={xi_val:g}$")
    ax.semilogy(tau_fm_vals, PL_on / PT_on, ls, color="r", linewidth=2, label=fr"$B\neq0$, $\xi={xi_val:g}$")

ax.set_xlabel(r"$\tau$ [fm]")
ax.set_ylabel(r"$P_L/P_T$")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=10, loc="upper right")

# LHC's B(tau) is monotonically falling (negative powers), so its peak is at tau0
B_peak_GeV2 = B_of_tau(tau0, case)
info_text = f"Centrality: 60-80%\n" + fr"$B$ = {B_peak_GeV2:.4f} GeV$^2$ ({B_peak_GeV2/m_pi2:.3f} $m_\pi^2$)"
ax.text(0.03, 0.03, info_text, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.6", alpha=0.85))

fig.tight_layout()
fig.savefig("LHC_PLPT_xi.png", dpi=150, bbox_inches="tight")
