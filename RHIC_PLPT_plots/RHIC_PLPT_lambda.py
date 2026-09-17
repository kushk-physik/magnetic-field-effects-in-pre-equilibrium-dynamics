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
xi = 10.0
Nc = 3.0
dA = Nc**2 - 1.0
nu_g = 2.0 * dA
c_ell = 1.25
pT_mean = 1.8 * Qs
mq = 0.100
n0 = 1.674
e_abs = 0.30282212
qe_abs = e_abs / 3.0
CF = 4.0 / 3.0
m_pi2 = 0.13957**2
REG2 = 1.0e-24

def make_coupling(lam):
    alpha_s = lam / (4.0 * np.pi * Nc)
    gs = np.sqrt(4.0 * np.pi * alpha_s)
    return dict(lam=lam, alpha_s=alpha_s, qc_abs=gs * np.sqrt(CF))

# RHIC broken-power-law B(tau)/mpi^2 = C0 (tau/tauB)^k / [1 + (tau/tauB)^(k+m)]
RHIC = {
    "0-20":  dict(AT_fm2=120.0, dETdy=573.0, C0=5.5322, tauB_fm=0.08270, k=0.5182, m=3.7744),
    "20-40": dict(AT_fm2=75.0,  dETdy=257.0, C0=8.9788, tauB_fm=0.08663, k=0.7112, m=3.7538),
    "60-80": dict(AT_fm2=21.0,  dETdy=33.0,  C0=13.2165, tauB_fm=0.08632, k=0.9409, m=3.6869),
}
for _c in RHIC.values():
    _c["AT"] = _c["AT_fm2"] * FM_TO_GEV**2
    _c["tauB"] = _c["tauB_fm"] * FM_TO_GEV

def B_of_tau(tau, case):
    if case is None:
        return 0.0
    r = max(tau / case["tauB"], 1.0e-10)
    return m_pi2 * case["C0"] * r**case["k"] / (1.0 + r**(case["k"] + case["m"]))

def eps_parallel(p_tau, B, coupling):
    qeB = qe_abs * B
    return 1.0 + (3.0 * n0 * coupling["alpha_s"] / (8.0 * np.pi)) * (qeB / mq**2) * np.exp(-p_tau**2 / (2.0 * qeB))

def a_B(tau, p_tau, case, coupling):
    if case is None:
        return np.zeros_like(p_tau)
    B = B_of_tau(tau, case)
    if B == 0.0:
        return np.zeros_like(p_tau)
    eps = eps_parallel(p_tau, B, coupling)
    pref = (coupling["qc_abs"] * qe_abs / mq**2) * B * np.sqrt((2.0 / tau0) * case["dETdy"]) / (case["AT"] * Qs)
    return pref * eps**0.75

def p_tau_of(px, py, peta, tau):
    return np.sqrt(px**2 + py**2 + (peta / tau)**2 + REG2)

def build_grid(N, pxmax, pymax, petamax):
    pxe = np.linspace(-pxmax, pxmax, N + 1); px = 0.5 * (pxe[:-1] + pxe[1:])
    pye = np.linspace(-pymax, pymax, N + 1); py = 0.5 * (pye[:-1] + pye[1:])
    pee = np.linspace(-petamax, petamax, N + 1); peta = 0.5 * (pee[:-1] + pee[1:])
    return px, py, peta, px[1] - px[0], py[1] - py[0], peta[1] - peta[0]

def _rhs(tau, y, PY_flat, case, coupling):
    n = PY_flat.size
    px, peta = y[:n], y[n:]
    ptau = np.sqrt(px**2 + PY_flat**2 + (peta / tau)**2 + REG2)
    pperp = np.sqrt(px**2 + (peta / tau)**2 + REG2)
    aB = a_B(tau, ptau, case, coupling)
    dpx = aB * PY_flat * peta / (tau * ptau * pperp)
    dpeta = -tau * aB * PY_flat * px / (ptau * pperp)
    return np.concatenate([dpx, dpeta])

def backward_trace(tau_f, PX, PY, PETA, case, coupling):
    if case is None or tau_f <= tau0:
        return PX.copy(), PETA.copy()
    shape = PX.shape
    y0 = np.concatenate([PX.ravel(), PETA.ravel()])
    sol = solve_ivp(_rhs, [tau_f, tau0], y0, args=(PY.ravel(), case, coupling),
                     method="RK45", rtol=1.0e-7, atol=1.0e-10)
    yf = sol.y[:, -1]
    n = PX.size
    return yf[:n].reshape(shape), yf[n:].reshape(shape)

def kurkela_shape(px, py, peta, tau0_, xi_val=xi):
    pT = np.sqrt(px**2 + py**2)
    pz0 = -peta / tau0_
    rhat2 = (pT / pT_mean)**2 + (xi_val * pz0 / pT_mean)**2
    rhat = np.sqrt(rhat2 + 1.0e-30)
    return np.exp(-(2.0 / 3.0) * rhat2) / rhat

def energy_integral(shape_vals, PX, PY, PETA, dpx, dpy, dpeta, tau_):
    ptau = p_tau_of(PX, PY, PETA, tau_)
    return nu_g * np.sum(ptau * shape_vals) * dpx * dpy * (dpeta / tau_) / (2.0 * np.pi)**3

def normalize_amplitude(shape_vals, PX, PY, PETA, dpx, dpy, dpeta, coupling):
    target = pT_mean * c_ell * dA * Qs**2 / (np.pi * coupling["lam"])
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

coupling10 = make_coupling(10.0)
coupling1 = make_coupling(1.0)
shape0 = kurkela_shape(PX, PY, PETA, tau0)
A10 = normalize_amplitude(shape0, PX, PY, PETA, dpx, dpy, dpeta, coupling10)
A1 = normalize_amplitude(shape0, PX, PY, PETA, dpx, dpy, dpeta, coupling1)

def evolve(case, coupling, A, tau_vals):
    PL_arr = np.empty_like(tau_vals)
    PT_arr = np.empty_like(tau_vals)
    for i, tau_f in enumerate(tau_vals):
        px0, peta0 = backward_trace(tau_f, PX, PY, PETA, case, coupling)
        fvals = A * kurkela_shape(px0, PY, peta0, tau0)
        PL_arr[i], PT_arr[i] = pressures_and_number(fvals, PX, PY, PETA, tau_f, dpx, dpy, dpeta)
    return PL_arr, PT_arr

tau_fm_vals = np.linspace(tau0_fm, 0.5, 80)
tau_vals = tau_fm_vals * FM_TO_GEV
PL_B0, PT_B0 = evolve(None, coupling10, A10, tau_vals)

fig, axes = plt.subplots(1, len(RHIC), figsize=(5 * len(RHIC), 5.5), squeeze=False)
axes = axes.ravel()
for ax, (cent_name, case) in zip(axes, RHIC.items()):
    PL10, PT10 = evolve(case, coupling10, A10, tau_vals)
    PL1, PT1 = evolve(case, coupling1, A1, tau_vals)

    ax.semilogy(tau_fm_vals, PL_B0 / PT_B0, "C0-", linewidth=2, label="$B=0$")
    ax.semilogy(tau_fm_vals, PL10 / PT10, "r-", linewidth=2, label=r"$B\neq0$, $\lambda=10$")
    ax.semilogy(tau_fm_vals, PL1 / PT1, "r--", linewidth=2, label=r"$B\neq0$, $\lambda=1$")
    ax.set_xlabel(r"$\tau$ [fm]")
    ax.set_ylabel(r"$P_L/P_T$")
    ax.grid(alpha=0.3, which="both")
    main_legend = ax.legend(fontsize=11, loc="lower left")
    ax.add_artist(main_legend)

    # B_peak scanned over the plotted tau range, not just its value at tau0
    tau_scan = np.linspace(tau0, tau_vals[-1], 2000)
    B_peak_GeV2 = max(B_of_tau(t, case) for t in tau_scan)
    info_text = (f"Centrality: {cent_name}%\n"
                 fr"$B_{{\rm peak}}$ = {B_peak_GeV2:.4f} GeV$^2$ ({B_peak_GeV2/m_pi2:.3f} $m_\pi^2$)")
    ax.legend([plt.Line2D([], [], linestyle="none")], [info_text],
              loc="upper right", fontsize=11, handlelength=0, handletextpad=0, framealpha=0.85)

fig.tight_layout()
fig.savefig("RHIC_PLPT_lambda.png", dpi=150, bbox_inches="tight")
