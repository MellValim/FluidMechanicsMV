import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

D_DUTO = 0.009
R = D_DUTO / 2.0
T = 300.0
Q = 0.9375e-3 / 60.0
U = Q / (np.pi * R**2)
DIAMETROS_NM = [5.0, 20.0, 40.0]
L_MAX = 3.0
N_PART = 60000
SEED = 2026

KB = 1.380649e-23
P_ATM = 101325.0
M_AR = 0.02897
R_GAS = 8.314462618

MU_AR = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
RHO_AR = P_ATM * M_AR / (R_GAS * T)
C_MEDIA = np.sqrt(8 * R_GAS * T / (np.pi * M_AR))
LAMBDA = MU_AR / (0.499 * RHO_AR * C_MEDIA)


def cunningham(d):
    kn = 2.0 * LAMBDA / d
    return 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))


def difusividade(d):
    return KB * T * cunningham(d) / (3.0 * np.pi * MU_AR * d)


def penetracao_GK(mu):
    mu = np.asarray(mu, dtype=float)
    p_baixo = 1 - 2.56 * mu ** (2 / 3) + 1.2 * mu + 0.177 * mu ** (4 / 3)
    p_alto = (0.819 * np.exp(-3.657 * mu) + 0.0975 * np.exp(-22.3 * mu)
              + 0.0325 * np.exp(-57.0 * mu))
    return np.where(mu < 0.02, p_baixo, p_alto)


def eficiencia_GK(D, L):
    mu = np.pi * D * L / Q
    return 1.0 - penetracao_GK(mu)


def simular(d, n, l_max, rng):
    D = difusividade(d)
    s = 1.0 - np.sqrt(1.0 - rng.random(n))
    r0 = R * np.sqrt(s)
    th = 2 * np.pi * rng.random(n)
    x = r0 * np.cos(th)
    y = r0 * np.sin(th)
    z = np.zeros(n)
    z_dep = np.full(n, np.inf)

    sig_min = 1e-3 * R
    dt_min = sig_min**2 / (2 * D)
    dt_max = 0.05
    ativos = np.arange(n)

    while ativos.size:
        xa, ya, za = x[ativos], y[ativos], z[ativos]
        r = np.hypot(xa, ya)
        dist = R - r
        dt = (np.maximum(dist, 0.0) / 3.0) ** 2 / (2 * D)
        dt = np.clip(dt, dt_min, dt_max)
        sig = np.sqrt(2 * D * dt)
        u = 2 * U * (1 - (r / R) ** 2)

        xa = xa + sig * rng.standard_normal(xa.size)
        ya = ya + sig * rng.standard_normal(ya.size)
        za = za + u * dt

        depositou = np.hypot(xa, ya) >= R
        saiu = za >= l_max
        z_dep[ativos[depositou]] = za[depositou]

        x[ativos], y[ativos], z[ativos] = xa, ya, za
        ativos = ativos[~(depositou | saiu)]
    return z_dep


def animar(d_nm, L=1.0, n=400, t_inj=4.0, t_tot=14.0, dt=0.02, skip=7, seed=1):
    d = d_nm * 1e-9
    D = difusividade(d)
    rng = np.random.default_rng(seed)
    s = 1.0 - np.sqrt(1.0 - rng.random(n))
    r0 = R * np.sqrt(s)
    th = 2 * np.pi * rng.random(n)
    x, y, z = r0 * np.cos(th), r0 * np.sin(th), np.zeros(n)
    t_ini = np.sort(rng.random(n) * t_inj)
    estado = np.zeros(n, dtype=int)
    sig = np.sqrt(2 * D * dt)

    quadros = []
    n_pass = int(round(t_tot / dt))
    for k in range(n_pass + 1):
        t = k * dt
        if k % skip == 0:
            quadros.append((t, x.copy(), y.copy(), z.copy(), estado.copy(), t >= t_ini))
        act = (estado == 0) & (t >= t_ini)
        r = np.hypot(x[act], y[act])
        u = 2 * U * (1 - (r / R) ** 2)
        x[act] += sig * rng.standard_normal(act.sum())
        y[act] += sig * rng.standard_normal(act.sum())
        z[act] += u * dt
        rn = np.hypot(x, y)
        dep = act & (rn >= R)
        x[dep] *= R / rn[dep]
        y[dep] *= R / rn[dep]
        estado[dep] = 1
        estado[act & ~dep & (z >= L)] = 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [3.2, 1]})
    ax1.set_xlim(0, L); ax1.set_ylim(-1.25 * R * 1e3, 1.25 * R * 1e3)
    ax1.axhline(R * 1e3, color="k", lw=2); ax1.axhline(-R * 1e3, color="k", lw=2)
    ax1.set_xlabel("z [m]"); ax1.set_ylabel("y [mm]")
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.2 * R * 1e3, 1.2 * R * 1e3); ax2.set_ylim(-1.2 * R * 1e3, 1.2 * R * 1e3)
    ax2.add_patch(plt.Circle((0, 0), R * 1e3, fill=False, lw=2))
    ax2.set_xlabel("x [mm]"); ax2.set_ylabel("y [mm]"); ax2.set_title("Seção transversal", fontsize=10)
    sc1 = ax1.scatter([], [], s=9)
    sc2 = ax2.scatter([], [], s=9)
    titulo = ax1.set_title("")
    cores = np.array(["tab:blue", "tab:red", "tab:green"])

    def atualizar(i):
        t, xx, yy, zz, st, vis = quadros[i]
        m = vis & (st != 2)
        sc1.set_offsets(np.c_[zz[m], yy[m] * 1e3]); sc1.set_color(cores[st[m]])
        sc2.set_offsets(np.c_[xx[m] * 1e3, yy[m] * 1e3]); sc2.set_color(cores[st[m]])
        nd, ns = int((st == 1).sum()), int((st == 2).sum())
        eta = 100 * nd / max(nd + ns, 1)
        titulo.set_text(f"d = {d_nm:g} nm | t = {t:5.2f} s | depositadas: {nd} | "
                        f"saíram: {ns} | η parcial = {eta:4.1f}%")
        return sc1, sc2

    ani = FuncAnimation(fig, atualizar, frames=len(quadros), blit=False)
    arq = f"animacao_{d_nm:g}nm.gif"
    ani.save(arq, writer=PillowWriter(fps=15), dpi=70)
    plt.close(fig)
    nd, ns = int((estado == 1).sum()), int((estado == 2).sum())
    print(f"{arq}: {nd} depositadas, {ns} saíram, {n - nd - ns} no duto")


def main():
    print(f"U = {U:.4f} m/s | Re = {RHO_AR * U * D_DUTO / MU_AR:.1f} | mu_ar = {MU_AR:.3e} Pa.s | "
          f"lambda = {LAMBDA * 1e9:.1f} nm")
    rng = np.random.default_rng(SEED)
    Ls = np.linspace(0.0, L_MAX, 91)
    res = {}
    for dnm in DIAMETROS_NM:
        d = dnm * 1e-9
        D = difusividade(d)
        z_dep = simular(d, N_PART, L_MAX, rng)
        eta_sim = np.array([np.mean(z_dep <= L) for L in Ls])
        eta_ana = eficiencia_GK(D, Ls)
        res[dnm] = (D, eta_sim, eta_ana)
        i1 = np.argmin(abs(Ls - 1.0))
        print(f"d = {dnm:4.0f} nm | Cc = {cunningham(d):6.2f} | D = {D:.3e} m²/s | "
              f"Pe = {U * R / D:.2e} | η(L=1 m): sim = {100 * eta_sim[i1]:5.2f}% | "
              f"ana = {100 * eta_ana[i1]:5.2f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    for (dnm, (D, es, ea)), c in zip(res.items(), ["tab:red", "tab:green", "tab:blue"]):
        ax.plot(Ls, 100 * ea, "-", color=c, label=f"{dnm:g} nm - analítica")
        ax.plot(Ls[::3], 100 * es[::3], "o", color=c, ms=5, mfc="none", label=f"{dnm:g} nm - Lagrangeana")
    ax.set_xlabel("Comprimento do duto L [m]"); ax.set_ylabel("Eficiência de deposição η [%]")
    ax.set_title("Deposição de nanogotas - duto D = 9 mm, Q = 0,9375 L/min, T = 300 K")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_xlim(0, L_MAX); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig("graficos_eficiencia.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    mu_ref = np.linspace(0, 0.3, 300)
    ax.plot(mu_ref, 100 * (1 - penetracao_GK(mu_ref)), "k-", label="Gormley-Kennedy")
    for (dnm, (D, es, ea)), c in zip(res.items(), ["tab:red", "tab:green", "tab:blue"]):
        mu = np.pi * D * Ls / Q
        ax.plot(mu[::2], 100 * es[::2], "o", color=c, ms=5, mfc="none", label=f"{dnm:g} nm - Lagrangeana")
    ax.set_xlim(0, 0.3)
    ax.set_xlabel(r"$\mu = \pi D L / Q$"); ax.set_ylabel("Eficiência de deposição η [%]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig("grafico_eficiencia_mu.png", dpi=150); plt.close(fig)

    for dnm in DIAMETROS_NM:
        animar(dnm)


if __name__ == "__main__":
    main()