#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAMADA DE MISTURA PLANA TURBULENTA: SIMULAÇÃO x EXPERIMENTO
============================================================
Exercício de Escoamentos Turbulentos:
  1) simular uma camada de mistura;
  2) buscar dados experimentais de referência;
  3) comparar resultados computacionais e experimentais.

SIMULAÇÃO (item 1)
  Equações RANS na aproximação de camada fina (parabólicas), integradas
  espacialmente em x (marcha), com o modelo k-epsilon padrão (Launder-Spalding).
  Solução por volumes finitos em y e esquema implícito em x.

EXPERIMENTO (itens 2 e 3)
  Delville, Bellin, Garem & Bonnet (1989): camada de mistura plana,
  U1 = 41,54 m/s (rápida), U2 = 22,40 m/s (lenta), r = U2/U1 = 0,539.
  Dados obtidos do NASA Turbulence Modeling Resource (TMR, caso 2DML),
  também disponíveis no ERCOFTAC Classic Collection (caso C.34).
  Os arquivos são baixados automaticamente do repositório GitHub do TMR.

Uso:   python camada_de_mistura.py
Requer: numpy, scipy, matplotlib
"""
import os
import urllib.request

import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARÂMETROS DO CASO (fonte: página 2DML do NASA TMR)
# =============================================================================
U1 = 41.54            # m/s, corrente rápida (y > 0)
U2 = 22.40            # m/s, corrente lenta  (y < 0)
DU = U1 - U2          # m/s
NU = 41.54 * 1.0e-3 / 2900.0   # m²/s, de Re = 2900 com L = 1 mm e U_ref = 41,54 m/s
DELTA1, THETA1 = 9.6e-3, 1.0e-3     # camada limite superior em x = -10 mm (m)
DELTA2, THETA2 = 6.3e-3, 0.73e-3    # camada limite inferior em x = -10 mm (m)
TU_INF = 0.003        # intensidade de turbulência da corrente livre (0,3 %)
NUT_NU_INF = 10.0     # razão nu_t/nu na corrente livre (hipótese numérica)

# Constantes do modelo k-epsilon padrão (não foram ajustadas)
CMU, C1E, C2E, SIGK, SIGE, KAPPA = 0.09, 1.44, 1.92, 1.0, 1.3, 0.41

# Malha e marcha
H = 0.15              # m, meia-altura do domínio (túnel tem 300 mm)
NY = 400              # células em y
BETA = 3.5            # concentração de pontos em torno de y = 0
X_FIM = 1.0e3 * 1e-3  # m (1000 mm)
DX0, DX_MAX, DX_CRESC = 1.0e-5, 5.0e-4, 1.03   # m
ESTACOES_MM = [1, 50, 200, 650, 950]           # onde há dados experimentais

URL_BASE = ("https://raw.githubusercontent.com/TMBWG/turbmodels/main/"
            "Delvilleshear_grids/")
ARQUIVOS = ["delville_exp_u.dat", "delville_exp_turb.dat",
            "delville_exp_delomega.dat"]
PASTA_DADOS = "dados_delville"


# =============================================================================
# 2. DADOS EXPERIMENTAIS: download e leitura
# =============================================================================
def baixar_dados(pasta=PASTA_DADOS):
    """Baixa os arquivos do TMR (se ainda não existirem localmente)."""
    os.makedirs(pasta, exist_ok=True)
    for nome in ARQUIVOS:
        destino = os.path.join(pasta, nome)
        if os.path.exists(destino):
            continue
        print(f"Baixando {nome} ...")
        try:
            urllib.request.urlretrieve(URL_BASE + nome, destino)
        except Exception as erro:
            raise SystemExit(
                f"Não consegui baixar {nome} ({erro}).\n"
                f"Baixe manualmente em {URL_BASE}{nome}\n"
                f"e coloque na pasta '{pasta}/'.")
    return pasta


def ler_zonas(caminho):
    """Lê arquivo formato Tecplot com zonas 'ZONE T="x =200mm"'.
    Retorna {x_mm: array(linhas, colunas)}."""
    zonas, atual = {}, None
    with open(caminho) as f:
        for linha in f:
            s = linha.strip()
            if not s or s.startswith("#") or s.upper().startswith("VARIABLES"):
                continue
            if s.upper().startswith("ZONE"):
                atual = float(s.split("=")[-1].replace("mm", "").replace('"', ""))
                zonas[atual] = []
            else:
                zonas[atual].append([float(v) for v in s.split()])
    return {x: np.array(v) for x, v in zonas.items()}


def ler_delomega(caminho):
    """Espessura de vorticidade experimental: colunas x (mm), delta_omega (mm)."""
    dados = []
    with open(caminho) as f:
        for linha in f:
            s = linha.strip()
            if not s or s.startswith("#") or s.lower().startswith("variables"):
                continue
            dados.append([float(v) for v in s.split()[:2]])
    return np.array(dados)


def centro_da_camada(y, s):
    """Posição y onde s = (U-U2)/DU = 0,5 (ajuste linear local em 0,3 < s < 0,7)."""
    m = (s > 0.3) & (s < 0.7)
    a, b = np.polyfit(s[m], y[m], 1)
    return a * 0.5 + b


# =============================================================================
# 3. SIMULAÇÃO: RANS parabólico com k-epsilon
# =============================================================================
def expoente_potencia(delta, theta):
    """n do perfil U/Ue = (y/delta)^(1/n) que reproduz theta/delta (bisseção)."""
    alvo = theta / delta
    lo, hi = 2.0, 30.0
    for _ in range(80):
        n = 0.5 * (lo + hi)
        if n / ((n + 1.0) * (n + 2.0)) > alvo:
            lo = n
        else:
            hi = n
    return n


def condicao_de_entrada(yc):
    """Perfis em x = 0 (bordo de fuga): duas camadas limites turbulentas
    modeladas por leis de potência com delta e theta do experimento."""
    U = np.empty_like(yc)
    k = np.empty_like(yc)
    eps = np.empty_like(yc)
    for lado, (Ue, delta, theta) in {"sup": (U1, DELTA1, THETA1),
                                     "inf": (U2, DELTA2, THETA2)}.items():
        n = expoente_potencia(delta, theta)
        m = yc > 0 if lado == "sup" else yc <= 0
        d = np.abs(yc[m])
        eta = np.minimum(d / delta, 1.0)
        U[m] = Ue * eta ** (1.0 / n)
        # atrito (Ludwieg-Tillmann) -> u_tau
        Hf = (n + 2.0) / n
        re_th = Ue * theta / NU
        cf = 0.246 * 10 ** (-0.678 * Hf) * re_th ** (-0.268)
        u_tau = Ue * np.sqrt(cf / 2.0)
        k_inf = 1.5 * (TU_INF * Ue) ** 2
        k[m] = np.maximum(k_inf, (u_tau ** 2 / np.sqrt(CMU)) * (1.0 - eta) ** 2)
        comp_mistura = np.minimum(KAPPA * d, 0.09 * delta)
        eps[m] = CMU ** 0.75 * k[m] ** 1.5 / comp_mistura
        eps_inf = CMU * k_inf ** 2 / (NUT_NU_INF * NU)
        eps[m] = np.maximum(eps[m], eps_inf)
    return U, k, eps


def resolver_transporte(phi_old, Uc, Vc, gam, dy, dyc, dx, Sc, Sp, phi_baixo, phi_cima):
    """Um passo em x da equação Uc*dphi/dx + Vc*dphi/dy = d/dy(gam dphi/dy) + Sc - Sp*phi.
    Convecção em y contra o vento; contornos de Dirichlet nas extremidades."""
    N = len(phi_old)
    gf = 0.5 * (gam[1:] + gam[:-1])               # difusividade nas faces internas
    aW = np.zeros(N)
    aE = np.zeros(N)
    aW[1:] = gf / (dy[1:] * dyc) + np.maximum(Vc[1:], 0.0) / dyc
    aE[:-1] = gf / (dy[:-1] * dyc) + np.maximum(-Vc[:-1], 0.0) / dyc
    diag = Uc / dx + aW + aE + Sp
    rhs = Uc / dx * phi_old + Sc
    # contornos
    diag[0], diag[-1] = 1.0, 1.0
    aE[0], aW[-1] = 0.0, 0.0
    aW[0], aE[-1] = 0.0, 0.0
    rhs[0], rhs[-1] = phi_baixo, phi_cima
    ab = np.zeros((3, N))
    ab[0, 1:] = -aE[:-1]      # diagonal superior
    ab[1, :] = diag
    ab[2, :-1] = -aW[1:]      # diagonal inferior
    return solve_banded((1, 1), ab, rhs)


def simular():
    """Marcha em x. Retorna histórico de delta_omega(x) e perfis nas estações."""
    # malha em y (concentrada em torno de y = 0)
    s = np.linspace(-1.0, 1.0, NY + 1)
    yf = H * np.sinh(BETA * s) / np.sinh(BETA)
    yc = 0.5 * (yf[1:] + yf[:-1])
    dy = np.diff(yf)
    dyc = np.diff(yc)

    U, k, eps = condicao_de_entrada(yc)
    V = np.zeros(NY)
    k_baixo, k_cima = 1.5 * (TU_INF * U2) ** 2, 1.5 * (TU_INF * U1) ** 2
    e_baixo = CMU * k_baixo ** 2 / (NUT_NU_INF * NU)
    e_cima = CMU * k_cima ** 2 / (NUT_NU_INF * NU)

    hist_x, hist_dw, hist_ymid = [], [], []
    perfis = {}
    estacoes = sorted(e * 1e-3 for e in ESTACOES_MM)
    x, dx, ie = 0.0, DX0, 0

    while x < X_FIM - 1e-12:
        dx = min(dx * DX_CRESC, DX_MAX)
        prox = estacoes[ie] if ie < len(estacoes) else X_FIM
        if x + dx >= prox:
            dx = prox - x
        U_old, k_old, e_old = U.copy(), k.copy(), eps.copy()
        Uc = np.maximum(U_old, 0.05)

        Un, kn, en = U_old.copy(), k_old.copy(), e_old.copy()
        for _ in range(2):                                  # 2 iterações de Picard
            nut = CMU * kn ** 2 / en
            nut = np.minimum(nut, 1.0e4 * NU)
            Un = resolver_transporte(U_old, Uc, V, NU + nut, dy, dyc, dx,
                                     np.zeros(NY), np.zeros(NY), U2, U1)
            # continuidade: V a partir de dU/dx
            dUdx = (Un - U_old) / dx
            Vf = np.concatenate(([0.0], np.cumsum(-dUdx * dy)))
            V = 0.5 * (Vf[1:] + Vf[:-1])
            # k e epsilon
            dUdy = np.gradient(Un, yc)
            P = nut * dUdy ** 2
            kk = np.maximum(k_old, 1e-10)
            ee = np.maximum(e_old, 1e-8)
            kn = resolver_transporte(k_old, Uc, V, NU + nut / SIGK, dy, dyc, dx,
                                     P, ee / kk, k_baixo, k_cima)
            kn = np.maximum(kn, 1e-10)
            en = resolver_transporte(e_old, Uc, V, NU + nut / SIGE, dy, dyc, dx,
                                     C1E * ee / kk * P, C2E * ee / kk, e_baixo, e_cima)
            en = np.maximum(en, 1e-8)
        U, k, eps = Un, kn, en
        x += dx

        dUdy = np.gradient(U, yc)
        dw = DU / np.max(dUdy)
        ymid = np.interp(0.5, (U - U2) / DU, yc)
        hist_x.append(x * 1e3)
        hist_dw.append(dw * 1e3)
        hist_ymid.append(ymid * 1e3)

        if ie < len(estacoes) and abs(x - estacoes[ie]) < 1e-9:
            nut = CMU * k ** 2 / eps
            perfis[round(x * 1e3)] = dict(y=yc * 1e3, U=U.copy(), k=k.copy(),
                                          nut=nut, dUdy=dUdy.copy(), dw=dw * 1e3,
                                          ymid=ymid * 1e3)
            print(f"  x = {x*1e3:7.1f} mm | delta_omega = {dw*1e3:6.2f} mm "
                  f"| centro em y = {ymid*1e3:5.2f} mm")
            ie += 1
    return np.array(hist_x), np.array(hist_dw), perfis


# =============================================================================
# 4. COMPARAÇÃO SIMULAÇÃO x EXPERIMENTO
# =============================================================================
def taxa_de_espalhamento(x, dw, x_ini=400.0, x_fim=1000.0):
    m = (x >= x_ini) & (x <= x_fim)
    return np.polyfit(x[m], dw[m], 1)[0]


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def comparar(hist_x, hist_dw, perfis, pasta):
    zu = ler_zonas(os.path.join(pasta, "delville_exp_u.dat"))
    zt = ler_zonas(os.path.join(pasta, "delville_exp_turb.dat"))
    dwe = ler_delomega(os.path.join(pasta, "delville_exp_delomega.dat"))

    cores = {200: "tab:blue", 650: "tab:green", 950: "tab:red"}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle("Camada de mistura plana: k-ε parabólico (linhas) × experimento "
                 "de Delville, NASA TMR (símbolos)\n"
                 f"U1 = {U1} m/s, U2 = {U2} m/s, r = {U2/U1:.3f}", fontsize=13)

    # (a) espessura de vorticidade
    a = ax[0, 0]
    a.plot(dwe[:, 0], dwe[:, 1], "ko", ms=4, label="Experimento")
    a.plot(hist_x[hist_x > 5], hist_dw[hist_x > 5], "r-", label="Simulação k-ε")
    a.set_xlabel("x (mm)"); a.set_ylabel(r"$\delta_\omega$ (mm)")
    a.set_title("(a) Espessura de vorticidade"); a.legend(); a.grid(alpha=.3)

    # (b) perfis dimensionais perto da placa
    a = ax[0, 1]
    for x, c in [(1, "tab:purple"), (50, "tab:orange")]:
        d = zu[float(x)]
        a.plot(d[::3, 2], d[::3, 1], "o", ms=3, color=c, label=f"exp x = {x} mm")
        p = perfis[x]
        a.plot(p["U"], p["y"], "-", color=c, label=f"sim x = {x} mm")
    a.set_ylim(-25, 25)
    a.set_xlabel("U (m/s)"); a.set_ylabel("y (mm)")
    a.set_title("(b) Região próxima à placa"); a.legend(fontsize=8); a.grid(alpha=.3)

    # (c) perfil médio normalizado
    a = ax[0, 2]
    metricas = {}
    for x in (200, 650, 950):
        d = zu[float(x)]
        # colunas: 0=x, 1=y_corrigido (mm), 2=U, 3=y/delta_w, 4=(U-U_lenta)/DU
        y_e, s_e = d[:, 1], d[:, 4]
        ym_e = centro_da_camada(y_e, s_e)
        dw_e = np.interp(x, dwe[:, 0], dwe[:, 1])
        eta_e = (y_e - ym_e) / dw_e
        p = perfis[x]
        s_s = (p["U"] - U2) / DU
        eta_s = (p["y"] - p["ymid"]) / p["dw"]
        a.plot(eta_e[::2], s_e[::2], "o", ms=3, color=cores[x], label=f"exp x = {x}")
        a.plot(eta_s, s_s, "-", color=cores[x], label=f"sim x = {x}")
        m = np.abs(eta_e) < 2
        metricas[x] = dict(erro_U=rms(s_e[m], np.interp(eta_e[m], eta_s, s_s)),
                           dw_e=dw_e, dw_s=p["dw"])
    a.set_xlim(-2.5, 2.5)
    a.set_xlabel(r"$(y-y_{1/2})/\delta_\omega$"); a.set_ylabel(r"$(U-U_2)/\Delta U$")
    a.set_title("(c) Velocidade média (autossimilaridade)")
    a.legend(fontsize=8, ncol=2); a.grid(alpha=.3)

    # (d) tensão de Reynolds e (e) energia cinética turbulenta
    for j, (titulo, chave) in enumerate([("(d) Tensão de Reynolds", "uv"),
                                         ("(e) Energia cinética turbulenta", "k")]):
        a = ax[1, j]
        for x in (200, 650, 950):
            d = zt[float(x)]
            y_e = d[:, 1]
            du_ = zu[float(x)]
            ym_e = centro_da_camada(du_[:, 1], du_[:, 4])
            dw_e = np.interp(x, dwe[:, 0], dwe[:, 1])
            eta_e = (y_e - ym_e) / dw_e
            p = perfis[x]
            eta_s = (p["y"] - p["ymid"]) / p["dw"]
            if chave == "uv":
                exp_v = d[:, 4]                                        # u'v'/DU²
                sim_v = -p["nut"] * p["dUdy"] / DU ** 2
            else:
                exp_v = 0.5 * (d[:, 6] + d[:, 8] + d[:, 10])           # k/DU²
                sim_v = p["k"] / DU ** 2
            a.plot(eta_e, exp_v, "o", ms=3, color=cores[x], label=f"exp x = {x}")
            a.plot(eta_s, sim_v, "-", color=cores[x], label=f"sim x = {x}")
            m = np.abs(eta_e) < 1.5
            pico_e = exp_v[m].min() if chave == "uv" else exp_v[m].max()
            pico_s = sim_v.min() if chave == "uv" else sim_v.max()
            metricas[x]["pico_" + chave] = (pico_e, pico_s)
        a.set_xlim(-2.5, 2.5)
        a.set_xlabel(r"$(y-y_{1/2})/\delta_\omega$")
        a.set_ylabel(r"$\langle u'v'\rangle/\Delta U^2$" if chave == "uv"
                     else r"$k/\Delta U^2$")
        a.set_title(titulo); a.legend(fontsize=8, ncol=2); a.grid(alpha=.3)

    # (f) parâmetro de estrutura -u'v'/k
    a = ax[1, 2]
    for x in (650, 950):
        d = zt[float(x)]
        du_ = zu[float(x)]
        ym_e = centro_da_camada(du_[:, 1], du_[:, 4])
        dw_e = np.interp(x, dwe[:, 0], dwe[:, 1])
        eta_e = (d[:, 1] - ym_e) / dw_e
        k_e = 0.5 * (d[:, 6] + d[:, 8] + d[:, 10])
        m = k_e > 0.15 * k_e.max()
        a.plot(eta_e[m], -d[m, 4] / k_e[m], "o", ms=3, color=cores[x], label=f"exp x = {x}")
        p = perfis[x]
        eta_s = (p["y"] - p["ymid"]) / p["dw"]
        ms_ = p["k"] > 0.15 * p["k"].max()
        a.plot(eta_s[ms_], (p["nut"] * p["dUdy"] / p["k"])[ms_], "-", color=cores[x],
               label=f"sim x = {x}")
    a.axhline(np.sqrt(CMU), color="gray", ls="--", lw=1, label=r"$\sqrt{C_\mu}=0{,}3$")
    a.set_xlim(-1.5, 1.5); a.set_ylim(0, 0.6)
    a.set_xlabel(r"$(y-y_{1/2})/\delta_\omega$"); a.set_ylabel(r"$-\langle u'v'\rangle/k$")
    a.set_title("(f) Parâmetro de estrutura"); a.legend(fontsize=8); a.grid(alpha=.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("comparacao_camada_mistura.png", dpi=150)

    # ---------------- resumo numérico ----------------
    S_exp = taxa_de_espalhamento(dwe[:, 0], dwe[:, 1])
    S_sim = taxa_de_espalhamento(hist_x, hist_dw)
    lam = DU / (U1 + U2)
    print("\n================ RESUMO DA COMPARAÇÃO ================")
    print(f"Taxa de espalhamento d(delta_omega)/dx  (400 <= x <= 1000 mm)")
    print(f"   experimento: {S_exp:.4f}  (= {S_exp/lam:.3f} * lambda)")
    print(f"   simulação  : {S_sim:.4f}  (= {S_sim/lam:.3f} * lambda)")
    print(f"   diferença  : {100*(S_sim-S_exp)/S_exp:+.1f} %")
    print("\n  x(mm) | delta_w exp | delta_w sim | RMS[(U-U2)/DU] | pico u'v'/DU² exp / sim"
          " | pico k/DU² exp / sim")
    for x in (200, 650, 950):
        m = metricas[x]
        print(f"  {x:5d} | {m['dw_e']:9.2f}   | {m['dw_s']:9.2f}   | {m['erro_U']:12.4f}   "
              f"| {m['pico_uv'][0]:9.4f} / {m['pico_uv'][1]:8.4f}   "
              f"| {m['pico_k'][0]:8.4f} / {m['pico_k'][1]:7.4f}")
    print("\nFigura salva em: comparacao_camada_mistura.png")


# =============================================================================
if __name__ == "__main__":
    print("Passo 1: dados experimentais (NASA TMR / Delville)")
    pasta = baixar_dados()
    print("\nPasso 2: simulação RANS k-epsilon parabólica")
    hx, hdw, perfis = simular()
    print("\nPasso 3: comparação com o experimento")
    comparar(hx, hdw, perfis, pasta)
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()
