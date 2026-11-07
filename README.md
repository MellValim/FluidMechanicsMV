# Trabalho 2: Deposição de nanopartículas em um escoamento laminar dado

Cálculo da eficiência de deposição de gotas de água com diâmetro de **5, 20 e 40 nm** em escoamento laminar completamente desenvolvido em um duto de seção circular, usando a **abordagem Lagrangeana** (passeio aleatório browniano).

## Dados do problema

| Parâmetro | Valor |
|---|---|
| Diâmetro do duto | 0,009 m |
| Gás | Ar, pressão atmosférica |
| Temperatura | 300 K |
| Vazão | 0,9375 L/min |
| Velocidade média | 0,246 m/s |
| Reynolds | ≈ 141 (laminar) |

## Modelo

- **Escoamento:** perfil de Poiseuille, `u(r) = 2U(1 − r²/R²)`, com `U = Q/(πR²)`.
- **Movimento das gotas:** advecção pelo escoamento mais passeio aleatório browniano, `Δx = √(2DΔt)·N(0,1)`.
- **Difusividade:** Stokes-Einstein com correção de Cunningham, `D = kT·Cc / (3πμd)`.
- **Parede:** absorvedora; a gota que atinge `r ≥ R` é considerada depositada.
- **Entrada:** concentração uniforme (gotas entram proporcionalmente à vazão local).
- **Passo de tempo:** adaptativo, menor perto da parede.
- **Validação:** solução analítica de Gormley-Kennedy (série de Graetz).

## Resultados (L = 1 m)

| d (nm) | D (m²/s) | η Lagrangeana | η analítica |
|---|---|---|---|
| 5 | 2,15×10⁻⁷ | 25,5 % | 26,1 % |
| 20 | 1,40×10⁻⁸ | 4,6 % | 4,8 % |
| 40 | 3,68×10⁻⁹ | 2,0 % | 2,0 % |

Quanto menor a gota, maior a difusividade e maior a deposição.

## Arquivos

- `deposicao_lagrangeana.py`: código-fonte.
- `graficos_eficiencia.png`: eficiência × comprimento do duto.
- `grafico_eficiencia_mu.png`: eficiência × parâmetro adimensional `μ = πDL/Q`.
- `animacao_5nm.gif`, `animacao_20nm.gif`, `animacao_40nm.gif`: animações do movimento das gotas.

## Como executar

```bash
pip install numpy matplotlib pillow
python deposicao_lagrangeana.py
```

O script imprime os resultados no terminal e gera os gráficos e as animações na pasta atual.

## Observações

- O enunciado não informa o comprimento do duto. As curvas variam de 0 a 3 m e as animações usam L = 1 m. Para alterar, edite `L_MAX` e o parâmetro `L` da função `animar`.
- Consideradas apenas difusão browniana (sem gravidade, eletrostática ou termoforese) e gotas esféricas.
