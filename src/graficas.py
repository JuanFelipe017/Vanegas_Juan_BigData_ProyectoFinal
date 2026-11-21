"""Gráficas estáticas (matplotlib) para el notebook y el reporte PDF.

Principios de la Actividad 7 aplicados en todas:
- **Tipo según la tarea**: líneas para evolución temporal, barras ordenadas para comparar categorías, histograma
  para distribuciones, mapa de calor secuencial para dos dimensiones, diagrama de calibración para relaciones.
- **Paleta**: los tres resultados usan siempre el mismo color (Local verde aguamarina, Empate naranja, Visitante
  azul), validados como seguros para daltonismo (ΔE CVD ≥ 9,2 en todos los pares). El verde queda bajo 3:1 de
  contraste sobre blanco, por eso las series llevan **etiqueta directa** además de la leyenda.
- **Alto data-ink ratio**: sin marcos superiores/derechos, rejilla horizontal tenue, sin 3D ni sombras.
- **Títulos con el hallazgo** (no "Gráfico de X") y ejes en español con unidades.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

COLORES = {"Local": "#1baf7a", "Empate": "#eb6834", "Visitante": "#2a78d6"}
NEUTRO = "#8a8984"
TINTA = "#1d1d1b"
TINTA_2 = "#52514e"
ACENTO = "#2a78d6"
SECUENCIAL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
FUENTE = "Fuente: Football-Data.co.uk, ClubElo y Transfermarkt · Elaboración propia"


def estilo() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 30, "axes.labelcolor": TINTA_2, "axes.labelsize": 10,
        "axes.edgecolor": "#c9c8c2", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#ecebe7", "grid.linewidth": 0.8,
        "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
    })


def _cerrar(fig, nombre: str, subtitulo: str | None = None) -> Path:
    if subtitulo:
        fig.axes[0].text(0, 1.025, subtitulo, transform=fig.axes[0].transAxes, fontsize=9.5, color=TINTA_2, va="bottom")
    fig.text(0.01, 0.005, FUENTE, fontsize=7.5, color=NEUTRO, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ruta = C.FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight")
    return ruta


def num(x: float, dec: int = 1) -> str:
    """Formato numérico en español (coma decimal)."""
    return f"{x:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ==================================================================================================
# EDA
# ==================================================================================================
def ventaja_local(m: pd.DataFrame):
    """Evolución temporal: % de victorias local / empate / visitante en las 5 grandes ligas."""
    grandes = m[m["liga_codigo"].isin(["E0", "SP1", "I1", "D1", "F1"]) & (m["temporada_inicio"] <= 2025)]
    t = pd.crosstab(grandes["temporada_inicio"], grandes["resultado"], normalize="index")[C.ORDEN_RESULTADOS] * 100
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for r in C.ORDEN_RESULTADOS:
        ax.plot(t.index, t[r], color=COLORES[r], marker="o", markersize=4)
        ax.text(t.index[-1] + 0.4, t[r].iloc[-1], f"{r} {num(t[r].iloc[-1], 0)} %", color=TINTA, va="center", fontsize=9)
    ax.axvspan(2019.6, 2020.6, color="#f0efec", zorder=0)
    covid = t.loc[2020]
    ax.annotate(f"2020/21: estadios vacíos\nlocal {num(covid['Local'], 0)} %", xy=(2020, covid["Local"]),
                xytext=(2014.3, 36.5), fontsize=8.5, color=TINTA_2, arrowprops={"arrowstyle": "-", "color": NEUTRO})
    ax.set_xlim(t.index.min() - 0.5, t.index.max() + 3.2)
    ax.set_ylim(15, 52)
    ax.set_xlabel("Temporada (año de inicio)")
    ax.set_ylabel("% de partidos")
    ax.set_title(f"La ventaja de jugar en casa se encogió: de {num(t['Local'].iloc[:3].mean(), 0)} % a "
                 f"{num(t['Local'].iloc[-3:].mean(), 0)} % de victorias locales")
    return fig, _cerrar(fig, "01_ventaja_local", "Resultado de los partidos en Premier, LaLiga, Serie A, Bundesliga y Ligue 1")


def distribucion_goles(m: pd.DataFrame):
    """Distribución: goles por partido vs. lo que predice una Poisson con la misma media."""
    from math import exp, factorial
    tot = m["total_goles"].astype(int).clip(upper=9)
    frec = tot.value_counts(normalize=True).sort_index() * 100
    lam = m["total_goles"].mean()
    poisson = [exp(-lam) * lam ** k / factorial(k) * 100 for k in frec.index]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(frec.index, frec.values, color=ACENTO, width=0.72, label="Observado")
    ax.plot(frec.index, poisson, color=TINTA, marker="o", markersize=5, linewidth=1.4, label=f"Poisson (λ = {num(lam, 2)})")
    for x, y in frec.items():
        ax.text(x, y + 0.5, f"{num(y, 1)}", ha="center", fontsize=8, color=TINTA_2)
    ax.set_xticks(frec.index, [str(i) if i < 9 else "9+" for i in frec.index])
    ax.set_xlabel("Goles totales en el partido")
    ax.set_ylabel("% de partidos")
    ax.legend(loc="upper right")
    moda = int(frec.idxmax())
    ax.set_title(f"El marcador típico tiene {moda} goles y los goles siguen casi una Poisson")
    return fig, _cerrar(fig, "02_distribucion_goles", f"{num(len(m), 0)} partidos, 38 ligas, 2000–2026")


def goles_por_liga(m: pd.DataFrame):
    """Comparación por categoría: goles por partido en primeras divisiones (barras horizontales ordenadas)."""
    primera = m[m["nivel"] == 1]
    t = primera.groupby("liga", observed=True)["total_goles"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8.5, 6.6))
    colores = [ACENTO if i >= len(t) - 3 else "#9ec5f4" for i in range(len(t))]
    ax.barh(t.index, t.values, color=colores, height=0.7)
    for i, v in enumerate(t.values):
        ax.text(v + 0.02, i, num(v, 2), va="center", fontsize=8, color=TINTA_2)
    ax.axvline(primera["total_goles"].mean(), color=NEUTRO, linewidth=1, linestyle="--")
    ax.text(primera["total_goles"].mean(), len(t) - 0.2, " promedio", fontsize=8, color=TINTA_2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(2.0, t.max() + 0.35)
    ax.set_xlabel("Goles promedio por partido")
    ax.set_title(f"{t.index[-1]} es la liga más goleadora ({num(t.iloc[-1], 2)}) y {t.index[0]} la más cerrada ({num(t.iloc[0], 2)})")
    return fig, _cerrar(fig, "03_goles_por_liga", "Primeras divisiones, todas las temporadas disponibles")


def calibracion_casa(m: pd.DataFrame):
    """Relación entre variables: probabilidad implícita de la casa vs. frecuencia real (diagrama de calibración)."""
    d = m.dropna(subset=["prob_casa_local"])
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot([0, 1], [0, 1], color=NEUTRO, linewidth=1, linestyle="--")
    ax.text(0.80, 0.86, "calibración\nperfecta", fontsize=8, color=NEUTRO, rotation=0)
    for r in C.ORDEN_RESULTADOS:
        p = d[f"prob_casa_{r.lower()}"]
        bins = pd.cut(p, np.linspace(0, 1, 21))
        t = pd.DataFrame({"p": p, "real": (d["resultado"] == r)}).groupby(bins, observed=True).agg(p=("p", "mean"), real=("real", "mean"), n=("p", "size"))
        t = t[t["n"] >= 200]
        ax.plot(t["p"], t["real"], color=COLORES[r], marker="o", markersize=4, label=r)
        ax.text(t["p"].iloc[-1] + 0.015, t["real"].iloc[-1], r, color=TINTA, fontsize=9, va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Probabilidad implícita de Bet365 (sin margen)")
    ax.set_ylabel("Frecuencia real del resultado")
    ax.legend(loc="upper left")
    ax.set_title("Las casas de apuestas están casi perfectamente calibradas")
    return fig, _cerrar(fig, "04_calibracion_casa", "Cada punto agrupa partidos con probabilidad implícita similar (≥ 200 partidos)")


def margen_casa(m: pd.DataFrame):
    """Evolución temporal del margen de la casa (overround)."""
    d = m[(m["nivel"] == 1) & m["margen_casa"].notna() & (m["temporada_inicio"] <= 2025)]
    europa = d[d["continente"] == "Europa"].groupby("temporada_inicio")["margen_casa"].mean() * 100
    america = d[d["continente"] != "Europa"].groupby("temporada_inicio")["margen_casa"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.plot(europa.index, europa.values, color=ACENTO, marker="o", markersize=4)
    ax.text(europa.index[-1] + 0.3, europa.iloc[-1], "Primeras europeas", fontsize=9, va="center")
    ax.plot(america.index, america.values, color="#eb6834", marker="o", markersize=4)
    ax.text(america.index[-1] + 0.3, america.iloc[-1], "América y Asia", fontsize=9, va="center")
    ax.set_xlim(europa.index.min() - 0.5, europa.index.max() + 4.5)
    ax.set_ylim(0, max(europa.max(), america.max()) * 1.15)
    ax.set_xlabel("Temporada (año de inicio)")
    ax.set_ylabel("Margen promedio de Bet365 (%)")
    ax.set_title(f"El margen de la casa cayó de {num(europa.iloc[0], 1)} % a {num(europa.iloc[-1], 1)} % en Europa")
    return fig, _cerrar(fig, "05_margen_casa", "Margen = Σ(1/cuota) − 1: lo que la casa se queda en promedio por unidad apostada")


def valor_mercado(m: pd.DataFrame):
    """Relación: diferencia de valor de plantilla vs. resultado (barras apiladas al 100 % por decil)."""
    d = m.dropna(subset=["log_ratio_valor"]).copy()
    d["ratio"] = np.exp(d["log_ratio_valor"])
    d["decil"] = pd.qcut(d["log_ratio_valor"], 10, labels=False)
    t = pd.crosstab(d["decil"], d["resultado"], normalize="index")[C.ORDEN_RESULTADOS] * 100
    etiquetas = d.groupby("decil")["ratio"].median()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    abajo = np.zeros(len(t))
    for r in C.ORDEN_RESULTADOS:
        ax.bar(t.index, t[r], bottom=abajo, color=COLORES[r], width=0.78, edgecolor="white", linewidth=2, label=r)
        for x, (b, v) in enumerate(zip(abajo, t[r])):
            if v > 9:
                ax.text(x, b + v / 2, f"{num(v, 0)}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        abajo += t[r].to_numpy()
    ax.set_xticks(t.index, [f"×{num(v, 2) if v < 1 else num(v, 1)}" for v in etiquetas], fontsize=8.5)
    ax.set_xlabel("Valor de plantilla local ÷ visitante (mediana de cada decil)")
    ax.set_ylabel("% de partidos")
    ax.set_ylim(0, 100)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title(f"Con la plantilla más cara, el local gana {num(t['Local'].iloc[-1], 0)} % (vs. {num(t['Local'].iloc[0], 0)} % cuando es la más barata)")
    return fig, _cerrar(fig, "06_valor_mercado", "Partidos con valor de mercado de Transfermarkt para ambos equipos")


def mapa_pais_decada(m: pd.DataFrame):
    """Agregación OLAP país × década: % de victorias locales (mapa de calor secuencial)."""
    d = m[m["nivel"] == 1]
    t = d.pivot_table(index="pais", columns="decada", values="resultado", aggfunc=lambda s: (s == "Local").mean() * 100, observed=True)
    t = t.loc[t.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(7.2, 8.2))
    im = ax.imshow(t.values, cmap=SECUENCIAL, aspect="auto", vmin=35, vmax=55)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, num(v, 0), ha="center", va="center", fontsize=8.5, color="white" if v > 46 else TINTA)
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(t.shape[0]), t.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    barra = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    barra.set_label("% de victorias locales")
    barra.outline.set_visible(False)
    ultima = t.columns[-1]
    bajan = int((t[ultima] < t[t.columns[-2]]).sum())
    ax.set_title(f"{t.index[0]} y {t.index[1]} son las mayores fortalezas locales;\n"
                 f"{bajan} de {int(t[[t.columns[-2], ultima]].notna().all(axis=1).sum())} países bajan en los {ultima}")
    return fig, _cerrar(fig, "07_olap_pais_decada", "Cubo país × década (primeras divisiones)")


# ==================================================================================================
# Resultados del modelo
# ==================================================================================================
def comparacion_modelos(metricas: dict):
    filas = [("Siempre local", metricas["lineas_base"]["Siempre local"]["prueba"]["log_loss"], NEUTRO),
             ("Favorito de la casa", metricas["lineas_base"]["Favorito de la casa"]["prueba"]["log_loss"], NEUTRO)]
    for conjunto, nombre_conj, color in [("sin_cuotas", "sin cuotas", "#9ec5f4"), ("con_cuotas", "con cuotas", ACENTO)]:
        for modelo, r in metricas["resultados"][conjunto].items():
            if not modelo.startswith("_"):
                filas.append((f"{modelo} ({nombre_conj})", r["prueba"]["log_loss"], color))
    t = pd.DataFrame(filas, columns=["modelo", "log_loss", "color"]).sort_values("log_loss", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(t["modelo"], t["log_loss"], color=t["color"], height=0.66)
    for i, v in enumerate(t["log_loss"]):
        ax.text(v + 0.0008, i, num(v, 4), va="center", fontsize=8.5, color=TINTA_2)
    casa = metricas["lineas_base"]["Favorito de la casa"]["prueba"]["log_loss"]
    ax.axvline(casa, color=TINTA, linewidth=1, linestyle="--")
    ax.set_xlim(0.99, t["log_loss"].max() + 0.012)
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("Log-loss en prueba (menor es mejor)")
    mejor = t.iloc[-1]
    ax.set_title(f"Con cuotas, los modelos igualan a la casa (mejor log-loss {num(mejor['log_loss'], 4)} vs. {num(casa, 4)})")
    return fig, _cerrar(fig, "08_comparacion_modelos", "Temporadas 2023/24 en adelante · gris = líneas base")


def matriz_confusion(metricas: dict, conjunto: str = "con_cuotas"):
    mejor = metricas["resultados"][conjunto]["_mejor"]
    mc = np.array(mejor["prueba_con_umbral"]["matriz_confusion"])
    pct = mc / mc.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.imshow(pct, cmap=SECUENCIAL, vmin=0, vmax=80)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{num(pct[i, j], 0)} %\n({mc[i, j]:,})".replace(",", "."), ha="center", va="center",
                    fontsize=9, color="white" if pct[i, j] > 45 else TINTA)
    ax.set_xticks(range(3), C.ORDEN_RESULTADOS); ax.set_yticks(range(3), C.ORDEN_RESULTADOS)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Resultado real")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    rec = mejor["prueba_con_umbral"]["por_clase"]
    ax.set_title(f"El empate es lo más difícil: recall {num(rec['Empate']['recall'] * 100, 0)} %")
    return fig, _cerrar(fig, "09_matriz_confusion", f"{mejor['modelo']} con cuotas · umbral de empate {num(mejor['umbral_empate'], 3)}")


def importancia(metricas: dict, conjunto: str = "sin_cuotas", top: int = 10):
    t = pd.DataFrame(metricas["importancias"][conjunto]).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.barh(t["etiqueta"], t["importancia"], xerr=t["desviacion"], color=ACENTO, height=0.62,
            error_kw={"ecolor": NEUTRO, "elinewidth": 1})
    ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    ax.set_xlabel("Aumento del log-loss al desordenar la variable (importancia por permutación)")
    nombre = "fútbol puro (sin cuotas)" if conjunto == "sin_cuotas" else "con cuotas"
    ax.set_title(f"Lo que más pesa en el modelo {nombre}: {t['etiqueta'].iloc[-1]}")
    return fig, _cerrar(fig, f"10_importancia_{conjunto}", "Muestra de 12.000 partidos de prueba, 3 repeticiones")


def curva_roi(metricas: dict):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    nombres = {"con_cuotas": ("Modelo con cuotas", ACENTO), "sin_cuotas": ("Modelo sin cuotas", "#6250d6"),
               "casa": ("Control: probabilidades de la casa", NEUTRO)}
    for conjunto, (etiqueta, color) in nombres.items():
        if conjunto not in metricas["simulacion"]:
            continue
        t = pd.DataFrame(metricas["simulacion"][conjunto]["curva"])
        t = t[t["apuestas"] >= 100]
        ax.plot(t["umbral_valor"] * 100, t["roi"] * 100, color=color, marker="o", markersize=4)
        ax.text(t["umbral_valor"].iloc[-1] * 100 + 0.4, t["roi"].iloc[-1] * 100, etiqueta, fontsize=8.5, va="center")
    ax.axhline(0, color=TINTA, linewidth=1)
    ax.set_xlim(-0.5, 42)
    ax.set_xlabel("Umbral de valor esperado para apostar (%)")
    ax.set_ylabel("ROI (%)")
    sim = metricas["simulacion"]["con_cuotas"]["umbral_5pct"]
    ax.set_title(f"Apostando con valor ≥ 5 %, el modelo con cuotas rinde {num(sim['roi'] * 100, 1)} % en {num(sim['apuestas'], 0)} apuestas")
    return fig, _cerrar(fig, "11_curva_roi", "Cuotas máximas del mercado · 1 unidad por apuesta · temporadas de prueba")


def generar_todas(m: pd.DataFrame | None = None, metricas: dict | None = None) -> list[dict]:
    """Genera las 11 figuras y devuelve su descripción (se guarda en MongoDB como visualizaciones_generadas)."""
    estilo()
    m = pd.read_parquet(C.PARQUET_PARTIDOS) if m is None else m
    metricas = json.loads(C.JSON_MODELO.read_text(encoding="utf-8")) if metricas is None else metricas
    especificacion = [
        (ventaja_local, (m,), "Evolución temporal", "Líneas", "Ventaja local por temporada en las 5 grandes ligas"),
        (distribucion_goles, (m,), "Distribución", "Histograma + curva", "Goles por partido vs. Poisson"),
        (goles_por_liga, (m,), "Comparación por categoría", "Barras horizontales ordenadas", "Goles por partido por liga"),
        (calibracion_casa, (m,), "Relación entre variables", "Diagrama de calibración", "Probabilidad implícita vs. frecuencia real"),
        (margen_casa, (m,), "Evolución temporal", "Líneas", "Margen de la casa por temporada"),
        (valor_mercado, (m,), "Relación entre variables", "Barras apiladas al 100 %", "Valor de plantilla vs. resultado"),
        (mapa_pais_decada, (m,), "Agregación OLAP", "Mapa de calor secuencial", "País × década: % de victorias locales"),
        (comparacion_modelos, (metricas,), "Resultados del modelo", "Barras horizontales", "Log-loss de modelos y líneas base"),
        (matriz_confusion, (metricas,), "Resultados del modelo", "Mapa de calor", "Matriz de confusión del mejor modelo"),
        (importancia, (metricas,), "Resultados del modelo", "Barras con error", "Importancia de variables (sin cuotas)"),
        (curva_roi, (metricas,), "Resultados del modelo", "Líneas", "ROI de la estrategia de apuestas por umbral"),
    ]
    salida = []
    for funcion, args, objetivo, tipo, descripcion in especificacion:
        fig, ruta = funcion(*args)
        salida.append({"archivo": f"reports/figuras/{ruta.name}", "titulo": fig.axes[0].get_title(loc="left"), "objetivo": objetivo,
                       "tipo": tipo, "descripcion": descripcion})
        plt.close(fig)
    return salida


if __name__ == "__main__":
    for v in generar_todas():
        print(v["archivo"], "->", v["titulo"])
